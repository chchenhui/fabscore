"""
ArXiv API client for literature search system.

This module provides a robust client for interacting with the ArXiv API,
including rate limiting, error handling, pagination, and XML response parsing.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import urllib.parse
import time
import re
from loguru import logger

from ..models.query_models import SearchQuery, SortOrder
from ..models.literature_models import Paper, Author, ArXivResponse
from ..config.literature_config import get_config
from ..utils.validation_utils import validate_arxiv_id, validate_search_query


class RateLimiter:
    """Rate limiter for ArXiv API requests."""
    
    def __init__(self, rate_limit_seconds: float = 3.0):
        self.rate_limit_seconds = rate_limit_seconds
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.rate_limit_seconds:
                wait_time = self.rate_limit_seconds - time_since_last_request
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class ArXivClient:
    """
    Robust ArXiv API client with intelligent querying capabilities.
    
    Features:
    - Rate limiting compliance with ArXiv usage policy
    - Automatic retry with exponential backoff
    - XML response parsing and data extraction
    - Pagination handling for large result sets
    - Query optimization and validation
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.session = None
        self.rate_limiter = RateLimiter(self.config.arxiv.rate_limit_seconds)
        
        # XML namespaces used by ArXiv API
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.arxiv.timeout_seconds)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'OneSim-Literature-Query/1.0'}
            )
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def search_papers(self, search_query: SearchQuery) -> ArXivResponse:
        """
        Execute search against ArXiv API.
        
        Args:
            search_query: Structured search query
            
        Returns:
            ArXivResponse containing papers and metadata
            
        Raises:
            ValueError: If search query is invalid
            aiohttp.ClientError: If API request fails
        """
        await self._ensure_session()
        
        # Validate search query
        if not search_query.keywords:
            raise ValueError("Search query must contain keywords")
        
        # Build search URL
        query_url = self._build_search_url(search_query)
        
        start_time = time.time()
        
        try:
            # Execute request with rate limiting and retries
            response_text = await self._execute_request_with_retries(query_url)
            
            # Parse XML response
            papers = self._parse_arxiv_response(response_text)
            
            # Extract metadata from response
            total_results, start_index, items_per_page = self._extract_metadata(response_text)
            
            response_time = time.time() - start_time
            
            return ArXivResponse(
                papers=papers,
                total_results=total_results,
                start_index=start_index,
                items_per_page=items_per_page,
                query_url=query_url,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"ArXiv API search failed: {e}")
            raise
    
    def _build_search_url(self, search_query: SearchQuery) -> str:
        """
        Build ArXiv API search URL from structured query.
        
        Args:
            search_query: Structured search parameters
            
        Returns:
            Complete ArXiv API URL
        """
        base_url = self.config.arxiv.base_url
        
        # Build search string
        search_terms = []
        
        # Process keywords based on suggested fields
        if search_query.search_fields:
            for field in search_query.search_fields:
                field_prefix = self.config.get_search_prefix(field)
                
                if field_prefix == "all":
                    # Use keywords as-is for 'all' field
                    for keyword in search_query.keywords[:5]:  # Limit to avoid too long URLs
                        search_terms.append(f'all:"{keyword}"')
                else:
                    # Use field-specific search
                    keyword_str = " OR ".join(f'"{kw}"' for kw in search_query.keywords[:3])
                    search_terms.append(f'{field_prefix}:({keyword_str})')
        else:
            # Default: search in title and abstract
            title_terms = " OR ".join(f'"{kw}"' for kw in search_query.keywords[:3])
            abs_terms = " OR ".join(f'"{kw}"' for kw in search_query.keywords[3:6])
            
            search_terms.append(f'ti:({title_terms})')
            if abs_terms:
                search_terms.append(f'abs:({abs_terms})')
        
        # Add category filters if specified
        if search_query.domain_categories:
            category_terms = " OR ".join(search_query.domain_categories)
            search_terms.append(f'cat:({category_terms})')
        
        # Combine search terms
        search_string = " AND ".join(search_terms)
        
        # Build query parameters
        params = {
            'search_query': search_string,
            'start': 0,  # Always start from 0, pagination handled separately
            'max_results': min(search_query.max_results, self.config.arxiv.max_results_per_query),
            'sortBy': search_query.sort_by.value,
            'sortOrder': search_query.sort_order
        }
        
        # Add URL encoding
        encoded_params = urllib.parse.urlencode(params)
        return f"{base_url}?{encoded_params}"
    
    async def _execute_request_with_retries(self, url: str) -> str:
        """
        Execute HTTP request with rate limiting and retry logic.
        
        Args:
            url: URL to request
            
        Returns:
            Response text
            
        Raises:
            aiohttp.ClientError: If all retries fail
        """
        for attempt in range(self.config.arxiv.max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed()
                
                # Make request
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
                    
                    # Check for ArXiv API errors in response
                    if self._is_error_response(content):
                        error_msg = self._extract_error_message(content)
                        raise aiohttp.ClientError(f"ArXiv API error: {error_msg}")
                    
                    return content
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.arxiv.max_retries:
                    # Exponential backoff
                    wait_time = self.config.arxiv.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _is_error_response(self, content: str) -> bool:
        """Check if response contains an error."""
        return 'id>http://arxiv.org/api/errors#' in content
    
    def _extract_error_message(self, content: str) -> str:
        """Extract error message from ArXiv error response."""
        try:
            root = ET.fromstring(content)
            for entry in root.findall('.//atom:entry', self.namespaces):
                title = entry.find('atom:title', self.namespaces)
                summary = entry.find('atom:summary', self.namespaces)
                
                if title is not None and title.text == "Error":
                    return summary.text if summary is not None else "Unknown error"
        except ET.ParseError:
            pass
        
        return "Unknown ArXiv API error"
    
    def _extract_metadata(self, content: str) -> Tuple[int, int, int]:
        """
        Extract metadata from ArXiv response.
        
        Args:
            content: XML response content
            
        Returns:
            Tuple of (total_results, start_index, items_per_page)
        """
        try:
            root = ET.fromstring(content)
            
            total_results = 0
            start_index = 0
            items_per_page = 0
            
            # Extract OpenSearch metadata
            total_elem = root.find('.//opensearch:totalResults', self.namespaces)
            if total_elem is not None:
                total_results = int(total_elem.text)
            
            start_elem = root.find('.//opensearch:startIndex', self.namespaces)
            if start_elem is not None:
                start_index = int(start_elem.text)
            
            items_elem = root.find('.//opensearch:itemsPerPage', self.namespaces)
            if items_elem is not None:
                items_per_page = int(items_elem.text)
            
            return total_results, start_index, items_per_page
            
        except (ET.ParseError, ValueError) as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return 0, 0, 0
    
    def _parse_arxiv_response(self, content: str) -> List[Paper]:
        """
        Parse ArXiv XML response into Paper objects.
        
        Args:
            content: XML response content
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        try:
            root = ET.fromstring(content)
            
            # Find all entry elements (each represents a paper)
            entries = root.findall('.//atom:entry', self.namespaces)
            
            for entry in entries:
                try:
                    paper = self._parse_paper_entry(entry)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper entry: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv response XML: {e}")
            raise
        
        return papers
    
    def _parse_paper_entry(self, entry: ET.Element) -> Optional[Paper]:
        """
        Parse individual paper entry from XML.
        
        Args:
            entry: XML entry element
            
        Returns:
            Paper object or None if parsing fails
        """
        try:
            # Extract basic fields
            title_elem = entry.find('atom:title', self.namespaces)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            if not title:
                return None
            
            # Extract ArXiv ID from entry ID
            id_elem = entry.find('atom:id', self.namespaces)
            if id_elem is None:
                return None
            
            entry_id = id_elem.text
            arxiv_id = self._extract_arxiv_id_from_url(entry_id)
            
            if not arxiv_id:
                return None
            
            # Extract authors
            authors = self._parse_authors(entry)
            if not authors:
                return None
            
            # Extract summary (abstract)
            summary_elem = entry.find('atom:summary', self.namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            if not abstract:
                logger.warning(f"Paper {arxiv_id} has no abstract")
                abstract = "No abstract available"
            
            # Extract dates
            published_elem = entry.find('atom:published', self.namespaces)
            updated_elem = entry.find('atom:updated', self.namespaces)
            
            published_date = self._parse_datetime(published_elem.text) if published_elem is not None else datetime.now()
            updated_date = self._parse_datetime(updated_elem.text) if updated_elem is not None else published_date
            
            # Extract categories
            categories, primary_category = self._parse_categories(entry)
            
            # Extract links
            pdf_url, abstract_url, doi_url = self._parse_links(entry, arxiv_id)
            
            # Extract optional fields
            comment_elem = entry.find('arxiv:comment', self.namespaces)
            comment = comment_elem.text.strip() if comment_elem is not None else None
            
            journal_elem = entry.find('arxiv:journal_ref', self.namespaces)
            journal_ref = journal_elem.text.strip() if journal_elem is not None else None
            
            doi_elem = entry.find('arxiv:doi', self.namespaces)
            doi = doi_elem.text.strip() if doi_elem is not None else None
            
            return Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                primary_category=primary_category,
                published_date=published_date,
                updated_date=updated_date,
                pdf_url=pdf_url,
                abstract_url=abstract_url,
                doi=doi,
                journal_ref=journal_ref,
                comments=comment
            )
            
        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None
    
    def _extract_arxiv_id_from_url(self, url: str) -> Optional[str]:
        """Extract ArXiv ID from entry URL."""
        if not url:
            return None
        
        # URL format: http://arxiv.org/abs/1234.5678v1
        parts = url.split('/')
        if len(parts) >= 2:
            arxiv_id_with_version = parts[-1]
            
            # Remove version number (e.g., v1, v2, etc.)
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id_with_version)
            
            # Validate the extracted ID
            is_valid, _ = validate_arxiv_id(arxiv_id)
            if is_valid:
                return arxiv_id
        
        return None
    
    def _parse_authors(self, entry: ET.Element) -> List[Author]:
        """Parse author information from entry."""
        authors = []
        
        author_elements = entry.findall('atom:author', self.namespaces)
        
        for author_elem in author_elements:
            name_elem = author_elem.find('atom:name', self.namespaces)
            if name_elem is not None and name_elem.text:
                name = name_elem.text.strip()
                
                # Extract affiliation if available
                affiliation_elem = author_elem.find('arxiv:affiliation', self.namespaces)
                affiliation = affiliation_elem.text.strip() if affiliation_elem is not None else None
                
                authors.append(Author(name=name, affiliation=affiliation))
        
        return authors
    
    def _parse_categories(self, entry: ET.Element) -> Tuple[List[str], str]:
        """Parse category information from entry."""
        categories = []
        primary_category = ""
        
        # Primary category
        primary_elem = entry.find('arxiv:primary_category', self.namespaces)
        if primary_elem is not None:
            primary_category = primary_elem.get('term', '')
            if primary_category:
                categories.append(primary_category)
        
        # All categories
        category_elements = entry.findall('atom:category', self.namespaces)
        for cat_elem in category_elements:
            term = cat_elem.get('term')
            if term and term not in categories:
                categories.append(term)
        
        # Ensure primary category is set
        if not primary_category and categories:
            primary_category = categories[0]
        
        return categories, primary_category
    
    def _parse_links(self, entry: ET.Element, arxiv_id: str) -> Tuple[str, str, Optional[str]]:
        """Parse link information from entry."""
        pdf_url = ""
        abstract_url = ""
        doi_url = None
        
        link_elements = entry.findall('atom:link', self.namespaces)
        
        for link_elem in link_elements:
            href = link_elem.get('href', '')
            rel = link_elem.get('rel', '')
            title = link_elem.get('title', '')
            
            if rel == 'alternate':
                abstract_url = href
            elif rel == 'related' and title == 'pdf':
                pdf_url = href
            elif rel == 'related' and title == 'doi':
                doi_url = href
        
        # Generate default URLs if not provided
        if not abstract_url:
            abstract_url = f"http://arxiv.org/abs/{arxiv_id}"
        
        if not pdf_url:
            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return pdf_url, abstract_url, doi_url
    
    def _parse_datetime(self, date_string: str) -> datetime:
        """Parse datetime string from ArXiv response."""
        try:
            # ArXiv uses ISO format: 2023-01-15T10:30:00Z
            if date_string.endswith('Z'):
                date_string = date_string[:-1] + '+00:00'
            
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Failed to parse date: {date_string}")
            return datetime.now()
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            Paper object or None if not found
        """
        # Validate ArXiv ID
        is_valid, error = validate_arxiv_id(arxiv_id)
        if not is_valid:
            raise ValueError(f"Invalid ArXiv ID: {error}")
        
        # Create search query for specific ID
        search_query = SearchQuery(
            keywords=[arxiv_id],
            max_results=1,
            search_fields=["id"]
        )
        
        try:
            response = await self.search_papers(search_query)
            return response.papers[0] if response.papers else None
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None
    
    async def search_by_author(self, author_name: str, max_results: int = 50) -> List[Paper]:
        """
        Search papers by author name.
        
        Args:
            author_name: Author name to search for
            max_results: Maximum number of results
            
        Returns:
            List of papers by the author
        """
        search_query = SearchQuery(
            keywords=[author_name],
            max_results=max_results,
            search_fields=["authors"]
        )
        
        response = await self.search_papers(search_query)
        return response.papers
    
    async def get_bibtex_by_id(self, arxiv_id: str) -> Optional[str]:
        """
        Get BibTeX entry for a specific ArXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2010.11929")
            
        Returns:
            BibTeX string or None if not found/error
        """
        # Validate ArXiv ID
        is_valid, error = validate_arxiv_id(arxiv_id)
        if not is_valid:
            raise ValueError(f"Invalid ArXiv ID: {error}")
        
        await self._ensure_session()
        
        # ArXiv BibTeX API endpoint
        bibtex_url = f"https://arxiv.org/bibtex/{arxiv_id}"
        
        try:
            # Apply rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Make request
            async with self.session.get(bibtex_url) as response:
                response.raise_for_status()
                bibtex_content = await response.text()
                
                # Check if it's actually BibTeX content
                if bibtex_content.strip().startswith('@'):
                    return bibtex_content.strip()
                else:
                    logger.warning(f"Invalid BibTeX response for {arxiv_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to fetch BibTeX for {arxiv_id}: {e}")
            return None
    
    async def get_bibtex_entries(self, arxiv_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Get BibTeX entries for multiple ArXiv papers.
        
        Args:
            arxiv_ids: List of ArXiv paper IDs
            
        Returns:
            Dictionary mapping arxiv_id to BibTeX string (None if failed)
        """
        results = {}
        
        for arxiv_id in arxiv_ids:
            try:
                bibtex = await self.get_bibtex_by_id(arxiv_id)
                results[arxiv_id] = bibtex
            except Exception as e:
                logger.error(f"Error fetching BibTeX for {arxiv_id}: {e}")
                results[arxiv_id] = None
        
        return results
    
    def extract_bibtex_key(self, bibtex_content: str) -> Optional[str]:
        """
        Extract BibTeX key from BibTeX content.
        
        Args:
            bibtex_content: BibTeX entry content
            
        Returns:
            BibTeX key or None if not found
        """
        import re
        
        # Look for @misc{key, pattern
        match = re.search(r'@\w+\{([^,\s]+)', bibtex_content)
        if match:
            return match.group(1)
        
        return None


# Usage example and testing
if __name__ == "__main__":
    async def main():
        config = get_config()
        
        async with ArXivClient(config) as client:
            # Test search
            search_query = SearchQuery(
                keywords=["machine learning", "transformers"],
                max_results=5,
                search_fields=["title", "abstract"]
            )
            
            try:
                response = await client.search_papers(search_query)
                print(f"Found {len(response.papers)} papers:")
                
                for paper in response.papers[:3]:
                    print(f"\nTitle: {paper.title}")
                    print(f"Authors: {', '.join(paper.author_names)}")
                    print(f"ArXiv ID: {paper.arxiv_id}")
                    print(f"Categories: {', '.join(paper.categories)}")
                    print(f"Abstract: {paper.abstract[:200]}...")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    # Run the example
    asyncio.run(main())