"""
Query processing engine for intelligent literature search orchestration.

This module coordinates keyword extraction, ArXiv searching, result processing,
and relevance ranking to provide comprehensive literature query capabilities.
"""

import asyncio
import time
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
from loguru import logger

from ..models.query_models import (
    QueryOptions, SearchMode, SearchQuery, SearchStrategy, 
    KeywordResult, QueryMetrics
)
from ..models.literature_models import Paper, LiteratureResult
from ..config.literature_config import get_config
from ..utils.text_utils import calculate_text_similarity, clean_text
from .keyword_extractor import KeywordExtractor
from .arxiv_client import ArXivClient


class QueryProcessor:
    """
    Intelligent query orchestration and optimization engine.
    
    Coordinates all aspects of literature query processing including:
    - LLM-powered keyword extraction
    - Multi-strategy ArXiv searching
    - Result fusion and deduplication  
    - Relevance ranking and filtering
    - Performance optimization
    """
    
    def __init__(
        self,
        keyword_extractor: KeywordExtractor = None,
        arxiv_client: ArXivClient = None,
        cache_service = None,
        model_config_name: str = None
    ):
        """
        Initialize query processor.
        
        Args:
            keyword_extractor: LLM-powered keyword extraction engine
            arxiv_client: ArXiv API client
            cache_service: Optional caching service (ignored in this version)
            model_config_name: OneSim model configuration name
        """
        self.keyword_extractor = keyword_extractor or KeywordExtractor(model_config_name)
        self.arxiv_client = arxiv_client or ArXivClient()
        self.cache_service = None  # Caching disabled for simplification
        self.config = get_config()
    
    async def process_literature_query(
        self,
        user_input: str,
        options: QueryOptions
    ) -> LiteratureResult:
        """
        End-to-end query processing pipeline.
        
        Args:
            user_input: Natural language research query
            options: Query processing options and parameters
            
        Returns:
            Complete literature search results
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")
        
        user_input = clean_text(user_input)
        
        # Initialize metrics tracking
        metrics = QueryMetrics(query_start_time=datetime.now())
        
        logger.info(f"Processing literature query: '{user_input}' (mode: {options.mode.value})")
        
        try:
            # Check cache first if enabled
            cache_key = None
            if options.enable_caching and self.cache_service:
                cache_key = self._generate_cache_key(user_input, options)
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result:
                    logger.info("Returning cached result")
                    metrics.cache_hits = 1
                    cached_result.cache_hit = True
                    return cached_result
                else:
                    metrics.cache_misses = 1
            
            # Step 1: Extract keywords using LLM
            extraction_start = time.time()
            keyword_result = await self.keyword_extractor.extract_keywords(
                text=user_input,
                domain_hint=self._infer_domain(user_input),
                mode=options.mode
            )
            metrics.keyword_extraction_time = time.time() - extraction_start
            
            logger.info(f"Extracted {keyword_result.keyword_count} keywords "
                       f"(confidence: {keyword_result.confidence_score:.2f})")
            
            # Step 2: Generate multiple search strategies
            search_strategies = self._generate_search_strategies(
                keyword_result, options
            )
            
            logger.info(f"Generated {len(search_strategies)} search strategies")
            
            # Step 3: Execute parallel searches
            search_start = time.time()
            all_papers = await self._execute_multi_strategy_search(
                search_strategies, options, metrics
            )
            metrics.arxiv_query_time = time.time() - search_start
            
            logger.info(f"Found {len(all_papers)} papers from all strategies")
            
            # Step 4: Deduplicate and merge results
            processing_start = time.time()
            unique_papers = self._deduplicate_papers(all_papers)
            
            logger.info(f"After deduplication: {len(unique_papers)} unique papers")
            
            # Step 5: Calculate relevance scores
            scored_papers = await self._calculate_relevance_scores(
                unique_papers, user_input, keyword_result
            )
            
            # Step 6: Filter and rank results
            filtered_papers = self._filter_and_rank_papers(
                scored_papers, options
            )
            
            metrics.result_processing_time = time.time() - processing_start
            metrics.total_papers_found = len(all_papers)
            metrics.papers_after_deduplication = len(unique_papers)
            metrics.papers_after_filtering = len(filtered_papers)
            
            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                filtered_papers, keyword_result, options
            )
            
            # Create final result
            result = LiteratureResult(
                original_query=user_input,
                extracted_keywords=keyword_result,
                papers=filtered_papers,
                total_found=len(unique_papers),
                search_strategies_used=[s.name for s in search_strategies],
                processing_time=metrics.total_processing_time,
                recommendations=recommendations
            )
            
            # Cache result if enabled
            if options.enable_caching and self.cache_service and cache_key:
                await self._cache_result(cache_key, result)
            
            # Finalize metrics
            metrics.mark_completed()
            logger.info(f"Query processing completed in {metrics.total_processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            metrics.mark_completed()
            raise
    
    def _generate_cache_key(self, user_input: str, options: QueryOptions) -> str:
        """Generate cache key for query and options."""
        import hashlib
        
        # Include key parameters that affect results
        cache_components = [
            user_input.lower().strip(),
            options.mode.value,
            str(options.max_results),
            str(options.relevance_threshold),
            str(options.include_summaries)
        ]
        
        cache_string = "|".join(cache_components)
        return f"lit_query:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[LiteratureResult]:
        """Retrieve cached result if available."""
        if not self.cache_service:
            return None
        
        try:
            return await self.cache_service.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: LiteratureResult) -> None:
        """Cache query result."""
        if not self.cache_service:
            return
        
        try:
            await self.cache_service.set(
                cache_key, 
                result, 
                ttl_hours=self.config.cache.ttl_hours
            )
        except Exception as e:
            logger.warning(f"Result caching failed: {e}")
    
    def _infer_domain(self, user_input: str) -> Optional[str]:
        """Infer research domain from user input."""
        user_input_lower = user_input.lower()
        
        # Simple keyword-based domain inference
        domain_keywords = {
            "machine learning": ["machine learning", "ml", "deep learning", "neural network", "ai"],
            "computer science": ["algorithm", "computation", "programming", "software"],
            "physics": ["quantum", "particle", "wave", "energy", "physics"],
            "mathematics": ["theorem", "proof", "equation", "mathematical", "optimization"],
            "biology": ["protein", "dna", "genome", "molecular", "biological"],
            "finance": ["trading", "portfolio", "risk", "financial", "market"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return domain
        
        return None
    
    def _generate_search_strategies(
        self,
        keyword_result: KeywordResult,
        options: QueryOptions
    ) -> List[SearchStrategy]:
        """Generate multiple complementary search strategies."""
        
        strategies = []
        
        # Strategy 1: Primary keywords with title focus
        if keyword_result.primary_keywords:
            primary_query = " AND ".join(f'"{kw}"' for kw in keyword_result.primary_keywords[:3])
            strategies.append(SearchStrategy(
                name="primary_title_focused",
                query_string=f"ti:({primary_query})",
                search_fields=["title"],
                expected_results=min(options.max_results // 2, 50),
                priority=1.0,
                description="Primary keywords in titles"
            ))
        
        # Strategy 2: Mixed primary/secondary with broader scope
        if keyword_result.primary_keywords and keyword_result.secondary_keywords:
            mixed_terms = keyword_result.primary_keywords[:2] + keyword_result.secondary_keywords[:3]
            mixed_query = " OR ".join(f'"{kw}"' for kw in mixed_terms)
            strategies.append(SearchStrategy(
                name="mixed_comprehensive",
                query_string=f"abs:({mixed_query})",
                search_fields=["abstract"],
                expected_results=min(options.max_results, 100),
                priority=0.8,
                description="Mixed keywords in abstracts"
            ))
        
        # Strategy 3: Secondary keywords for discovery
        if keyword_result.secondary_keywords and options.mode != SearchMode.FOCUSED:
            secondary_query = " OR ".join(f'"{kw}"' for kw in keyword_result.secondary_keywords[:5])
            strategies.append(SearchStrategy(
                name="secondary_discovery",
                query_string=f"all:({secondary_query})",
                search_fields=["all"],
                expected_results=min(options.max_results, 75),
                priority=0.6,
                description="Secondary keywords for broader discovery"
            ))
        
        # Strategy 4: Domain-specific category search
        if keyword_result.domain_category:
            domain_categories = self.config.get_domain_categories(keyword_result.domain_category)
            if domain_categories:
                # Combine category filter with primary keywords
                primary_terms = " OR ".join(f'"{kw}"' for kw in keyword_result.primary_keywords[:4])
                cat_filter = " OR ".join(domain_categories[:3])
                combined_query = f"({primary_terms}) AND cat:({cat_filter})"
                
                strategies.append(SearchStrategy(
                    name="domain_filtered",
                    query_string=combined_query,
                    search_fields=["all", "categories"],
                    expected_results=min(options.max_results, 60),
                    priority=0.9,
                    description=f"Domain-filtered search for {keyword_result.domain_category}"
                ))
        
        # Strategy 5: Recent papers focus (if in exploratory mode)
        if options.mode == SearchMode.EXPLORATORY and keyword_result.primary_keywords:
            recent_query = " OR ".join(f'"{kw}"' for kw in keyword_result.primary_keywords[:3])
            strategies.append(SearchStrategy(
                name="recent_focus",
                query_string=f"all:({recent_query})",
                search_fields=["all"],
                expected_results=min(options.max_results // 2, 30),
                priority=0.7,
                description="Recent papers with primary keywords"
            ))
        
        # Limit number of strategies based on performance settings
        max_strategies = min(len(strategies), options.parallel_strategies)
        
        # Sort by priority and take top strategies
        strategies.sort(key=lambda s: s.priority, reverse=True)
        return strategies[:max_strategies]
    
    async def _execute_multi_strategy_search(
        self,
        strategies: List[SearchStrategy],
        options: QueryOptions,
        metrics: QueryMetrics
    ) -> List[Paper]:
        """Execute multiple search strategies in parallel."""
        
        # Create search tasks
        tasks = []
        for strategy in strategies:
            search_query = SearchQuery(
                keywords=self._extract_keywords_from_query(strategy.query_string),
                max_results=strategy.expected_results,
                search_fields=strategy.search_fields
            )
            
            task = self._execute_single_search(strategy, search_query, metrics)
            tasks.append(task)
        
        # Execute searches with concurrency control
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_requests)
        
        async def bounded_search(task):
            async with semaphore:
                return await task
        
        # Execute all searches
        search_results = await asyncio.gather(
            *[bounded_search(task) for task in tasks],
            return_exceptions=True
        )
        
        # Combine results from all strategies
        all_papers = []
        successful_strategies = 0
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy '{strategies[i].name}' failed: {result}")
                metrics.failed_requests += 1
            else:
                all_papers.extend(result)
                successful_strategies += 1
                metrics.successful_requests += 1
        
        logger.info(f"Completed {successful_strategies}/{len(strategies)} search strategies")
        
        return all_papers
    
    async def _execute_single_search(
        self,
        strategy: SearchStrategy,
        search_query: SearchQuery,
        metrics: QueryMetrics
    ) -> List[Paper]:
        """Execute a single search strategy."""
        
        try:
            logger.debug(f"Executing strategy: {strategy.name}")
            
            response = await self.arxiv_client.search_papers(search_query)
            
            metrics.total_arxiv_requests += 1
            
            logger.debug(f"Strategy '{strategy.name}' returned {len(response.papers)} papers")
            
            return response.papers
            
        except Exception as e:
            logger.error(f"Search strategy '{strategy.name}' failed: {e}")
            raise
    
    def _extract_keywords_from_query(self, query_string: str) -> List[str]:
        """Extract keywords from ArXiv query string for SearchQuery."""
        import re
        
        # Simple extraction of quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query_string)
        
        if quoted_terms:
            return quoted_terms
        
        # Fallback: extract terms after field prefixes
        field_terms = re.findall(r'(?:ti|abs|all|cat):\(([^)]*)\)', query_string)
        if field_terms:
            # Split by OR/AND and clean
            terms = []
            for term_group in field_terms:
                parts = re.split(r'\s+(?:OR|AND)\s+', term_group)
                terms.extend([t.strip(' "') for t in parts])
            return terms
        
        # Ultimate fallback
        return ["research"]
    
    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on ArXiv ID."""
        
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)
        
        return unique_papers
    
    async def _calculate_relevance_scores(
        self,
        papers: List[Paper],
        original_query: str,
        keyword_result: KeywordResult
    ) -> List[Paper]:
        """Calculate relevance scores for papers."""
        
        logger.info("Calculating relevance scores...")
        
        # Prepare query text for comparison
        query_lower = original_query.lower()
        all_keywords = keyword_result.all_keywords
        keyword_text = " ".join(all_keywords).lower()
        
        for paper in papers:
            # Multiple relevance factors
            title_similarity = calculate_text_similarity(
                query_lower, paper.title.lower()
            )
            
            abstract_similarity = calculate_text_similarity(
                keyword_text, paper.abstract.lower()
            )
            
            # Keyword matching score
            matched_keywords = []
            paper_text = (paper.title + " " + paper.abstract).lower()
            
            for keyword in all_keywords:
                if keyword.lower() in paper_text:
                    matched_keywords.append(keyword)
            
            keyword_match_score = len(matched_keywords) / len(all_keywords) if all_keywords else 0
            
            # Category relevance (if domain specified)
            category_score = 0.0
            if keyword_result.domain_category:
                relevant_categories = self.config.get_domain_categories(keyword_result.domain_category)
                if any(cat in paper.categories for cat in relevant_categories):
                    category_score = 0.2
            
            # Recent papers get slight boost
            recency_score = 0.1 if paper.is_recent(90) else 0.0
            
            # Combine scores with weights
            relevance_score = (
                title_similarity * 0.3 +
                abstract_similarity * 0.3 +
                keyword_match_score * 0.25 +
                category_score +
                recency_score
            )
            
            paper.relevance_score = min(1.0, relevance_score)
            paper.keywords_matched = matched_keywords
        
        return papers
    
    def _filter_and_rank_papers(
        self,
        papers: List[Paper],
        options: QueryOptions
    ) -> List[Paper]:
        """Filter and rank papers based on options."""
        
        # Filter by relevance threshold
        filtered_papers = [
            paper for paper in papers 
            if paper.relevance_score >= options.relevance_threshold
        ]
        
        # Filter by abstract length if specified
        if options.min_abstract_length:
            filtered_papers = [
                paper for paper in filtered_papers
                if paper.abstract_word_count >= options.min_abstract_length
            ]
        
        # Filter by age if specified
        if options.max_paper_age_days:
            filtered_papers = [
                paper for paper in filtered_papers
                if paper.is_recent(options.max_paper_age_days)
            ]
        
        # Filter by preferred categories if specified
        if options.preferred_categories:
            filtered_papers = [
                paper for paper in filtered_papers
                if any(cat in paper.categories for cat in options.preferred_categories)
            ]
        
        # Sort by relevance score (descending)
        filtered_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        # Limit to max results
        return filtered_papers[:options.max_results]
    
    def _generate_recommendations(
        self,
        papers: List[Paper],
        keyword_result: KeywordResult,
        options: QueryOptions
    ) -> List[str]:
        """Generate search recommendations based on results."""
        
        recommendations = []
        
        if not papers:
            recommendations.append("No papers found. Try using broader keywords or different search terms.")
            if keyword_result.secondary_keywords:
                recommendations.append(f"Consider searching for: {', '.join(keyword_result.secondary_keywords[:3])}")
        
        elif len(papers) < 5:
            recommendations.append("Few results found. Consider broadening your search.")
            recommendations.append("Try using more general terms or including related concepts.")
        
        else:
            # Analyze result patterns
            top_categories = defaultdict(int)
            top_authors = defaultdict(int)
            
            for paper in papers[:20]:  # Analyze top papers
                for category in paper.categories:
                    top_categories[category] += 1
                
                for author in paper.authors[:2]:  # First two authors
                    top_authors[author.name] += 1
            
            # Recommend related categories
            if top_categories:
                most_common_cat = max(top_categories.items(), key=lambda x: x[1])
                if most_common_cat[1] >= 3:
                    recommendations.append(f"Consider exploring more papers in '{most_common_cat[0]}' category")
            
            # Recommend prolific authors
            if top_authors:
                prolific_authors = [
                    author for author, count in top_authors.items() 
                    if count >= 2
                ]
                if prolific_authors:
                    recommendations.append(f"Key researchers in this area: {', '.join(prolific_authors[:3])}")
            
            # Suggest refinement if too many results
            if len(papers) >= options.max_results * 0.9:
                recommendations.append("Many relevant papers found. Consider adding more specific keywords to narrow results.")
        
        return recommendations


# Usage example and testing
if __name__ == "__main__":
    async def main():
        # Create processor with real model
        processor = QueryProcessor(model_config_name="chat_load_balancer")
        
        # Test query
        test_query = "How can transformers be applied to time series forecasting?"
        options = QueryOptions(
            mode=SearchMode.COMPREHENSIVE,
            max_results=10,
            include_summaries=False
        )
        
        try:
            async with processor.arxiv_client:
                result = await processor.process_literature_query(test_query, options)
                
                print(f"\nQuery: {result.original_query}")
                print(f"Found {len(result.papers)} papers")
                print(f"Processing time: {result.processing_time:.2f}s")
                print(f"Strategies used: {', '.join(result.search_strategies_used)}")
                
                if result.papers:
                    print("\nTop papers:")
                    for i, paper in enumerate(result.papers[:3], 1):
                        print(f"\n{i}. {paper.title}")
                        print(f"   Authors: {', '.join(paper.author_names)}")
                        print(f"   Relevance: {paper.relevance_score:.3f}")
                        print(f"   ArXiv ID: {paper.arxiv_id}")
                
                if result.recommendations:
                    print(f"\nRecommendations:")
                    for rec in result.recommendations:
                        print(f"- {rec}")
                        
        except Exception as e:
            print(f"Error: {e}")
    
    # Run the example
    asyncio.run(main())