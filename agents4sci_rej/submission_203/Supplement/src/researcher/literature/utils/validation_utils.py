"""
Validation utilities for the literature query system.

This module provides validation functions for ArXiv IDs, search queries,
and other data integrity checks used throughout the system.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date
from urllib.parse import urlparse


def validate_arxiv_id(arxiv_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate ArXiv ID format.
    
    Args:
        arxiv_id: ArXiv ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not arxiv_id or not isinstance(arxiv_id, str):
        return False, "ArXiv ID must be a non-empty string"
    
    arxiv_id = arxiv_id.strip()
    
    # Modern format: YYMM.NNNNN (e.g., 2301.12345)
    modern_pattern = r'^\d{4}\.\d{4,5}$'
    
    # Legacy format: subject-class/YYMMnnn (e.g., cs.AI/0301012, math-ph/0301012)
    legacy_pattern = r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$'
    
    if re.match(modern_pattern, arxiv_id):
        # Validate year and month for modern format
        year_month = arxiv_id[:4]
        year = int(year_month[:2])
        month = int(year_month[2:])
        
        # ArXiv started using 4-digit format around 2007 (07xx)
        if year < 7:
            return False, "Invalid year in ArXiv ID (too early for 4-digit format)"
        
        if month < 1 or month > 12:
            return False, "Invalid month in ArXiv ID (must be 01-12)"
        
        return True, None
    
    elif re.match(legacy_pattern, arxiv_id):
        # Validate subject class for legacy format
        parts = arxiv_id.split('/')
        subject_class = parts[0]
        
        # Common ArXiv subject classes
        valid_subjects = {
            'astro-ph', 'cond-mat', 'cs', 'econ', 'eess', 'gr-qc', 'hep-ex',
            'hep-lat', 'hep-ph', 'hep-th', 'math', 'math-ph', 'nlin', 'nucl-ex',
            'nucl-th', 'physics', 'q-bio', 'q-fin', 'quant-ph', 'stat'
        }
        
        base_subject = subject_class.split('.')[0]
        if base_subject not in valid_subjects:
            return False, f"Unknown subject class: {base_subject}"
        
        return True, None
    
    else:
        return False, "ArXiv ID format not recognized (should be YYMM.NNNNN or subject-class/YYMMnnn)"


def validate_search_query(query: str, max_length: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Validate search query string.
    
    Args:
        query: Search query to validate
        max_length: Maximum allowed query length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not isinstance(query, str):
        return False, "Search query must be a non-empty string"
    
    query = query.strip()
    
    if not query:
        return False, "Search query cannot be empty or only whitespace"
    
    if len(query) > max_length:
        return False, f"Search query too long (max {max_length} characters)"
    
    # Check for potentially problematic patterns
    if query.count('"') % 2 != 0:
        return False, "Unmatched quotation marks in search query"
    
    if query.count('(') != query.count(')'):
        return False, "Unmatched parentheses in search query"
    
    # Check for excessive special characters that might break ArXiv API
    special_char_ratio = len(re.findall(r'[^\w\s\-_\.\(\)\":]', query)) / len(query)
    if special_char_ratio > 0.3:
        return False, "Too many special characters in search query"
    
    return True, None


def validate_keyword_list(keywords: List[str], min_count: int = 1, max_count: int = 50) -> Tuple[bool, Optional[str]]:
    """
    Validate list of keywords.
    
    Args:
        keywords: List of keywords to validate
        min_count: Minimum number of keywords required
        max_count: Maximum number of keywords allowed
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(keywords, list):
        return False, "Keywords must be provided as a list"
    
    if len(keywords) < min_count:
        return False, f"At least {min_count} keyword(s) required"
    
    if len(keywords) > max_count:
        return False, f"Too many keywords (max {max_count})"
    
    for i, keyword in enumerate(keywords):
        if not isinstance(keyword, str):
            return False, f"Keyword at index {i} must be a string"
        
        keyword = keyword.strip()
        if not keyword:
            return False, f"Keyword at index {i} cannot be empty"
        
        if len(keyword) > 100:
            return False, f"Keyword at index {i} too long (max 100 characters)"
    
    return True, None


def validate_date_range(start_date: Optional[date], end_date: Optional[date]) -> Tuple[bool, Optional[str]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date (optional)
        end_date: End date (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_date is None and end_date is None:
        return True, None  # No date range specified is valid
    
    # Validate individual dates
    today = date.today()
    arxiv_start_date = date(1991, 8, 1)  # ArXiv started in August 1991
    
    if start_date is not None:
        if not isinstance(start_date, date):
            return False, "start_date must be a date object"
        
        if start_date < arxiv_start_date:
            return False, "start_date cannot be before ArXiv existed (1991-08-01)"
        
        if start_date > today:
            return False, "start_date cannot be in the future"
    
    if end_date is not None:
        if not isinstance(end_date, date):
            return False, "end_date must be a date object"
        
        if end_date < arxiv_start_date:
            return False, "end_date cannot be before ArXiv existed (1991-08-01)"
        
        if end_date > today:
            return False, "end_date cannot be in the future"
    
    # Validate date range relationship
    if start_date is not None and end_date is not None:
        if start_date > end_date:
            return False, "start_date cannot be after end_date"
        
        # Check for excessively long range
        delta = end_date - start_date
        if delta.days > 365 * 10:  # More than 10 years
            return False, "Date range too large (max 10 years)"
    
    return True, None


def validate_url(url: str, allowed_schemes: List[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format and scheme.
    
    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes (default: ['http', 'https'])
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"
    
    if not parsed.scheme:
        return False, "URL must have a scheme (http/https)"
    
    if parsed.scheme.lower() not in allowed_schemes:
        return False, f"URL scheme must be one of: {allowed_schemes}"
    
    if not parsed.netloc:
        return False, "URL must have a domain"
    
    return True, None


def validate_arxiv_categories(categories: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate ArXiv category codes.
    
    Args:
        categories: List of ArXiv categories to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(categories, list):
        return False, "Categories must be provided as a list"
    
    if not categories:
        return False, "At least one category must be specified"
    
    # Valid ArXiv subject classes and their subcategories
    valid_categories = {
        # Computer Science
        'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
        'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL',
        'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
        'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
        'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY',
        
        # Mathematics
        'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO', 'math.CT',
        'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GR',
        'math.GT', 'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG', 'math.MP',
        'math.NA', 'math.NT', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RA',
        'math.RT', 'math.SG', 'math.SP', 'math.ST',
        
        # Physics
        'physics.acc-ph', 'physics.ao-ph', 'physics.app-ph', 'physics.atm-clus',
        'physics.atom-ph', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph',
        'physics.comp-ph', 'physics.data-an', 'physics.ed-ph', 'physics.flu-dyn',
        'physics.gen-ph', 'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det',
        'physics.med-ph', 'physics.optics', 'physics.plasm-ph', 'physics.pop-ph',
        'physics.soc-ph', 'physics.space-ph',
        
        # Other subjects
        'astro-ph', 'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE',
        'astro-ph.IM', 'astro-ph.SR',
        'cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other',
        'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el',
        'cond-mat.supr-con',
        'econ.EM', 'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY',
        'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph',
        'nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI',
        'nucl-ex', 'nucl-th',
        'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT',
        'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO',
        'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR',
        'q-fin.RM', 'q-fin.ST', 'q-fin.TR',
        'quant-ph',
        'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT', 'stat.TH'
    }
    
    for category in categories:
        if not isinstance(category, str):
            return False, "All categories must be strings"
        
        category = category.strip()
        if not category:
            return False, "Categories cannot be empty"
        
        if category not in valid_categories:
            return False, f"Unknown ArXiv category: {category}"
    
    return True, None


def validate_confidence_score(score: float) -> Tuple[bool, Optional[str]]:
    """
    Validate confidence score.
    
    Args:
        score: Confidence score to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(score, (int, float)):
        return False, "Confidence score must be a number"
    
    if score < 0.0 or score > 1.0:
        return False, "Confidence score must be between 0.0 and 1.0"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "untitled"
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
    
    return sanitized


def is_safe_query(query: str) -> bool:
    """
    Check if query is safe (doesn't contain potential injection attempts).
    
    Args:
        query: Query string to check
        
    Returns:
        True if query appears safe
    """
    if not query:
        return True
    
    # Check for potential script injection
    dangerous_patterns = [
        r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
        r'eval\s*\(', r'function\s*\(', r'setTimeout', r'setInterval'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    
    return True


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ArXiv ID validation:")
    test_ids = ["2301.12345", "cs.AI/0301012", "invalid_id", "2399.99999"]
    
    for arxiv_id in test_ids:
        is_valid, error = validate_arxiv_id(arxiv_id)
        print(f"  {arxiv_id}: {'✓' if is_valid else '✗'} {error or ''}")
    
    print("\nTesting search query validation:")
    test_queries = [
        "machine learning transformers",
        "neural networks AND deep learning",
        'query with "unmatched quotes',
        "(" * 10 + "unmatched parens",
        ""
    ]
    
    for query in test_queries:
        is_valid, error = validate_search_query(query)
        print(f"  '{query}': {'✓' if is_valid else '✗'} {error or ''}")
    
    print("\nTesting ArXiv category validation:")
    test_categories = [
        ["cs.AI", "cs.LG"],
        ["cs.AI", "invalid.category"],
        ["cs.CV", "math.OC", "stat.ML"],
        []
    ]
    
    for categories in test_categories:
        is_valid, error = validate_arxiv_categories(categories)
        print(f"  {categories}: {'✓' if is_valid else '✗'} {error or ''}")