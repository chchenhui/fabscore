"""
Text processing utilities for the literature query system.

This module provides text cleaning, keyword extraction, and processing
functions used throughout the literature search system.
"""

import re
import string
from typing import List, Set, Dict, Tuple
from collections import Counter


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Input text to clean
        preserve_newlines: Whether to preserve newline characters
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove or replace common Unicode characters
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '--')  # em-dash
    text = text.replace('\u2018', "'")   # left single quote
    text = text.replace('\u2019', "'")   # right single quote
    text = text.replace('\u201c', '"')   # left double quote
    text = text.replace('\u201d', '"')   # right double quote
    
    # Remove excessive whitespace
    if preserve_newlines:
        # Collapse multiple spaces but preserve newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    else:
        # Collapse all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_phrases(text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
    """
    Extract meaningful phrases from text.
    
    Args:
        text: Input text
        min_length: Minimum number of words in phrase
        max_length: Maximum number of words in phrase
        
    Returns:
        List of extracted phrases
    """
    if not text:
        return []
    
    # Clean text first
    text = clean_text(text)
    
    # Split into words and filter
    words = text.split()
    words = [word.strip(string.punctuation) for word in words]
    words = [word for word in words if word and len(word) > 1]
    
    phrases = []
    
    # Extract phrases of different lengths
    for length in range(min_length, max_length + 1):
        for i in range(len(words) - length + 1):
            phrase = ' '.join(words[i:i + length])
            if _is_valid_phrase(phrase):
                phrases.append(phrase)
    
    return phrases


def _is_valid_phrase(phrase: str) -> bool:
    """Check if phrase is meaningful (not just stop words/common terms)."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    words = phrase.lower().split()
    
    # Must contain at least one non-stop word
    non_stop_words = [word for word in words if word not in stop_words]
    if not non_stop_words:
        return False
    
    # Must not be all common words
    common_academic_words = {
        'paper', 'study', 'research', 'analysis', 'method', 'approach',
        'result', 'conclusion', 'finding', 'work', 'model', 'system'
    }
    
    if all(word in common_academic_words for word in non_stop_words):
        return False
    
    return True


def normalize_keywords(keywords: List[str]) -> List[str]:
    """
    Normalize and clean a list of keywords.
    
    Args:
        keywords: List of keywords to normalize
        
    Returns:
        List of normalized keywords
    """
    if not keywords:
        return []
    
    normalized = []
    
    for keyword in keywords:
        if not keyword:
            continue
        
        # Clean the keyword
        keyword = clean_text(keyword.strip())
        
        if not keyword:
            continue
        
        # Convert to lowercase for normalization
        keyword_lower = keyword.lower()
        
        # Remove common prefixes/suffixes that don't add meaning
        keyword_lower = re.sub(r'^(using|with|for|in|on|by|a|an|the)\s+', '', keyword_lower)
        keyword_lower = re.sub(r'\s+(method|approach|algorithm|technique|system)s?$', '', keyword_lower)
        
        # Handle acronyms - preserve case if all caps
        if keyword.isupper() and len(keyword) <= 6:
            normalized_keyword = keyword
        else:
            normalized_keyword = keyword_lower
        
        # Remove duplicates
        if normalized_keyword not in [nk.lower() for nk in normalized]:
            normalized.append(normalized_keyword)
    
    return normalized


def extract_technical_terms(text: str, min_freq: int = 1) -> List[Tuple[str, int]]:
    """
    Extract technical terms from text based on patterns.
    
    Args:
        text: Input text
        min_freq: Minimum frequency for term inclusion
        
    Returns:
        List of (term, frequency) tuples sorted by frequency
    """
    if not text:
        return []
    
    text = clean_text(text)
    
    # Patterns for technical terms
    patterns = [
        # Hyphenated terms (e.g., "multi-task", "state-of-the-art")
        r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b',
        
        # Capitalized terms (e.g., "Neural Network", "SVM")
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        
        # Acronyms (2-6 uppercase letters)
        r'\b[A-Z]{2,6}\b',
        
        # Terms with numbers (e.g., "GPT-3", "ResNet-50")
        r'\b[A-Za-z]+[-_]?\d+\b',
        
        # Mathematical terms (ending with common suffixes)
        r'\b\w+(?:tion|ism|ity|ness|ment|ence|ance|ing)\b'
    ]
    
    all_terms = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        all_terms.extend(matches)
    
    # Count frequencies
    term_counts = Counter(all_terms)
    
    # Filter by minimum frequency and remove common non-technical terms
    common_terms = {
        'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where',
        'Why', 'How', 'Which', 'Who', 'While', 'During', 'After', 'Before'
    }
    
    filtered_terms = [
        (term, count) for term, count in term_counts.items()
        if count >= min_freq and term not in common_terms and len(term) > 2
    ]
    
    # Sort by frequency (descending)
    return sorted(filtered_terms, key=lambda x: x[1], reverse=True)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = clean_text(text1.lower())
    text2 = clean_text(text2.lower())
    
    # Extract words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def highlight_keywords_in_text(text: str, keywords: List[str]) -> str:
    """
    Highlight keywords in text using markdown formatting.
    
    Args:
        text: Input text
        keywords: Keywords to highlight
        
    Returns:
        Text with keywords highlighted
    """
    if not text or not keywords:
        return text
    
    highlighted_text = text
    
    # Sort keywords by length (longest first) to avoid partial replacements
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        if not keyword:
            continue
        
        # Create case-insensitive pattern
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        # Replace with highlighted version
        highlighted_text = pattern.sub(
            lambda m: f"**{m.group()}**",
            highlighted_text
        )
    
    return highlighted_text


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length, trying to break at word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum length
        add_ellipsis: Whether to add "..." if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length]
    
    # Find the last space within the limit
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break reasonably close to limit
        truncated = truncated[:last_space]
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def extract_doi_from_text(text: str) -> List[str]:
    """
    Extract DOI (Digital Object Identifier) from text.
    
    Args:
        text: Input text
        
    Returns:
        List of found DOIs
    """
    if not text:
        return []
    
    # DOI pattern
    doi_pattern = r'10\.\d+/[^\s]+'
    
    matches = re.findall(doi_pattern, text, re.IGNORECASE)
    
    # Clean up matches (remove trailing punctuation)
    cleaned_dois = []
    for doi in matches:
        # Remove trailing punctuation
        doi = re.sub(r'[.,;:)\]}]+$', '', doi)
        if doi:
            cleaned_dois.append(doi)
    
    return list(set(cleaned_dois))  # Remove duplicates


def format_author_names(authors: List[str]) -> str:
    """
    Format author names for display.
    
    Args:
        authors: List of author names
        
    Returns:
        Formatted author string
    """
    if not authors:
        return ""
    
    if len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    elif len(authors) <= 5:
        return f"{', '.join(authors[:-1])}, and {authors[-1]}"
    else:
        return f"{authors[0]} et al."


if __name__ == "__main__":
    # Example usage and testing
    test_text = """
    This paper presents a novel deep learning approach for natural language processing.
    We introduce a transformer-based architecture that achieves state-of-the-art
    performance on multiple NLP tasks including machine translation and text summarization.
    """
    
    print("Original text:")
    print(test_text)
    
    print("\nCleaned text:")
    print(clean_text(test_text))
    
    print("\nExtracted phrases:")
    phrases = extract_phrases(test_text, min_length=2, max_length=3)
    for phrase in phrases[:10]:
        print(f"- {phrase}")
    
    print("\nTechnical terms:")
    terms = extract_technical_terms(test_text)
    for term, freq in terms:
        print(f"- {term}: {freq}")
    
    keywords = ["deep learning", "transformer", "NLP"]
    print(f"\nHighlighted text with keywords {keywords}:")
    print(highlight_keywords_in_text(test_text, keywords))