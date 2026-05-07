"""
External data sources for Phase 1 collection (FDA, ClinicalTrials.gov, SEC EDGAR).
Simple, pragmatic fetchers with on-disk caching and polite headers.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "AI.MED Research Bot contact@example.com")


def _cache_key(url: str, params: Optional[Dict[str, Any]] = None) -> Path:
    key_src = url + (json.dumps(params, sort_keys=True) if params else "")
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _cached_get(url: str, params: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                ttl_seconds: int = 7 * 24 * 3600) -> Dict[str, Any]:
    """GET with simple JSON file cache."""
    path = _cache_key(url, params)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code == 429:
        time.sleep(2)
        resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    # Prefer JSON; if text, wrap
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


# --- FDA (openFDA Drugs@FDA) -------------------------------------------------

def fetch_fda_drug(brand_name: str) -> Dict[str, Any]:
    """
    Fetch comprehensive FDA approval metadata for a brand name via openFDA.
    
    Extracts key fields useful for pharmaceutical forecasting:
    - Approval dates and regulatory timeline
    - Drug characteristics (route, mechanism, dosage form)
    - Competitive intelligence (supplemental approvals)
    - Market access indicators (sponsor, priority review)
    
    Returns:
        Dict with original FDA response plus extracted fields under 'extracted' key
    """
    url = "https://api.fda.gov/drug/drugsfda.json"
    params = {"search": f"openfda.brand_name:{brand_name}", "limit": 5}
    
    try:
        raw_response = _cached_get(url, params=params)
        
        # Initialize extracted data
        extracted = {
            'original_approval_date': None,
            'review_priority': None,
            'application_number': None,
            'sponsor_name': None,
            'generic_name': None,
            'route': None,
            'dosage_form': None,
            'mechanism_of_action': None,
            'marketing_status': None,
            'approved_supplementals_count': 0,
            'first_supplemental_date': None,
            'most_recent_activity': None,
            'manufacturer_name': None
        }
        
        if 'results' in raw_response and raw_response['results']:
            result = raw_response['results'][0]
            
            # 1. Core approval data
            extracted['application_number'] = result.get('application_number')
            extracted['sponsor_name'] = result.get('sponsor_name')
            
            # 2. OpenFDA metadata
            if 'openfda' in result:
                openfda = result['openfda']
                
                # Extract first item from lists (FDA returns arrays)
                def safe_extract_first(field_list):
                    return field_list[0] if isinstance(field_list, list) and field_list else field_list
                
                extracted['generic_name'] = safe_extract_first(openfda.get('generic_name'))
                extracted['route'] = safe_extract_first(openfda.get('route'))
                extracted['mechanism_of_action'] = safe_extract_first(openfda.get('pharm_class_moa'))
                extracted['manufacturer_name'] = safe_extract_first(openfda.get('manufacturer_name'))
            
            # 3. Product information
            if 'products' in result and result['products']:
                product = result['products'][0]
                extracted['dosage_form'] = product.get('dosage_form')
                extracted['marketing_status'] = product.get('marketing_status')
                
                # Override route if not found in openfda
                if not extracted['route']:
                    extracted['route'] = product.get('route')
            
            # 4. Submissions analysis (approval timeline)
            if 'submissions' in result:
                submissions = result['submissions']
                
                # Find original approval
                for sub in submissions:
                    if (sub.get('submission_type') == 'ORIG' and 
                        sub.get('submission_status') == 'AP' and
                        sub.get('submission_status_date')):
                        
                        date_str = sub['submission_status_date']
                        # Convert YYYYMMDD to YYYY-MM-DD format
                        if len(date_str) == 8 and date_str.isdigit():
                            extracted['original_approval_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        
                        extracted['review_priority'] = sub.get('review_priority', '').upper()
                        break
                
                # Count approved supplementals
                approved_suppls = [s for s in submissions 
                                 if s.get('submission_type') == 'SUPPL' and s.get('submission_status') == 'AP']
                extracted['approved_supplementals_count'] = len(approved_suppls)
                
                # First supplemental approval
                if approved_suppls:
                    # Sort by date to find first
                    dated_suppls = [s for s in approved_suppls if s.get('submission_status_date')]
                    if dated_suppls:
                        first_suppl = min(dated_suppls, key=lambda x: x['submission_status_date'])
                        date_str = first_suppl['submission_status_date']
                        if len(date_str) == 8 and date_str.isdigit():
                            extracted['first_supplemental_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                
                # Most recent activity
                all_dated = [s for s in submissions if s.get('submission_status_date')]
                if all_dated:
                    most_recent = max(all_dated, key=lambda x: x['submission_status_date'])
                    date_str = most_recent['submission_status_date']
                    if len(date_str) == 8 and date_str.isdigit():
                        extracted['most_recent_activity'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Add extracted data to response
        enhanced_response = raw_response.copy()
        enhanced_response['extracted'] = extracted
        
        return enhanced_response
        
    except Exception as e:
        return {"error": str(e), "brand": brand_name, "extracted": {}}


# --- ClinicalTrials.gov v2 ---------------------------------------------------

def fetch_ctgov_summary(query: str) -> Dict[str, Any]:
    """Fetch a small summary from ClinicalTrials.gov v2 API for a condition/drug."""
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"q": query, "pageSize": 5}
    try:
        return _cached_get(url, params=params)
    except Exception as e:
        return {"error": str(e), "query": query}


# --- SEC EDGAR helpers -------------------------------------------------------

def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Map ticker to CIK using SEC-provided JSON file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    data = _cached_get(url, headers=headers, ttl_seconds=14 * 24 * 3600)
    try:
        # Data is {"0": {"ticker": "A", "cik_str": 0, ...}, ...}
        for _, row in data.items():
            if str(row.get("ticker", "")).upper() == ticker.upper():
                return str(row.get("cik_str")).zfill(10)
    except Exception:
        pass
    return None


def fetch_company_submissions(cik: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    return _cached_get(url, headers=headers, ttl_seconds=7 * 24 * 3600)


def fetch_company_xbrl_facts(cik: str) -> Dict[str, Any]:
    """
    Fetch structured financial data from SEC EDGAR XBRL API.
    
    This is much better than parsing SEC filing texts - gives us clean,
    structured revenue data by fiscal year.
    
    Args:
        cik: Company CIK (e.g., "0000078003" for Pfizer)
    
    Returns:
        Dict with XBRL financial facts
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        return _cached_get(url, headers=headers, ttl_seconds=7 * 24 * 3600)
    except Exception as e:
        return {"error": str(e), "cik": cik}


def extract_revenue_from_xbrl(xbrl_data: Dict[str, Any], 
                             approval_year: int,
                             brand_aliases: List[str]) -> Dict[str, float]:
    """
    Extract revenue data from XBRL facts, mapped to years since launch.
    Uses deterministic segment-aware extraction.
    
    Args:
        xbrl_data: Result from fetch_company_xbrl_facts
        approval_year: Drug approval year (e.g., 2019)
        brand_aliases: List of brand names to search for
    
    Returns:
        Dict mapping year_since_launch to revenue amount
    """
    from .xbrl_extractor import extract_product_revenues, validate_extraction_quality
    
    # Use the enhanced deterministic extractor
    extraction_result = extract_product_revenues(xbrl_data, approval_year, brand_aliases)
    
    # Handle both old and new formats
    if isinstance(extraction_result, dict) and '_enhanced_extraction' in extraction_result:
        # New enhanced format
        revenues_data = extraction_result['revenues']
        metadata = extraction_result['metadata']
        
        # Convert enhanced format to simple format for backward compatibility
        simple_revenues = {}
        for year_since, data in revenues_data.items():
            if isinstance(data, dict):
                simple_revenues[year_since] = data.get('revenue', 0)
            else:
                simple_revenues[year_since] = data
        
        # Log extraction metadata
        brand_name = brand_aliases[0] if brand_aliases else "Unknown"
        if metadata['total_records_found'] > 0:
            currencies = ", ".join(metadata['currencies_found'])
            print(f"[XBRL] Enhanced extraction for {brand_name}: {metadata['total_records_found']} records, currencies: {currencies}")
            
            # Log fiscal-calendar mappings if any
            if metadata['fiscal_calendar_mappings']:
                print(f"[XBRL] Fiscal-calendar mappings: {metadata['fiscal_calendar_mappings']}")
        
        revenues = simple_revenues
    else:
        # Old format - handle as before
        revenues = extraction_result
    
    # Validate quality and log issues
    brand_name = brand_aliases[0] if brand_aliases else "Unknown"
    validation = validate_extraction_quality(revenues, brand_name)
    
    if validation['issues']:
        print(f"XBRL extraction issues for {brand_name}: {validation['issues']}")
    
    # Remove metadata before returning
    clean_revenues = {k: v for k, v in revenues.items() if k not in ['_source', '_enhanced_extraction']}
    
    return clean_revenues


def fetch_10k_texts(cik: str, max_docs: int = 5) -> List[str]:
    """Fetch recent 10-K/10-Q document texts (HTML as text) for a company CIK."""
    subs = fetch_company_submissions(cik)
    filings = subs.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    texts: List[str] = []
    base = "https://www.sec.gov/Archives/edgar/data"
    headers = {"User-Agent": SEC_USER_AGENT}
    for f, acc, doc in zip(forms, accession, primary_docs):
        if f not in ("10-K", "10-Q"):
            continue
        acc_nodash = acc.replace("-", "")
        # CIK may start with zeros omitted in path; strip leading zeros when constructing path folder id
        path = f"{base}/{int(cik)}/{acc_nodash}/{doc}"
        try:
            data = _cached_get(path, headers=headers, ttl_seconds=30 * 24 * 3600)
            text = data.get("text", "")
            if text:
                texts.append(text)
            if len(texts) >= max_docs:
                break
        except Exception:
            continue
    return texts


_CURRENCY_RE = re.compile(r"\$?\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)\s*(million|billion|thousand|m|bn|b|k)\b",
                          re.IGNORECASE)

# Fallback regex for amounts without units (assumes millions in tables)
# Avoid matching year numbers by requiring comma-separated numbers
_CURRENCY_NO_UNIT_RE = re.compile(r"\$?\s?([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{1,2})?)\b",
                                  re.IGNORECASE)


def _parse_amount(raw: str) -> Optional[float]:
    m = _CURRENCY_RE.search(raw)
    if not m:
        # Try fallback regex for amounts without units
        m = _CURRENCY_NO_UNIT_RE.search(raw)
        if not m:
            return None
        # Assume millions for amounts without units (common in tables)
        num = m.group(1).replace(",", "")
        try:
            val = float(num)
        except ValueError:
            return None
        return val * 1e6  # Assume millions
    
    num = m.group(1).replace(",", "")
    unit = (m.group(2) or "").lower()
    try:
        val = float(num)
    except ValueError:
        return None
    if unit in ("million", "m"):
        val *= 1e6
    elif unit in ("billion", "bn", "b"):
        val *= 1e9
    elif unit in ("thousand", "k"):
        val *= 1e3
    return val


def extract_brand_revenue(doc_text: str, brand_aliases: List[str]) -> Dict[str, float]:
    """Extract year->revenue near brand aliases with XBRL filtering."""
    results: Dict[str, float] = {}
    
    # Core issue: Brand mentions in XBRL metadata tags, not narrative text
    # Solution: Filter out XBRL sections before applying original logic
    text = doc_text
    
    # Remove inline XBRL tags that contain brand references but no financial narrative
    text = re.sub(r'<xbrli:.*?</xbrli:[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<xbrldi:.*?</xbrldi:[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    lower = text.lower()
    for alias in brand_aliases:
        a = alias.lower()
        start = 0
        while True:
            idx = lower.find(a, start)
            if idx == -1:
                break
            
            # Original window size + XBRL filtering
            window = text[max(0, idx - 400): idx + 600]
            
            # Skip if this window is mostly HTML/XBRL markup
            markup_ratio = window.count('<') / max(len(window), 1)
            if markup_ratio > 0.05:  # > 5% markup chars, likely XBRL section
                start = idx + len(a)
                continue
            
            # Original financial context check
            if re.search(r"revenue|net sales|product sales|sales|generated", window, re.IGNORECASE):
                # Original extraction logic
                years = re.findall(r"(20\d{2})", window)
                amounts = [m.group(0) for m in _CURRENCY_RE.finditer(window)]
                if not amounts:
                    amounts = [m.group(0) for m in _CURRENCY_NO_UNIT_RE.finditer(window)]
                if years and amounts:
                    # Original mapping with reasonable bounds
                    for y, amt_raw in zip(years[:3], amounts[:3]):
                        amt = _parse_amount(amt_raw)
                        if amt and 1e6 <= amt <= 50e9:  # Reasonable product revenue range
                            results[y] = max(results.get(y, 0.0), amt)
            start = idx + len(a)
    return results


__all__ = [
    "fetch_fda_drug",
    "fetch_ctgov_summary",
    "get_cik_for_ticker",
    "fetch_company_submissions", 
    "fetch_company_xbrl_facts",
    "extract_revenue_from_xbrl",
    "fetch_10k_texts",
    "extract_brand_revenue",
]


