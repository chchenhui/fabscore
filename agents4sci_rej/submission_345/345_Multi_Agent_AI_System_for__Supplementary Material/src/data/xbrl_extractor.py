"""
Deterministic XBRL extractor for product-specific pharmaceutical revenues.
Parses segment reports properly instead of company totals.
Enhanced with fiscal-calendar mapping, currency handling, and anchoring per GPT-5 guidance.
"""

from __future__ import annotations

import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


def map_fiscal_to_calendar_year(fiscal_year: int, fiscal_year_end: Optional[str] = None) -> int:
    """
    Map fiscal year to calendar year based on fiscal year-end.
    Per GPT-5 guidance: explicit fiscal→calendar year mapping.
    
    Args:
        fiscal_year: Fiscal year (e.g., 2023)
        fiscal_year_end: Fiscal year-end date (e.g., "December 31", "March 31")
    
    Returns:
        Calendar year corresponding to the fiscal year
    """
    if not fiscal_year_end:
        # Default assumption: December fiscal year-end = same calendar year
        return fiscal_year
    
    # Parse common fiscal year-end patterns
    fiscal_year_end_lower = fiscal_year_end.lower()
    
    # Q1 fiscal year ends (Jan-Mar) typically represent previous calendar year
    if any(month in fiscal_year_end_lower for month in ['january', 'february', 'march']):
        return fiscal_year - 1
    
    # Q2-Q4 fiscal year ends typically represent same calendar year  
    return fiscal_year


def parse_currency_and_convert(record: Dict[str, Any], target_currency: str = 'USD') -> Tuple[float, str]:
    """
    Parse currency from XBRL record and convert to target currency.
    Per GPT-5 guidance: Currency/FX handling.
    
    Args:
        record: XBRL record with currency information
        target_currency: Target currency (default: USD)
    
    Returns:
        (converted_value, original_currency)
    """
    val = record.get('val', 0)
    
    # Check for currency in record or assume USD
    # XBRL typically stores currency in unit ID
    original_currency = 'USD'  # Default assumption for US companies
    
    # Simple currency detection patterns
    unit_id = record.get('unit', '')
    if isinstance(unit_id, str):
        if 'eur' in unit_id.lower():
            original_currency = 'EUR'
        elif 'gbp' in unit_id.lower():
            original_currency = 'GBP'
        elif 'jpy' in unit_id.lower():
            original_currency = 'JPY'
    
    # For now, return as-is since most pharma companies report in USD
    # TODO: Add actual FX conversion when needed
    converted_value = val
    
    return converted_value, original_currency


def create_extraction_anchor(record: Dict[str, Any], 
                           accession: Optional[str] = None,
                           table_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Create extraction anchor for traceability.
    Per GPT-5 guidance: Store anchors (accession, table title, snippet).
    
    Args:
        record: XBRL record
        accession: SEC accession number
        table_title: Table or section title
    
    Returns:
        Anchor information for provenance tracking
    """
    return {
        'accession': accession or record.get('accn', 'unknown'),
        'filing_date': record.get('filed', 'unknown'),
        'form_type': record.get('form', 'unknown'),
        'fiscal_year': record.get('fy', 'unknown'),
        'fiscal_period': record.get('fp', 'unknown'),
        'table_title': table_title or 'segment_revenue',
        'xbrl_tag': record.get('tag', 'unknown'),
        'context_id': record.get('context', 'unknown'),
        'extraction_timestamp': datetime.now().isoformat()
    }


def parse_unit_scale(record: Dict[str, Any]) -> float:
    """
    Parse XBRL record and apply proper scaling based on context.
    
    Args:
        record: Full XBRL record with val and metadata
    
    Returns:
        Value scaled to USD (base units)
    """
    val = record.get('val', 0)
    if not val:
        return 0.0
    
    # Check for explicit scale in the record
    scale = record.get('scale', 0)
    if scale:
        return val * (10 ** scale)
    
    # XBRL values are typically in the actual unit stated
    # Check magnitude to detect likely scaling issues
    abs_val = abs(val)
    
    # Values > $1T are likely corrupted or wrong scale
    if abs_val > 1e12:
        return 0.0  # Flag as invalid
    
    # Values in reasonable range - return as-is
    return val


def extract_segment_revenues(xbrl_data: Dict[str, Any], 
                           brand_aliases: List[str],
                           include_anchors: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Extract segment-specific revenues from XBRL data with enhanced anchoring.
    
    Args:
        xbrl_data: XBRL facts from SEC API
        brand_aliases: List of brand names to search for
        include_anchors: Whether to include extraction anchors for traceability
    
    Returns:
        Dict[segment_name, Dict[fiscal_year, revenue_data_with_anchors]]
    """
    segment_revenues = {}
    
    if 'error' in xbrl_data or 'facts' not in xbrl_data:
        return segment_revenues
    
    facts = xbrl_data['facts']
    
    # Check both us-gaap and company-specific namespaces
    namespaces = []
    if 'us-gaap' in facts:
        namespaces.append(('us-gaap', facts['us-gaap']))
    
    # Add company-specific namespaces (e.g., 'pfe' for Pfizer)
    for ns_key, ns_data in facts.items():
        if ns_key != 'us-gaap' and isinstance(ns_data, dict):
            namespaces.append((ns_key, ns_data))
    
    # Revenue field patterns for segment reporting
    segment_revenue_fields = [
        'SegmentReportingInformationRevenue',  # Primary segment revenue field
        'RevenueFromContractWithCustomerByProductOrService',
        'DisaggregatedRevenue',
        'SegmentRevenues',
        'ProductSales',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'NetSales',
        'SalesRevenueNet',
        'Revenues'  # Fallback to consolidated
    ]
    
    for ns_name, ns_data in namespaces:
        for field_name in segment_revenue_fields:
            if field_name not in ns_data:
                continue
                
            field_data = ns_data[field_name]
            
            if 'units' not in field_data:
                continue
            
            # Process USD values
            if 'USD' in field_data['units']:
                usd_records = field_data['units']['USD']
                
                for record in usd_records:
                    # Skip quarterly data - only annual
                    form = record.get('form', '')
                    if form not in ['10-K']:
                        continue
                    
                    fy = record.get('fy')
                    if not fy:
                        continue
                    
                    val = record.get('val', 0)
                    if not val or val <= 0:
                        continue
                    
                    # Check for segment context
                    segment_name = extract_segment_name(record, brand_aliases)
                    if segment_name:
                        if segment_name not in segment_revenues:
                            segment_revenues[segment_name] = {}
                        
                        # Apply enhanced processing
                        scaled_val = parse_unit_scale(record)
                        converted_val, currency = parse_currency_and_convert(record)
                        
                        # Use converted value if different from scaled
                        final_val = converted_val if converted_val != scaled_val else scaled_val
                        
                        # Create revenue data with metadata
                        revenue_data = {
                            'revenue': final_val,
                            'currency': currency,
                            'fiscal_year': fy
                        }
                        
                        # Add extraction anchor if requested
                        if include_anchors:
                            accession = xbrl_data.get('cik', {}).get('accession', 'unknown')
                            revenue_data['anchor'] = create_extraction_anchor(
                                record, accession, f"{segment_name}_segment_revenue"
                            )
                        
                        # Keep the highest value for each year (restatements)
                        current_data = segment_revenues[segment_name].get(fy, {})
                        current_revenue = current_data.get('revenue', 0) if isinstance(current_data, dict) else current_data
                        
                        if final_val > current_revenue:
                            segment_revenues[segment_name][fy] = revenue_data
    
    return segment_revenues


def extract_segment_name(record: Dict[str, Any], brand_aliases: List[str]) -> Optional[str]:
    """
    Extract segment name from XBRL record context.
    
    Args:
        record: XBRL record with context information
        brand_aliases: List of brand names to match
    
    Returns:
        Segment name if found, None otherwise
    """
    # Check for explicit segment dimensions
    segment_indicators = [
        'ProductOrServiceAxis',
        'BusinessSegmentAxis', 
        'RevenueStreamAxis',
        'ProductLineAxis'
    ]
    
    # Look in record properties for segment info
    for key, value in record.items():
        if isinstance(value, str):
            value_lower = value.lower()
            
            # Check if any brand alias appears in the context
            for alias in brand_aliases:
                if alias.lower() in value_lower:
                    return alias
            
            # Check for segment axis indicators
            for indicator in segment_indicators:
                if indicator.lower() in value_lower:
                    # Extract the segment value
                    segment_match = re.search(r'(\w+)Member', value)
                    if segment_match:
                        return segment_match.group(1)
    
    return None


def extract_product_revenues(xbrl_data: Dict[str, Any],
                           approval_year: int,
                           brand_aliases: List[str],
                           fiscal_year_end: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract product-specific revenues mapped to years since launch with enhanced features.
    
    Args:
        xbrl_data: XBRL facts from SEC API
        approval_year: Drug approval year
        brand_aliases: List of brand names
        fiscal_year_end: Company fiscal year-end for calendar mapping
    
    Returns:
        Dict with revenue data and enhanced metadata
    """
    result_revenues = {}
    extraction_metadata = {
        'total_records_found': 0,
        'fiscal_calendar_mappings': {},
        'currencies_found': set(),
        'extraction_anchors': []
    }
    
    # First try segment-specific extraction with anchoring
    segment_revenues = extract_segment_revenues(xbrl_data, brand_aliases, include_anchors=True)
    
    # Look for exact brand matches first
    for alias in brand_aliases:
        if alias in segment_revenues:
            for fy_str, revenue_data in segment_revenues[alias].items():
                try:
                    fy = int(fy_str)
                    
                    # Apply fiscal-to-calendar year mapping
                    calendar_year = map_fiscal_to_calendar_year(fy, fiscal_year_end)
                    year_since = calendar_year - approval_year
                    
                    if 0 <= year_since <= 10:  # Reasonable range
                        # Extract revenue amount (handle both old and new formats)
                        if isinstance(revenue_data, dict):
                            revenue_amount = revenue_data.get('revenue', 0)
                            currency = revenue_data.get('currency', 'USD')
                            anchor = revenue_data.get('anchor', {})
                        else:
                            revenue_amount = revenue_data  # Backward compatibility
                            currency = 'USD'
                            anchor = {}
                        
                        year_key = str(year_since)
                        current_data = result_revenues.get(year_key, {})
                        current_revenue = current_data.get('revenue', 0) if isinstance(current_data, dict) else 0
                        
                        if revenue_amount > current_revenue:
                            result_revenues[year_key] = {
                                'revenue': revenue_amount,
                                'fiscal_year': fy,
                                'calendar_year': calendar_year,
                                'currency': currency,
                                'brand_alias': alias,
                                'anchor': anchor
                            }
                            
                            # Update metadata
                            extraction_metadata['total_records_found'] += 1
                            extraction_metadata['fiscal_calendar_mappings'][str(fy)] = calendar_year
                            extraction_metadata['currencies_found'].add(currency)
                            if anchor:
                                extraction_metadata['extraction_anchors'].append(anchor)
                        
                except (ValueError, TypeError):
                    continue
    
    # NO FALLBACK to consolidated data - GPT-5 guidance: 
    # "If only consolidated totals exist, mark product revenue unknown 
    # rather than contaminating labels"
    
    # Convert currencies_found set to list for JSON serialization
    extraction_metadata['currencies_found'] = list(extraction_metadata['currencies_found'])
    
    # Return both revenue data and metadata
    return {
        'revenues': result_revenues,
        'metadata': extraction_metadata,
        '_enhanced_extraction': True  # Flag for enhanced version
    }


def extract_consolidated_revenues(xbrl_data: Dict[str, Any], 
                                approval_year: int) -> Dict[str, float]:
    """
    Fallback: extract consolidated company revenues.
    This is what the old code was doing - getting company totals instead of product-specific.
    """
    revenues = {}
    
    if 'error' in xbrl_data or 'facts' not in xbrl_data:
        return revenues
    
    facts = xbrl_data['facts']
    if 'us-gaap' not in facts:
        return revenues
    
    gaap = facts['us-gaap']
    
    # Standard revenue fields for consolidated reporting
    revenue_fields = [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet'
    ]
    
    for field_name in revenue_fields:
        if field_name not in gaap:
            continue
            
        field_data = gaap[field_name]
        
        if 'units' not in field_data or 'USD' not in field_data['units']:
            continue
        
        usd_records = field_data['units']['USD']
        
        for record in usd_records:
            # Only annual data
            form = record.get('form', '')
            if form not in ['10-K']:
                continue
            
            fy = record.get('fy')
            val = record.get('val', 0)
            
            if not fy or not val or val <= 0:
                continue
            
            try:
                fiscal_year = int(fy)
                year_since = fiscal_year - approval_year
                
                if 0 <= year_since <= 10:
                    scaled_val = parse_unit_scale(record)
                    current = revenues.get(str(year_since), 0)
                    revenues[str(year_since)] = max(current, scaled_val)
                    
            except (ValueError, TypeError):
                continue
    
    return revenues


def validate_extraction_quality(revenues: Dict[str, float], 
                               brand_name: str,
                               expected_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Validate the quality of extracted revenue data.
    
    Args:
        revenues: Extracted revenue data
        brand_name: Brand name for logging
        expected_range: Optional (min, max) expected revenue range
    
    Returns:
        Validation report
    """
    report = {
        'brand': brand_name,
        'years_extracted': len([k for k in revenues.keys() if k != '_source']),
        'source_type': revenues.get('_source', 'segment'),
        'issues': []
    }
    
    # Check for reasonable data coverage
    if report['years_extracted'] < 2:
        report['issues'].append('Insufficient data coverage (<2 years)')
    
    # Check for reasonable revenue values
    revenue_values = [v for k, v in revenues.items() if k != '_source']
    if revenue_values:
        max_rev = max(revenue_values)
        min_rev = min(revenue_values)
        
        # Flag suspiciously high values (likely consolidated)
        if max_rev > 50e9:  # >$50B suggests company total
            report['issues'].append(f'Revenue too high: ${max_rev/1e9:.1f}B (likely consolidated)')
        
        # Check range if provided
        if expected_range:
            exp_min, exp_max = expected_range
            if max_rev < exp_min or min_rev > exp_max:
                report['issues'].append(f'Revenue outside expected range: {exp_min/1e9:.1f}-{exp_max/1e9:.1f}B')
    
    report['quality_score'] = max(0, 100 - len(report['issues']) * 25)
    
    return report


if __name__ == "__main__":
    # Test with cached XBRL data
    cache_dir = Path("cache")
    
    # Example test with known XBRL files
    test_files = list(cache_dir.glob("*.json"))[:3]
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                xbrl_data = json.load(f)
            
            if 'facts' in xbrl_data:
                segments = extract_segment_revenues(xbrl_data, ['Keytruda', 'Eliquis'])
                print(f"\nFile: {test_file.name}")
                print(f"Segments found: {list(segments.keys())}")
                
                for seg_name, seg_data in segments.items():
                    print(f"  {seg_name}: {len(seg_data)} years")
                    
        except Exception as e:
            print(f"Error processing {test_file.name}: {e}")