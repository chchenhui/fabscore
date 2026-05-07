"""
Deterministic 10-K table parser for product-level revenue extraction.
Follows GPT-5's guidance: structured parsing first, LLM only for disambiguation.
"""

from __future__ import annotations

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag
import pandas as pd


@dataclass
class TableRevenue:
    """Structured revenue extraction with full provenance."""
    brand: str
    year: int
    amount_usd: float
    fiscal_year: int
    currency_code: str
    header_units: str
    accession_id: str
    table_title: str
    row_snippet: str
    confidence: float


class DeterministicTableParser:
    """Deterministic parser for 10-K product revenue tables."""
    
    # Target table section titles
    PRODUCT_TABLE_PATTERNS = [
        r"net product sales",
        r"product sales",
        r"net sales by product",
        r"key products",
        r"principal products",
        r"product revenues",
        r"revenues by product",
        r"pharmaceutical sales"
    ]
    
    # Header unit patterns
    UNIT_PATTERNS = {
        'millions': 1e6,
        'million': 1e6,
        'billions': 1e9,
        'billion': 1e9,
        'thousands': 1e3,
        'thousand': 1e3
    }
    
    # Rows to reject (company/segment totals)
    REJECT_ROW_PATTERNS = [
        r"total",
        r"consolidated",
        r"segment",
        r"division",
        r"other products",
        r"all other",
        r"remaining",
        r"subtotal"
    ]
    
    def __init__(self):
        self.debug = False
        
    def extract_product_revenues(
        self, 
        doc_html: str, 
        brand_aliases: List[str],
        accession_id: str = ""
    ) -> List[TableRevenue]:
        """Extract product revenues using deterministic table parsing."""
        
        revenues = []
        soup = BeautifulSoup(doc_html, 'html.parser')
        
        # Step 1: Find candidate tables by section titles
        candidate_tables = self._find_product_tables(soup)
        
        if self.debug:
            print(f"Found {len(candidate_tables)} candidate product tables")
        
        # Step 2: Parse each table deterministically
        for table_info in candidate_tables:
            table_revenues = self._parse_product_table(
                table_info, brand_aliases, accession_id
            )
            revenues.extend(table_revenues)
        
        # Step 3: Multi-year validation and deduplication
        validated_revenues = self._validate_multi_year(revenues)
        
        return validated_revenues
    
    def _find_product_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Find tables with product sales by scanning section headers."""
        candidate_tables = []
        
        # Look for tables near section headers that mention products
        for pattern in self.PRODUCT_TABLE_PATTERNS:
            # Find headers (h1-h6, div with certain classes, table captions)
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'p'], 
                                   string=re.compile(pattern, re.IGNORECASE))
            
            # Also check text content of elements
            text_matches = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
            
            for match in headers + text_matches:
                if isinstance(match, str):
                    # String match - find parent element
                    parent = match.parent if hasattr(match, 'parent') else None
                else:
                    parent = match
                    
                if not parent:
                    continue
                    
                # Find nearest table after this header
                table = self._find_nearest_table(parent)
                if table:
                    candidate_tables.append({
                        'table': table,
                        'title': str(match)[:100],
                        'pattern_matched': pattern
                    })
        
        # Deduplicate by table element
        seen_tables = set()
        unique_tables = []
        for table_info in candidate_tables:
            table_id = id(table_info['table'])
            if table_id not in seen_tables:
                seen_tables.add(table_id)
                unique_tables.append(table_info)
        
        return unique_tables
    
    def _find_nearest_table(self, element: Tag) -> Optional[Tag]:
        """Find the nearest table element after a header."""
        # Search in siblings and descendants
        current = element
        
        # First, try siblings after this element
        for sibling in element.find_next_siblings():
            table = sibling.find('table')
            if table:
                return table
            if sibling.name == 'table':
                return sibling
        
        # Then try parent's siblings
        parent = element.parent
        if parent:
            for sibling in parent.find_next_siblings():
                table = sibling.find('table')
                if table:
                    return table
                if sibling.name == 'table':
                    return sibling
        
        # Finally, try finding any table in the next 5000 characters
        element_pos = str(element.parent).find(str(element)) if element.parent else 0
        remaining_html = str(element.parent)[element_pos:element_pos + 5000] if element.parent else ""
        
        if remaining_html:
            mini_soup = BeautifulSoup(remaining_html, 'html.parser')
            return mini_soup.find('table')
        
        return None
    
    def _parse_product_table(
        self, 
        table_info: Dict[str, Any], 
        brand_aliases: List[str],
        accession_id: str
    ) -> List[TableRevenue]:
        """Parse a specific table for product revenues."""
        
        table = table_info['table']
        title = table_info['title']
        
        if self.debug:
            print(f"\nParsing table: {title[:50]}...")
        
        revenues = []
        
        # Step 1: Extract header information (units, years)
        header_info = self._extract_table_headers(table)
        
        if not header_info['years']:
            if self.debug:
                print("  No years found in headers, skipping")
            return revenues
        
        # Step 2: Find brand rows
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:  # Need at least brand name + 1 data column
                continue
                
            # First cell typically contains product name
            first_cell_text = cells[0].get_text(strip=True).lower()
            
            # Check if this row contains any of our brand aliases
            matching_brand = None
            for brand in brand_aliases:
                if brand.lower() in first_cell_text:
                    # Reject if it's a total/consolidated row
                    if any(reject_pattern in first_cell_text 
                           for reject_pattern in [re.compile(p, re.IGNORECASE) 
                                                 for p in self.REJECT_ROW_PATTERNS]):
                        if self.debug:
                            print(f"    Rejecting total/consolidated row: {first_cell_text[:50]}")
                        break
                    matching_brand = brand
                    break
            
            if not matching_brand:
                continue
                
            if self.debug:
                print(f"    Found brand row: {matching_brand} -> {first_cell_text[:50]}")
            
            # Step 3: Extract revenue values from this row
            row_revenues = self._extract_row_revenues(
                cells, header_info, matching_brand, title, accession_id, first_cell_text
            )
            revenues.extend(row_revenues)
        
        return revenues
    
    def _extract_table_headers(self, table: Tag) -> Dict[str, Any]:
        """Extract year columns and unit information from table headers."""
        
        header_info = {
            'years': [],
            'year_columns': {},  # column_index -> year
            'units_multiplier': 1.0,
            'currency': 'USD',
            'header_text': ''
        }
        
        # Get all header rows (first few tr elements, or those with th)
        header_rows = []
        rows = table.find_all('tr')
        
        for i, row in enumerate(rows[:5]):  # Check first 5 rows for headers
            if row.find('th') or i < 2:  # Has th elements or is in first 2 rows
                header_rows.append(row)
        
        # Extract text from all header rows for unit detection
        all_header_text = ' '.join(row.get_text() for row in header_rows)
        header_info['header_text'] = all_header_text
        
        # Find units in header text
        for unit_name, multiplier in self.UNIT_PATTERNS.items():
            if re.search(rf'\bin {unit_name}\b', all_header_text, re.IGNORECASE):
                header_info['units_multiplier'] = multiplier
                break
        
        # Find year columns
        for row in header_rows:
            cells = row.find_all(['th', 'td'])
            for col_idx, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                
                # Look for 4-digit years
                year_match = re.search(r'(20\d{2})', cell_text)
                if year_match:
                    year = int(year_match.group(1))
                    header_info['years'].append(year)
                    header_info['year_columns'][col_idx] = year
        
        # Remove duplicates and sort years
        header_info['years'] = sorted(list(set(header_info['years'])))
        
        return header_info
    
    def _extract_row_revenues(
        self,
        cells: List[Tag],
        header_info: Dict[str, Any],
        brand: str,
        table_title: str,
        accession_id: str,
        row_text: str
    ) -> List[TableRevenue]:
        """Extract revenue amounts from a product row."""
        
        revenues = []
        
        # For each data cell, try to match it to a year column
        for col_idx, cell in enumerate(cells[1:], 1):  # Skip first cell (product name)
            
            # Check if this column corresponds to a year
            year = header_info['year_columns'].get(col_idx)
            if not year:
                continue
            
            # Extract numeric value from cell
            cell_text = cell.get_text(strip=True)
            amount = self._parse_cell_amount(cell_text)
            
            if amount is None or amount <= 0:
                continue
            
            # Apply units multiplier
            amount_usd = amount * header_info['units_multiplier']
            
            # Sanity check: reasonable product revenue range
            if not (1e6 <= amount_usd <= 50e9):  # $1M to $50B
                if self.debug:
                    print(f"      Rejecting unreasonable amount: ${amount_usd/1e9:.1f}B")
                continue
            
            # Create revenue record
            revenue = TableRevenue(
                brand=brand,
                year=year,
                amount_usd=amount_usd,
                fiscal_year=year,  # Assume calendar year for now
                currency_code=header_info['currency'],
                header_units=f"x{header_info['units_multiplier']:g}",
                accession_id=accession_id,
                table_title=table_title[:100],
                row_snippet=row_text[:200],
                confidence=0.9  # High confidence for deterministic extraction
            )
            
            revenues.append(revenue)
            
            if self.debug:
                print(f"      Extracted: {year} -> ${amount_usd/1e9:.2f}B")
        
        return revenues
    
    def _parse_cell_amount(self, cell_text: str) -> Optional[float]:
        """Parse numeric amount from a table cell."""
        
        # Remove common formatting
        clean_text = cell_text.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        clean_text = re.sub(r'[^\d.-]', '', clean_text)
        
        if not clean_text:
            return None
            
        try:
            return float(clean_text)
        except ValueError:
            return None
    
    def _validate_multi_year(self, revenues: List[TableRevenue]) -> List[TableRevenue]:
        """Validate multi-year consistency and flag anomalies."""
        
        # Group by brand
        by_brand = {}
        for rev in revenues:
            if rev.brand not in by_brand:
                by_brand[rev.brand] = []
            by_brand[rev.brand].append(rev)
        
        validated = []
        
        for brand, brand_revenues in by_brand.items():
            # Sort by year
            brand_revenues.sort(key=lambda x: x.year)
            
            # Require at least 2 years for validation
            if len(brand_revenues) < 2:
                if self.debug:
                    print(f"  {brand}: Only {len(brand_revenues)} years, keeping")
                validated.extend(brand_revenues)
                continue
            
            # Check for reasonable year-over-year changes
            for i in range(1, len(brand_revenues)):
                prev_rev = brand_revenues[i-1]
                curr_rev = brand_revenues[i]
                
                yoy_change = abs(curr_rev.amount_usd - prev_rev.amount_usd) / prev_rev.amount_usd
                
                # Flag but don't reject large changes (>200% YoY)
                if yoy_change > 2.0:
                    curr_rev.confidence *= 0.7  # Reduce confidence
                    if self.debug:
                        print(f"  {brand}: Large YoY change {yoy_change:.1%} in {curr_rev.year}")
            
            validated.extend(brand_revenues)
        
        return validated


def extract_product_revenues_deterministic(
    doc_html: str,
    brand_aliases: List[str],
    accession_id: str = "",
    debug: bool = False
) -> Dict[str, float]:
    """
    Main entry point for deterministic product revenue extraction.
    
    Returns:
        Dict mapping year -> revenue_usd (compatible with existing interface)
    """
    
    parser = DeterministicTableParser()
    parser.debug = debug
    
    revenues = parser.extract_product_revenues(doc_html, brand_aliases, accession_id)
    
    # Convert to simple year -> amount mapping (take max per year)
    year_revenues = {}
    for rev in revenues:
        year_str = str(rev.year)
        year_revenues[year_str] = max(year_revenues.get(year_str, 0.0), rev.amount_usd)
    
    return year_revenues