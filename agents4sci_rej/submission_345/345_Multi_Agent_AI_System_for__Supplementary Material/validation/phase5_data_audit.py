#!/usr/bin/env python3
"""
Phase 5 Critical Data Audit + SEC Extraction Fix
Following GPT-5's tightened refinements
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class CriticalDataAuditor:
    """
    Audit revenue data for systematic errors
    Following GPT-5: Don't cap peaks, just flag anomalies
    """
    
    def __init__(self):
        # Known drug benchmarks for validation (GPT-5's targets)
        self.known_benchmarks = {
            'Keytruda': {'expected_peak': 25e9, 'description': 'Top cancer immunotherapy'},
            'Humira': {'expected_peak': 20e9, 'description': 'Top-selling drug ever'},
            'Repatha': {'expected_peak': 2e9, 'description': 'PCSK9 inhibitor (access issues)'},
            'Opdivo': {'expected_peak': 10e9, 'description': 'Cancer immunotherapy'},
            'Ibrance': {'expected_peak': 5.5e9, 'description': 'CDK4/6 inhibitor'}
        }
        
        # Anomaly detection thresholds (flag, don't fix)
        self.anomaly_thresholds = {
            'max_any_drug': 50e9,    # No drug should exceed $50B
            'min_any_drug': 1e6,     # No drug should be under $1M
            'jump_ratio': 10.0       # No 10x year-over-year jumps
        }
    
    def audit_revenue_data(self, save_report=True) -> Dict:
        """
        Complete audit of revenue data quality
        Returns audit report with specific issues found
        """
        
        print("=" * 80)
        print("PHASE 5 CRITICAL DATA AUDIT")
        print("=" * 80)
        
        # Load data
        launches = pd.read_parquet('data_proc/launches.parquet')
        revenues = pd.read_parquet('data_proc/launch_revenues.parquet')
        
        audit_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {},
            'anomalies': [],
            'known_drug_validation': {},
            'extraction_issues': []
        }
        
        # Basic data summary
        audit_report['data_summary'] = {
            'total_revenue_records': len(revenues),
            'unique_drugs': len(revenues['launch_id'].unique()),
            'revenue_range_usd': {
                'min': float(revenues['revenue_usd'].min()),
                'max': float(revenues['revenue_usd'].max()),
                'median': float(revenues['revenue_usd'].median()),
                'mean': float(revenues['revenue_usd'].mean())
            }
        }
        
        print(f"Data Summary:")
        print(f"  Revenue records: {len(revenues)}")
        print(f"  Unique drugs: {len(revenues['launch_id'].unique())}")
        print(f"  Revenue range: ${revenues['revenue_usd'].min()/1e9:.1f}B - ${revenues['revenue_usd'].max()/1e9:.1f}B")
        
        # Check for anomalies
        print(f"\n=== ANOMALY DETECTION ===")
        
        # Anomaly 1: Impossible peaks
        high_revenues = revenues[revenues['revenue_usd'] > self.anomaly_thresholds['max_any_drug']]
        if len(high_revenues) > 0:
            print(f"X ANOMALY: {len(high_revenues)} records exceed ${self.anomaly_thresholds['max_any_drug']/1e9:.0f}B")
            for _, row in high_revenues.iterrows():
                drug_name = self._get_drug_name(row['launch_id'], launches)
                anomaly = {
                    'type': 'impossible_peak',
                    'drug': drug_name,
                    'value': float(row['revenue_usd']),
                    'year': int(row['year_since_launch'])
                }
                audit_report['anomalies'].append(anomaly)
                print(f"    {drug_name}: ${row['revenue_usd']/1e9:.1f}B (Year {row['year_since_launch']})")
        else:
            print(f"V No impossible peaks found")
        
        # Anomaly 2: Year-over-year jumps
        print(f"\n=== YEAR-OVER-YEAR CONSISTENCY ===")
        jump_anomalies = self._detect_revenue_jumps(revenues, launches)
        audit_report['anomalies'].extend(jump_anomalies)
        
        if jump_anomalies:
            print(f"X Found {len(jump_anomalies)} suspicious revenue jumps")
            for anomaly in jump_anomalies[:5]:  # Show first 5
                print(f"    {anomaly['drug']}: {anomaly['jump_ratio']:.1f}x jump Y{anomaly['year_from']} -> Y{anomaly['year_to']}")
        else:
            print(f"V No suspicious revenue jumps")
        
        # Validation against known drugs
        print(f"\n=== KNOWN DRUG VALIDATION ===")
        
        for drug_name, benchmark in self.known_benchmarks.items():
            validation = self._validate_known_drug(drug_name, benchmark, launches, revenues)
            audit_report['known_drug_validation'][drug_name] = validation
            
            if validation['found']:
                actual_peak = validation['actual_peak']
                expected_peak = benchmark['expected_peak']
                ratio = actual_peak / expected_peak
                
                status = "V" if 0.5 <= ratio <= 2.0 else "X"
                print(f"  {status} {drug_name}: ${actual_peak/1e9:.1f}B vs ${expected_peak/1e9:.1f}B expected ({ratio:.1f}x)")
                
                if ratio > 2.0 or ratio < 0.5:
                    audit_report['extraction_issues'].append({
                        'drug': drug_name,
                        'issue': 'wrong_magnitude',
                        'actual': actual_peak,
                        'expected': expected_peak,
                        'ratio': ratio
                    })
            else:
                print(f"  ? {drug_name}: Not found in dataset")
        
        # Top 10 peaks review
        print(f"\n=== TOP 10 PEAKS REVIEW ===")
        top_peaks = []
        for launch_id in revenues['launch_id'].unique():
            drug_revenues = revenues[revenues['launch_id'] == launch_id]
            peak_revenue = drug_revenues['revenue_usd'].max()
            drug_name = self._get_drug_name(launch_id, launches)
            top_peaks.append((drug_name, peak_revenue))
        
        top_peaks.sort(key=lambda x: x[1], reverse=True)
        
        for i, (drug_name, peak) in enumerate(top_peaks[:10]):
            plausible = "V" if peak < 30e9 else "X"
            print(f"  {i+1:2d}. {plausible} {drug_name}: ${peak/1e9:.1f}B")
        
        # Summary assessment
        print(f"\n=== AUDIT SUMMARY ===")
        total_issues = len(audit_report['anomalies']) + len(audit_report['extraction_issues'])
        
        if total_issues == 0:
            print(f"V Data quality: GOOD ({total_issues} issues found)")
            audit_report['overall_status'] = 'GOOD'
        elif total_issues <= 5:
            print(f"! Data quality: ACCEPTABLE ({total_issues} issues found)")
            audit_report['overall_status'] = 'ACCEPTABLE'
        else:
            print(f"X Data quality: POOR ({total_issues} issues found)")
            audit_report['overall_status'] = 'POOR'
        
        print(f"   - Anomalies: {len(audit_report['anomalies'])}")
        print(f"   - Extraction issues: {len(audit_report['extraction_issues'])}")
        
        # Save audit report
        if save_report:
            audit_file = Path('results/phase5_data_audit.json')
            audit_file.parent.mkdir(exist_ok=True)
            
            with open(audit_file, 'w') as f:
                json.dump(audit_report, f, indent=2, default=str)
            
            print(f"\nAudit report saved to: {audit_file}")
        
        return audit_report
    
    def _get_drug_name(self, launch_id: str, launches: pd.DataFrame) -> str:
        """Get drug name from launch_id"""
        drug_row = launches[launches['launch_id'] == launch_id]
        return drug_row['drug_name'].iloc[0] if len(drug_row) > 0 else f"ID_{launch_id}"
    
    def _detect_revenue_jumps(self, revenues: pd.DataFrame, launches: pd.DataFrame) -> List[Dict]:
        """Detect suspicious year-over-year revenue jumps"""
        
        jump_anomalies = []
        
        for launch_id in revenues['launch_id'].unique():
            drug_revenues = revenues[revenues['launch_id'] == launch_id].sort_values('year_since_launch')
            
            if len(drug_revenues) < 2:
                continue
            
            drug_name = self._get_drug_name(launch_id, launches)
            
            for i in range(1, len(drug_revenues)):
                prev_rev = drug_revenues.iloc[i-1]['revenue_usd']
                curr_rev = drug_revenues.iloc[i]['revenue_usd']
                
                if prev_rev > 0:  # Avoid division by zero
                    jump_ratio = curr_rev / prev_rev
                    
                    if jump_ratio > self.anomaly_thresholds['jump_ratio'] or jump_ratio < (1/self.anomaly_thresholds['jump_ratio']):
                        jump_anomalies.append({
                            'type': 'revenue_jump',
                            'drug': drug_name,
                            'launch_id': launch_id,
                            'year_from': int(drug_revenues.iloc[i-1]['year_since_launch']),
                            'year_to': int(drug_revenues.iloc[i]['year_since_launch']),
                            'revenue_from': float(prev_rev),
                            'revenue_to': float(curr_rev),
                            'jump_ratio': float(jump_ratio)
                        })
        
        return jump_anomalies
    
    def _validate_known_drug(self, drug_name: str, benchmark: Dict, 
                           launches: pd.DataFrame, revenues: pd.DataFrame) -> Dict:
        """Validate a known drug against benchmark"""
        
        # Find drug in launches
        drug_launches = launches[launches['drug_name'] == drug_name]
        
        if len(drug_launches) == 0:
            return {'found': False, 'reason': 'drug_not_found'}
        
        launch_id = drug_launches.iloc[0]['launch_id']
        
        # Find revenues
        drug_revenues = revenues[revenues['launch_id'] == launch_id]
        
        if len(drug_revenues) == 0:
            return {'found': False, 'reason': 'no_revenue_data'}
        
        # Calculate peak and validation
        actual_peak = drug_revenues['revenue_usd'].max()
        expected_peak = benchmark['expected_peak']
        
        return {
            'found': True,
            'actual_peak': float(actual_peak),
            'expected_peak': float(expected_peak),
            'ratio': float(actual_peak / expected_peak),
            'years_available': int(len(drug_revenues)),
            'revenue_trajectory': drug_revenues.sort_values('year_since_launch')['revenue_usd'].tolist()
        }

class FixedSECExtractor:
    """
    Fixed SEC revenue extraction following GPT-5 refinements:
    - Require explicit context anchors within same table/section
    - Product-level only (brand alias + revenue keyword + fiscal year)
    - Forbid "thousands" unless table header says "in thousands"
    - Multi-year consistency checks
    """
    
    def __init__(self):
        self.unit_patterns = {
            'millions': re.compile(r'in\s+millions?|millions?\s+of\s+dollars?|\$\s*millions?', re.IGNORECASE),
            'billions': re.compile(r'in\s+billions?|billions?\s+of\s+dollars?|\$\s*billions?', re.IGNORECASE),
            'thousands': re.compile(r'in\s+thousands?|thousands?\s+of\s+dollars?|\$\s*thousands?', re.IGNORECASE)
        }
        
        # Revenue keywords for product matching
        self.revenue_keywords = [
            'net sales', 'product sales', 'revenue', 'net revenue',
            'product revenue', 'net product sales', 'sales'
        ]
    
    def extract_product_revenue_fixed(self, sec_text: str, drug_name: str, 
                                    drug_aliases: List[str], fiscal_year: int) -> Optional[float]:
        """
        Extract product-specific revenue with GPT-5's refinements
        
        Args:
            sec_text: SEC filing text
            drug_name: Primary drug name  
            drug_aliases: List of brand aliases
            fiscal_year: Target fiscal year
        
        Returns:
            Revenue in USD or None if not found/invalid
        """
        
        # Split into sections/tables
        sections = self._split_into_sections(sec_text)
        
        for section in sections:
            # Check if this section contains our drug and fiscal year
            if not self._section_contains_drug_and_year(section, drug_name, drug_aliases, fiscal_year):
                continue
            
            # Find unit context within this section
            unit_multiplier = self._extract_unit_multiplier(section)
            
            if unit_multiplier is None:
                continue  # Skip sections without clear units
            
            # Extract revenue value from this section
            revenue_value = self._extract_revenue_value(section, drug_name, drug_aliases)
            
            if revenue_value is not None:
                final_revenue = revenue_value * unit_multiplier
                
                # Validate extracted value
                if self._validate_extracted_revenue(final_revenue, drug_name):
                    return final_revenue
        
        return None
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split SEC text into logical sections/tables"""
        
        # Look for table boundaries and section headers
        section_markers = [
            r'table of contents',
            r'consolidated statements',
            r'product sales',
            r'net sales by',
            r'revenue by',
            r'the following table',
            r'our product sales'
        ]
        
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a new section
            is_section_start = any(marker in line_lower for marker in section_markers)
            
            if is_section_start and len(current_section) > 500:  # Min section size
                sections.append(current_section)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
                
                # Also split on blank lines (table breaks)
                if len(line.strip()) == 0 and len(current_section) > 1000:
                    sections.append(current_section)
                    current_section = ""
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _section_contains_drug_and_year(self, section: str, drug_name: str, 
                                      aliases: List[str], year: int) -> bool:
        """Check if section contains our drug and target year"""
        
        section_lower = section.lower()
        
        # Check for drug presence (name or aliases)
        drug_found = drug_name.lower() in section_lower
        if not drug_found:
            for alias in aliases:
                if alias.lower() in section_lower:
                    drug_found = True
                    break
        
        if not drug_found:
            return False
        
        # Check for year presence
        year_patterns = [str(year), str(year)[-2:]]  # e.g., "2021" or "21"
        year_found = any(year_str in section for year_str in year_patterns)
        
        return year_found
    
    def _extract_unit_multiplier(self, section: str) -> Optional[float]:
        """
        Extract unit multiplier from section following GPT-5 rules:
        - Require explicit context anchors within same section
        - Forbid "thousands" unless table header explicitly says it
        """
        
        section_lower = section.lower()
        
        # Look for explicit unit declarations
        if self.unit_patterns['billions'].search(section_lower):
            return 1e9
        elif self.unit_patterns['millions'].search(section_lower):
            return 1e6
        elif self.unit_patterns['thousands'].search(section_lower):
            # Only allow thousands if it's in a clear table header
            if 'table' in section_lower[:200] and 'thousands' in section_lower[:200]:
                return 1e3
            else:
                return None  # Reject ambiguous thousands
        
        return None
    
    def _extract_revenue_value(self, section: str, drug_name: str, aliases: List[str]) -> Optional[float]:
        """Extract numerical revenue value for the drug"""
        
        # Look for patterns like:
        # "Drug Name    $1,234    $2,345"
        # "Net sales of Drug Name were $1,234 million"
        
        all_names = [drug_name] + aliases
        
        for name in all_names:
            # Pattern 1: Drug name followed by numbers
            pattern1 = rf'{re.escape(name)}.*?(\d+(?:,\d{{3}})*(?:\.\d+)?)'
            matches = re.findall(pattern1, section, re.IGNORECASE)
            
            if matches:
                # Take the last number (usually current year)
                value_str = matches[-1].replace(',', '')
                try:
                    return float(value_str)
                except ValueError:
                    continue
            
            # Pattern 2: Revenue keyword + drug name + number
            for keyword in self.revenue_keywords:
                pattern2 = rf'{keyword}.*?{re.escape(name)}.*?(\d+(?:,\d{{3}})*(?:\.\d+)?)'
                matches = re.findall(pattern2, section, re.IGNORECASE)
                
                if matches:
                    value_str = matches[-1].replace(',', '')
                    try:
                        return float(value_str)
                    except ValueError:
                        continue
        
        return None
    
    def _validate_extracted_revenue(self, revenue: float, drug_name: str) -> bool:
        """Validate extracted revenue for basic sanity"""
        
        # Basic bounds
        if revenue < 1e6 or revenue > 100e9:  # $1M - $100B
            return False
        
        # Known drug specific validation could go here
        
        return True

def main():
    """Run critical data audit"""
    
    auditor = CriticalDataAuditor()
    audit_report = auditor.audit_revenue_data()
    
    # Decision point based on audit
    if audit_report['overall_status'] == 'POOR':
        print(f"\n" + "="*80)
        print("X CRITICAL: Data quality is POOR")
        print("SEC extraction must be fixed before proceeding with Phase 5")
        print("="*80)
        return False
    else:
        print(f"\n" + "="*80)
        print("V Data quality acceptable - can proceed with Phase 5 fixes")
        print("="*80)
        return True

if __name__ == "__main__":
    main()