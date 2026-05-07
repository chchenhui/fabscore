"""
Collect real launches and 5-year revenues using FDA, CT.gov, and SEC endpoints.
Writes normalized parquet tables satisfying G1 requirements (when data suffices).
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd

from .sources import (
    fetch_fda_drug,
    fetch_ctgov_summary,
    get_cik_for_ticker,
    fetch_company_xbrl_facts,
    extract_revenue_from_xbrl,
    fetch_10k_texts,
    extract_brand_revenue,
)

try:
    from ..models.ta_priors import apply_ta_priors
except ImportError:
    # Fallback if ta_priors not available
    def apply_ta_priors(drug_row, imputation_log=None):
        return drug_row


def load_brands(seed_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(seed_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_real_dataset(seed_csv: Path,
                       years: Tuple[int, int] = (2015, 2024),
                       output_dir: Path = Path("data_proc")) -> bool:
    brands = load_brands(seed_csv)

    launches_rows: List[Dict[str, Any]] = []
    revenues_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(brands):
        brand = row.get("brand", "").strip()
        company = row.get("company", "").strip()
        ticker = row.get("ticker", "").strip()
        aliases = [a.strip() for a in (row.get("aliases", "") or "").split(";") if a.strip()]
        ta = row.get("therapeutic_area", "Unknown")

        if not brand or not company:
            continue

        launch_id = f"REAL_{idx:04d}"

        # FDA info (comprehensive extraction)
        fda = fetch_fda_drug(brand)
        fda_extracted = fda.get('extracted', {})
        
        # Core approval data
        approval_date = fda_extracted.get('original_approval_date')
        review_priority = fda_extracted.get('review_priority')
        application_number = fda_extracted.get('application_number')
        
        # Drug characteristics
        route = fda_extracted.get('route')
        mechanism = fda_extracted.get('mechanism_of_action')
        dosage_form = fda_extracted.get('dosage_form')
        generic_name = fda_extracted.get('generic_name')
        
        # Competitive intelligence
        approved_supplementals_count = fda_extracted.get('approved_supplementals_count', 0)
        first_supplemental_date = fda_extracted.get('first_supplemental_date')
        
        # Market access indicators
        sponsor_name = fda_extracted.get('sponsor_name')
        marketing_status = fda_extracted.get('marketing_status')
        manufacturer_name = fda_extracted.get('manufacturer_name')

        # CT.gov summary (optional enrichment)
        ct = fetch_ctgov_summary(brand)
        
        # Use therapeutic area as indication fallback (set above in launches row)

        # SEC revenues using XBRL API (much better than text parsing)
        cik = get_cik_for_ticker(ticker) if ticker else None
        
        # Get approval year for mapping fiscal years to launch-relative years
        approval_year = None
        if approval_date:
            try:
                approval_year = int(str(approval_date)[:4])
            except Exception:
                approval_year = None

        # Extract revenues: First try 10-K text (product-specific), then XBRL (company totals)
        revenue_by_year: Dict[str, float] = {}
        if cik and approval_year:
            # Primary: Extract from 10-K text documents (product-specific mentions)
            doc_texts = fetch_10k_texts(cik, max_docs=3)
            for doc_text in doc_texts:
                text_revenues = extract_brand_revenue(doc_text, [brand] + aliases)
                # Map calendar years to years since launch
                for cal_year_str, amt in text_revenues.items():
                    try:
                        cal_year = int(cal_year_str)
                        year_since = cal_year - approval_year
                        if 0 <= year_since <= 15:  # Expand range to capture mature drugs
                            current = revenue_by_year.get(str(year_since), 0.0)
                            revenue_by_year[str(year_since)] = max(current, amt)
                    except (ValueError, TypeError):
                        continue
            
            # Fallback: If no text extraction, use XBRL (but flag as company total)
            if not revenue_by_year:
                xbrl_data = fetch_company_xbrl_facts(cik)
                xbrl_revenues = extract_revenue_from_xbrl(
                    xbrl_data, 
                    approval_year, 
                    [brand] + aliases
                )
                if xbrl_revenues:
                    revenue_by_year = xbrl_revenues
                    revenue_by_year['_source'] = 'xbrl_company_total'

        # Build launches row with comprehensive FDA data
        launches_rows.append({
            # Core identifiers
            "launch_id": launch_id,
            "drug_name": brand,
            "company": company,
            "ticker": ticker,
            
            # FDA regulatory data
            "approval_date": approval_date or "",
            "review_priority": review_priority or "",
            "application_number": application_number or "",
            "generic_name": generic_name or "",
            
            # Drug characteristics  
            "indication": ta,  # Use therapeutic area as indication
            "mechanism": mechanism or "",
            "route": route or "",
            "dosage_form": dosage_form or "",
            "therapeutic_area": ta,
            
            # Competitive intelligence
            "approved_supplementals_count": approved_supplementals_count,
            "first_supplemental_date": first_supplemental_date or "",
            
            # Market access indicators
            "sponsor_name": sponsor_name or company,  # Use CSV company as fallback
            "marketing_status": marketing_status or "",
            "manufacturer_name": manufacturer_name or "",
            
            # Fields still needing enrichment (pricing/market data)
            "eligible_patients_at_launch": 0,
            "market_size_source": "",
            "list_price_month_usd_launch": 0.0,
            "net_gtn_pct_launch": 0.0,
            "access_tier_at_launch": "",
            "price_source": "",
            "competitor_count_at_launch": 0,
            "clinical_efficacy_proxy": 0.0,
            "safety_black_box": False,
            
            # Data sources
            "source_urls": json.dumps([f"FDA:{application_number}" if application_number else "FDA:Unknown"]),
        })

        # Revenues: mapped to year_since_launch
        if revenue_by_year:
            source_type = revenue_by_year.get('_source', '10k_text')
            for year_since_str, amt in revenue_by_year.items():
                if year_since_str == '_source':  # Skip metadata
                    continue
                try:
                    year_since = int(year_since_str)
                    # Include Y0 through Y10 (extended to capture available data)
                    if 0 <= year_since <= 10:
                        source_url = f"SEC 10-K CIK {cik}" if source_type == '10k_text' else f"SEC XBRL CIK {cik}"
                        revenues_rows.append({
                            "launch_id": launch_id,
                            "year_since_launch": year_since,
                            "revenue_usd": float(amt),
                            "source_url": source_url
                        })
                except Exception:
                    continue

    # Apply TA priors to fill missing market/pricing data
    if launches_rows:
        launches_df = pd.DataFrame(launches_rows)
        print(f"Applying TA priors to {len(launches_df)} drugs...")
        
        imputation_logs = []
        enhanced_launches = []
        
        for idx, drug_row in launches_df.iterrows():
            imputation_log = {}
            enhanced_drug = apply_ta_priors(drug_row, imputation_log)
            enhanced_launches.append(enhanced_drug)
            
            if imputation_log:
                imputation_log['launch_id'] = enhanced_drug['launch_id']
                imputation_log['drug_name'] = enhanced_drug['drug_name']
                imputation_logs.append(imputation_log)
        
        launches_df = pd.DataFrame(enhanced_launches)
        
        # Log imputation summary
        if imputation_logs:
            print(f"Applied TA priors to {len(imputation_logs)} drugs")
            price_imputations = sum(1 for log in imputation_logs if log.get('price_imputed'))
            gtn_imputations = sum(1 for log in imputation_logs if log.get('gtn_imputed'))
            market_imputations = sum(1 for log in imputation_logs if log.get('market_size_imputed'))
            print(f"  - Price priors: {price_imputations} drugs")
            print(f"  - GTN priors: {gtn_imputations} drugs")
            print(f"  - Market size priors: {market_imputations} drugs")

    # Write outputs
    output_dir.mkdir(exist_ok=True, parents=True)
    if launches_rows:
        launches_df.to_parquet(output_dir / "launches.parquet", index=False)
        
        # Save imputation log for audit trail
        if imputation_logs:
            pd.DataFrame(imputation_logs).to_parquet(output_dir / "ta_imputations.parquet", index=False)
    
    if revenues_rows:
        pd.DataFrame(revenues_rows).to_parquet(output_dir / "launch_revenues.parquet", index=False)

    # Analog table is optional here; can be created by existing builder later
    return bool(launches_rows)


if __name__ == "__main__":
    # Minimal manual test (expects data_raw/brands.csv)
    seed = Path("data_raw/brands.csv")
    if seed.exists():
        ok = build_real_dataset(seed)
        print("Build status:", ok)
    else:
        print("brands.csv not found; create data_raw/brands.csv first")


