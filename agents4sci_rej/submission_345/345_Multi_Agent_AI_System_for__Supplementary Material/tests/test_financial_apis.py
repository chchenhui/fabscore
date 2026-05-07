#!/usr/bin/env python3
"""
Test financial data APIs as alternatives to SEC text parsing
"""

import requests
import json
from typing import Dict, Any

def test_alpha_vantage():
    """Test Alpha Vantage earnings API"""
    # Note: Requires API key, but has free tier
    print("Alpha Vantage: Requires API key but has structured earnings data")
    print("  - 5 calls/minute free")
    print("  - Annual/quarterly earnings")
    print("  - Example: GET https://www.alphavantage.co/query?function=EARNINGS&symbol=MRCK&apikey=YOUR_KEY")

def test_yahoo_finance():
    """Test Yahoo Finance via yfinance"""
    try:
        import yfinance as yf
        print("Yahoo Finance (yfinance): Available")
        
        # Test with Merck (Keytruda maker)
        ticker = yf.Ticker("MRK")
        financials = ticker.financials
        print(f"  - Can get revenue data for multiple years")
        print(f"  - Example revenue columns: {list(financials.columns)[:3] if not financials.empty else 'No data'}")
        return True
    except ImportError:
        print("Yahoo Finance (yfinance): Not installed")
        print("  - Install: pip install yfinance")
        return False

def test_financial_modeling_prep():
    """Test Financial Modeling Prep API"""
    print("Financial Modeling Prep: Free tier available")
    print("  - 250 calls/day free")
    print("  - Income statements API")
    print("  - Example: GET https://financialmodelingprep.com/api/v3/income-statement/MRK?limit=120&apikey=YOUR_KEY")

def test_sec_edgar_api():
    """Test newer SEC EDGAR API"""
    print("SEC EDGAR API: Structured financial data")
    print("  - https://data.sec.gov/api/xbrl/")
    print("  - XBRL facts API for structured data")
    print("  - No API key required")
    
    # Test basic endpoint
    try:
        url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000078003.json"
        headers = {"User-Agent": "AI.MED Research Bot contact@example.com"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  SUCCESS: Working! Found company facts for CIK 0000078003")
            
            # Check if we have revenue data
            if 'facts' in data and 'us-gaap' in data['facts']:
                gaap_facts = data['facts']['us-gaap']
                revenue_fields = [k for k in gaap_facts.keys() if 'revenue' in k.lower()]
                print(f"  - Revenue fields available: {revenue_fields[:3]}")
                return True
            return False
        else:
            print(f"  FAILED: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    print("=== Financial Data API Options ===\n")
    
    print("1. SEC EDGAR XBRL API (Structured Data)")
    edgar_works = test_sec_edgar_api()
    
    print("\n2. Yahoo Finance")
    yahoo_works = test_yahoo_finance()
    
    print("\n3. Alpha Vantage")
    test_alpha_vantage()
    
    print("\n4. Financial Modeling Prep")
    test_financial_modeling_prep()
    
    print("\n=== Recommendation ===")
    if edgar_works:
        print("SUCCESS: SEC EDGAR XBRL API: Best option - free, structured, official")
        print("  - No API key required")
        print("  - Structured XBRL financial data")
        print("  - Can get historical revenue by year")
    elif yahoo_works:
        print("SUCCESS: Yahoo Finance: Good fallback - free, easy to use")
        print("  - No API key required")
        print("  - Annual financial statements")
    else:
        print("WARNING: Consider paid APIs or install yfinance")

if __name__ == "__main__":
    main()