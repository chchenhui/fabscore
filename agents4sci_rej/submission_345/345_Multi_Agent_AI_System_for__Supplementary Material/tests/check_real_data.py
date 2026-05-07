#!/usr/bin/env python3
"""
Quick check of real pharmaceutical data.
"""

import pandas as pd

print('=== REAL PHARMACEUTICAL DATA EXPLORATION ===')
print()

# Load data
launches = pd.read_parquet('data_proc/launches.parquet')
revenues = pd.read_parquet('data_proc/launch_revenues.parquet')
analogs = pd.read_parquet('data_proc/analogs.parquet')

print(f'LAUNCHES: {len(launches)} drugs')
print(f'REVENUES: {len(revenues)} records')
print(f'ANALOGS: {len(analogs)} mappings')
print()

print('THERAPEUTIC AREAS:')
ta_counts = launches['therapeutic_area'].value_counts()
print(ta_counts)
print(f'Total TAs: {len(ta_counts)}')
print()

print('SAMPLE DRUGS:')
for i, row in launches.head(5).iterrows():
    print(f'  {row["drug_name"]} ({row["company"]}) - {row["therapeutic_area"]}')
print()

print('REVENUE RANGE:')
revenue_stats = revenues.groupby('launch_id')['revenue_usd'].agg(['min', 'max', 'sum'])
print(f'Min total revenue: ${revenue_stats["sum"].min():,.0f}')
print(f'Max total revenue: ${revenue_stats["sum"].max():,.0f}')
print(f'Median total revenue: ${revenue_stats["sum"].median():,.0f}')
print()

print('DATA VALIDATION:')
print(f'Missing values in launches: {launches.isnull().sum().sum()}')
print(f'Missing values in revenues: {revenues.isnull().sum().sum()}')
print(f'Unique drugs with revenues: {revenues["launch_id"].nunique()}')
print()

# Check if this meets G1 criteria
n_launches = len(launches)
n_tas = len(ta_counts)
print('GATE G1 CHECK:')
print(f'Launches: {n_launches} (need >=50) - {"PASS" if n_launches >= 50 else "FAIL"}')
print(f'Therapeutic Areas: {n_tas} (need >=5) - {"PASS" if n_tas >= 5 else "FAIL"}')
print(f'Overall G1: {"PASS" if n_launches >= 50 and n_tas >= 5 else "FAIL"}')