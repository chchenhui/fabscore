# Load and preprocess Music Arena pairwise preference data from HuggingFace.
# Dataset: music-arena/music-arena-dataset (~3000 battles, 4-way outcomes).
# Provides chronological train/test split, stats reporting, and stratified subsets.

import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets

CONFIGS = ['2025_07-08', '2025_09', '2025_10', '2025_11', '2025_12', '2026_01']
COLUMNS = ['date', 'system_a', 'system_b', 'preference', 'is_instrumental']
OUTCOMES = ['A', 'B', 'TIE', 'BOTH_BAD']


def load_all_battles() -> pd.DataFrame:
    all_ds = []
    for cfg in CONFIGS:
        ds = load_dataset('music-arena/music-arena-dataset', cfg, split='train')
        all_ds.append(ds)
    combined = concatenate_datasets(all_ds)
    df = combined.to_pandas()
    df = df[COLUMNS].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    split_idx = int(n * train_frac)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def compute_stats(df: pd.DataFrame, label: str = "all") -> dict:
    n = len(df)
    outcome_counts = df['preference'].value_counts().to_dict()
    for o in OUTCOMES:
        outcome_counts.setdefault(o, 0)
    systems = set(df['system_a'].unique()) | set(df['system_b'].unique())
    bothbad_count = outcome_counts.get('BOTH_BAD', 0)
    bothbad_rate = bothbad_count / n if n > 0 else 0.0
    return {
        'label': label,
        'total_battles': n,
        'unique_systems': len(systems),
        'outcome_counts': {o: int(outcome_counts.get(o, 0)) for o in OUTCOMES},
        'bothbad_count': int(bothbad_count),
        'bothbad_rate': float(bothbad_rate),
    }


def check_underpower(test_stats: dict) -> dict:
    flag = False
    reasons = []
    if test_stats['bothbad_count'] < 50:
        flag = True
        reasons.append(f"BOTH_BAD count ({test_stats['bothbad_count']}) < 50")
    if test_stats['bothbad_rate'] < 0.05:
        flag = True
        reasons.append(f"BOTH_BAD rate ({test_stats['bothbad_rate']:.4f}) < 5%")
    return {'underpowered': flag, 'reasons': reasons}


def load_and_split(train_frac: float = 0.7):
    df = load_all_battles()
    train_df, test_df = chronological_split(df, train_frac)

    all_stats = compute_stats(df, 'all')
    train_stats = compute_stats(train_df, 'train')
    test_stats = compute_stats(test_df, 'test')
    underpower = check_underpower(test_stats)

    train_instrumental = train_df[train_df['is_instrumental'] == True].reset_index(drop=True)
    train_vocal = train_df[train_df['is_instrumental'] == False].reset_index(drop=True)
    test_instrumental = test_df[test_df['is_instrumental'] == True].reset_index(drop=True)
    test_vocal = test_df[test_df['is_instrumental'] == False].reset_index(drop=True)

    stats = {
        'all': all_stats,
        'train': train_stats,
        'test': test_stats,
        'underpower_gate': underpower,
        'train_instrumental': compute_stats(train_instrumental, 'train_instrumental'),
        'train_vocal': compute_stats(train_vocal, 'train_vocal'),
        'test_instrumental': compute_stats(test_instrumental, 'test_instrumental'),
        'test_vocal': compute_stats(test_vocal, 'test_vocal'),
    }

    splits = {
        'train': train_df,
        'test': test_df,
        'train_instrumental': train_instrumental,
        'train_vocal': train_vocal,
        'test_instrumental': test_instrumental,
        'test_vocal': test_vocal,
    }

    return splits, stats
