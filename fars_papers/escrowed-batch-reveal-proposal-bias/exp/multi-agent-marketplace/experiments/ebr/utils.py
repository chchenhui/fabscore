"""Shared utilities for Escrowed Batch Reveal experiments.

Common functions for result parsing, metric computation, and statistical tests
used across all experimental conditions (SoftWait, HardGate, EBR).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_experiment_db(db_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load all tables from an experiment SQLite database."""
    conn = sqlite3.connect(str(db_path))
    tables = {}
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (name,) in cursor.fetchall():
        tables[name] = pd.read_sql_query(f"SELECT * FROM [{name}]", conn)
    conn.close()
    return tables


def compute_earliest_arrival_rate(results: list[dict]) -> float:
    """Compute fraction of runs where the first-arriving proposal was paid."""
    if not results:
        return 0.0
    earliest_chosen = sum(1 for r in results if r.get("earliest_arrival_chosen", False))
    return earliest_chosen / len(results)


def compute_completion_rate(results: list[dict]) -> float:
    """Compute fraction of runs that completed with a payment."""
    if not results:
        return 0.0
    completed = sum(1 for r in results if r.get("payment_made", False))
    return completed / len(results)


def fisher_exact_test(rate_a: float, n_a: int, rate_b: float, n_b: int) -> dict:
    """Run Fisher's exact test comparing two proportions."""
    table = np.array([
        [int(rate_a * n_a), n_a - int(rate_a * n_a)],
        [int(rate_b * n_b), n_b - int(rate_b * n_b)],
    ])
    odds_ratio, p_value = stats.fisher_exact(table)
    return {"odds_ratio": odds_ratio, "p_value": p_value, "table": table.tolist()}


def chi_squared_uniformity_test(counts: list[int]) -> dict:
    """Chi-squared test for uniformity of position selection rates."""
    expected = np.mean(counts)
    chi2, p_value = stats.chisquare(counts)
    return {"chi2": chi2, "p_value": p_value, "expected": expected}
