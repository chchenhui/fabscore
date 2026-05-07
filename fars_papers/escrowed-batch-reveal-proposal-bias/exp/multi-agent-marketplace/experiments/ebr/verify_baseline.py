#!/usr/bin/env python3
"""Minimal baseline verification: run one position-bias experiment.

Verifies the end-to-end pipeline: database, API calls, agent loop, SQLite export.
Uses gemini-2.5-flash via MAAS proxy (openai provider), contractors_first, 1 run.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "packages" / "magentic-marketplace" / "src"))

load_dotenv(repo_root / ".env")

os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "gemini-2.5-flash"

from magentic_marketplace.experiments.run_experiment import run_marketplace_experiment


async def main():
    data_dir = repo_root / "data" / "position_bias" / "contractors_first"
    export_dir = repo_root / "experiments" / "ebr" / "results"
    export_dir.mkdir(exist_ok=True, parents=True)

    experiment_name = "verify_baseline_gemini_2_5_flash_r1"
    db_filename = "verify_baseline_gemini_2_5_flash_run1.db"
    db_path = export_dir / db_filename

    if db_path.exists():
        db_path.unlink()

    print(f"Running baseline verification: {experiment_name}")
    print(f"Data dir: {data_dir}")
    print(f"Export: {db_path}")
    print(f"Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"Model: {os.environ.get('LLM_MODEL')}")
    print(f"Base URL: {os.environ.get('OPENAI_BASE_URL', 'default')}")

    await run_marketplace_experiment(
        data_dir=data_dir,
        experiment_name=experiment_name,
        search_algorithm="simple",
        search_bandwidth=10,
        customer_max_steps=100,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_password="",
        override=True,
        export_sqlite=True,
        export_dir=str(export_dir),
        export_filename=db_filename,
    )

    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nSQLite DB created successfully: {db_path}")
        print(f"Tables: {tables}")
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
            print(f"  {t}: {count} rows")
        conn.close()
        print("\nBaseline verification PASSED")
    else:
        print("\nERROR: SQLite DB not created!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
