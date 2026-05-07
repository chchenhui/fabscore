"""Shared experiment runner for all EBR conditions (SoftWait, HardGate, EBR).

Parameterized by condition name, model config, data folders, and number of runs.
Monkey-patches CustomerAgent for non-standard conditions before calling
run_marketplace_experiment().
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "magentic-marketplace" / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

load_dotenv(REPO_ROOT / ".env")

from magentic_marketplace.experiments.run_experiment import run_marketplace_experiment

DEFAULT_DATA_FOLDERS = ["contractors_first", "contractors_second", "contractors_third"]


def _get_agent_class(condition: str):
    if condition == "softwait":
        from ebr.agents.softwait_agent import SoftWaitCustomerAgent
        return SoftWaitCustomerAgent
    if condition == "its":
        from ebr.agents.its_agent import ITSCustomerAgent
        return ITSCustomerAgent
    if condition == "hardgate":
        from ebr.agents.hardgate_agent import HardGateCustomerAgent
        return HardGateCustomerAgent
    if condition == "ebr":
        from ebr.agents.ebr_agent import EBRCustomerAgent
        return EBRCustomerAgent
    return None


async def run_condition_experiments(
    condition: str,
    model_config: dict,
    data_folders: list[str] | None = None,
    num_runs: int = 10,
    K: int = 3,
    export_dir: str = "experiments/ebr/results",
):
    if data_folders is None:
        data_folders = DEFAULT_DATA_FOLDERS

    provider = model_config.get("provider", "openai")
    model = model_config.get("model", "gemini-2.5-flash")
    clean_model = model.replace("-", "_").replace(".", "_").replace("/", "_")

    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    os.environ["LLM_TEMPERATURE"] = "0.7"

    export_path = REPO_ROOT / export_dir
    export_path.mkdir(exist_ok=True, parents=True)

    agent_class = _get_agent_class(condition)

    total = len(data_folders) * num_runs
    completed = 0

    for folder in data_folders:
        for run_num in range(1, num_runs + 1):
            experiment_name = f"ebr_{condition}_{folder}_{clean_model}_r{run_num}"
            db_filename = f"ebr_{condition}_{folder}_{clean_model}_run{run_num}.db"
            db_path = export_path / db_filename

            if db_path.exists():
                completed += 1
                print(f"[{completed}/{total}] Skipping {experiment_name} -- already exists")
                continue

            completed += 1
            print(f"\n[{completed}/{total}] Running: {experiment_name}")

            data_dir = REPO_ROOT / "data" / "position_bias" / folder

            run_kwargs = dict(
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
                export_dir=str(export_path),
                export_filename=db_filename,
            )

            try:
                if agent_class is not None:
                    with patch(
                        "magentic_marketplace.experiments.run_experiment.CustomerAgent",
                        agent_class,
                    ):
                        await run_marketplace_experiment(**run_kwargs)
                else:
                    await run_marketplace_experiment(**run_kwargs)
                print(f"  -> Completed: {db_filename}")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nAll {condition} experiments finished. Results in: {export_path}")


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["softwait", "its", "hardgate", "ebr"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--export-dir", default="experiments/ebr/results/softwait_gemini")
    parser.add_argument("--folders", nargs="+", default=None)
    args = parser.parse_args()

    await run_condition_experiments(
        condition=args.condition,
        model_config={"provider": args.provider, "model": args.model},
        data_folders=args.folders,
        num_runs=args.num_runs,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
