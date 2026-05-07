import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from loguru import logger
import re


class SimpleSceneMetricsCollector:
    """
    Simplified collector:
    - For each metric category under each step, only read the latest JSON file (by mtime)
      within that step (do not keep intra-step timestamps).
    - Support collecting from the latest N runs (run folders named with timestamps) per group.
    - Output one merged JSON per category across all groups/runs/steps.
    """

    def __init__(
        self,
        groups_base_path: str,
        output_dir: str,
        latest_runs: int = 1,
    ):
        self.groups_base_path = Path(groups_base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latest_runs = max(1, int(latest_runs))

        self.group_dirs = self._discover_group_directories()
        logger.info(f"Found {len(self.group_dirs)} groups: {[g.name for g in self.group_dirs]}")

        # Stats containers
        self.stats: Dict[str, Any] = {
            "selected_runs_per_group": {},  # group -> [run]
            "steps_per_group_run": {},      # (group, run) -> [step]
            "categories": set(),
            "actual_keys": set(),          # (category, group, run, step)
        }

    def _discover_group_directories(self) -> List[Path]:
        group_dirs: List[Path] = []
        if self.groups_base_path.exists():
            for item in self.groups_base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    group_dirs.append(item)
        return sorted(group_dirs)

    def _discover_run_directories(self, group_path: Path) -> List[Path]:
        """Return all TIMESTAMP-named run directories under group/runs sorted by timestamp desc."""
        runs_path = group_path / "runs"
        if not runs_path.exists():
            logger.warning(f"No runs directory for group: {group_path}")
            return []
        run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]

        def extract_ts(name: str) -> Optional[str]:
            m = re.search(r"\d{8}_\d{6}", name)
            return m.group(0) if m else None

        # Only keep timestamp-named run folders
        ts_runs = [d for d in run_dirs if extract_ts(d.name) is not None]
        # sort by timestamp descending (lex order works for YYYYMMDD_HHMMSS)
        return sorted(ts_runs, key=lambda p: extract_ts(p.name), reverse=True)

    def _discover_steps(self, run_path: Path) -> List[Tuple[str, Path]]:
        metrics_plots_path = run_path / "metrics_plots"
        if not metrics_plots_path.exists():
            logger.warning(f"No metrics_plots directory found: {metrics_plots_path}")
            return []
        steps: List[Tuple[str, Path]] = []
        for d in metrics_plots_path.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                steps.append((d.name, d))

        def step_key(item: Tuple[str, Path]) -> Tuple[int, int | str]:
            name = item[0]
            m = re.search(r"step_(\d+)", name)
            if m:
                try:
                    return (0, int(m.group(1)))
                except Exception:
                    pass
            return (1, name)

        return sorted(steps, key=step_key)

    def _extract_ts_from_filename(self, file_path: Path) -> Optional[str]:
        """Extract YYYYMMDD_HHMMSS from filename; return None if not found."""
        try:
            m = re.search(r"\d{8}_\d{6}", file_path.name)
            return m.group(0) if m else None
        except Exception:
            return None

    def _select_latest_file(self, files: List[Path]) -> Optional[Path]:
        """
        Select the latest file by filename timestamp; fallback to mtime if none contain timestamps.
        """
        if not files:
            return None
        files_with_ts = [(f, self._extract_ts_from_filename(f)) for f in files]
        with_ts = [pair for pair in files_with_ts if pair[1] is not None]
        if with_ts:
            # pick by max timestamp (lex order OK for YYYYMMDD_HHMMSS)
            return max(with_ts, key=lambda p: p[1])[0]
        # fallback to mtime
        try:
            return max(files, key=lambda f: f.stat().st_mtime)
        except Exception as e:
            logger.error(f"Failed to select latest file: {e}")
            return None

    def _collect_step_metrics_once(self, group_name: str, run_name: str, step_name: str, step_dir: Path) -> Dict[str, Any]:
        """
        For each category under scene_metrics, pick only the latest JSON file by mtime,
        and do not keep intra-step timestamps.
        """
        scene_metrics_dir = step_dir / "scene_metrics"
        result: Dict[str, List[Dict[str, Any]]] = {}
        if not scene_metrics_dir.exists():
            logger.warning(f"No scene_metrics directory found: {scene_metrics_dir}")
            return result

        for category_dir in scene_metrics_dir.iterdir():
            if not category_dir.is_dir():
                continue
            json_files = list(category_dir.glob("*.json"))
            if not json_files:
                continue
            latest_file = self._select_latest_file(json_files)
            if latest_file is None:
                continue
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Collapse time-series payloads with timestamp xAxis into a single last value
                data = self._collapse_time_series_payload(data)
                item = {
                    "group_name": group_name,
                    "run": run_name,
                    "step": step_name,
                    "category": category_dir.name,
                    "filename": latest_file.name,
                    "data": data,
                }
                result.setdefault(category_dir.name, []).append(item)
            except Exception as e:
                logger.error(f"Failed to read {latest_file}: {e}")
        return result

    def _is_timestamp_like(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        patterns = [
            r"^\d{2}:\d{2}:\d{2}$",                    # HH:MM:SS
            r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$",  # YYYY-MM-DD HH:MM(:SS)
            r"^\d{8}_\d{6}$",                           # YYYYMMDD_HHMMSS
            r"^\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}(:\d{2})?$",  # YYYY/MM/DD HH:MM(:SS)
        ]
        for p in patterns:
            if re.match(p, value):
                return True
        return False

    def _collapse_time_series_payload(self, payload: Any) -> Any:
        """
        If payload looks like a time-series with timestamp xAxis and series values,
        collapse to the last numeric value. Otherwise return payload unchanged.
        Supported shapes:
        - { xAxis: [...timestamps...], series: { <name>: [v1, v2, ...] } }
        - { xAxis: [...timestamps...], series: [v1, v2, ...] }
        - { xAxis: [...timestamps...], series: [ { data: [...] }, ... ] } -> pick first series.data
        """
        try:
            if not isinstance(payload, dict):
                return payload
            x_axis = payload.get("xAxis")
            series = payload.get("series")
            if not isinstance(x_axis, list) or not x_axis:
                return payload
            # verify xAxis looks timestamp-like for majority of entries
            sample = [v for v in x_axis[:3] if isinstance(v, str)]
            if not sample or not all(self._is_timestamp_like(s) for s in sample):
                return payload

            # extract series values vector
            values: Optional[List[Any]] = None
            if isinstance(series, dict) and series:
                # prefer 'default' key if present, else first key
                chosen_key = 'default' if 'default' in series else sorted(series.keys())[0]
                maybe = series.get(chosen_key)
                if isinstance(maybe, list):
                    values = maybe
            elif isinstance(series, list) and series:
                # could be list of numbers
                if all(isinstance(x, (int, float)) for x in series):
                    values = series  # type: ignore
                else:
                    # or list of series objects with data
                    first = series[0]
                    if isinstance(first, dict) and isinstance(first.get('data'), list):
                        values = first.get('data')  # type: ignore

            if not isinstance(values, list) or not values:
                return payload

            # align lengths just in case
            last_index = min(len(x_axis), len(values)) - 1
            if last_index < 0:
                return payload
            last_value = values[last_index]

            # ensure numeric if possible
            if isinstance(last_value, (int, float)):
                return last_value
            # try to coerce to number
            try:
                return float(last_value)
            except Exception:
                return last_value
        except Exception as e:
            logger.debug(f"Payload collapse skipped due to error: {e}")
            return payload

    def collect(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect metrics across groups, limited to latest N runs, picking only latest file per step/category."""
        all_data_by_category: Dict[str, List[Dict[str, Any]]] = {}
        # Reset stats
        self.stats["selected_runs_per_group"] = {}
        self.stats["steps_per_group_run"] = {}
        self.stats["categories"] = set()
        self.stats["actual_keys"] = set()

        for group_path in self.group_dirs:
            logger.info(f"Processing group: {group_path.name}")
            all_runs = self._discover_run_directories(group_path)
            if not all_runs:
                logger.warning(f"Group {group_path.name} has no available runs, skipping")
                continue
            selected_runs = all_runs[: self.latest_runs]
            logger.debug(f"Selected runs: {[r.name for r in selected_runs]}")
            self.stats["selected_runs_per_group"][group_path.name] = [r.name for r in selected_runs]
            for run_path in selected_runs:
                steps = self._discover_steps(run_path)
                self.stats["steps_per_group_run"][(group_path.name, run_path.name)] = [s for s, _ in steps]
                for step_name, step_dir in steps:
                    logger.debug(f"Step: {step_name}")
                    step_data = self._collect_step_metrics_once(group_path.name, run_path.name, step_name, step_dir)
                    for category, items in step_data.items():
                        self.stats["categories"].add(category)
                        for it in items:
                            self.stats["actual_keys"].add((category, it["group_name"], it["run"], it["step"]))
                        all_data_by_category.setdefault(category, []).extend(items)
        return all_data_by_category

    def _generate_file_header(self, category_name: str) -> Dict[str, Any]:
        return {
            "file_info": {
                "generated_at": datetime.now().isoformat(),
                "category": category_name,
                "source_experiment": "dynamic_culture_dissemination",
                "description": f"Collected {category_name} metric data across groups, latest N runs, latest file per step.",
            }
        }

    def save(self) -> None:
        collected_data = self.collect()
        saved_files: List[Dict[str, Any]] = []

        for category, category_data in collected_data.items():
            if not category_data:
                continue
            safe_filename = category.replace(" ", "_").replace("/", "_")
            output_file = self.output_dir / f"{safe_filename}_all_groups.json"
            output_data = {
                **self._generate_file_header(category),
                "data": category_data,
                "summary": {
                    "total_data_points": len(category_data),
                    "groups_with_data": sorted(list({item['group_name'] for item in category_data})),
                    "steps_with_data": sorted(list({item['step'] for item in category_data})),
                },
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            saved_files.append({
                "category": category,
                "file_path": str(output_file),
                "data_points": len(category_data),
            })
            logger.debug(f"Saved: {output_file} ({len(category_data)} data points)")

        summary_file = self.output_dir / "collection_summary_all_groups.json"
        summary_data = {
            "collection_info": {
                "generated_at": datetime.now().isoformat(),
                "source_path": str(self.groups_base_path),
                "output_directory": str(self.output_dir),
                "total_groups_processed": len(self.group_dirs),
                "groups_processed": [g.name for g in self.group_dirs],
                "latest_runs": self.latest_runs,
            },
            "files_generated": saved_files,
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        # Statistics & validation
        self._compute_and_log_expectations(collected_data)

        logger.success("Collection completed!")
        logger.info(f"Groups processed: {len(self.group_dirs)} | Metric files generated: {len(saved_files)} | Summary file: {summary_file} | Output directory: {self.output_dir}")

    def _compute_and_log_expectations(self, collected_data: Dict[str, List[Dict[str, Any]]]) -> None:
        categories: List[str] = sorted(list(self.stats["categories"]))
        steps_total = sum(len(v) for v in self.stats["steps_per_group_run"].values())
        groups_with_runs = [g for g, runs in self.stats["selected_runs_per_group"].items() if runs]
        total_groups = len(groups_with_runs)
        total_runs = sum(len(runs) for runs in self.stats["selected_runs_per_group"].values())

        # Expected count: for each (group, run, step), there should be one item per category
        expected_per_category = steps_total
        expected_total = expected_per_category * len(categories)

        actual_total = sum(len(v) for v in collected_data.values())

        logger.debug("Statistics check:")
        logger.debug(f"Number of categories: {len(categories)} -> {categories}")
        logger.debug(f"Number of groups: {total_groups}")
        logger.debug(f"Total runs collected (all groups): {total_runs}")
        logger.debug(f"Total steps (sum over all (group, run)): {steps_total}")
        logger.debug(f"Expected total data points: {expected_total} (= categories({len(categories)}) * steps_total({steps_total}))")
        logger.debug(f"Actual total data points: {actual_total}")

        if actual_total != expected_total:
            logger.warning("Actual total data points differ from expectation. Check missing combinations.")

        # Per-category check
        for category in categories:
            actual = len(collected_data.get(category, []))
            if actual != expected_per_category:
                logger.warning(f"[Category {category}] actual {actual} != expected {expected_per_category}")

        # Find missing combination samples
        expected_keys = set()
        for category in categories:
            for (group_name, run_name), steps in self.stats["steps_per_group_run"].items():
                for step_name in steps:
                    expected_keys.add((category, group_name, run_name, step_name))

        actual_keys = self.stats["actual_keys"]
        missing = list(expected_keys - actual_keys)
        if missing:
            logger.warning(f"Missing combination count: {len(missing)} (first 20 shown)")
            for tup in missing[:20]:
                category, group_name, run_name, step_name = tup
                logger.warning(f"  Missing -> category:{category}, group:{group_name}, run:{run_name}, step:{step_name}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scene metrics (simple): latest file per step per metric; support latest N runs",
    )
    parser.add_argument(
        "--groups-path",
        type=str,
        required=False,
        help="Path to the groups directory (ignored if --project-name is provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Output directory (ignored if --project-name is provided)",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project name. If provided, input becomes projects/{project_name}/groups and output becomes projects/{project_name}/analysis/data/processed",
    )
    parser.add_argument(
        "--latest-runs",
        type=int,
        default=1,
        help="Select the latest N runs (timestamp-named folders) per group",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs",
    )
    return parser.parse_args()


def main(groups_base_path: Optional[str] = None, output_dir: Optional[str] = None, latest_runs: int = 1):
    # Parse arguments if paths not provided
    if groups_base_path is None or output_dir is None:
        args = parse_arguments()
        project_name = getattr(args, 'project_name', None)
        if project_name:
            base_dir = Path.cwd() / 'projects' / project_name
            groups_base_path = str(base_dir / 'groups')
            output_dir = str(base_dir / 'analysis' / 'data' / 'processed')
            logger.info(f"Using --project-name derived directories. Base dir={base_dir}")
            logger.info(f"groups_path={groups_base_path}")
            logger.info(f"output_dir={output_dir}")
        else:
            groups_base_path = groups_base_path or args.groups_path
            output_dir = output_dir or args.output_dir
            if not groups_base_path or not output_dir:
                logger.error("When --project-name is not provided, both --groups-path and --output-dir are required.")
                return 1
        latest_runs = int(getattr(args, 'latest_runs', latest_runs))
        verbose = bool(getattr(args, 'verbose', False))
    else:
        verbose = False

    # 统一使用 loguru 的统一格式
    groups_path = Path(groups_base_path)
    if not groups_path.exists():
        logger.error(f"Groups directory does not exist: {groups_base_path}")
        return 1

    try:
        collector = SimpleSceneMetricsCollector(
            groups_base_path=groups_base_path,
            output_dir=output_dir,
            latest_runs=latest_runs,
        )
        collector.save()
        return 0
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


