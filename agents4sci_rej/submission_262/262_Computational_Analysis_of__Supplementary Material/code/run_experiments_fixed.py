from __future__ import annotations

import os
from hash_function_analysis_fixed import run_and_save


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    results_dir = os.path.join(root, "results")
    run_and_save(results_dir)


if __name__ == "__main__":
    main()





