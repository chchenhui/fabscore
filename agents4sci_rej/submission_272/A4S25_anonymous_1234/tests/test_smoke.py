
import json, os, subprocess, sys

def test_pipeline_runs(tmp_path):
    # Run the experiment
    subprocess.check_call([sys.executable, "code/run_experiments.py", "--random-seed", "123", "--results-dir", str(tmp_path / "results")])
    # Check metrics
    mpath = tmp_path / "results" / "metrics.json"
    assert mpath.exists(), "metrics.json not found"
    metrics = json.loads(mpath.read_text())
    for k in ["AUROC","AUPRC","Hit@10"]:
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0 or (k in metrics and not isinstance(metrics[k], (int,float))), "Metric out of range"
