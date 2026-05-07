import os, json
from greenhouse.figures import ensure_dir, plot_bench_with_err, plot_linreg_with_err, plot_bench_lines, plot_linreg_lines
from greenhouse.dsl import Rule

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "artifacts"); ensure_dir(ART)

best = json.load(open(os.path.join(ART, "best_rule_v02.json")))
best_rule = Rule(**best["best_rule"])

# Appendix B (error bands)
plot_bench_with_err(best_rule, "rastrigin", os.path.join(ART, "bench_rastrigin_v02_err.png"))
plot_bench_with_err(best_rule, "rosenbrock", os.path.join(ART, "bench_rosenbrock_v02_err.png"))
plot_bench_with_err(best_rule, "ackley", os.path.join(ART, "bench_ackley_v02_err.png"))
plot_linreg_with_err(best_rule, os.path.join(ART, "bench_linreg_v02_err.png"))

# Main text (lines only)
plot_bench_lines(best_rule, "rastrigin", os.path.join(ART, "bench_rastrigin_v02_lines.png"))
plot_bench_lines(best_rule, "rosenbrock", os.path.join(ART, "bench_rosenbrock_v02_lines.png"))
plot_bench_lines(best_rule, "ackley", os.path.join(ART, "bench_ackley_v02_lines.png"))
plot_linreg_lines(best_rule, os.path.join(ART, "bench_linreg_v02_lines.png"))

print("Saved all figures (error bands + lines) in artifacts/")
