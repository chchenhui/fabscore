"""
Analyze the most frequent fabrication patterns (Data / Experiment / Result)
across all five paper collections combined.

For each fabrication type, entries are grouped into named pattern buckets via
keyword matching on the explanation text.  The top 1-3 buckets by count are
reported with 1-2 representative examples each.
"""

import json, glob, os, re
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = [
    "agents4sci_acc",
    "agents4sci_rej",
    "aiscientist_papers",
    "fars_papers",
    "mlragent_papers",
]
FAB_TYPES = ["Data Fabrication", "Experiment Fabrication", "Result Fabrication"]

# Each keyword is a regex passed to re.search() (case-insensitive) against
# the combined explanation+claim text. An entry is assigned to the FIRST
# pattern whose keyword list contains at least one match (OR logic).
# The final "Other / unclassified" bucket catches all unmatched entries.
PATTERNS = {
    "Data Fabrication": [
        (
            "Synthetic or mock data generation used instead of real data described in the paper",
            ["generate_mock", "generate_synthetic", "_generate_synthetic",
             "np\\.random", "randomly generated", "synthetically generated",
             "mock data", "fabricated dataset", "synthetic dataset",
             "synthetic.*cohort", "simulated data", "randomly sampled",
             "mock formula", "from mock.*formula", "directly from.*mock",
             "directly match.*mock", "random\\.uniform.*time",
             "numbers.*match mock formula", "values.*match mock formula",
             "closely match.*formula output", "match.*mock formula output",
             "directly from.*formula output",
             "synthetic.*formula", "formula.*factor", "use_api.*true",
             "use_api=true"],
        ),
        (
            "Non-existent or wrong dataset referenced in the paper",
            ["does not exist", "never existed", "not exist", "404",
             "absent", "unavailable", "cannot be found", "load_dataset",
             "wrong dataset", "different dataset", "instead of.*claimed",
             "unrelated dataset", "datasetnotfounderror",
             "no.*dataset.*files", "no.*data.*files", "no icdar",
             "no twitter", "dataset.*not.*found", "huggingface.*not exist"],
        ),
        (
            "Specific data values claimed in the paper conflict with actual data files",
            ["golden_drugs", "golden.*csv", "correct actual",
             "using.*correct.*denominator", "correct.*denominator",
             "actual.*revenue.*wrong", "actual.*denominator.*wrong",
             "mape.*denominator", "denominator.*wrong", "wrong.*denominator",
             "calibrated with.*as.*target", "calibrated.*target.*not.*real",
             "non-blind", "evaluation non-blind",
             "fabricated.*actual", "fabricated.*reference",
             "hardcoded.*as.*actual", "code.*comment.*actual",
             "gpt5_orchestrator.*actual", "uses.*as.*the.*calibration",
             "repository.*shows.*not", "csv.*shows.*not.*paper",
             "data.*file.*contradicts", "actual.*data.*contradicts"],
        ),
        (
            "Dataset size or configuration mismatch between code and paper",
            ["config.*dataset_size", "dataset_size.*config",
             "num_test_examples", "paper claims.*samples.*config",
             "config.*sets.*not.*paper", "config.*5.*paper.*100",
             "config.*100.*paper.*212", "config.*mismatch",
             "not generated.*from.*config", "config.*conflict",
             "claimed.*100.*config.*5", "claimed.*212.*config.*100"],
        ),
        ("Other / unclassified", []),
    ],

    # P1: A specific experimental component (e.g. training loop, human-AI
    #     interaction) is replaced by mock/simulation/hardcoded output.
    # P2: The formula or metric is implemented but mathematically wrong,
    #     producing incorrect numerical values.
    # P3: Execution order is wrong, or a required step is entirely missing.
    #     Does NOT cover formula errors (→ P2) or missing resources (→ P5).
    # P4: A runtime/syntax bug causes incorrect or failed execution.
    #     Does NOT include missing-resource errors (→ P5).
    # P5: A model, dataset, or checkpoint referenced in the code does not
    #     exist (wrong HuggingFace ID, missing checkpoint file, etc.).
    "Experiment Fabrication": [
        (
            "Simulation or hardcoded values replace a specific experimental component",
            ["programmatic simulation", "programmatically simulated",
             "programmatically generated",
             "dummy traversal", "dummy link predictor",
             "random pseudo-advantage", "random targets.*skip",
             "generate_results\\.py",
             "placeholder.*target", "toy model", "SimpleCIMLRModel",
             "SimpleBaselineModel",
             "mr_phis = sims", "explicitly.*simulated",
             "no real training", "no actual gradient", "no.*gradient.*update",
             "skips.*gradient", "train_step.*pass",
             "no train.*method", "never trained",
             "data_generator\\.py", "_simulate_",
             "identical to.*stub", "identical to broken",
             "MCCFRExternal", "dummy stub", "is a stub",
             "same as index", "class.*: pass",
             "from synthetic data simulation", "from synthetic simulation",
             "synthetic simulation", "synthetic.*generator",
             "achievable with.*fabricat",
             "no real experiment", "no real data",
             "programmatic.*degradation",
             "paper.*inflated.*by", "paper.*reduced.*by",
             "inflated.*this.*by", "reduced.*this.*by",
             "experiments\\.tex shows.*paper claims",
             "not generated by.*the code", "not generated by.*code",
             "was not generated by running", "results\\.md.*not generated",
             "paper/results\\.json",
             "hardcoded.*score", "hardcoded.*trajectori",
             "hardcoded.*metric", "hardcoded.*result",
             "hardcoded.*value",
             "artificial.*boost", "artificial.*increment",
             "fixed random matric", "derived from fabricated",
             "skips.*training", "computed from.*simulation",
             "np\\.random\\.normal", "np\\.random\\.uniform",
             "np\\.random\\.seed.*synthetic", "synthetic.*np\\.random"],
        ),
        (
            "Formula or metric implementation produces incorrect numerical values",
            ["log-ratio", "simple ratio",
             "formula.*conflict", "formula.*mismatch", "formula.*discrepancy",
             "implementation.*conflict", "differs from.*formula",
             "formula.*differ", "conflicts with.*formula",
             "bcr formula",
             "binary_truth.*=.*\\[1",
             "precision.*always 1", "all ground truth.*positive",
             "attribution_f1_score.*accuracy",
             "content_originality_score.*accuracy",
             "should be equal but differ",
             "broken.*f1.*metric", "same broken.*f1",
             "f1.*would not match",
             "jaccard.*overlap", "jaccard word", "jaccard",
             "heuristic.*formula", "perplexity.*heuristic",
             "formulation incorrect",
             "precision.*1\\.0.*f1", "precision.*1\\.0.*recall",
             "wrong.*formula", "different formula", "eq\\.2",
             "paper.*formula.*conflict",
             "path_adherence.*linear.*path",
             "std/n", "code computes std/n",
             "claims.*95.*ci.*code computes", "paper claims 95.*ci",
             "does not match paper.*ratio ~",
             "stored stderr.*does not match", "standard error.*95.*ci"],
        ),
        (
            "Experimental execution logic or order inconsistent with the paper description",
            ["mislabeled", "mis-labeled",
             "presented as.*row", "labeled as.*row",
             "presented as.*xavier", "presented as.*gaussian",
             "presented as.*he.*row", "presented as.*uniform",
             "presented as.*condition", "misrepresented.*condition",
             "wrong.*condition", "wrong.*label", "wrong.*setup",
             "baseline.*augmentation", "augmentation.*baseline",
             "different.*evaluation.*set",
             "6 examples.*30", "6.*example.*30 example",
             "weights.*cite.*1\\.0",
             "dataset.*result.*presented",
             "no perplexity", "no f1 score",
             "no epoch.*convergence", "no epoch.*track", "no epoch-based",
             "no ablation run", "no ablation.*implemented",
             "not implemented in the repository", "not implemented in any",
             "completely absent from", "absent from the repository",
             "no .{0,25}computation exists",
             "does not compute", "does not track", "does not measure",
             "does not evaluate", "never computed", "never tracked",
             "no output files exist", "no.*evaluation script exists",
             "no table.*evaluation script",
             "never uses it anywhere", "never uses it.*training",
             "always true for any", "is always true",
             "is ignored", "action.*ignored",
             "ignores.*entirely", "ignores.*fairness",
             "ignores.*\u03bb",
             "all.*variants.*produce.*identical",
             "fairness.*penalty.*never", "stores.*but never uses",
             "never passes.*to.*transformer",
             "not reinitialized between",
             "hardcoded return", "hardcodes return",
             "unconditionally returns", "bypassing.*actual"],
        ),
        (
            "Code implementation bug causes incorrect or failed execution",
            ["attributeerror", "runtimeerror",
             "dimension mismatch",
             "before reaching", "before producing",
             "fatal error", "cannot run", "fails with.*error",
             "broken.*greedy.*eval", "broken eval",
             "no trajectory results",
             "all statistical tests.*nan", "f_stat=nan", "effect_size=nan",
             "0\\.000 metrics", "all methods returned 0",
             "gpu.*cpu.*mismatch", "device mismatch",
             "before any.*evaluation", "before.*evaluation.*can"],
        ),
        (
            "Unavailable referenced model or dataset",
            ["no module named",
             "failed.*due to missing",
             "datasetnotfounderror", "never existed on huggingface",
             "does not exist.*huggingface", "huggingface.*not.*exist",
             "wrong model", "different model", "not the.*model",
             "instead of.*gpt", "instead of.*ivtfuse",
             "film.*model.*instead", "net\\.film",
             "distilbert", "wrong.*architecture",
             "imports.*wrong", "wrong.*import",
             "checkpoint.*does not exist", "checkpoint.*not exist",
             "missing.*checkpoint", "model.*file.*not exist",
             "num_experts.*4.*paper.*8", "num_experts.*conflict",
             "wrong.*evaluation.*script", "wrong.*eval.*script",
             "different.*evaluation.*script"],
        ),
        ("Other / unclassified", []),
    ],

    # P1: Code was re-run and the output contradicts the paper.
    # P2: Stored artifacts (JSON/CSV/log/figure) already show a different
    #     value; no re-execution needed. Includes mislabeled runs.
    # P3: The value is mathematically impossible given the paper's own
    #     formula or methodology.
    "Result Fabrication": [
        (
            "Reported value conflicts with re-execution results",
            ["execution stage.*paper claims", "execution stage.*claims",
             "actual.*execution.*produces", "actual.*run.*produces",
             "re-running.*produces", "re-execution.*produces",
             "fresh.*execution.*produces", "independent execution.*produces",
             "execution.*produces.*paper.*claims",
             "running.*code.*produces", "running.*script.*produces",
             "p_perm.*vs.*paper.*claims",
             "execution.*discrepancy", "execution.*mismatch",
             "cannot produce.*claimed", "code.*cannot.*produce",
             "code.*never.*produce",
             "contradicts.*paper.*execution", "conflict.*paper.*execution",
             "execution.*gives.*paper.*claims",
             "fresh.*execution.*gives",
             "executing.*gives.*not.*claimed", "executing.*gives.*not [0-9]",
             "consistently produce", "three independent executions",
             "does not match the computed", "cannot be reproduced",
             "no threshold produces", "all.*runs show higher",
             "all [0-9].*runs show",
             "achieves.*—not [0-9]", "which rounds to.*not [0-9]",
             "produces.*×.*conflicting"],
        ),
        (
            "Reported value conflicts with existing execution logs or stored artifacts",
            ["final_info\\.json", "experiment_results\\.json",
             "comparison.*json", "results\\.json",
             "stored.*output", "stored.*result",
             "artifact.*shows", "log.*shows", "csv.*shows",
             "output.*shows", "repository.*csv.*shows",
             "consistently show", "both.*csv.*show",
             "stored.*stderr.*does not match",
             "code computes std/n", "paper claims 95.*ci.*code.*std",
             "ratio ~.*x", "ratio ~[0-9]",
             "sem.*95.*ci", "std.*ci.*mismatch",
             "claims.*95.*ci.*computes.*std",
             "internal inconsistency",
             "same.*value.*different.*protein", "identical.*across",
             "shared.*identically", "implausible.*bh",
             "misattributed",
             "table.*figure.*contra", "figure.*table.*contra",
             "wrong.*attributed", "attributed.*wrong",
             "mixes run", "mixes.*results.*from run",
             "mixing.*run", "paper mixes",
             "directly contradicts this claim", "figure.*in the paper pdf",
             "embedded.*paper pdf", "figure.*explicitly.*contradicts",
             "figure.*contradicts.*table", "table.*contradicts.*figure",
             "figure.*directly contradicts", "embedded in the paper",
             "no ablation run scripts", "no ablation.*directory.*exist",
             "no.*run.*produces.*claimed",
             "does not correspond to any.*run",
             "matches no.*existing run", "cannot be found in any run",
             "no.*run.*directory.*exist", "no corresponding run",
             "no ablation run exists",
             "matches run_[0-9]", "mis-maps run_",
             "uses run_.*'s.*for", "uses run_.*average for",
             "explicitly states.*total", "explicitly states.*passed",
             "explicitly states.*failed", "explicitly states.*gives",
             "is not present",
             "consistently record",
             "hardcoded fallback", "fallback.*constant",
             "fallback value.*matches", "matches.*fallback",
             "csv.*records.*not.*claimed", "csv.*records.*not [0-9]",
             "failure list.*contains",
             "differ.*from table [0-9]", "differ.*from.*table.*per-scenario",
             "figure.*differ.*substantially.*from table",
             "generate_figures.*hardcodes",
             "giving.*\\(stored\\)"],
        ),
        (
            "Mathematically impossible value (cannot be derived from described methodology)",
            ["mathematically inconsistent",
             "applying.*gives.*not.*claimed", "applying.*gives.*capped.*not",
             "mathematically impossible", "impossible from.*formula",
             "impossible.*formula", "formula.*impossible",
             "bounded.*\\[0,1\\]", "formula.*bounded",
             "bounded to \\[", "cannot exceed",
             "exceeds.*maximum", "exceeds.*bound",
             ">1 is", "impossible.*given.*p_perm",
             "q-value cannot", "bh.*cannot", "impossible.*bh",
             "physically impossible", "negative.*reduction.*impossible",
             "-700", "below.*minimum", "above.*maximum.*possible"],
        ),
        ("Other / unclassified", []),
    ],
}


def load_all_entries():
    entries = []
    for cat in CATEGORIES:
        pattern = os.path.join(BASE, cat, "**/fabscore_claude/fs_summary.json")
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                data = json.load(f)
            paper = os.path.basename(os.path.dirname(os.path.dirname(path)))
            for section in ("tables", "figures", "results_section"):
                for claim in data.get(section, []):
                    verdict = claim.get("verdict", "")
                    if "fabricat" in verdict.lower():
                        entries.append(
                            {
                                "category": cat,
                                "paper": paper,
                                "section": section,
                                "verdict": verdict,
                                "claim": claim.get("claim", ""),
                                "explanation": claim.get("explanation", ""),
                            }
                        )
    return entries


def classify(entry, fab_type):
    text = (entry["explanation"] + " " + entry["claim"]).lower()
    for bucket_name, keywords in PATTERNS.get(fab_type, []):
        for kw in keywords:
            if re.search(kw.lower(), text):
                return bucket_name
    return "Other"


def top_samples(entries, n=2):
    seen = set()
    samples = []
    for e in entries:
        if e["paper"] not in seen and len(samples) < n:
            seen.add(e["paper"])
            samples.append(e)
    return samples


def classify_all(fab_entries, fab_type):
    bucket_map = defaultdict(list)
    for e in fab_entries:
        text = (e["explanation"] + " " + e["claim"]).lower()
        assigned = None
        for bucket_name, keywords in PATTERNS.get(fab_type, []):
            if not keywords:  # catch-all bucket
                assigned = bucket_name
                break
            for kw in keywords:
                if re.search(kw.lower(), text):
                    assigned = bucket_name
                    break
            if assigned:
                break
        bucket_map[assigned or "Other / unclassified"].append(e)
    return bucket_map


def print_analysis(entries):
    total = len(entries)
    counts = defaultdict(int)
    for e in entries:
        counts[e["verdict"]] += 1

    print("=" * 70)
    print(f"Overall fabrication counts  (all collections combined, N={total})")
    print("=" * 70)
    for fab in FAB_TYPES:
        print(f"  {fab}: {counts.get(fab, 0)}")

    print("\n\n" + "=" * 70)
    print("All patterns per fabrication type  (percentages sum to 100%)")
    print("=" * 70)

    for fab in FAB_TYPES:
        fab_entries = [e for e in entries if e["verdict"] == fab]
        n_fab = len(fab_entries)
        print(f"\n\n{'─'*70}")
        print(f"  {fab}  (total: {n_fab})")
        print(f"{'─'*70}")

        bucket_map = classify_all(fab_entries, fab)
        ranked = sorted(
            bucket_map.items(),
            key=lambda x: (x[0] == "Other / unclassified", -len(x[1]))
        )

        check_sum = 0
        for rank, (bucket, members) in enumerate(ranked, 1):
            pct = 100 * len(members) / n_fab
            check_sum += len(members)
            print(f"\n  [{rank}] {bucket}")
            print(f"       Count: {len(members)} / {n_fab}  ({pct:.1f}%)")
            if len(members) >= 3 and bucket != "Other / unclassified":
                for s in top_samples(members, n=1):
                    claim_txt = s["claim"].replace("\n", " ").strip()
                    expl_txt  = s["explanation"].replace("\n", " ").strip()
                    print(f"       Example — {s['paper']}  [{s['category']}]")
                    print(f"         Claim  : {claim_txt}")
                    print(f"         Reason : {expl_txt}")

        assert check_sum == n_fab, f"BUG: bucket counts sum to {check_sum}, expected {n_fab}"
        print(f"\n       {'─'*40}")
        print(f"       Total: {n_fab} / {n_fab}  (100.0%)")


def main():
    entries = load_all_entries()
    print_analysis(entries)


if __name__ == "__main__":
    main()
