# Failure taxonomy analysis for non-executable FinMR instances.
# Computes failure category statistics, analyzes external dependencies,
# generates summary report with tables, bar chart, and Sankey-style flow diagram.

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
XBRL_PACKAGES_DIR = OUTPUT_DIR / "xbrl_packages"

RULE_ORDER = ["0015", "0117", "0126"]
RULE_NAMES = {
    "0015": "DQC_0015",
    "0117": "DQC_0117",
    "0126": "DQC_0126",
}

FAILURE_CATS = [
    "external_dependency",
    "missing_dts_artifact",
    "malformed_xml",
    "ambiguous_fact_selection",
    "rule_specific_insufficiency",
    "unknown",
]

FAILURE_NICE = {
    "external_dependency": "External Dependency",
    "missing_dts_artifact": "Missing DTS Artifact",
    "malformed_xml": "Malformed XML/Escaping",
    "ambiguous_fact_selection": "Ambiguous Fact Selection",
    "rule_specific_insufficiency": "Rule-Specific Insufficiency",
    "unknown": "Unknown",
}

FAILURE_COLORS = {
    "external_dependency": "#e377c2",
    "missing_dts_artifact": "#7f7f7f",
    "malformed_xml": "#bcbd22",
    "ambiguous_fact_selection": "#ff7f0e",
    "rule_specific_insufficiency": "#17becf",
    "unknown": "#aec7e8",
}

RECOMMENDATIONS = {
    "external_dependency": (
        "Include cached US-GAAP / SEC / SRT taxonomy packages (2020-2024) "
        "in the benchmark release so instances can resolve schema imports offline."
    ),
    "missing_dts_artifact": (
        "Ensure all referenced schema/linkbase files are included in the query. "
        "Some instances lack linkbase documents or have broken schemaRef paths."
    ),
    "malformed_xml": (
        "Fix double-escaped XML entities, unclosed tags, and encoding errors "
        "in the source XBRL snippets before publishing."
    ),
    "ambiguous_fact_selection": (
        "Add explicit fact-selection hints (context ID, period) to queries "
        "where multiple candidate facts match the target concept."
    ),
    "rule_specific_insufficiency": (
        "Verify that query text includes the necessary linkbase arcs "
        "(calculation children, dimensional relationships) for rule execution."
    ),
    "unknown": (
        "Investigate remaining failures individually; likely a mix of edge cases."
    ),
}

STANDARD_TAXONOMY_DOMAINS = [
    "xbrl.fasb.org",
    "fasb.org",
    "xbrl.sec.gov",
    "www.xbrl.org",
    "xbrl.org",
]


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _extract_urls_from_xbrl_package(instance_id):
    pkg_dir = XBRL_PACKAGES_DIR / str(instance_id)
    if not pkg_dir.exists():
        return set()
    urls = set()
    url_re = re.compile(r'(?:schemaLocation|href)\s*=\s*"(https?://[^"]+)"', re.IGNORECASE)
    for f in pkg_dir.iterdir():
        if f.suffix in (".xsd", ".xml"):
            try:
                text = f.read_text(errors="replace")
                urls.update(url_re.findall(text))
            except Exception:
                pass
    return urls


def _parse_domain(url):
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else url


def _is_standard_taxonomy_url(url):
    domain = _parse_domain(url)
    return any(d in domain for d in STANDARD_TAXONOMY_DOMAINS)


def _count_sections(query_text):
    return len(re.findall(r"##\w", query_text or ""))


def compute_taxonomy_stats(rows):
    non_exec = [r for r in rows if not r.get("executable")]
    total_non_exec = len(non_exec)

    overall_counts = Counter(r.get("failure_label", "unknown") for r in non_exec)
    overall_stats = {}
    for cat in FAILURE_CATS:
        cnt = overall_counts.get(cat, 0)
        pct = round(cnt / total_non_exec * 100, 2) if total_non_exec else 0
        overall_stats[cat] = {"count": cnt, "percentage": pct}

    per_rule = {}
    for family in RULE_ORDER:
        family_rows = [r for r in non_exec if (r.get("dqc_rule_family") or "") == family]
        n = len(family_rows)
        counts = Counter(r.get("failure_label", "unknown") for r in family_rows)
        per_rule[family] = {}
        for cat in FAILURE_CATS:
            cnt = counts.get(cat, 0)
            pct = round(cnt / n * 100, 2) if n else 0
            per_rule[family][cat] = {"count": cnt, "percentage": pct}

    return overall_stats, per_rule, total_non_exec


def compute_instance_characteristics(rows):
    stats_by_cat = defaultdict(lambda: {"query_lengths": [], "section_counts": []})
    for r in rows:
        if r.get("executable"):
            continue
        cat = r.get("failure_label", "unknown")
        query = r.get("query_text", "")
        stats_by_cat[cat]["query_lengths"].append(len(query))
        stats_by_cat[cat]["section_counts"].append(_count_sections(query))

    exec_rows = [r for r in rows if r.get("executable")]
    stats_by_cat["_executable"] = {
        "query_lengths": [len(r.get("query_text", "")) for r in exec_rows],
        "section_counts": [_count_sections(r.get("query_text", "")) for r in exec_rows],
    }

    summary = {}
    for cat, data in stats_by_cat.items():
        ql = data["query_lengths"]
        sc = data["section_counts"]
        summary[cat] = {
            "n": len(ql),
            "mean_query_length": round(np.mean(ql), 1) if ql else 0,
            "median_query_length": round(float(np.median(ql)), 1) if ql else 0,
            "mean_section_count": round(np.mean(sc), 2) if sc else 0,
        }
    return summary


def analyze_external_dependencies(rows):
    ext_dep_rows = [r for r in rows if not r.get("executable") and r.get("failure_label") == "external_dependency"]
    all_urls = set()
    per_instance_urls = {}
    domain_counter = Counter()

    for r in ext_dep_rows:
        iid = r["id"]
        urls = _extract_urls_from_xbrl_package(iid)
        ext_urls = {u for u in urls if _is_standard_taxonomy_url(u)}
        per_instance_urls[iid] = sorted(ext_urls)
        all_urls.update(ext_urls)
        for u in ext_urls:
            domain_counter[_parse_domain(u)] += 1

    domain_summary = {d: c for d, c in domain_counter.most_common()}

    url_by_domain = defaultdict(set)
    for u in all_urls:
        url_by_domain[_parse_domain(u)].add(u)
    url_catalog = {d: sorted(urls) for d, urls in sorted(url_by_domain.items(), key=lambda x: -len(x[1]))}

    n_ext = len(ext_dep_rows)
    n_only_standard = 0
    for r in ext_dep_rows:
        iid = r["id"]
        urls = _extract_urls_from_xbrl_package(iid)
        non_local_urls = {u for u in urls if u.startswith("http")}
        if non_local_urls and all(_is_standard_taxonomy_url(u) for u in non_local_urls):
            n_only_standard += 1

    return {
        "total_external_dependency_instances": n_ext,
        "distinct_external_urls": len(all_urls),
        "distinct_domains": len(domain_summary),
        "domain_frequency": domain_summary,
        "url_catalog_by_domain": url_catalog,
        "recoverable_with_taxonomy_package": n_only_standard,
        "recoverable_fraction": round(n_only_standard / n_ext * 100, 2) if n_ext else 0,
    }


def _make_bar_chart(overall_stats, total_non_exec):
    cats_with_counts = [(cat, overall_stats[cat]["count"]) for cat in FAILURE_CATS if overall_stats[cat]["count"] > 0]
    cats_with_counts.sort(key=lambda x: x[1], reverse=True)
    cats = [c[0] for c in cats_with_counts]
    counts = [c[1] for c in cats_with_counts]
    colors = [FAILURE_COLORS[c] for c in cats]
    labels = [FAILURE_NICE[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(cats))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Instances", fontsize=12)
    ax.set_title(f"Failure Category Distribution (N={total_non_exec} non-executable)", fontsize=13, fontweight="bold")

    for bar, cnt in zip(bars, counts):
        pct = cnt / total_non_exec * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlim(0, max(counts) * 1.25)
    plt.tight_layout()
    path = FIGURES_DIR / "failure_category_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _make_sankey(rows, overall_stats):
    total = len(rows)
    n_exec = sum(1 for r in rows if r.get("executable"))
    n_nonexec = total - n_exec

    non_exec_rows = [r for r in rows if not r.get("executable")]

    cat_counts = Counter(r.get("failure_label", "unknown") for r in non_exec_rows)
    active_cats = [c for c in FAILURE_CATS if cat_counts.get(c, 0) > 0]

    cat_rule_counts = defaultdict(Counter)
    for r in non_exec_rows:
        cat = r.get("failure_label", "unknown")
        family = r.get("dqc_rule_family", "?")
        cat_rule_counts[cat][family] += 1

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total + 10)
    ax.axis("off")
    ax.set_title("Instance Flow: Executability -> Failure Category -> DQC Rule",
                 fontsize=13, fontweight="bold", pad=20)

    col_x = [1, 4, 7.5]

    def _draw_block(x, y_start, height, color, label, alpha=0.85):
        rect = mpatches.FancyBboxPatch((x - 0.3, y_start), 0.6, height,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=alpha, edgecolor="grey", linewidth=0.5)
        ax.add_patch(rect)
        if height > 2:
            ax.text(x, y_start + height / 2, label, ha="center", va="center", fontsize=8, fontweight="bold")

    y_off = 2
    exec_y = y_off
    exec_h = n_exec * (total / total)
    nonexec_y = exec_y + exec_h + 2
    nonexec_h = n_nonexec * (total / total)

    _draw_block(col_x[0], exec_y, exec_h, "#2ca02c", f"Executable\n({n_exec})")
    _draw_block(col_x[0], nonexec_y, nonexec_h, "#d62728", f"Non-exec\n({n_nonexec})")

    cat_y = nonexec_y
    cat_positions = {}
    for cat in active_cats:
        cnt = cat_counts[cat]
        h = cnt
        _draw_block(col_x[1], cat_y, h, FAILURE_COLORS[cat],
                    f"{FAILURE_NICE[cat]}\n({cnt})")
        cat_positions[cat] = (cat_y, h)
        cat_y += h + 1

    rule_y = nonexec_y
    rule_positions = {}
    for family in RULE_ORDER:
        cnt = sum(1 for r in non_exec_rows if (r.get("dqc_rule_family") or "") == family)
        if cnt == 0:
            continue
        h = cnt
        _draw_block(col_x[2], rule_y, h, "#1f77b4",
                    f"DQC_{family}\n({cnt})")
        rule_positions[family] = (rule_y, h)
        rule_y += h + 1

    def _draw_flow(x1, y1_start, y1_h, x2, y2_start, y2_h, flow_n, color, total_ref):
        if flow_n == 0:
            return
        from matplotlib.patches import FancyArrowPatch
        mid_y1 = y1_start + y1_h / 2
        mid_y2 = y2_start + y2_h / 2
        alpha = max(0.15, min(0.6, flow_n / total_ref * 2))
        xs = np.linspace(x1 + 0.3, x2 - 0.3, 50)
        ys_top = np.interp(xs, [x1 + 0.3, (x1 + x2) / 2, x2 - 0.3],
                           [y1_start + y1_h, (y1_start + y1_h + y2_start + y2_h) / 2, y2_start + y2_h])
        ys_bot = np.interp(xs, [x1 + 0.3, (x1 + x2) / 2, x2 - 0.3],
                           [y1_start, (y1_start + y2_start) / 2, y2_start])
        width_scale = flow_n / max(cat_counts.values()) if cat_counts else 1
        for i in range(len(xs) - 1):
            ax.fill_between([xs[i], xs[i+1]],
                            [ys_bot[i], ys_bot[i+1]],
                            [ys_top[i], ys_top[i+1]],
                            color=color, alpha=alpha, linewidth=0)

    _draw_flow(col_x[0], nonexec_y, nonexec_h, col_x[1],
               nonexec_y, cat_y - nonexec_y - 1, n_nonexec, "#d62728", total)

    for cat in active_cats:
        cy, ch = cat_positions[cat]
        for family in RULE_ORDER:
            if family not in rule_positions:
                continue
            cnt = cat_rule_counts[cat].get(family, 0)
            if cnt == 0:
                continue
            ry, rh = rule_positions[family]
            _draw_flow(col_x[1], cy, ch, col_x[2], ry, rh, cnt, FAILURE_COLORS[cat], total)

    ax.text(col_x[0], y_off - 1.5, "Executability", ha="center", fontsize=11, fontweight="bold")
    ax.text(col_x[1], y_off - 1.5, "Failure Category", ha="center", fontsize=11, fontweight="bold")
    ax.text(col_x[2], y_off - 1.5, "DQC Rule", ha="center", fontsize=11, fontweight="bold")

    max_y = max(cat_y, rule_y, nonexec_y + nonexec_h) + 5
    ax.set_ylim(-2, max_y)

    plt.tight_layout()
    path = FIGURES_DIR / "failure_sankey.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _make_per_rule_failure_stacked_bar(per_rule_stats):
    fig, ax = plt.subplots(figsize=(10, 5))
    active_cats = [c for c in FAILURE_CATS
                   if any(per_rule_stats[f][c]["count"] > 0 for f in RULE_ORDER)]

    x = np.arange(len(RULE_ORDER))
    width = 0.5
    bottoms = np.zeros(len(RULE_ORDER))

    for cat in active_cats:
        vals = [per_rule_stats[f][cat]["count"] for f in RULE_ORDER]
        ax.bar(x, vals, width, bottom=bottoms, label=FAILURE_NICE[cat],
               color=FAILURE_COLORS[cat], edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"DQC_{f}" for f in RULE_ORDER], fontsize=11)
    ax.set_ylabel("Number of Non-Executable Instances", fontsize=11)
    ax.set_title("Failure Categories by DQC Rule Family", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    for i, f in enumerate(RULE_ORDER):
        total = sum(per_rule_stats[f][c]["count"] for c in FAILURE_CATS)
        if total > 0:
            ax.text(i, bottoms[i] + 0.5, str(total), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "failure_per_rule_stacked.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def build_summary_table(overall_stats, per_rule_stats):
    table = []
    for cat in FAILURE_CATS:
        cnt = overall_stats[cat]["count"]
        if cnt == 0:
            continue
        pct = overall_stats[cat]["percentage"]
        top_rules = []
        for f in RULE_ORDER:
            rc = per_rule_stats[f][cat]["count"]
            if rc > 0:
                top_rules.append(f"DQC_{f}({rc})")
        table.append({
            "failure_category": FAILURE_NICE[cat],
            "count": cnt,
            "percentage": pct,
            "top_affected_dqc_rules": ", ".join(top_rules),
            "recommended_fix": RECOMMENDATIONS[cat],
        })
    return table


def load_query_texts():
    try:
        from executable_finmr.data.load_finmr import load_finmr
        instances = load_finmr()
        return {inst.id: inst.raw_query for inst in instances}
    except Exception as e:
        print(f"Warning: could not load FinMR dataset for query texts: {e}")
        return {}


def main():
    print("Loading results...")
    rows = _load_jsonl(OUTPUT_DIR / "arelle_baseline_results.jsonl")
    print(f"Loaded {len(rows)} instances ({sum(1 for r in rows if r.get('executable'))} executable)")

    query_texts = load_query_texts()
    for r in rows:
        r["query_text"] = query_texts.get(r["id"], "")

    print("\n=== Step 1: Failure Taxonomy Statistics ===")
    overall_stats, per_rule_stats, total_non_exec = compute_taxonomy_stats(rows)
    print(f"\nTotal non-executable: {total_non_exec}")
    for cat in FAILURE_CATS:
        s = overall_stats[cat]
        if s["count"] > 0:
            print(f"  {FAILURE_NICE[cat]}: {s['count']} ({s['percentage']}%)")

    print("\nPer-DQC rule breakdown:")
    for family in RULE_ORDER:
        n_fail = sum(per_rule_stats[family][c]["count"] for c in FAILURE_CATS)
        print(f"  DQC_{family} (N_fail={n_fail}):")
        for cat in FAILURE_CATS:
            c = per_rule_stats[family][cat]["count"]
            if c > 0:
                print(f"    {FAILURE_NICE[cat]}: {c} ({per_rule_stats[family][cat]['percentage']}%)")

    char_stats = compute_instance_characteristics(rows)
    print("\nInstance characteristics by failure category:")
    for cat, s in char_stats.items():
        label = FAILURE_NICE.get(cat, cat)
        print(f"  {label}: n={s['n']}, mean_qlen={s['mean_query_length']}, "
              f"median_qlen={s['median_query_length']}, mean_sections={s['mean_section_count']}")

    print("\n=== Step 2: External Dependency Analysis ===")
    ext_analysis = analyze_external_dependencies(rows)
    print(f"External dependency instances: {ext_analysis['total_external_dependency_instances']}")
    print(f"Distinct external URLs: {ext_analysis['distinct_external_urls']}")
    print(f"Distinct domains: {ext_analysis['distinct_domains']}")
    print(f"Domain frequency:")
    for d, c in ext_analysis["domain_frequency"].items():
        print(f"  {d}: referenced by {c} instances")
    print(f"Recoverable with taxonomy package: {ext_analysis['recoverable_with_taxonomy_package']} "
          f"({ext_analysis['recoverable_fraction']}%)")

    print("\n=== Step 3: Generating Report and Figures ===")
    summary_table = build_summary_table(overall_stats, per_rule_stats)

    _make_bar_chart(overall_stats, total_non_exec)
    _make_sankey(rows, overall_stats)
    _make_per_rule_failure_stacked_bar(per_rule_stats)

    n_total = len(rows)
    n_exec = sum(1 for r in rows if r.get("executable"))
    report = {
        "summary": {
            "total_instances": n_total,
            "executable": n_exec,
            "non_executable": total_non_exec,
            "executability_pct": round(n_exec / n_total * 100, 2),
        },
        "failure_taxonomy_overall": overall_stats,
        "failure_taxonomy_per_rule": per_rule_stats,
        "instance_characteristics": char_stats,
        "external_dependency_analysis": {
            "total_instances": ext_analysis["total_external_dependency_instances"],
            "distinct_external_urls": ext_analysis["distinct_external_urls"],
            "distinct_domains": ext_analysis["distinct_domains"],
            "domain_frequency": ext_analysis["domain_frequency"],
            "recoverable_with_taxonomy_package": ext_analysis["recoverable_with_taxonomy_package"],
            "recoverable_fraction_pct": ext_analysis["recoverable_fraction"],
            "url_catalog_by_domain": ext_analysis["url_catalog_by_domain"],
        },
        "summary_table": summary_table,
        "recommendations": [
            "Include US-GAAP taxonomy packages (2020-2024 vintages) in the benchmark release.",
            "Include SEC DEI, SRT, Country taxonomy packages.",
            f"This alone would recover ~{ext_analysis['recoverable_with_taxonomy_package']} of "
            f"{ext_analysis['total_external_dependency_instances']} external-dependency failures "
            f"({ext_analysis['recoverable_fraction']}%).",
            f"Fix malformed XML in {overall_stats['malformed_xml']['count']} instances "
            "(double-escaped entities, unclosed tags).",
            f"Resolve {overall_stats['missing_dts_artifact']['count']} instances with missing DTS artifacts "
            "(broken schemaRef, missing linkbase files).",
            f"Investigate {overall_stats.get('unknown', {}).get('count', 0)} unknown-category failures.",
        ],
        "figures": [
            "results/figures/failure_category_bar.png",
            "results/figures/failure_sankey.png",
            "results/figures/failure_per_rule_stacked.png",
        ],
    }

    report_path = RESULTS_DIR / "failure_taxonomy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")

    print("\n=== Summary Table ===")
    header = f"{'Category':<30} {'Count':>5} {'Pct':>6} {'Top Rules':<30} {'Fix'}"
    print(header)
    print("-" * len(header))
    for row in summary_table:
        print(f"{row['failure_category']:<30} {row['count']:>5} {row['percentage']:>5.1f}% "
              f"{row['top_affected_dqc_rules']:<30} {row['recommended_fix'][:60]}")

    print("\nDone.")
    return report


if __name__ == "__main__":
    main()
