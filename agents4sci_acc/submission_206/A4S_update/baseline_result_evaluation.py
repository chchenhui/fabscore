
import os, sys, glob, json, re

def process(all_path: str) -> None:
    m = re.fullmatch(r"all_baseline_(.+)\.jsonl", os.path.basename(all_path))
    if not m:
        return
    model = m.group(1)
    summary_path = f"results_summary_baseline_{model}.txt"
    failed_path  = f"failed_baseline_{model}.jsonl"

    total = passed = failed = 0
    failed_lines = []

    with open(all_path, "r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1
            if obj.get("passed", False):
                passed += 1
            else:
                failed += 1
         
                failed_lines.append(line.rstrip("\n"))

    with open(summary_path, "w", encoding="utf-8") as fsum:
        fsum.write(f"Total: {total}, Passed: {passed}, Failed: {failed}\n")

    with open(failed_path, "w", encoding="utf-8") as ff:
        for s in failed_lines:
            ff.write(s + "\n")

    print(f"✓ {all_path} -> {summary_path}, {failed_path}  (Total={total} Passed={passed} Failed={failed})")

def main() -> None:
    paths = sorted(glob.glob("all_baseline_*.jsonl"))
    if not paths:
        print("❌ Not Found all_baseline_*.jsonl", file=sys.stderr)
        sys.exit(1)
    for p in paths:
        process(p)

if __name__ == "__main__":
    main()
