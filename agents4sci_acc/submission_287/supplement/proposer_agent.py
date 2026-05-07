# Proposer Agent: Identifies and ranks most common confounders across drug-symptom associations

import json
import argparse
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser(description="Find top N most common confounders (count based on max per symptom).")
    parser.add_argument("--file", help="Path to the JSON file")
    parser.add_argument("-n", type=int, default=10, help="Number of top confounders to show (default: 10)")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

    confounder_counts = Counter()
    seen = set()
    total = defaultdict(set)
    confounder_ass = defaultdict(set)

    for drug, symptoms in data.items():
        for symptom, entries in symptoms.items():
            for entry in entries:
                try:
                    for conf in entry.get("confounders", []):
                        seen.add((conf, symptom))
                        total[conf].add(drug)
                        confounder_ass[conf].add((drug, symptom))
                except:
                    continue

    for c, s in seen:
        if(len(total[c]) < 3):
            continue
        confounder_counts[c] += 1/len(total[c])
    print(f"Top {args.n} confounders:")
    for conf, count in confounder_counts.most_common(args.n):
        print(f"{conf}: {count}, {total[conf]}")
        print(confounder_ass[conf])
        print("\n")

if __name__ == "__main__":
    main()