"""Extract reveal-position data from EBR experiment SQLite DBs.

Parses the '[EBR] Released batch' log message from each run to reconstruct the
mapping from reveal position (shuffled order) to arrival rank, and determines
which reveal position was ultimately paid. Outputs a CSV for downstream analysis.
"""

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path


def parse_ebr_log(log_message: str) -> dict | None:
    m = re.search(
        r"Arrival order: \[([^\]]+)\], Shuffled order: \[([^\]]+)\]",
        log_message,
    )
    if not m:
        return None
    arrival_ids = [s.strip().strip("'\"") for s in m.group(1).split(",")]
    shuffled_ids = [s.strip().strip("'\"") for s in m.group(2).split(",")]
    return {"arrival_order": arrival_ids, "shuffled_order": shuffled_ids}


def get_paid_proposal_id(cursor) -> str | None:
    cursor.execute("""
    SELECT json_extract(data, '$.request.parameters.message.proposal_message_id')
    FROM actions
    WHERE json_extract(data, '$.request.name') = 'SendMessage'
        AND json_extract(data, '$.request.parameters.message.type') = 'payment'
    LIMIT 1
    """)
    row = cursor.fetchone()
    return row[0] if row else None


def analyze_reveal_order(db_path: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT data FROM logs WHERE data LIKE '%Released batch%'")
    row = c.fetchone()
    if not row:
        conn.close()
        return None

    log_data = json.loads(row[0])
    parsed = parse_ebr_log(log_data["message"])
    if not parsed:
        conn.close()
        return None

    paid_id = get_paid_proposal_id(c)
    conn.close()

    if not paid_id:
        return None

    arrival_order = parsed["arrival_order"]
    shuffled_order = parsed["shuffled_order"]

    paid_arrival_rank = arrival_order.index(paid_id) + 1 if paid_id in arrival_order else None
    paid_reveal_position = shuffled_order.index(paid_id) + 1 if paid_id in shuffled_order else None

    reveal_pos_to_arrival = {}
    for reveal_pos, sid in enumerate(shuffled_order, start=1):
        if sid in arrival_order:
            reveal_pos_to_arrival[reveal_pos] = arrival_order.index(sid) + 1

    return {
        "paid_arrival_rank": paid_arrival_rank,
        "paid_reveal_position": paid_reveal_position,
        "reveal_position_to_arrival_rank": reveal_pos_to_arrival,
        "n_proposals": len(shuffled_order),
    }


def extract_all(results_dir: str, output_csv: str) -> list[dict]:
    results_path = Path(results_dir)
    rows = []

    for db_file in sorted(results_path.glob("*.db")):
        if db_file.name.endswith(("-shm", "-wal")):
            continue

        name = db_file.stem
        scenario = "unknown"
        for folder in ["contractors_first", "contractors_second", "contractors_third"]:
            if folder in name:
                scenario = folder
                break

        run_match = re.search(r"run(\d+)$", name)
        run_id = f"{scenario}_run{run_match.group(1)}" if run_match else name

        result = analyze_reveal_order(str(db_file))
        if result is None:
            print(f"  SKIP (no reveal data): {db_file.name}")
            continue

        mapping = result["reveal_position_to_arrival_rank"]
        row = {
            "run_id": run_id,
            "scenario": scenario,
            "paid_arrival_rank": result["paid_arrival_rank"],
            "paid_reveal_position": result["paid_reveal_position"],
            "reveal_position_1_arrival_rank": mapping.get(1, ""),
            "reveal_position_2_arrival_rank": mapping.get(2, ""),
            "reveal_position_3_arrival_rank": mapping.get(3, ""),
        }
        rows.append(row)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id", "scenario", "paid_arrival_rank", "paid_reveal_position",
        "reveal_position_1_arrival_rank", "reveal_position_2_arrival_rank",
        "reveal_position_3_arrival_rank",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nExtracted {len(rows)} runs -> {output_csv}")
    return rows


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "experiments" / "ebr" / "results" / "ebr_gemini_v2"
    output_csv = repo_root / "experiments" / "ebr" / "results" / "ebr_gemini" / "reveal_order_data.csv"

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_csv = Path(sys.argv[2])

    rows = extract_all(str(results_dir), str(output_csv))

    print(f"\nSummary:")
    from collections import Counter
    valid = [r for r in rows if r["paid_reveal_position"] is not None]
    invalid = len(rows) - len(valid)
    reveal_counts = Counter(r["paid_reveal_position"] for r in valid)
    arrival_counts = Counter(r["paid_arrival_rank"] for r in valid)
    print(f"  Valid runs: {len(valid)}, excluded: {invalid}")
    print(f"  Reveal-position selection counts: {dict(sorted(reveal_counts.items()))}")
    print(f"  Arrival-rank selection counts: {dict(sorted(arrival_counts.items()))}")
