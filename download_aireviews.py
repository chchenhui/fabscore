"""
Download all AI-generated reviews (signatures containing 'AIRev') from the
Agents4Science 2025 conference on OpenReview.

Output: agents4sci_aireviews/<submission_number>_<forum_id>.json
Each file contains all AIRev notes for that submission.
"""

import requests
import json
import os
import time

API_BASE = "https://api2.openreview.net"
VENUE = "Agents4Science/2025/Conference"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents4sci_aireviews")
os.makedirs(OUT_DIR, exist_ok=True)


def get_all_submissions():
    submissions = []
    offset = 0
    limit = 200
    while True:
        url = f"{API_BASE}/notes?invitation={VENUE}/-/Submission&limit={limit}&offset={offset}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("notes", [])
        submissions.extend(batch)
        print(f"  Fetched {len(submissions)} submissions so far...")
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.3)
    return submissions


def get_airev_notes(forum_id):
    url = f"{API_BASE}/notes?forum={forum_id}&limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    notes = resp.json().get("notes", [])
    airev_notes = [
        n for n in notes
        if any("AIRev" in s for s in n.get("signatures", []))
    ]
    return airev_notes


def main():
    print("Fetching all submissions...")
    submissions = get_all_submissions()
    print(f"Total submissions: {len(submissions)}\n")

    total_saved = 0
    for sub in submissions:
        forum_id = sub["id"]
        number = sub.get("number", "unknown")
        title = sub.get("content", {}).get("title", {})
        if isinstance(title, dict):
            title = title.get("value", "")

        airev_notes = get_airev_notes(forum_id)
        if not airev_notes:
            continue

        out = {
            "forum_id": forum_id,
            "submission_number": number,
            "title": title,
            "airev_notes": [
                {
                    "note_id": n["id"],
                    "signature": n.get("signatures", []),
                    "content": {
                        k: v.get("value", v) if isinstance(v, dict) else v
                        for k, v in n.get("content", {}).items()
                    },
                }
                for n in airev_notes
            ],
        }

        fname = os.path.join(OUT_DIR, f"submission{number}_{forum_id}.json")
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"  [submission {number}] {title[:60]} → {len(airev_notes)} AIRev notes saved")
        total_saved += 1
        time.sleep(0.2)

    print(f"\nDone. Saved {total_saved} submissions with AIRev notes to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
