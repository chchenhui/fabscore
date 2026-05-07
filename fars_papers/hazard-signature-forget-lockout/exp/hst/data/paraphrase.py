# Paraphrase generation via DeepSeek API.
# Generates 5 paraphrases per poisoned seed (50 total) using the OpenAI-compatible
# endpoint. Each call returns a JSON list of 5 {req, resp} objects. Results are
# cached to hst/cache/paraphrases.json.

import json
import os
import re
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "paraphrases.json"

MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

SYSTEM_PROMPT = (
    "You are a paraphrase generator. Given a memory record with `req` (request) "
    "and `resp` (response) fields, generate exactly 5 paraphrases that preserve "
    "the semantic intent and behavioral pattern but vary surface form (word choice, "
    "sentence structure, phrasing). Return ONLY a JSON list of 5 objects, each with "
    '`req` and `resp` string fields. No markdown fences, no explanation.'
)


def _build_user_prompt(record: Dict) -> str:
    return (
        f"Paraphrase the following memory record 5 times:\n\n"
        f"req: {record['req']}\n"
        f"resp: {record['resp']}\n\n"
        f"Return a JSON list of 5 objects, each with `req` and `resp` fields."
    )


def _get_client() -> OpenAI:
    base_url = f"http://{os.environ['LEMMA_MAAS_BASE_URL']}/v1"
    api_key = os.environ["LEMMA_MAAS_API_KEY"]
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_json(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def generate_paraphrases(poisoned_records: List[Dict], n_para: int = 5,
                         force: bool = False) -> List[Dict]:
    if not force and CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        if len(cached) == len(poisoned_records) * n_para:
            return cached

    client = _get_client()
    all_paraphrases = []

    for record in poisoned_records:
        poison_idx = record["id"].split("_")[-1]
        print(f"  Generating paraphrases for {record['id']}...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(record)},
            ],
            temperature=0,
            top_p=1,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content
        paras = _extract_json(raw)
        assert len(paras) == n_para, (
            f"Expected {n_para} paraphrases for {record['id']}, got {len(paras)}"
        )

        for y, p in enumerate(paras, start=1):
            all_paraphrases.append({
                "id": f"para_poison_{poison_idx}_{y}",
                "req": p["req"],
                "resp": p["resp"],
                "tag": record.get("tag", ""),
                "source_id": record["id"],
            })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(all_paraphrases, f, indent=2)
    print(f"  Cached {len(all_paraphrases)} paraphrases to {CACHE_FILE}")

    return all_paraphrases


def load_paraphrases() -> List[Dict]:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Paraphrases not cached yet. Run generate_paraphrases() first.")
    with open(CACHE_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    from hst.data.download_seeds import load_seeds

    _, poisoned = load_seeds()
    paras = generate_paraphrases(poisoned, force=False)
    print(f"\nTotal paraphrases: {len(paras)}")
    for p in paras[:3]:
        print(f"  {p['id']}: req={p['req'][:60]}...")
