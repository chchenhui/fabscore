# Data loader for TheFinAI/FinMR dataset (332 instances).
# Loads from HuggingFace, JSON-decodes query/answer fields, parses DQC rule IDs,
# extracts section boundaries and the target concept+period from Question text.

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from executable_finmr.configs.settings import FINMR_DATASET_NAME, HF_TOKEN


@dataclass
class FinMRInstance:
    id: int
    dqc_id: str
    raw_query: str
    query: str  # JSON-decoded query
    gold_answer: dict  # {"extracted_value": ..., "calculated_value": ...}
    target_concept: Optional[str] = None
    target_period_start: Optional[str] = None
    target_period_end: Optional[str] = None
    target_instant: Optional[str] = None
    sections: dict = field(default_factory=dict)

    @property
    def dqc_rule_family(self) -> str:
        m = re.search(r"(\d{4})", self.dqc_id)
        return m.group(1) if m else "unknown"


_DQC_ID_RE = re.compile(r"DQC[._]?(?:US[._]?)?(\d{4})", re.IGNORECASE)

_TARGET_CONCEPT_RE = re.compile(
    r"(?:reported value of|actual value of)\s+([\w\-]+:[\w\-]+)",
    re.IGNORECASE,
)

_PERIOD_DURATION_RE = re.compile(
    r"period\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

_PERIOD_INSTANT_RE = re.compile(
    r"(?:as of|instant|date|period)\s+(\d{4}-\d{2}-\d{2})(?:\s*\?|\s*,|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_sections(query: str) -> dict:
    parts = re.split(r"(?=##\w)", query)
    sections = {}
    for part in parts:
        m = re.match(r"##(\S[^\n]*)\n?(.*)", part, re.DOTALL)
        if m:
            key = m.group(1).strip()
            sections[key] = m.group(2).strip()
        elif not sections:
            sections["_preamble"] = part.strip()
    return sections


def _parse_target(query: str):
    concept = None
    period_start = period_end = instant = None

    m = _TARGET_CONCEPT_RE.search(query)
    if m:
        concept = m.group(1)

    m = _PERIOD_DURATION_RE.search(query)
    if m:
        period_start, period_end = m.group(1), m.group(2)
    else:
        m = _PERIOD_INSTANT_RE.search(query)
        if m:
            instant = m.group(1)

    return concept, period_start, period_end, instant


def load_finmr() -> list[FinMRInstance]:
    ds = load_dataset(FINMR_DATASET_NAME, split="test", token=HF_TOKEN or None)
    instances = []
    for row in ds:
        raw_query = row["query"]
        try:
            query = json.loads(raw_query)
        except (json.JSONDecodeError, TypeError):
            query = raw_query

        try:
            gold = json.loads(row["answer"])
        except (json.JSONDecodeError, TypeError):
            gold = {"extracted_value": None, "calculated_value": None}

        dqc_id = row["dqc_id"].strip('" ')

        sections = _parse_sections(query)
        concept, ps, pe, instant = _parse_target(query)

        inst = FinMRInstance(
            id=row["id"],
            dqc_id=dqc_id,
            raw_query=raw_query,
            query=query,
            gold_answer=gold,
            target_concept=concept,
            target_period_start=ps,
            target_period_end=pe,
            target_instant=instant,
            sections=sections,
        )
        instances.append(inst)

    return instances


if __name__ == "__main__":
    instances = load_finmr()
    print(f"Loaded {len(instances)} instances")
    from collections import Counter

    dqc_counts = Counter(i.dqc_id for i in instances)
    print(f"DQC distribution: {dict(sorted(dqc_counts.items()))}")

    for dqc in sorted(dqc_counts):
        sample = next(i for i in instances if i.dqc_id == dqc)
        print(f"\n--- {dqc} (id={sample.id}) ---")
        print(f"  target_concept: {sample.target_concept}")
        print(f"  period: {sample.target_period_start} to {sample.target_period_end}")
        print(f"  instant: {sample.target_instant}")
        print(f"  gold: {sample.gold_answer}")
        print(f"  sections: {list(sample.sections.keys())}")
