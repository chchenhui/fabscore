# Extracts targeting metadata from FinMR instances for use by the Arelle engine.
# Uses already-parsed fields from FinMRInstance (concept, period, dqc_id) plus
# additional parsing of the query text for axis/member info needed by DQC_0117.

import re
from dataclasses import dataclass, field
from typing import Optional

from executable_finmr.data.load_finmr import FinMRInstance


@dataclass
class TargetInfo:
    instance_id: int
    dqc_rule_family: str
    concept_prefix: str
    concept_local_name: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant: Optional[str] = None
    unit: Optional[str] = None
    axis_names: list[str] = field(default_factory=list)
    member_names: list[str] = field(default_factory=list)

    @property
    def concept_qname_str(self) -> str:
        return f"{self.concept_prefix}:{self.concept_local_name}"


_UNIT_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", re.IGNORECASE)


def extract_target(inst: FinMRInstance) -> TargetInfo:
    concept = inst.target_concept or ""
    if ":" in concept:
        prefix, local = concept.split(":", 1)
    else:
        prefix, local = "us-gaap", concept

    unit = None
    m = _UNIT_RE.search(inst.sections.get("_preamble", ""))
    if m:
        unit = m.group(1).upper()

    info = TargetInfo(
        instance_id=inst.id,
        dqc_rule_family=inst.dqc_rule_family,
        concept_prefix=prefix,
        concept_local_name=local,
        period_start=inst.target_period_start,
        period_end=inst.target_period_end,
        instant=inst.target_instant,
        unit=unit,
    )

    return info
