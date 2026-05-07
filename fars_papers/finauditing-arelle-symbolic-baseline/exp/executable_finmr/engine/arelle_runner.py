# Arelle engine wrapper: loads reconstructed XBRL packages, checks executability,
# retrieves target facts, and provides access to relationship sets.

import datetime
from dataclasses import dataclass, field
from typing import Optional

from arelle import XbrlConst
from arelle.api.Session import Session
from arelle.RuntimeOptions import RuntimeOptions

from executable_finmr.data.target_extractor import TargetInfo


@dataclass
class ArelleFact:
    qname_str: str
    value: str
    context_id: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant: Optional[str] = None
    has_dimensions: bool = False
    dimensions: dict = field(default_factory=dict)
    decimals: Optional[str] = None


@dataclass
class ArelleResult:
    executable: bool
    target_fact: Optional[ArelleFact] = None
    all_target_facts: list = field(default_factory=list)
    error_log: str = ""
    n_facts: int = 0
    n_concepts: int = 0
    model: object = None
    session: object = None

    def close(self):
        if self.session:
            self.session.close()
            self.session = None
            self.model = None


def _date_str(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, (datetime.datetime, datetime.date)):
        return dt.strftime("%Y-%m-%d")
    return str(dt)


def _extract_fact_info(fact) -> ArelleFact:
    ctx = fact.context
    period_start = period_end = instant = None
    has_dims = False
    dims = {}

    if ctx is not None:
        if ctx.isInstantPeriod:
            instant = _date_str(ctx.instantDatetime - datetime.timedelta(days=1) if ctx.instantDatetime else None)
        elif ctx.isStartEndPeriod:
            period_start = _date_str(ctx.startDatetime)
            period_end = _date_str(ctx.endDatetime - datetime.timedelta(days=1) if ctx.endDatetime else None)

        if ctx.qnameDims:
            has_dims = True
            for dim_qname, dim_val in ctx.qnameDims.items():
                member = dim_val.memberQname if hasattr(dim_val, "memberQname") and dim_val.memberQname else None
                dims[str(dim_qname)] = str(member) if member else str(dim_val)

    return ArelleFact(
        qname_str=str(fact.qname),
        value=fact.value if fact.value else "",
        context_id=fact.contextID or "",
        period_start=period_start,
        period_end=period_end,
        instant=instant,
        has_dimensions=has_dims,
        dimensions=dims,
        decimals=fact.decimals if hasattr(fact, "decimals") else None,
    )


def _match_period(af: ArelleFact, target: TargetInfo) -> bool:
    if target.period_start and target.period_end:
        return af.period_start == target.period_start and af.period_end == target.period_end
    if target.instant:
        return af.instant == target.instant
    return True


def _find_target_qname(model, target: TargetInfo):
    local_name = target.concept_local_name
    for qname in model.factsByQname:
        if qname.localName == local_name:
            return qname
    for qname in model.qnameConcepts:
        if qname.localName == local_name:
            return qname
    return None


def _select_fact(period_matched: list[ArelleFact], target: TargetInfo) -> ArelleFact:
    candidates = sorted(period_matched, key=lambda af: af.context_id)

    if target.dqc_rule_family == "0015":
        neg = [af for af in candidates if _is_negative(af.value)]
        if len(neg) == 1:
            return neg[0]
        if neg:
            non_dim_neg = [af for af in neg if not af.has_dimensions]
            if non_dim_neg:
                return non_dim_neg[0]
            return neg[0]

    non_dim = [af for af in candidates if not af.has_dimensions]
    if non_dim:
        return non_dim[0]
    return candidates[0]


def _is_negative(value_str: str) -> bool:
    try:
        return float(str(value_str).replace(",", "")) < 0
    except (ValueError, TypeError):
        return False


def load_and_analyze(instance_path: str, target: TargetInfo) -> ArelleResult:
    session = Session()
    opts = RuntimeOptions(
        entrypointFile=instance_path,
        internetConnectivity="offline",
        keepOpen=True,
        logFile="logToBuffer",
    )
    success = session.run(opts)

    error_log = ""
    try:
        error_log = session.get_logs("text")
    except (ValueError, Exception):
        error_log = ""

    models = session.get_models()
    if not models or not success:
        return ArelleResult(
            executable=False,
            error_log=error_log or "model_load_failed",
            session=session,
        )

    model = models[0]
    n_facts = len(model.facts)
    n_concepts = len(model.qnameConcepts)

    if n_facts == 0:
        return ArelleResult(
            executable=False,
            error_log=error_log or "no_facts_loaded",
            n_facts=0,
            n_concepts=n_concepts,
            model=model,
            session=session,
        )

    target_qname = _find_target_qname(model, target)
    if target_qname is None:
        return ArelleResult(
            executable=False,
            error_log=error_log or f"target_concept_not_found:{target.concept_local_name}",
            n_facts=n_facts,
            n_concepts=n_concepts,
            model=model,
            session=session,
        )

    raw_facts = model.factsByQname.get(target_qname, set())
    all_target_facts = [_extract_fact_info(f) for f in raw_facts]
    period_matched = [af for af in all_target_facts if _match_period(af, target)]

    if not period_matched:
        return ArelleResult(
            executable=False,
            error_log=error_log or f"no_period_match:found_{len(all_target_facts)}_facts",
            n_facts=n_facts,
            n_concepts=n_concepts,
            all_target_facts=all_target_facts,
            model=model,
            session=session,
        )

    chosen = _select_fact(period_matched, target)

    return ArelleResult(
        executable=True,
        target_fact=chosen,
        all_target_facts=all_target_facts,
        error_log=error_log,
        n_facts=n_facts,
        n_concepts=n_concepts,
        model=model,
        session=session,
    )


def get_calc_children(model, target: TargetInfo) -> list[tuple[str, float, str]]:
    target_qname = _find_target_qname(model, target)
    if target_qname is None:
        return []

    concept = model.qnameConcepts.get(target_qname)
    if concept is None:
        return []

    calc_set = model.relationshipSet(XbrlConst.summationItem)
    children = []

    for linkrole in calc_set.linkRoleUris:
        role_set = model.relationshipSet(XbrlConst.summationItem, linkrole)
        rels = role_set.fromModelObject(concept)
        for rel in rels:
            child_concept = rel.toModelObject
            if child_concept is not None:
                children.append((str(child_concept.qname), rel.weight, linkrole))

    return children


def get_dimensional_facts(model, target: TargetInfo) -> list[ArelleFact]:
    target_qname = _find_target_qname(model, target)
    if target_qname is None:
        return []

    raw_facts = model.factsByQname.get(target_qname, set())
    all_facts = [_extract_fact_info(f) for f in raw_facts]
    period_matched = [af for af in all_facts if _match_period(af, target)]
    dimensional = [af for af in period_matched if af.has_dimensions]

    return dimensional


def get_fact_value_by_qname_str(model, qname_str: str, target: TargetInfo) -> Optional[ArelleFact]:
    local_name = qname_str.split(":")[-1] if ":" in qname_str else qname_str
    qname = None
    for q in model.factsByQname:
        if q.localName == local_name:
            qname = q
            break
    if qname is None:
        return None

    raw_facts = model.factsByQname.get(qname, set())
    all_facts = [_extract_fact_info(f) for f in raw_facts]
    period_matched = sorted(
        [af for af in all_facts if _match_period(af, target)],
        key=lambda af: af.context_id,
    )

    non_dim = [af for af in period_matched if not af.has_dimensions]
    if non_dim:
        return non_dim[0]
    if period_matched:
        return period_matched[0]
    return None
