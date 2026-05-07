# Regex message-only baseline for FinMR (Level 0 heuristic).
# Extracts extracted_value and calculated_value from the query text using only
# regex and simple arithmetic -- no XML parser, no XBRL tooling.

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from executable_finmr.data.load_finmr import FinMRInstance


def normalize_numeric(s: str) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "").replace(" ", "")
    if s.startswith("$"):
        s = s[1:]
    if not s:
        return None
    try:
        d = Decimal(s)
        if neg:
            d = -d
        return _fmt_decimal(d)
    except InvalidOperation:
        return None


def _fmt_decimal(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    return str(d)


def _build_context_map(instance_doc: str) -> dict:
    ctx_map = {}
    for m in re.finditer(
        r'<context\s+id="([^"]+)">(.*?)</context>', instance_doc, re.DOTALL
    ):
        ctx_id = m.group(1)
        body = m.group(2)

        period_start = period_end = instant = None
        pm = re.search(r"<startDate>(\d{4}-\d{2}-\d{2})</startDate>", body)
        if pm:
            period_start = pm.group(1)
        pm = re.search(r"<endDate>(\d{4}-\d{2}-\d{2})</endDate>", body)
        if pm:
            period_end = pm.group(1)
        pm = re.search(r"<instant>(\d{4}-\d{2}-\d{2})</instant>", body)
        if pm:
            instant = pm.group(1)

        has_segment = bool(re.search(r"<(?:\w+:)?segment>", body))

        ctx_map[ctx_id] = {
            "start": period_start,
            "end": period_end,
            "instant": instant,
            "has_segment": has_segment,
        }
    return ctx_map


def _extract_facts(instance_doc: str) -> list[dict]:
    facts = []
    for m in re.finditer(
        r"<([\w-]+:[\w-]+)\s+contextRef=\"([^\"]+)\"[^>]*>([^<]+)</",
        instance_doc,
    ):
        tag = m.group(1)
        ctx = m.group(2)
        val = m.group(3).strip()
        facts.append({"tag": tag, "contextRef": ctx, "value": val})
    return facts


def _ctx_info_from_id(ctx_id: str) -> dict:
    info = {"start": None, "end": None, "instant": None, "has_segment": False}
    m = re.match(
        r"(?:From|D)(\d{4}-?\d{2}-?\d{2})(?:to|-)(\d{4}-?\d{2}-?\d{2})",
        ctx_id,
        re.IGNORECASE,
    )
    if m:
        s = m.group(1)
        e = m.group(2)
        if len(s) == 8:
            s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if len(e) == 8:
            e = f"{e[:4]}-{e[4:6]}-{e[6:]}"
        info["start"] = s
        info["end"] = e
        if "_" in ctx_id[m.end():]:
            info["has_segment"] = True
        return info
    m = re.match(r"(?:AsOf|I|i[0-9a-f]+_I)(\d{4}-?\d{2}-?\d{2})", ctx_id)
    if m:
        d = m.group(1)
        if len(d) == 8:
            d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        info["instant"] = d
        if "_" in ctx_id[m.end():]:
            info["has_segment"] = True
        return info
    return info


def _get_ctx_info(ctx_id: str, ctx_map: dict) -> dict:
    if ctx_id in ctx_map:
        return ctx_map[ctx_id]
    return _ctx_info_from_id(ctx_id)


def _match_period(ctx_info: dict, inst: FinMRInstance) -> bool:
    if inst.target_period_start and inst.target_period_end:
        return (
            ctx_info.get("start") == inst.target_period_start
            and ctx_info.get("end") == inst.target_period_end
        )
    if inst.target_instant:
        return ctx_info.get("instant") == inst.target_instant
    return False


def _find_target_fact_value(
    facts: list[dict],
    ctx_map: dict,
    target_concept: str,
    inst: FinMRInstance,
    require_no_segment: bool = True,
) -> Optional[str]:
    local_name = target_concept.split(":")[-1] if ":" in target_concept else target_concept
    candidates = []
    for f in facts:
        f_local = f["tag"].split(":")[-1] if ":" in f["tag"] else f["tag"]
        if f_local.lower() != local_name.lower():
            continue
        ctx_info = _get_ctx_info(f["contextRef"], ctx_map)
        if not _match_period(ctx_info, inst):
            continue
        if require_no_segment and ctx_info.get("has_segment"):
            continue
        candidates.append(f)

    if not candidates and require_no_segment:
        return _find_target_fact_value(facts, ctx_map, target_concept, inst, False)

    if candidates:
        return candidates[0]["value"]
    return None


def _extract_calc_children(calc_doc: str, parent_concept: str) -> list[tuple[str, float]]:
    parent_local = parent_concept.split(":")[-1] if ":" in parent_concept else parent_concept
    parent_label_pat = re.compile(
        rf'xlink:label="(loc_[\w-]*{re.escape(parent_local)}[^"]*)"', re.IGNORECASE
    )
    parent_labels = set()
    for m in parent_label_pat.finditer(calc_doc):
        parent_labels.add(m.group(1))

    if not parent_labels:
        return []

    children = []
    seen = set()
    for plabel in parent_labels:
        arc_pat = re.compile(
            rf'<link:calculationArc[^>]*'
            rf'xlink:from="{re.escape(plabel)}"[^>]*'
            rf'xlink:to="([^"]+)"[^>]*'
            rf'weight="([^"]+)"',
            re.DOTALL,
        )
        for am in arc_pat.finditer(calc_doc):
            child_label = am.group(1)
            weight = float(am.group(2))
            if child_label in seen:
                continue
            seen.add(child_label)
            loc_pat = re.compile(
                rf'xlink:label="{re.escape(child_label)}"[^>]*'
                rf'xlink:href="[^"]*#([^"]+)"',
            )
            cm = loc_pat.search(calc_doc)
            if not cm:
                loc_pat2 = re.compile(
                    rf'xlink:href="[^"]*#([^"]+)"[^>]*'
                    rf'xlink:label="{re.escape(child_label)}"',
                )
                cm = loc_pat2.search(calc_doc)
            if cm:
                concept_id = cm.group(1)
                ns_local = concept_id.replace("_", ":", 1)
                children.append((ns_local, weight))

    return children


def _extract_dimensional_members(
    instance_doc: str,
    ctx_map: dict,
    facts: list[dict],
    target_concept: str,
    inst: FinMRInstance,
) -> list[str]:
    local_name = target_concept.split(":")[-1] if ":" in target_concept else target_concept
    member_values = []
    for f in facts:
        f_local = f["tag"].split(":")[-1] if ":" in f["tag"] else f["tag"]
        if f_local.lower() != local_name.lower():
            continue
        ctx_info = _get_ctx_info(f["contextRef"], ctx_map)
        if not _match_period(ctx_info, inst):
            continue
        if ctx_info.get("has_segment"):
            member_values.append(f["value"])
    return member_values


def run_regex_baseline_single(inst: FinMRInstance) -> dict:
    instance_doc = inst.sections.get("Instance document", "")
    calc_doc = inst.sections.get("Calculation linkbase document", "")

    if not inst.target_concept:
        return {}

    ctx_map = _build_context_map(instance_doc)
    facts = _extract_facts(instance_doc)

    if not facts:
        return {}

    rule_family = inst.dqc_rule_family

    extracted_value = _find_target_fact_value(
        facts, ctx_map, inst.target_concept, inst
    )
    if extracted_value is None:
        return {}

    ev_norm = normalize_numeric(extracted_value)

    if rule_family == "0015":
        if ev_norm is not None:
            try:
                cv = _fmt_decimal(abs(Decimal(ev_norm)))
                return {
                    "extracted_value": extracted_value,
                    "calculated_value": cv,
                }
            except InvalidOperation:
                pass
        return {"extracted_value": extracted_value, "calculated_value": "0"}

    elif rule_family == "0126":
        children = _extract_calc_children(calc_doc, inst.target_concept)
        if children:
            total = Decimal(0)
            all_found = True
            for child_concept, weight in children:
                child_val = _find_target_fact_value(
                    facts, ctx_map, child_concept, inst
                )
                if child_val is not None:
                    cv_norm = normalize_numeric(child_val)
                    if cv_norm is not None:
                        total += Decimal(cv_norm) * Decimal(str(weight))
                    else:
                        all_found = False
                else:
                    all_found = False
            if all_found or total != Decimal(0):
                return {
                    "extracted_value": extracted_value,
                    "calculated_value": _fmt_decimal(total),
                }
        return {"extracted_value": extracted_value, "calculated_value": "0"}

    elif rule_family == "0117":
        member_values = _extract_dimensional_members(
            instance_doc, ctx_map, facts, inst.target_concept, inst
        )
        if member_values:
            total = Decimal(0)
            for mv in member_values:
                mv_norm = normalize_numeric(mv)
                if mv_norm is not None:
                    total += Decimal(mv_norm)
            return {
                "extracted_value": extracted_value,
                "calculated_value": _fmt_decimal(total),
            }
        return {"extracted_value": extracted_value, "calculated_value": "0"}

    return {}


def run_regex_baseline(instances: list[FinMRInstance]) -> list[dict]:
    results = []
    for inst in instances:
        pred = run_regex_baseline_single(inst)
        results.append(
            {
                "id": inst.id,
                "dqc_id": inst.dqc_id,
                "prediction": pred,
                "gold": inst.gold_answer,
            }
        )
    return results


if __name__ == "__main__":
    from executable_finmr.data.load_finmr import load_finmr

    instances = load_finmr()
    for inst in instances[:5]:
        pred = run_regex_baseline_single(inst)
        print(f"id={inst.id} dqc={inst.dqc_id}")
        print(f"  gold: {inst.gold_answer}")
        print(f"  pred: {pred}")
        print()
