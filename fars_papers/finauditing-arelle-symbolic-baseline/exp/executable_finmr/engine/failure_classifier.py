# Classifies non-executable instances by failure cause, using Arelle's error log
# and the execution state. Assigns exactly one label from a priority-ordered taxonomy.

import re


FAILURE_LABELS = [
    "missing_dts_artifact",
    "external_dependency",
    "malformed_xml",
    "ambiguous_fact_selection",
    "rule_specific_insufficiency",
    "unknown",
]


def classify_failure(error_log: str, rule_exec_result: dict | None = None) -> str:
    log = error_log.lower() if error_log else ""

    if any(p in log for p in [
        "could not load",
        "schemaref",
        "linkbaseref",
        "missing schema",
        "import error",
        "schemaimportmissing",
    ]):
        if "http://" in log or "https://" in log:
            return "external_dependency"
        return "missing_dts_artifact"

    if any(p in log for p in [
        "could not load file from local filesystem",
        "disable offline mode",
    ]):
        return "external_dependency"

    if any(p in log for p in [
        "xml syntax",
        "xmlsyntax",
        "parser error",
        "not well-formed",
        "encoding error",
        "xmlschema",
        "elementunexpected",
        "lxml",
    ]):
        return "malformed_xml"

    if "no_facts_loaded" in log or "model_load_failed" in log:
        if "http" in log:
            return "external_dependency"
        return "malformed_xml"

    if "target_concept_not_found" in log:
        return "external_dependency"

    if "no_period_match" in log:
        return "ambiguous_fact_selection"

    if rule_exec_result and not rule_exec_result.get("success", False):
        reason = rule_exec_result.get("reason", "")
        if reason in ["no_calc_children", "no_dimensional_facts", "no_child_facts_found"]:
            return "rule_specific_insufficiency"
        if reason == "no_parseable_dim_values":
            return "rule_specific_insufficiency"
        if reason == "unparseable_value":
            return "malformed_xml"

    return "unknown"
