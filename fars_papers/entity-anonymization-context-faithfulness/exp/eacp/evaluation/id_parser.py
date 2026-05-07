r"""ID output parser for conditions B and C. Extracts predicted entity ID
(ENT_\d+ or UNKNOWN) from model raw output, then maps to entity label."""

import re


_ID_PATTERN = re.compile(r"\b(ENT_\d+|UNKNOWN)\b")


def parse_id_output(raw_output: str) -> str | None:
    r"""Extract first ENT_\d+ or UNKNOWN from raw model output.
    Returns None if no valid ID is found."""
    match = _ID_PATTERN.search(raw_output)
    if match:
        return match.group(1)
    return None


def map_id_to_entity(predicted_id: str | None, id_to_entity_map: dict[str, str]) -> str:
    """Map predicted ID to entity label string. Returns the raw ID if not in map,
    or empty string if predicted_id is None (scored as incorrect)."""
    if predicted_id is None:
        return ""
    if predicted_id == "UNKNOWN":
        return "UNKNOWN"
    return id_to_entity_map.get(predicted_id, predicted_id)
