"""Parse .lp format files to extract constraints, objective, bounds, and integrality sections."""

import re
from dataclasses import dataclass, field


@dataclass
class LPModel:
    objective_sense: str = "minimize"
    objective_name: str = ""
    objective_text: str = ""
    constraints: dict[str, str] = field(default_factory=dict)
    constraint_order: list[str] = field(default_factory=list)
    bounds_text: str = ""
    integrality_text: str = ""
    has_integers: bool = False


_SECTION_PATTERNS = [
    (r"(?i)^(minimize|min)\b", "objective"),
    (r"(?i)^(maximize|max)\b", "objective"),
    (r"(?i)^(subject\s+to|s\.t\.|st)\b", "constraints"),
    (r"(?i)^(bounds?)\b", "bounds"),
    (r"(?i)^(general|generals)\b", "integrality"),
    (r"(?i)^(integer|integers)\b", "integrality"),
    (r"(?i)^(binary|binaries)\b", "integrality"),
    (r"(?i)^(end)\b", "end"),
]

_CONSTRAINT_NAME_RE = re.compile(r"^\s*(\S+?)\s*:\s*(.+)$")


def _detect_section(line: str) -> str | None:
    stripped = line.strip()
    for pattern, section in _SECTION_PATTERNS:
        if re.match(pattern, stripped):
            return section
    return None


def parse_lp_string(lp_text: str) -> LPModel:
    """Parse an LP-format string into structured components."""
    model = LPModel()
    lines = lp_text.splitlines()

    current_section = None
    section_lines: dict[str, list[str]] = {
        "objective": [],
        "constraints": [],
        "bounds": [],
        "integrality": [],
    }

    for line in lines:
        raw = line.rstrip()
        if not raw.strip() or raw.strip().startswith("\\"):
            continue

        detected = _detect_section(raw)
        if detected is not None:
            if detected == "end":
                break
            if detected == "objective":
                sense_match = re.match(r"(?i)(minimize|min|maximize|max)", raw.strip())
                if sense_match:
                    s = sense_match.group(1).lower()
                    model.objective_sense = "maximize" if s.startswith("max") else "minimize"
            current_section = detected
            remainder = re.sub(r"(?i)^(minimize|min|maximize|max|subject\s+to|s\.t\.|st|bounds?|general|generals|integer|integers|binary|binaries)\s*", "", raw.strip())
            if remainder.strip():
                section_lines[current_section].append(remainder.strip())
            continue

        if current_section and current_section in section_lines:
            section_lines[current_section].append(raw)

    _parse_objective(model, section_lines["objective"])
    _parse_constraints(model, section_lines["constraints"])
    model.bounds_text = "\n".join(section_lines["bounds"]).strip()
    model.integrality_text = "\n".join(section_lines["integrality"]).strip()
    model.has_integers = bool(model.integrality_text)

    return model


def _parse_objective(model: LPModel, lines: list[str]) -> None:
    full = " ".join(l.strip() for l in lines)
    m = re.match(r"^\s*(\S+?)\s*:\s*(.+)$", full)
    if m:
        model.objective_name = m.group(1)
        model.objective_text = m.group(2).strip()
    else:
        model.objective_text = full.strip()


def _parse_constraints(model: LPModel, lines: list[str]) -> None:
    combined_lines = _combine_continuation_lines(lines)
    auto_idx = 0
    for cline in combined_lines:
        cline = cline.strip()
        if not cline:
            continue
        m = _CONSTRAINT_NAME_RE.match(cline)
        if m:
            name = m.group(1)
            text = m.group(2).strip()
        else:
            name = f"c{auto_idx:04d}"
            text = cline
        model.constraints[name] = text
        model.constraint_order.append(name)
        auto_idx += 1


def _combine_continuation_lines(lines: list[str]) -> list[str]:
    """Combine continuation lines (lines starting with whitespace that don't have a constraint name)."""
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if line[0] in (" ", "\t") and result and not _CONSTRAINT_NAME_RE.match(stripped):
            result[-1] = result[-1] + " " + stripped
        else:
            result.append(stripped)
    return result


def parse_lp_file(path: str) -> LPModel:
    """Parse an LP-format file."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_lp_string(f.read())


def get_constraint_text(model: LPModel, name_or_index: str | int) -> str | None:
    """Get constraint text by name or 0-based index."""
    if isinstance(name_or_index, int):
        if 0 <= name_or_index < len(model.constraint_order):
            name = model.constraint_order[name_or_index]
            return model.constraints[name]
        return None
    return model.constraints.get(name_or_index)


def get_constraint_name(model: LPModel, index: int) -> str | None:
    """Get constraint name by 0-based index."""
    if 0 <= index < len(model.constraint_order):
        return model.constraint_order[index]
    return None


def build_name_to_text_map(model: LPModel) -> dict[str, str]:
    """Build a mapping from constraint name to full constraint text."""
    return dict(model.constraints)


def strip_integrality(lp_text: str) -> str:
    """Remove integer/binary/general sections from LP text to produce LP relaxation."""
    lines = lp_text.splitlines()
    result = []
    skip = False
    for line in lines:
        detected = _detect_section(line)
        if detected == "integrality":
            skip = True
            continue
        if detected is not None and detected != "integrality":
            skip = False
        if not skip:
            result.append(line)
    return "\n".join(result)
