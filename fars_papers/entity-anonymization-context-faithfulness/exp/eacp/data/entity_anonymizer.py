"""Entity anonymization module for condition C (EACP). Replaces all entity
surface forms in cf_context and question with anonymized placeholders (ENT_k),
producing texts where no real entity names remain.

Fixes applied over original version:
  1. Parenthetical stripping: labels like "Mike Flanagan (filmmaker)" also
     generate "Mike Flanagan" (without disambiguation) as a surface form.
  2. Partial-name replacement for person entities only: multi-word person
     names add individual name parts as surface forms to catch back-references
     like "Mahrez" when the full name is "Riyad Mahrez"."""

import logging
import re
from typing import Any

from eacp.data.confiqa_loader import extract_path_entities
from eacp.data.entity_inventory import build_entity_inventory

logger = logging.getLogger(__name__)

STOPWORDS = frozenset({
    "the", "of", "and", "in", "on", "at", "to", "for", "a", "an", "is", "by",
    "new", "north", "south", "east", "west", "city", "united", "states", "state",
    "republic", "kingdom", "island", "general", "national", "san", "de", "la",
    "el", "del", "las", "los", "van", "von", "den", "der", "des", "du", "le",
    "al", "ibn", "bin", "di", "da", "do", "das", "grand", "great", "old",
    "men", "men's", "women", "women's", "team", "association", "club",
    "international", "program", "technical", "cooperation", "series",
    "film", "song", "game", "video", "novel", "album", "tv",
    "his", "her", "its", "our", "their", "your", "my",
    "king", "queen", "prince", "princess", "lord", "lady", "sir", "saint",
    "head", "god", "grace", "faith", "other", "third", "first", "second",
    "defender", "territories", "realms", "commonwealth", "majesty",
    "war", "life", "star", "world", "story", "dark", "last", "dead", "lost",
    "from", "with", "into", "that", "this", "were", "been", "have", "will",
    "born", "earl", "duke", "count",
})

SPLITTABLE_TYPES = frozenset({"person", "entity"})
MAX_NAME_WORDS_FOR_SPLITTING = 6
MIN_PART_LENGTH = 4


def _strip_parenthetical(name: str) -> str | None:
    m = re.match(r"^(.+?)\s*\(.*\)\s*$", name)
    if m:
        stripped = m.group(1).strip()
        if len(stripped) >= 2:
            return stripped
    return None


def _build_surface_forms(
    entity_label: str, aliases: set[str], entity_type: str
) -> list[str]:
    forms = set()
    raw_names = {entity_label} | aliases

    for name in raw_names:
        forms.add(name)
        forms.add(name.lower())
        forms.add(name.title())

        stripped = _strip_parenthetical(name)
        if stripped:
            forms.add(stripped)
            forms.add(stripped.lower())
            forms.add(stripped.title())
            name_for_parts = stripped
        else:
            name_for_parts = name

        if entity_type in SPLITTABLE_TYPES:
            parts = name_for_parts.split()
            if 1 < len(parts) <= MAX_NAME_WORDS_FOR_SPLITTING:
                for part in parts:
                    clean = part.rstrip(",.;:")
                    if len(clean) >= MIN_PART_LENGTH and clean.lower() not in STOPWORDS:
                        forms.add(clean)
                        forms.add(clean.lower())
                        forms.add(clean.title())

    forms.discard("")
    return sorted(forms, key=lambda s: -len(s))


def _build_replacement_regex(surface_forms: list[str]) -> re.Pattern | None:
    if not surface_forms:
        return None
    escaped = [re.escape(f) for f in surface_forms]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def anonymize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Main entry point. Returns dict with:
      - anon_context: anonymized cf_context
      - anon_question: anonymized question
      - entity_map: {label: (ent_id, ent_type)} from build_entity_inventory
      - replacement_stats: {ent_id: count} of replacements made
      - warnings: list of warning strings
    """
    entity_map = build_entity_inventory(instance)
    entities_with_aliases = extract_path_entities(instance)

    replacement_rules: list[tuple[re.Pattern, str, str]] = []
    for label, (ent_id, ent_type) in entity_map.items():
        aliases = entities_with_aliases.get(label, {label})
        surface_forms = _build_surface_forms(label, aliases, ent_type)
        regex = _build_replacement_regex(surface_forms)
        if regex is not None:
            replacement_rules.append((regex, ent_id, label))

    replacement_rules.sort(key=lambda r: -max(len(alt) for alt in r[0].pattern.split("|")))

    anon_context = instance["cf_context"]
    anon_question = instance["question"]
    replacement_stats: dict[str, int] = {}
    warnings: list[str] = []

    for regex, ent_id, _label in replacement_rules:
        anon_context, n1 = regex.subn(ent_id, anon_context)
        anon_question, n2 = regex.subn(ent_id, anon_question)
        replacement_stats[ent_id] = n1 + n2

    for label, (ent_id, _) in entity_map.items():
        if replacement_stats.get(ent_id, 0) == 0:
            msg = f"Entity '{label}' ({ent_id}) not found in context or question"
            warnings.append(msg)
            logger.warning(msg)

    return {
        "anon_context": anon_context,
        "anon_question": anon_question,
        "entity_map": entity_map,
        "replacement_stats": replacement_stats,
        "warnings": warnings,
    }
