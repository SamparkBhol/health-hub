"""District-name glossary used to protect proper nouns during translation.

Neural MT mangles rare Indian proper nouns -- ``Khordha`` came back as
``गोरखा`` in an unprotected probe, which is unacceptable for a district-level
health product.  The Odisha gazetteer already carries every district name in
Latin, Devanagari and Odia script, so the same alias table is reused to swap a
district name out for an inert sentinel before translation and to put the
correct native spelling back afterwards.
"""

from __future__ import annotations

import csv
import functools

from . import models

GAZETTEER_PATH = models.REPOSITORY_ROOT / "data" / "gazetteer" / "odisha_district_aliases.csv"

_DEVANAGARI = range(0x0900, 0x0980)
_ODIA = range(0x0B00, 0x0B80)


def _script_of(value: str) -> str:
    if any(ord(character) in _ODIA for character in value):
        return "or"
    if any(ord(character) in _DEVANAGARI for character in value):
        return "hi"
    return "en"


@functools.lru_cache(maxsize=1)
def _aliases_by_district() -> tuple[tuple[str, dict[str, list[str]]], ...]:
    if not GAZETTEER_PATH.exists():  # pragma: no cover - deployment guard
        return ()
    rows: list[tuple[str, dict[str, list[str]]]] = []
    with GAZETTEER_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped: dict[str, list[str]] = {"en": [], "hi": [], "or": []}
            for alias in row["aliases"].split("|"):
                cleaned = alias.strip()
                if cleaned:
                    grouped[_script_of(cleaned)].append(cleaned)
            if not grouped["en"]:
                grouped["en"].append(row["canonical_name"])
            rows.append((row["district_id"], grouped))
    return tuple(rows)


@functools.lru_cache(maxsize=16)
def district_terms(source_language: str, target_language: str) -> dict[str, str]:
    """Map every district alias in the source language to its target spelling."""

    if source_language == target_language:
        return {}
    terms: dict[str, str] = {}
    for _, grouped in _aliases_by_district():
        targets = grouped.get(target_language) or []
        if not targets:
            continue
        replacement = targets[0]
        for alias in grouped.get(source_language) or []:
            terms.setdefault(alias, replacement)
    return terms


def district_display_name(district_id: str, language: str) -> str | None:
    """Preferred spelling of a district id in one of en/hi/or."""

    for identifier, grouped in _aliases_by_district():
        if identifier == district_id:
            names = grouped.get(language) or grouped.get("en") or []
            return names[0] if names else None
    return None
