from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Zero-width joiners and the BOM appear inside Odia and Devanagari conjuncts
# copied out of government CMS pages. They carry no lexical meaning, so both
# the lexicon and the document are stripped of them before matching.
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def normalise_text(value: str) -> str:
    """Fold case, normalise to NFC, drop zero-width marks, collapse whitespace."""

    folded = unicodedata.normalize("NFC", value).casefold()
    return " ".join(_ZERO_WIDTH.sub("", folded).split())


def _has_indic(value: str) -> bool:
    return any(
        0x0900 <= ord(character) <= 0x097F or 0x0B00 <= ord(character) <= 0x0B7F
        for character in value
    )


def _pattern(alias: str) -> str:
    # Inflection is commonly suffix-bound in native-script health reporting
    # (for example ଡେଙ୍ଗୁର, डेंगू से). Left-bound native matching preserves that
    # form; Latin terms and abbreviations retain both bounds so that "add" or
    # "ili" cannot match inside a longer word.
    escaped = re.escape(alias)
    return rf"(?<!\w){escaped}" if _has_indic(alias) else rf"(?<!\w){escaped}(?!\w)"


class DiseaseLexicon:
    def __init__(self, terms: dict[str, tuple[str, ...]], version: str) -> None:
        self.terms = terms
        self.version = version
        self._patterns: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
            disease: tuple((alias, re.compile(_pattern(alias))) for alias in aliases)
            for disease, aliases in terms.items()
        }

    @classmethod
    def load(cls, path: str | Path = "config/disease_lexicon.json") -> DiseaseLexicon:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        terms = {
            disease: tuple(
                dict.fromkeys(normalise_text(term) for term in aliases if term.strip())
            )
            for disease, aliases in value["diseases"].items()
        }
        return cls(terms=terms, version=str(value["schema_version"]))

    def find(self, text: str) -> tuple[str, ...]:
        """Return the disease groups mentioned in `text`, sorted and unique.

        A hit is a mention, never an incidence figure: counting matches would
        count how often a document repeats a word, not how many people fell ill.
        """

        return tuple(sorted({disease for disease, _ in self.find_terms(text)}))

    def find_terms(self, text: str) -> tuple[tuple[str, str], ...]:
        """Return (disease, matched surface term) pairs for evidence display."""

        folded = normalise_text(text)
        matches: list[tuple[str, str]] = []
        for disease, patterns in self._patterns.items():
            for alias, pattern in patterns:
                if pattern.search(folded):
                    matches.append((disease, alias))
                    break
        return tuple(sorted(matches))
