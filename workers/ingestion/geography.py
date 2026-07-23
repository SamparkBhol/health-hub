from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import DistrictMatch

# Zero-width joiners inside Odia and Devanagari conjuncts are invisible in the
# CMS but would otherwise split an alias in two once punctuation is blanked.
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")


def _normalise(value: str) -> str:
    value = _ZERO_WIDTH.sub("", unicodedata.normalize("NFC", value).casefold())
    return " ".join(re.sub(r"[^\w\u0900-\u097f\u0b00-\u0b7f]+", " ", value).split())


@dataclass(frozen=True, slots=True)
class Alias:
    district_id: str
    canonical_name: str
    display: str
    normalised: str


class DistrictGazetteer:
    def __init__(self, aliases: tuple[Alias, ...]) -> None:
        self.aliases = tuple(sorted(aliases, key=lambda item: len(item.normalised), reverse=True))

    @classmethod
    def load(
        cls, path: str | Path = "data/gazetteer/odisha_district_aliases.csv"
    ) -> DistrictGazetteer:
        aliases: list[Alias] = []
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for display in row["aliases"].split("|"):
                    aliases.append(
                        Alias(
                            district_id=row["district_id"],
                            canonical_name=row["canonical_name"],
                            display=display,
                            normalised=_normalise(display),
                        )
                    )
        return cls(tuple(aliases))

    def resolve(self, text: str) -> tuple[DistrictMatch, ...]:
        folded = _normalise(text)
        matches: dict[str, DistrictMatch] = {}
        for alias in self.aliases:
            has_indic = any(
                0x0900 <= ord(char) <= 0x097F or 0x0B00 <= ord(char) <= 0x0B7F
                for char in alias.normalised
            )
            # Native case/postposition markers can be suffix-bound (ଖୋର୍ଦ୍ଧାରେ).
            # The left boundary prevents a match inside a longer stem; the
            # right boundary is retained for Latin aliases.
            pattern = (
                rf"(?<!\w){re.escape(alias.normalised)}"
                if has_indic
                else rf"(?<!\w){re.escape(alias.normalised)}(?!\w)"
            )
            match = re.search(pattern, folded)
            if not match or alias.district_id in matches:
                continue
            # These are offsets into normalised text, explicitly not source evidence offsets.
            matches[alias.district_id] = DistrictMatch(
                district_id=alias.district_id,
                canonical_name=alias.canonical_name,
                matched_alias=alias.display,
                start=match.start(),
                end=match.end(),
            )
        return tuple(sorted(matches.values(), key=lambda item: item.district_id))
