from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from .models import LanguageRoute

ODIA_FIRST, ODIA_LAST = 0x0B00, 0x0B7F
DEVANAGARI_FIRST, DEVANAGARI_LAST = 0x0900, 0x097F

# Function words are the only evidence used to separate English from
# romanised Odia/Hindi.  Romanised Indic text is never guessed; it stays
# `und` so a reviewer or a dedicated model handles it.
ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "the",
        "and",
        "of",
        "in",
        "for",
        "on",
        "to",
        "with",
        "from",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "this",
        "that",
        "district",
        "districts",
        "health",
        "cases",
        "reported",
        "department",
        "government",
        "notification",
        "notifications",
        "circular",
        "circulars",
        "report",
        "state",
        "public",
        # Administrative English that appears on index pages carrying no prose.
        # None of these are plausible romanised Odia or Hindi tokens, so they
        # separate an English listing page from romanised Indic text.
        "week",
        "weeks",
        "weekly",
        "year",
        "outbreak",
        "outbreaks",
        "surveillance",
        "disease",
        "diseases",
        "download",
        "screen",
        "reader",
        "home",
        "about",
        "contact",
        "search",
        "sitemap",
        "help",
        "title",
        "date",
        "details",
        "page",
        "programme",
        "hospital",
        "medical",
        "office",
        "notice",
        "order",
        "list",
        "name",
        "number",
        "status",
    }
)

DEFAULT_DOMINANT_SHARE = 0.60
DEFAULT_BLOCK_SHARE = 0.60
DEFAULT_BILINGUAL_SHARE = 0.25


@dataclass(frozen=True, slots=True)
class ScriptProfile:
    """Per-script letter/mark weights for one document."""

    counts: Mapping[str, int]
    total: int

    def share(self, script: str) -> float:
        if self.total <= 0:
            return 0.0
        return self.counts.get(script, 0) / self.total

    @property
    def dominant(self) -> tuple[str | None, float]:
        if self.total <= 0:
            return None, 0.0
        script = max(self.counts, key=lambda key: (self.counts[key], key))
        return script, self.counts[script] / self.total


def script_of(character: str) -> str | None:
    """Classify one character, ignoring digits, punctuation and symbols.

    U+0964/U+0965 danda punctuation is shared by several Indic scripts and is
    category `Po`, so it is excluded here rather than counted as Devanagari.
    """

    category = unicodedata.category(character)
    if category[0] not in {"L", "M"}:
        return None
    point = ord(character)
    if ODIA_FIRST <= point <= ODIA_LAST:
        return "or"
    if DEVANAGARI_FIRST <= point <= DEVANAGARI_LAST:
        return "hi"
    if character.isalpha() and "LATIN" in unicodedata.name(character, ""):
        return "latin"
    return None


def script_counts(text: str) -> Counter[str]:
    """Count letters and marks per script; everything else is ignored."""

    counts: Counter[str] = Counter()
    for character in unicodedata.normalize("NFC", text):
        script = script_of(character)
        if script is not None:
            counts[script] += 1
    return counts


def _blocks(text: str) -> Iterator[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def script_profile(
    text: str,
    *,
    block_dominant_share: float = DEFAULT_BLOCK_SHARE,
) -> ScriptProfile:
    """Weight each text block by its length and vote for its dominant script.

    Site chrome (a bilingual masthead, a language switcher, a footer credit)
    arrives as short blocks, so it cannot outvote the body of a page. A block
    with no dominant script — a genuinely bilingual line — contributes its
    characters proportionally instead of voting once.
    """

    votes: Counter[str] = Counter()
    saw_block = False
    for block in _blocks(text):
        counts = script_counts(block)
        total = sum(counts.values())
        if total == 0:
            continue
        saw_block = True
        script, count = counts.most_common(1)[0]
        if count / total >= block_dominant_share:
            votes[script] += total
        else:
            votes.update(counts)
    if not saw_block:
        votes = script_counts(text)
    return ScriptProfile(counts=dict(votes), total=sum(votes.values()))


def has_english_function_words(text: str) -> bool:
    words = {word.strip(".,:;!?()[]{}\"'`").casefold() for word in text.split()}
    return bool(words & ENGLISH_FUNCTION_WORDS)


def route_unicode(
    text: str,
    *,
    minimum_script_characters: int = 2,
    dominant_share: float = DEFAULT_DOMINANT_SHARE,
    bilingual_share: float = DEFAULT_BILINGUAL_SHARE,
) -> LanguageRoute:
    """Route a document to the script that actually dominates it.

    Routing is proportional, not presence-based: an English page that carries
    `ଓଡ଼ିଶା ସରକାର` in its masthead is English, because the Odia share of its
    letters is a fraction of a percent. A page whose letters are genuinely
    split between two scripts is `mixed`, and Latin text without English
    function words stays `und` rather than being guessed as romanised Indic.
    """

    profile = script_profile(text)
    if profile.total < minimum_script_characters:
        return LanguageRoute.UNDETERMINED
    script, share = profile.dominant
    if script is not None and share >= dominant_share:
        count = profile.counts.get(script, 0)
        if count < minimum_script_characters:
            return LanguageRoute.UNDETERMINED
        if script == "or":
            return LanguageRoute.ODIA
        if script == "hi":
            return LanguageRoute.HINDI
        return (
            LanguageRoute.ENGLISH
            if has_english_function_words(text)
            else LanguageRoute.UNDETERMINED
        )
    substantive = [
        name
        for name, count in profile.counts.items()
        if count >= minimum_script_characters and count / profile.total >= bilingual_share
    ]
    if len(substantive) >= 2:
        return LanguageRoute.MIXED
    return LanguageRoute.UNDETERMINED
