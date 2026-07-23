from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Redaction, RedactionResult


@dataclass(frozen=True, slots=True)
class _Pattern:
    kind: str
    expression: re.Pattern[str]


_PATTERNS = (
    _Pattern("EMAIL", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    _Pattern("PHONE", re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)")),
    _Pattern("ID", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    _Pattern(
        "PERSON",
        re.compile(
            r"(?:(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+(?:[A-Z][a-z]+(?:\s+|$)){1,4})"
            r"|(?:\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}"
            r"(?=\s+(?:in|from)\s+[A-Z][a-z]+\s+(?:has|was|died|tested)\b))"
            r"|(?:(?:श्री|श्रीमती|डॉ\.?)[\s\u200c\u200d]+[\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){0,2})"
            r"|(?:(?:ଶ୍ରୀ|ଶ୍ରୀମତୀ|ଡା\.?)[\s\u200c\u200d]+[\u0B00-\u0B7F]{2,}(?:\s+[\u0B00-\u0B7F]{2,}){0,2})"
        ),
    ),
)


def redact_pii(text: str) -> RedactionResult:
    """Return model-visible text with obvious identifiers replaced.

    Offsets refer to the original input and raw matched values are not returned.
    This heuristic mitigates exposure; it does not claim measured PII recall.
    """

    matches: list[tuple[int, int, str]] = []
    for item in _PATTERNS:
        for match in item.expression.finditer(text):
            if any(match.start() < end and match.end() > start for start, end, _ in matches):
                continue
            matches.append((match.start(), match.end(), item.kind))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    counters: dict[str, int] = {}
    output: list[str] = []
    redactions: list[Redaction] = []
    cursor = 0
    for start, end, kind in matches:
        if start < cursor:
            continue
        counters[kind] = counters.get(kind, 0) + 1
        placeholder = f"[{kind}_{counters[kind]}]"
        output.extend((text[cursor:start], placeholder))
        redactions.append(Redaction(kind=kind, placeholder=placeholder, start=start, end=end))
        cursor = end
    output.append(text[cursor:])
    return RedactionResult(text="".join(output), redactions=tuple(redactions))
