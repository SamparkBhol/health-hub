from __future__ import annotations

import re
from datetime import date

from .models import AssertionClass

_NEGATED = (
    "no case",
    "no cases",
    "not detected",
    "not reported",
    "tested negative",
    "ruled out",
    "free of",
    "zero dengue",
    "zero malaria",
    "zero cholera",
    "zero aes",
    "no new case",
    "denied reports",
    "कोई मामला नहीं",
    "मामला नहीं मिला",
    "पुष्टि नहीं",
    "କୌଣସି ମାମଲା ମିଳିନାହିଁ",
    "ମାମଲା ନାହିଁ",
    "ଚିହ୍ନଟ ହୋଇନାହିଁ",
    "ପୁଷ୍ଟି ହୋଇନାହିଁ",
)
_SPECULATIVE = (
    "may cause",
    "may lead",
    "could lead",
    "risk of",
    "fear of",
    "suspected risk",
    "हो सकता",
    "आशंका",
    "खतरा",
    "ହୋଇପାରେ",
    "ଆଶଙ୍କା",
    "ବିପଦ",
)
_NON_CURRENT = (
    "last year",
    "previous outbreak",
    "in the past",
    "historical",
    "पिछले वर्ष",
    "पुराना प्रकोप",
    "ଗତ ବର୍ଷ",
    "ପୂର୍ବ ପ୍ରକୋପ",
    "ଇତିହାସରେ",
)


def classify_assertion(text: str, *, as_of: date | None = None) -> AssertionClass:
    """Classify a disease-bearing evidence span into the four v1 classes.

    This is deliberately a transparent triage rule, not a claim of validated
    Odia clinical NLP. A native-language gold set must gate autonomous use.
    """

    folded = " ".join(text.casefold().split())
    zero_case_pattern = re.search(
        r"\b(?:zero|0)\s+(?:new\s+)?(?:[a-z -]+\s+)?cases?\b", folded
    )
    if zero_case_pattern or any(cue in folded for cue in _NEGATED):
        return AssertionClass.NOT_AFFIRMED
    if any(cue in folded for cue in _SPECULATIVE):
        return AssertionClass.SPECULATIVE
    if any(cue in folded for cue in _NON_CURRENT):
        return AssertionClass.NON_CURRENT
    current = as_of or date.today()
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", folded)]
    if years and max(years) < current.year - 1:
        return AssertionClass.NON_CURRENT
    return AssertionClass.AFFIRMED
