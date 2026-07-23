"""On-device multilingual NLP: translation, semantic retrieval, grounded answers.

Every capability degrades to a typed state when its model is not downloaded, so
the API process runs unchanged on a machine with no models present.
"""

from __future__ import annotations

from typing import Any

from . import answer, models, retrieval, translate
from .answer import GroundedAnswer, answer_question
from .retrieval import EvidenceRecord, RankedRecord, RetrievalResult, rank
from .translate import TranslationResult

__all__ = [
    "EvidenceRecord",
    "GroundedAnswer",
    "RankedRecord",
    "RetrievalResult",
    "TranslationResult",
    "answer",
    "answer_question",
    "models",
    "nlp_status",
    "rank",
    "retrieval",
    "translate",
]


def nlp_status() -> dict[str, Any]:
    """One readiness payload for every model-backed capability."""

    return {
        "mode": models.nlp_mode(),
        "models_directory": str(models.models_directory()),
        "translation": translate.status(),
        "retrieval": retrieval.status(),
        "generation": answer.status(),
        "models": models.status(),
    }
