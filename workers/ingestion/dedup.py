from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import ExtractedSignal


@dataclass(frozen=True, slots=True)
class DedupCandidate:
    left_signal_id: str
    right_signal_id: str
    score: float
    reasons: tuple[str, ...]
    cross_language: bool
    disposition: str = "review_required_never_auto_merge"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w\u0900-\u097f\u0b00-\u0b7f]+", text.casefold())
        if len(token) > 1
    }


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _structural_overlap(left: ExtractedSignal, right: ExtractedSignal) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    left_diseases, right_diseases = set(left.diseases), set(right.diseases)
    if left_diseases and left_diseases & right_diseases:
        score += 0.40
        reasons.append("same_normalised_disease")
    left_places = {item.district_id for item in left.districts}
    right_places = {item.district_id for item in right.districts}
    if left_places and left_places & right_places:
        score += 0.40
        reasons.append("same_resolved_district")
    delta_hours = abs((left.retrieved_at - right.retrieved_at).total_seconds()) / 3600
    if delta_hours <= 72:
        score += 0.20 * (1 - delta_hours / 96)
        reasons.append("retrieved_within_72_hours")
    return score, reasons


def candidate_pair(left: ExtractedSignal, right: ExtractedSignal) -> DedupCandidate | None:
    if left.signal_id == right.signal_id:
        return None
    cross_language = left.language != right.language
    score, reasons = _structural_overlap(left, right)
    if not cross_language:
        lexical = _jaccard(left.redacted_evidence, right.redacted_evidence)
        score = min(1.0, score * 0.7 + lexical * 0.3)
        if lexical >= 0.70:
            reasons.append("high_within_language_token_overlap")
    # No multilingual encoder is smuggled in here. Cross-language pairs are
    # candidates only when their explicit event fields align.
    threshold = 0.70 if cross_language else 0.75
    if score < threshold:
        return None
    return DedupCandidate(
        left_signal_id=left.signal_id,
        right_signal_id=right.signal_id,
        score=round(score, 4),
        reasons=tuple(reasons),
        cross_language=cross_language,
    )


def generate_candidates(signals: list[ExtractedSignal]) -> tuple[DedupCandidate, ...]:
    candidates: list[DedupCandidate] = []
    for index, left in enumerate(signals):
        for right in signals[index + 1 :]:
            candidate = candidate_pair(left, right)
            if candidate:
                candidates.append(candidate)
    return tuple(
        sorted(
            candidates, key=lambda item: (-item.score, item.left_signal_id, item.right_signal_id)
        )
    )


def content_fingerprint(value: bytes | str) -> str:
    if isinstance(value, str):
        value = " ".join(value.casefold().split()).encode("utf-8")
    return hashlib.sha256(value).hexdigest()
