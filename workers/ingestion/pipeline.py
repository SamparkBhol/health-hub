from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

from .assertions import classify_assertion
from .diseases import DiseaseLexicon
from .geography import DistrictGazetteer
from .language import route_unicode
from .models import CoverageState, Document, ExtractedSignal, LanguageRoute
from .redact import redact_pii


@dataclass(slots=True)
class IngestionPipeline:
    gazetteer: DistrictGazetteer
    diseases: DiseaseLexicon
    minimum_ocr_confidence: float = 0.70

    @classmethod
    def default(cls) -> IngestionPipeline:
        return cls(gazetteer=DistrictGazetteer.load(), diseases=DiseaseLexicon.load())

    def process(self, document: Document, *, as_of: date | None = None) -> ExtractedSignal:
        language = route_unicode(document.text)
        redacted = redact_pii(document.text)
        # Every detected identifier is still replaced, everywhere, before
        # anything downstream sees the text.  Only the *escalation* question --
        # "did this publication disclose a personal detail?" -- is asked of the
        # article rather than of the site furniture wrapped around it.
        article_redactions = (
            redacted.redactions
            if document.article_text is None
            else redact_pii(document.privacy_scan_text).redactions
        )
        chrome_only = len(redacted.redactions) - len(article_redactions)
        # Extraction consumes only the redacted form. No person value can enter
        # a model feature, dedup fingerprint or exported evidence packet.
        diseases = self.diseases.find(redacted.text)
        districts = self.gazetteer.resolve(redacted.text)
        assertion = classify_assertion(redacted.text, as_of=as_of)
        warnings: list[str] = ["redaction_recall_unvalidated"]
        coverage = CoverageState.ACTIVE_DIRECT
        if language == LanguageRoute.UNDETERMINED:
            coverage = CoverageState.LANGUAGE_REVIEW_REQUIRED
            warnings.append("romanised_or_short_text_not_routed")
        if (
            document.ocr_confidence is not None
            and document.ocr_confidence < self.minimum_ocr_confidence
        ):
            coverage = CoverageState.LANGUAGE_REVIEW_REQUIRED
            warnings.append("ocr_confidence_below_floor")
        if document.ocr_confidence is not None:
            warnings.append("odia_hindi_ocr_accuracy_unmeasured_on_target_corpus")
        if article_redactions:
            coverage = CoverageState.PRIVACY_REVIEW_REQUIRED
            warnings.append("detected_personal_detail_requires_privacy_review")
        elif chrome_only > 0:
            warnings.append("personal_detail_found_only_in_site_chrome_and_redacted")
        eligible = bool(
            diseases
            and districts
            and assertion.value == "affirmed"
            and coverage == CoverageState.ACTIVE_DIRECT
        )
        if assertion.value != "affirmed":
            warnings.append(f"assertion_{assertion.value}_not_event_eligible")
        evidence_hash = hashlib.sha256(redacted.text.encode("utf-8")).hexdigest()
        signal_seed = f"{document.document_id}:{evidence_hash}".encode()
        return ExtractedSignal(
            signal_id=f"sig_{hashlib.sha256(signal_seed).hexdigest()[:20]}",
            document_id=document.document_id,
            source_id=document.source_id,
            canonical_url=document.canonical_url,
            retrieved_at=document.retrieved_at,
            language=language,
            assertion=assertion,
            diseases=diseases,
            districts=districts,
            redacted_evidence=redacted.text,
            redaction_state=redacted.state,
            evidence_sha256=evidence_hash,
            eligible_for_event_review=eligible,
            coverage_state=coverage,
            is_synthetic_fixture=document.is_synthetic_fixture,
            warnings=tuple(warnings),
            metadata={
                "redaction_count": len(redacted.redactions),
                "site_chrome_redaction_count": chrome_only,
                "disease_lexicon_version": self.diseases.version,
                "ocr_confidence": document.ocr_confidence,
            },
        )
