"""Hash-verified synthetic fixture replay through the real extraction pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

from packages.contracts.api import RedactedSignalInput, SourceReceiptInput

from .idsp import IdspCatalogueRow, parse_idsp_catalogue_text
from .models import Document
from .parse import parse_html
from .pipeline import IngestionPipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DOCUMENT_ROOT = FIXTURE_ROOT / "documents"
FIXED_RETRIEVAL = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
ProcessingState = Literal[
    "active_direct",
    "language_review_required",
    "privacy_review_required",
    "ambiguous_entity_linkage",
]


@dataclass(frozen=True, slots=True)
class FixtureWorkItem:
    fixture_name: str
    source_id: str
    receipt: SourceReceiptInput
    signal: RedactedSignalInput


@dataclass(frozen=True, slots=True)
class FixturePack:
    pack_id: str
    items: tuple[FixtureWorkItem, ...]
    catalogue_rows: tuple[IdspCatalogueRow, ...]
    catalogue_receipt: SourceReceiptInput
    manifest_sha256: str


def _source_shape(language: str) -> str:
    return {
        "or": "odisha_hfw_circulars_or",
        "hi": "pib_bhubaneswar_hi",
        "en": "nhm_odisha_notifications",
    }.get(language, "nhm_odisha_notifications")


def _verified_manifest() -> tuple[dict[str, Any], str]:
    manifest_path = FIXTURE_ROOT / "manifest.json"
    body = manifest_path.read_bytes()
    manifest = json.loads(body)
    documents = manifest.get("documents", {})
    if len(documents) != 12:
        raise RuntimeError("fixture manifest must pin exactly twelve documents")
    for relative, expected in documents.items():
        actual = hashlib.sha256((FIXTURE_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"fixture hash mismatch: {relative}")
    return manifest, hashlib.sha256(body).hexdigest()


def load_fixture_pack() -> FixturePack:
    """Build deterministic redacted records; no fixture bypasses extraction."""

    manifest, manifest_sha256 = _verified_manifest()
    pipeline = IngestionPipeline.default()
    work: list[FixtureWorkItem] = []
    catalogue: list[IdspCatalogueRow] = []
    catalogue_receipt: SourceReceiptInput | None = None
    for index, path in enumerate(sorted(DOCUMENT_ROOT.glob("*.json")), start=1):
        raw = path.read_bytes()
        fixture = json.loads(raw)
        expected = fixture["expected"]
        if "catalogue_rows" in expected:
            rows = parse_idsp_catalogue_text(fixture["ocr_text"])
            if len(rows) != expected["catalogue_rows"]:
                raise RuntimeError(f"catalogue fixture contract failed: {path.name}")
            catalogue.extend(rows)
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            catalogue_receipt = SourceReceiptInput(
                source_snapshot_id=f"fixture_snapshot_{raw_sha256[:24]}",
                source_id="idsp_weekly_outbreaks",
                requested_url=f"fixture://bundled/{path.name}",
                final_url=f"fixture://bundled/{path.name}",
                retrieved_at=FIXED_RETRIEVAL.isoformat().replace("+00:00", "Z"),
                status_code=200,
                content_type="application/json",
                byte_length=len(raw),
                sha256=raw_sha256,
                access_path="bundled_hash_verified_fixture",
                is_fixture=True,
            )
            continue
        if fixture["media_type"] == "text/html":
            parsed = parse_html(fixture["body"].encode("utf-8"))
            text = parsed.text
            confidence = None
        else:
            text = fixture["ocr_text"]
            confidence = fixture.get("ocr_confidence")
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        document = Document(
            document_id=f"fixture-{index:02d}",
            source_id=fixture["source_id"],
            canonical_url=f"fixture://bundled/{path.name}",
            retrieved_at=FIXED_RETRIEVAL,
            content_type=fixture["media_type"],
            text=text,
            sha256=raw_sha256,
            source_language_hint=fixture.get("source_language_hint"),
            ocr_confidence=confidence,
            is_synthetic_fixture=True,
        )
        extracted = pipeline.process(document, as_of=date(2026, 7, 21))
        language = extracted.language.value
        source_id = _source_shape(language)
        snapshot_id = f"fixture_snapshot_{raw_sha256[:24]}"
        signal = RedactedSignalInput(
            signal_id=f"fixture_{extracted.signal_id.removeprefix('sig_')}",
            source_id=source_id,
            source_snapshot_id=snapshot_id,
            district_id=extracted.districts[0].district_id if extracted.districts else None,
            disease=extracted.diseases[0] if extracted.diseases else None,
            assertion=extracted.assertion.value,
            evidence_text=extracted.redacted_evidence,
            evidence_start=0,
            evidence_end=len(extracted.redacted_evidence),
            content_sha256=extracted.evidence_sha256,
            retrieved_at=FIXED_RETRIEVAL.isoformat().replace("+00:00", "Z"),
            event_review_eligible=extracted.eligible_for_event_review,
            processing_state=cast(ProcessingState, extracted.coverage_state.value),
            redaction_state=cast(Literal["heuristic_unvalidated"], extracted.redaction_state),
            language=extracted.language.value,
            extractor_version="rules-v1",
        )
        receipt = SourceReceiptInput(
            source_snapshot_id=snapshot_id,
            source_id=source_id,
            requested_url=f"fixture://bundled/{path.name}",
            final_url=f"fixture://bundled/{path.name}",
            retrieved_at=FIXED_RETRIEVAL.isoformat().replace("+00:00", "Z"),
            status_code=200,
            content_type="application/json",
            byte_length=len(raw),
            sha256=raw_sha256,
            access_path="bundled_hash_verified_fixture",
            is_fixture=True,
        )
        work.append(
            FixtureWorkItem(
                fixture_name=path.name,
                source_id=source_id,
                receipt=receipt,
                signal=signal,
            )
        )
    if len(work) != 11 or len(catalogue) != 1 or catalogue_receipt is None:
        raise RuntimeError("fixture pack must contain eleven signals and one catalogue row")
    return FixturePack(
        pack_id=str(manifest.get("fixture_set", "synthetic_fixture_set_v1")),
        items=tuple(work),
        catalogue_rows=tuple(catalogue),
        catalogue_receipt=catalogue_receipt,
        manifest_sha256=manifest_sha256,
    )
