"""Live end-to-end probe for Objective 1 (crawl → parse → route → extract).

Not a pytest module: it performs real network retrieval and is run by hand.

    CRAWLER_CONTACT="mailto:you@example.org" .venv/bin/python tests/live_trilingual_probe.py

For every enabled registered source it fetches the index, parses it, routes the
language, extracts disease and district mentions, then follows the single
highest-ranked discovered link to show that the pipeline reaches an actual
notice rather than stopping at navigation chrome. Politeness gaps come from
each source's declared `minimum_interval_seconds`.

Extraction counts are mention counts. They are never case counts.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workers.ingestion.connectors import ingest_registered_url  # noqa: E402
from workers.ingestion.language import script_profile  # noqa: E402
from workers.ingestion.pipeline import IngestionPipeline  # noqa: E402
from workers.ingestion.registry import SourceRegistry, SourceSpec, load_registry  # noqa: E402
from workers.ingestion.safe_fetch import FetchError  # noqa: E402


@dataclass
class Row:
    source_id: str
    declared: str
    stage: str
    status: str
    route: str = "-"
    shares: str = "-"
    diseases: tuple[str, ...] = ()
    districts: tuple[str, ...] = ()
    detail: str = ""
    notes: list[str] = field(default_factory=list)


class Politeness:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def wait(self, url: str, seconds: int) -> None:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        previous = self._last.get(host)
        if previous is not None:
            remaining = seconds - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)
        self._last[host] = time.monotonic()


def _shares(text: str) -> str:
    profile = script_profile(text)
    if profile.total == 0:
        return "no letters"
    parts = sorted(profile.counts.items(), key=lambda item: -item[1])
    return " ".join(f"{name}:{count / profile.total:.0%}" for name, count in parts)


def probe_url(
    *,
    registry: SourceRegistry,
    source: SourceSpec,
    url: str,
    pipeline: IngestionPipeline,
    stage: str,
    politeness: Politeness,
    approved: frozenset[str] = frozenset(),
) -> tuple[Row, object | None]:
    politeness.wait(url, source.minimum_interval_seconds)
    row = Row(
        source_id=source.id,
        declared=",".join(source.languages),
        stage=stage,
        status="",
    )
    try:
        outcome = ingest_registered_url(
            registry=registry,
            source_id=source.id,
            url=url,
            pipeline=pipeline,
            approved_pdf_sha256s=approved,
        )
    except FetchError as exc:
        row.status = f"FETCH_FAIL {exc.code}"
        return row, None
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        row.status = f"ERROR {type(exc).__name__}"
        return row, None
    receipt = outcome.receipt
    row.status = f"{receipt.status_code} {receipt.content_type} {receipt.byte_length}B"
    if receipt.access_path != "live_origin":
        row.notes.append(f"access_path={receipt.access_path}")
    if outcome.signal is None:
        row.notes.append(outcome.processing_state)
        if outcome.catalogue_rows:
            row.notes.append(f"catalogue_rows={len(outcome.catalogue_rows)}")
        return row, outcome
    signal = outcome.signal
    row.route = signal.language.value
    row.shares = _shares(signal.redacted_evidence)
    row.diseases = signal.diseases
    row.districts = tuple(match.canonical_name for match in signal.districts)
    row.notes.append(f"assertion={signal.assertion.value}")
    row.notes.append(f"coverage={signal.coverage_state.value}")
    return row, outcome


def main() -> int:
    contact = os.getenv("CRAWLER_CONTACT", "").strip()
    if not contact or contact.endswith("example.invalid"):
        print("CRAWLER_CONTACT must be set to a monitored address before crawling.")
        return 2
    registry = load_registry()
    pipeline = IngestionPipeline.default()
    politeness = Politeness()
    rows: list[Row] = []
    enabled = [source for source in registry.sources if source.enabled]
    print(f"contact={contact}")
    print(f"enabled sources: {len(enabled)} of {len(registry.sources)}\n")

    for source in enabled:
        row, outcome = probe_url(
            registry=registry,
            source=source,
            url=source.url,
            pipeline=pipeline,
            stage="index",
            politeness=politeness,
        )
        rows.append(row)
        print(f"  probed index   {source.id}: {row.status}", flush=True)
        if outcome is None:
            continue
        candidates = [
            link for link in getattr(outcome, "discovered_links", ()) if link.score > 0
        ]
        if not candidates:
            continue
        best = candidates[0]
        detail_row, detail_outcome = probe_url(
            registry=registry,
            source=source,
            url=best.url,
            pipeline=pipeline,
            stage="detail",
            politeness=politeness,
        )
        detail_row.detail = f"score={best.score:g} {best.label[:70]}"
        if (
            detail_outcome is not None
            and getattr(detail_outcome, "processing_state", "")
            == "metadata_only_unapproved_pdf_hash"
        ):
            # The collector holds unknown PDFs at metadata until their hash is
            # approved. Approve this exact byte string and parse it, so the
            # probe exercises the same gate the runtime enforces.
            digest = detail_outcome.receipt.sha256
            detail_row, _ = probe_url(
                registry=registry,
                source=source,
                url=best.url,
                pipeline=pipeline,
                stage="detail",
                politeness=politeness,
                approved=frozenset({digest}),
            )
            detail_row.detail = f"score={best.score:g} {best.label[:70]}"
            detail_row.notes.append("pdf_hash_approved_for_probe")
        rows.append(detail_row)
        print(f"  probed detail  {source.id}: {detail_row.status}", flush=True)

    print()
    header = (
        f"{'source_id':30} {'decl':5} {'stage':6} {'route':5} "
        f"{'script shares':22} {'diseases':34} districts"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.source_id:30.30} {row.declared:5.5} {row.stage:6.6} {row.route:5.5} "
            f"{row.shares:22.22} {','.join(row.diseases) or '-':34.34} "
            f"{','.join(row.districts) or '-'}"
        )
        print(f"{'':30} status={row.status} {' '.join(row.notes)}")
        if row.detail:
            print(f"{'':30} link={row.detail}")
    print(
        "\nDisease and district values are mentions extracted from published "
        "documents. They are not case counts and not incidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
