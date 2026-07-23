#!/usr/bin/env python3
"""Exercise the durable repository contract against a real PostgreSQL server.

This is intentionally an additive, idempotent smoke check.  It creates the
application tables when absent and replays hash-pinned synthetic fixtures, but
never drops a database, schema, table, or pre-existing row.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Support direct execution from a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.api.database import Database
from workers.ingestion.registry import load_registry


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fixture_signal_ids(database: Database) -> tuple[str, ...]:
    rows = database.list_signals(fixture_mode="fixture_only", limit=500)
    return tuple(sorted(str(row["id"]) for row in rows))


def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgresql://", "postgres://", "postgresql+psycopg://")):
        raise SystemExit("DATABASE_URL must identify an isolated PostgreSQL smoke database")

    database: Database | None = None
    readers: list[Database] = []
    try:
        database = Database(database_url)
        require(database.backend == "postgresql", "repository did not select PostgreSQL")
        require(database.ready(), "PostgreSQL readiness query failed")

        sources = database.list_sources()
        registry = load_registry()
        expected_source_count = len(registry.sources)
        expected_enabled_count = sum(bool(source.enabled) for source in registry.sources)
        require(
            len(sources) == expected_source_count,
            f"expected {expected_source_count} registered sources, found {len(sources)}",
        )
        enabled_count = sum(bool(row["enabled"]) for row in sources)
        require(
            enabled_count == expected_enabled_count,
            f"expected {expected_enabled_count} enabled sources, found {enabled_count}",
        )
        database.replay_demo_fixtures()
        replay = database.replay_demo_fixtures()
        require(replay["created_signals"] == 0, "fixture replay is not signal-idempotent")
        require(
            replay["created_review_tasks"] == 0,
            "fixture replay is not review-task-idempotent",
        )
        require(
            replay["created_catalogue_events"] == 0,
            "fixture replay is not catalogue-idempotent",
        )

        signals = database.list_signals(fixture_mode="fixture_only", limit=500)
        require(len(signals) == 11, f"expected 11 fixture signals, found {len(signals)}")
        signal_ids = {str(row["id"]) for row in signals}
        require(len(signal_ids) == 11, "fixture signal identifiers are not unique")

        # The Hindi fixture deliberately uses the currently disabled PIB source
        # shape.  Replaying evidence fixtures must remain possible without
        # enabling outbound collection for that source.
        source_by_id = {str(row["source_id"]): row for row in sources}
        require(
            not bool(source_by_id["pib_bhubaneswar_hi"]["enabled"]),
            "disabled-source replay precondition changed",
        )
        require(
            any(row["source_id"] == "pib_bhubaneswar_hi" for row in signals),
            "disabled Hindi source shape was not replayed",
        )

        public_signals = database.list_signals(
            fixture_mode="fixture_only", public_only=True, limit=500
        )
        require(
            len(public_signals) == 9,
            f"expected 9 public-active fixture signals, found {len(public_signals)}",
        )
        require(
            all(row["processing_state"] == "active_direct" for row in public_signals),
            "public-only query crossed the processing-state release boundary",
        )

        fixture_tasks = [
            row
            for row in database.list_review_tasks(limit=500)
            if str(row["signal_id"]) in signal_ids
        ]
        require(
            len(fixture_tasks) == 9,
            f"expected 9 fixture review tasks, found {len(fixture_tasks)}",
        )
        require(
            len({str(row["signal_id"]) for row in fixture_tasks}) == 9,
            "review-task uniqueness per fixture signal failed",
        )

        fixture_catalogue = [
            row
            for row in database.list_catalogue_events(limit=500)
            if bool(row["event"].get("is_synthetic_fixture"))
        ]
        require(
            len(fixture_catalogue) == 1,
            f"expected 1 fixture catalogue event, found {len(fixture_catalogue)}",
        )
        require(
            bool(fixture_catalogue[0]["event"].get("positive_only_catalogue")),
            "catalogue row lost its positive-only warning",
        )

        # Use independent connections so this is a PostgreSQL concurrency smoke,
        # not merely repeated calls behind one repository lock.
        readers = [Database(database_url) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=len(readers)) as pool:
            snapshots = list(pool.map(fixture_signal_ids, readers))
        expected_ids = tuple(sorted(signal_ids))
        require(
            all(snapshot == expected_ids for snapshot in snapshots),
            "concurrent PostgreSQL readers observed inconsistent fixture sets",
        )

        report: dict[str, Any] = {
            "backend": database.backend,
            "catalogue_events": len(fixture_catalogue),
            "concurrent_readers": len(readers),
            "enabled_sources": enabled_count,
            "fixture_signals": len(signals),
            "public_active_signals": len(public_signals),
            "registered_sources": len(sources),
            "review_tasks": len(fixture_tasks),
            "status": "ok",
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        for reader in readers:
            reader.close()
        if database is not None:
            database.close()


if __name__ == "__main__":
    main()
