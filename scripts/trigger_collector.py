#!/usr/bin/env python3
"""Run durable collection jobs, either in-process or against a deployed API.

Two modes:

``--local sqlite:///live.sqlite3``
    Drive `CollectionRuntime` directly in this process. This is how a real
    collection run is reproduced on a workstation or in CI without a deployed
    service, and it prints the district/disease/language coverage it achieved.

``--api-base https://...``
    Wake a deployed free-tier web service and ask it to drain its own queue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import httpx

# `uv run python scripts/trigger_collector.py` executes with `scripts/` as
# sys.path[0] while this project is intentionally non-installable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.ingestion.registry import load_registry  # noqa: E402


def _coverage(database) -> dict[str, object]:  # noqa: ANN001 - services.api.database.Database
    rows = [
        row
        for row in database.list_signals(fixture_mode="live_only", limit=100000)
        if not row.get("is_fixture")
    ]
    districts = sorted({row["district_id"] for row in rows if row["district_id"]})
    diseases = sorted({row["disease"] for row in rows if row["disease"]})
    languages = Counter(str(row["language"]) for row in rows)
    return {
        "live_signals": len(rows),
        "districts_populated": len(districts),
        "districts": [item.replace("OD-DIST-", "") for item in districts],
        "diseases_populated": len(diseases),
        "diseases": diseases,
        "signals_by_language": dict(sorted(languages.items())),
        "note": (
            "counts are published-evidence mentions, never case counts or incidence"
        ),
    }


def run_local(database_url: str, rounds: int, jobs_per_tick: int, deadline: int) -> int:
    from services.api.collection_runtime import CollectionRuntime
    from services.api.database import Database

    database = Database(database_url)
    runtime = CollectionRuntime(database)
    stop_at = time.monotonic() + deadline
    reports: list[dict[str, object]] = []
    for round_number in range(rounds):
        if time.monotonic() > stop_at:
            break
        started = time.monotonic()
        result = runtime.tick(maximum_jobs=jobs_per_tick)
        processed = result.get("processed", [])
        completed = [item for item in processed if item.get("state") == "completed"]
        reports.append(
            {
                "round": round_number,
                "state": result["state"],
                "enqueued": result.get("enqueued", 0),
                "processed": len(processed),
                "completed": len(completed),
                "signals": sum(int(item.get("signal_count", 0)) for item in completed),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
        if result["state"] == "withheld" or not processed:
            break
    print(
        json.dumps(
            {"rounds": reports, "coverage": _coverage(database)},
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


def run_remote(args: argparse.Namespace, jobs_per_tick: int) -> int:
    token = os.getenv("COLLECTOR_API_TOKEN", "")
    parsed = urllib.parse.urlsplit(args.api_base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise SystemExit("PUBLIC_API_BASE_URL must be a credential-free HTTPS origin")
    if not token:
        raise SystemExit("COLLECTOR_API_TOKEN is required")
    base = args.api_base.rstrip("/")
    reports: list[dict[str, object]] = []
    if not 300 <= args.deadline_seconds <= 1500:
        raise SystemExit("--deadline-seconds must be between 300 and 1500")
    deadline = time.monotonic() + args.deadline_seconds
    enabled_sources = [source for source in load_registry().sources if source.enabled]
    # The API tick already paces every origin individually; a round is a batch
    # spread across many hosts, so the gap between rounds only has to cover the
    # politeness interval of a single host, not the sum of all of them.
    inter_round_seconds = max(
        (source.minimum_interval_seconds for source in enabled_sources), default=0
    )
    with httpx.Client(timeout=290, follow_redirects=False) as client:
        round_limit = max(1, min(args.rounds, 10))
        for round_number in range(round_limit):
            remaining = deadline - time.monotonic()
            if reports and remaining < 300:
                break
            response = client.post(
                f"{base}/api/v1/internal/collector/tick?maximum_jobs={jobs_per_tick}",
                headers={"X-Collector-Token": token, "Accept": "application/json"},
                timeout=min(290, max(1, remaining)),
            )
            response.raise_for_status()
            payload = response.json()["data"]
            reports.append(
                {
                    "state": payload["state"],
                    "enqueued": payload.get("enqueued", 0),
                    "processed": len(payload.get("processed", [])),
                    "reason_code": payload.get("reason_code"),
                }
            )
            if not payload.get("processed"):
                break
            if round_number + 1 < round_limit and inter_round_seconds:
                time.sleep(inter_round_seconds)
    print(json.dumps({"rounds": reports}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=os.getenv("PUBLIC_API_BASE_URL", ""))
    parser.add_argument(
        "--local",
        default="",
        help="Run in-process against this DATABASE_URL instead of a deployed API",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--deadline-seconds", type=int, default=1500)
    parser.add_argument(
        "--jobs-per-tick",
        type=int,
        default=int(os.getenv("COLLECTOR_JOBS_PER_TICK", "40")),
        help=(
            "Jobs drained per tick. One job per tick could never populate a "
            "hundred-plus routes across 30 districts."
        ),
    )
    args = parser.parse_args()
    jobs_per_tick = max(1, min(args.jobs_per_tick, 200))
    if args.local:
        return run_local(
            args.local,
            max(1, args.rounds),
            jobs_per_tick,
            max(60, args.deadline_seconds),
        )
    return run_remote(args, jobs_per_tick)


if __name__ == "__main__":
    raise SystemExit(main())
