"""Live-crawl the whole registered surface and report what was really reached.

Every enabled route in `config/sources.yaml` is fetched from its live origin,
routed to a language, and run through the disease lexicon and district
gazetteer. Requests are parallel across distinct origin hosts and strictly
serialised, robots-checked and paced within each host.

This stops at the registered route itself, which is why it finishes in a couple
of minutes. For the full pipeline (detail-page discovery, PDF download and OCR)
run the durable collector instead:

    .venv/bin/python -c "from services.api.database import Database; \
        from services.api.collection_runtime import CollectionRuntime; \
        print(CollectionRuntime(Database('sqlite:///live.sqlite3')).tick())"

No operator contact is required: `workers.ingestion.safe_fetch.crawler_contact`
always yields a self-identifying User-Agent, and CRAWLER_CONTACT can add a
monitored mailbox on top of it.

    .venv/bin/python scripts/live_crawl_demo.py [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Running `python scripts/live_crawl_demo.py` puts scripts/ on sys.path[0] rather
# than the project root, and this project is intentionally non-installable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.ingestion.diseases import DiseaseLexicon  # noqa: E402
from workers.ingestion.geography import DistrictGazetteer  # noqa: E402
from workers.ingestion.language import route_unicode  # noqa: E402
from workers.ingestion.parse import parse_html  # noqa: E402
from workers.ingestion.registry import SourceSpec, load_registry  # noqa: E402
from workers.ingestion.robots import HostRateLimiter, RobotsPolicy, host_of  # noqa: E402
from workers.ingestion.safe_fetch import crawler_contact, fetch_url  # noqa: E402


@dataclass(slots=True)
class Row:
    source_id: str
    declared: str
    district_id: str | None
    state: str = "pending"
    byte_length: int = 0
    route: str = "-"
    diseases: tuple[str, ...] = ()
    districts: tuple[str, ...] = ()


class _Probe:
    def __init__(self) -> None:
        self.lexicon = DiseaseLexicon.load()
        self.gazetteer = DistrictGazetteer.load()
        self.robots = RobotsPolicy()
        self.limiter = HostRateLimiter()

    def run(self, source: SourceSpec) -> Row:
        row = Row(
            source_id=source.id,
            declared=",".join(source.languages),
            district_id=source.district_id,
        )
        verdict = self.robots.evaluate(source.url)
        if not verdict.allowed:
            row.state = "robots_disallowed"
            return row
        host = host_of(source.url)
        delay = (
            verdict.crawl_delay
            if verdict.crawl_delay is not None
            else float(source.minimum_interval_seconds)
        )
        self.limiter.acquire(host, delay)
        try:
            response = fetch_url(
                source.url,
                source_id=source.id,
                allowed_hosts=list(source.allowed_hosts),
            )
        except Exception as exc:  # noqa: BLE001 - a dead origin is a normal outcome
            row.state = f"unreachable:{getattr(exc, 'code', type(exc).__name__)}"
            return row
        finally:
            self.limiter.release(host)
        text = parse_html(response.body).text
        row.state = "ok"
        row.byte_length = response.receipt.byte_length
        row.route = str(route_unicode(text))
        row.diseases = tuple(sorted(self.lexicon.find(text)))
        row.districts = tuple(
            sorted(match.district_id for match in self.gazetteer.resolve(text))
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="probe only N routes")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    registry = load_registry()
    sources = [source for source in registry.sources if source.enabled]
    if arguments.limit > 0:
        sources = sources[: arguments.limit]
    by_host: dict[str, list[SourceSpec]] = defaultdict(list)
    for source in sources:
        by_host[host_of(source.url)].append(source)

    print(f"crawler contact: {crawler_contact()}")
    print(f"enabled routes: {len(sources)} across {len(by_host)} origin hosts\n")
    probe = _Probe()
    started = time.time()
    rows: list[Row] = []

    def run_host(bucket: list[SourceSpec]) -> list[Row]:
        return [probe.run(source) for source in bucket]

    workers = max(1, min(arguments.workers, len(by_host)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for bucket_rows in pool.map(run_host, list(by_host.values())):
            rows.extend(bucket_rows)

    rows.sort(key=lambda row: row.source_id)
    for row in rows:
        if row.state != "ok":
            print(f"{row.source_id:<38} {row.declared:<6} {row.state}")
            continue
        districts = [item.replace("OD-DIST-", "") for item in row.districts]
        print(
            f"{row.source_id:<38} {row.declared:<6} routed={row.route:<5} "
            f"{row.byte_length:>7}B diseases={list(row.diseases)} "
            f"districts={districts[:5]}"
        )

    reached = [row for row in rows if row.state == "ok"]
    languages = Counter(row.route for row in reached)
    diseases: set[str] = set()
    districts_seen: set[str] = set()
    for row in reached:
        diseases.update(row.diseases)
        districts_seen.update(row.districts)
        if row.district_id:
            districts_seen.add(row.district_id)
    print(
        f"\n{len(reached)}/{len(sources)} routes reached in {time.time() - started:.1f}s"
        f"\n  routed languages: {dict(sorted(languages.items()))}"
        f"\n  distinct diseases mentioned: {len(diseases)} {sorted(diseases)}"
        f"\n  distinct districts addressable: {len(districts_seen)}"
        "\nExtracted values are mention counts from published pages. They are not "
        "case counts, and they are not incidence."
    )
    return 0 if reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
