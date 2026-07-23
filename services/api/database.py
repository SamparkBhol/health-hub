"""Durable repository for the public competition profile.

SQLite provides a zero-configuration local/offline store. The hosted profile
uses the same schema and invariants on PostgreSQL/Supabase: transactional job
leases, monotonically increasing fencing tokens, idempotent mutations and
append-only review decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from packages.contracts.api import (
    LIVE_EVIDENCE_PLACEHOLDER,
    LIVE_EVIDENCE_REDACTION_STATE,
    RedactedSignalInput,
    SourceReceiptInput,
)
from workers.ingestion.urls import canonicalize_discovered_url


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class RepositoryConflict(Exception):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


class RepositoryNotFound(Exception):
    pass


def verified_postgres_dsn(database_url: str) -> str:
    """Normalise PostgreSQL URLs and require hostname-verified TLS off-host."""

    normalised = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    normalised = normalised.replace("postgres://", "postgresql://", 1)
    parsed = urlsplit(normalised)
    if parsed.scheme != "postgresql" or not parsed.hostname:
        raise ValueError("invalid PostgreSQL DATABASE_URL")
    if parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}:
        return normalised
    parameters = parse_qs(parsed.query, keep_blank_values=True)
    if parameters.get("sslmode") != ["verify-full"]:
        raise ValueError("remote PostgreSQL DATABASE_URL must set sslmode=verify-full")
    if parameters.get("sslrootcert") != ["system"]:
        raise ValueError("remote PostgreSQL DATABASE_URL must set sslrootcert=system")
    return normalised


def load_registered_sources() -> tuple[dict[str, Any], ...]:
    """Load the checked-in registry, which is JSON-compatible YAML by design."""

    registry_path = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load source registry {registry_path}: {exc}") from exc
    sources = []
    for item in document.get("sources", []):
        sources.append(
            {
                "source_id": item["id"],
                "name": item["name"],
                "canonical_url": item["url"],
                "language": ",".join(item["languages"]),
                "content_type": item["kind"],
                "policy_state": "; ".join(
                    str(value)
                    for value in (
                        item["rights_state"],
                        item.get("availability_state"),
                        item.get("transport_state"),
                    )
                    if value
                ),
                "enabled": int(bool(item.get("enabled", False))),
            }
        )
    if not sources:
        raise RuntimeError("source registry contains no sources")
    return tuple(sources)


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_registry (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    language TEXT NOT NULL,
    content_type TEXT NOT NULL,
    policy_state TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_success_at TEXT,
    last_error_code TEXT
);

CREATE TABLE IF NOT EXISTS job (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    payload_ref TEXT,
    payload_hash TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at TEXT,
    fencing_token INTEGER NOT NULL DEFAULT 0,
    row_version INTEGER NOT NULL DEFAULT 0,
    completion_idempotency_key TEXT UNIQUE,
    source_snapshot_id TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_attempt (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES job(id),
    attempt INTEGER NOT NULL,
    owner TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    outcome TEXT,
    error_code TEXT
);

CREATE TABLE IF NOT EXISTS source_snapshot (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    requested_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    byte_length INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    access_path TEXT NOT NULL,
    archive_timestamp TEXT,
    archive_digest TEXT,
    fallback_reason TEXT,
    is_fixture INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovered_link (
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    url TEXT NOT NULL,
    url_sha256 TEXT NOT NULL,
    label_sha256 TEXT NOT NULL,
    content_hint TEXT NOT NULL,
    priority_rank INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    queue_mode TEXT,
    job_id TEXT REFERENCES job(id),
    source_snapshot_id TEXT REFERENCES source_snapshot(id),
    observed_content_sha256 TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    fetched_at TEXT,
    last_error_code TEXT,
    row_version INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(source_id, url),
    CHECK(state IN ('pending','queued','pending_approval','fetched','failed')),
    CHECK(queue_mode IS NULL OR queue_mode IN ('discovery','approved'))
);

CREATE TABLE IF NOT EXISTS signal (
    id TEXT PRIMARY KEY,
    content_key TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL REFERENCES source_registry(source_id),
    source_snapshot_id TEXT NOT NULL,
    district_id TEXT,
    disease TEXT,
    assertion TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_start INTEGER NOT NULL,
    evidence_end INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    event_review_eligible INTEGER NOT NULL DEFAULT 0,
    processing_state TEXT NOT NULL DEFAULT 'privacy_review_required',
    redaction_state TEXT NOT NULL DEFAULT 'content_not_retained_unvalidated_pii',
    language TEXT NOT NULL DEFAULT 'und',
    extractor_version TEXT NOT NULL DEFAULT 'rules-v1',
    created_at TEXT NOT NULL,
    CHECK(assertion IN ('affirmed','not_affirmed','speculative','non_current'))
);

CREATE TABLE IF NOT EXISTS review_task (
    id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL UNIQUE REFERENCES signal(id),
    task_kind TEXT NOT NULL DEFAULT 'quality_hold',
    state TEXT NOT NULL,
    claimed_by TEXT,
    claim_expires_at TEXT,
    row_version INTEGER NOT NULL DEFAULT 0,
    current_decision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(state IN ('open','claimed','decided')),
    CHECK(task_kind IN ('event_verification','quality_hold'))
);

CREATE TABLE IF NOT EXISTS review_transition (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES review_task(id),
    from_state TEXT,
    to_state TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    row_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_decision (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES review_task(id),
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    supersedes_id TEXT REFERENCES review_decision(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    event_json TEXT,
    created_at TEXT NOT NULL,
    CHECK(decision IN ('verified','rejected','needs_more_information','duplicate'))
);

CREATE TABLE IF NOT EXISTS verified_event (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES review_task(id),
    decision_id TEXT NOT NULL UNIQUE REFERENCES review_decision(id),
    district_id TEXT,
    disease TEXT,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS official_catalogue_event (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_snapshot_id TEXT,
    district_id TEXT,
    disease TEXT,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mutation_idempotency (
    scope TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS audit_event (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_eligible ON job(state, available_at, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_review_state ON review_task(state, created_at);
CREATE INDEX IF NOT EXISTS idx_signal_district ON signal(district_id, created_at);
CREATE INDEX IF NOT EXISTS idx_snapshot_source ON source_snapshot(source_id, retrieved_at);
CREATE INDEX IF NOT EXISTS idx_discovered_pending
    ON discovered_link(source_id, state, priority_rank, first_seen_at);
"""


def postgres_schema() -> str:
    """Translate the one SQLite-only identity declaration for PostgreSQL."""

    return SCHEMA.replace("PRAGMA foreign_keys = ON;", "").replace(
        "sequence INTEGER PRIMARY KEY AUTOINCREMENT", "sequence BIGSERIAL PRIMARY KEY"
    )


class PostgresConnectionAdapter:
    """Expose the repository's small qmark-style interface over psycopg."""

    def __init__(self, connection: Any) -> None:
        self.raw = connection

    def execute(self, query: str, parameters: Any = ()) -> Any:
        return self.raw.execute(query.replace("?", "%s"), parameters)

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.raw.execute(statement)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()


class Database:
    """Thread-safe repository over SQLite or PostgreSQL."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._lock = threading.RLock()
        self._raw_connection: Any
        self._connection: Any
        self._psycopg: Any = None
        self._postgres_dsn: str | None = None
        self._postgres_row_factory: Any = None
        if database_url.startswith(("postgresql://", "postgres://", "postgresql+psycopg://")):
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:  # pragma: no cover - exercised by deployment packaging
                raise RuntimeError("PostgreSQL DATABASE_URL requires psycopg[binary]") from exc
            normalised = verified_postgres_dsn(database_url)
            self.backend = "postgresql"
            self.path = None
            self._psycopg = psycopg
            self._postgres_dsn = normalised
            self._postgres_row_factory = dict_row
            self._replace_postgres_connection()
            with self._lock:
                self._connection.executescript(postgres_schema())
                self._migrate_schema()
                self._seed_sources()
        else:
            self.backend = "sqlite"
            self.path = self._sqlite_path(database_url)
            if self.path != ":memory:":
                Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._raw_connection = sqlite3.connect(
                self.path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._raw_connection.row_factory = sqlite3.Row
            self._connection = self._raw_connection
            with self._lock:
                if self.path != ":memory:":
                    self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.executescript(SCHEMA)
                self._migrate_schema()
                self._seed_sources()

    def _replace_postgres_connection(self) -> None:
        """Open a fresh PostgreSQL connection and atomically replace the adapter."""

        if self._psycopg is None or self._postgres_dsn is None:
            raise RuntimeError("PostgreSQL connection metadata is unavailable")
        raw = self._psycopg.connect(
            self._postgres_dsn,
            autocommit=True,
            row_factory=self._postgres_row_factory,
        )
        self._raw_connection = raw
        self._connection = PostgresConnectionAdapter(raw)

    def _migrate_schema(self) -> None:
        """Apply the tiny forward-only compatibility migration used by this profile."""

        def column_exists(table: str, column: str) -> bool:
            if self.backend == "postgresql":
                return (
                    self._connection.execute(
                        """
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema=current_schema() AND table_name=?
                          AND column_name=?
                        """,
                        (table, column),
                    ).fetchone()
                    is not None
                )
            return any(
                row["name"] == column
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            )

        for table, column in (
            ("job", "source_snapshot_id"),
            ("official_catalogue_event", "source_snapshot_id"),
            ("signal", "event_review_eligible"),
            ("signal", "processing_state"),
            ("signal", "redaction_state"),
            ("signal", "language"),
            ("signal", "extractor_version"),
            ("review_task", "task_kind"),
            ("discovered_link", "queue_mode"),
            ("discovered_link", "observed_content_sha256"),
        ):
            if not column_exists(table, column):
                declarations = {
                    ("signal", "event_review_eligible"): "INTEGER NOT NULL DEFAULT 0",
                    ("signal", "processing_state"): (
                        "TEXT NOT NULL DEFAULT 'privacy_review_required'"
                    ),
                    ("signal", "redaction_state"): (
                        "TEXT NOT NULL DEFAULT 'content_not_retained_unvalidated_pii'"
                    ),
                    ("review_task", "task_kind"): (
                        "TEXT NOT NULL DEFAULT 'quality_hold'"
                    ),
                    ("signal", "language"): "TEXT NOT NULL DEFAULT 'und'",
                    ("signal", "extractor_version"): (
                        "TEXT NOT NULL DEFAULT 'rules-v1'"
                    ),
                    ("discovered_link", "queue_mode"): "TEXT",
                    ("discovered_link", "observed_content_sha256"): "TEXT",
                }
                declaration = declarations.get((table, column), "TEXT")
                self._connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )
        self._migrate_global_discovered_url_ownership()
        self._migrate_cross_route_signal_contamination()
        self._migrate_live_evidence_to_non_retention()

    def _migrate_cross_route_signal_contamination(self) -> None:
        """Quarantine legacy route-inherited signals made from identical bytes.

        Before sentence-scoped linking, the same navigation/category document
        could be fetched through several district routes and inherit a different
        district from each route. Identical non-fixture bytes cannot support
        mutually different district facts, so every member of such a group is
        conservatively moved off public maps and into quality review.
        """

        with self.transaction() as connection:
            groups = connection.execute(
                """
                SELECT p.content_sha256
                FROM signal s
                JOIN source_snapshot p ON p.id=s.source_snapshot_id
                WHERE COALESCE(p.is_fixture, 0)=0
                  AND p.content_sha256 IS NOT NULL
                  AND s.district_id IS NOT NULL
                GROUP BY p.content_sha256
                HAVING COUNT(DISTINCT s.source_id)>1
                   AND COUNT(DISTINCT s.district_id)>1
                """
            ).fetchall()
            for group in groups:
                digest = group["content_sha256"]
                connection.execute(
                    """
                    UPDATE signal
                    SET district_id=NULL, event_review_eligible=0,
                        processing_state='ambiguous_entity_linkage'
                    WHERE source_snapshot_id IN (
                      SELECT id FROM source_snapshot
                      WHERE content_sha256=? AND COALESCE(is_fixture, 0)=0
                    )
                    """,
                    (digest,),
                )
                connection.execute(
                    """
                    UPDATE review_task SET task_kind='quality_hold'
                    WHERE signal_id IN (
                      SELECT s.id FROM signal s
                      JOIN source_snapshot p ON p.id=s.source_snapshot_id
                      WHERE p.content_sha256=? AND COALESCE(p.is_fixture, 0)=0
                    )
                    """,
                    (digest,),
                )

    def _migrate_global_discovered_url_ownership(self) -> None:
        """Give each canonical detail URL one durable owner across all routes.

        District sections of the same publisher routinely expose the same site
        navigation links.  Keeping `(source_id, url)` as the effective identity
        fetched and counted those pages once per district. Existing exact URL
        duplicates are collapsed before the unique index is installed.
        """

        state_priority = {
            "fetched": 0,
            "pending_approval": 1,
            "queued": 2,
            "pending": 3,
            "failed": 4,
        }
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT source_id,url,state,first_seen_at FROM discovered_link
                ORDER BY first_seen_at,source_id,url
                """
            ).fetchall()
            groups: dict[str, list[Any]] = {}
            for row in rows:
                groups.setdefault(canonicalize_discovered_url(row["url"]), []).append(row)
            for canonical_url, candidates in groups.items():
                candidates.sort(
                    key=lambda item: (
                        state_priority.get(str(item["state"]), 9),
                        str(item["first_seen_at"]),
                        str(item["source_id"]),
                    )
                )
                owner = candidates[0]
                for loser in candidates[1:]:
                    payload_ref = f"registered-link:{loser['url']}"
                    connection.execute(
                        """
                        UPDATE job SET state='dead', lease_owner=NULL,
                          lease_expires_at=NULL,last_error_code='DUPLICATE_URL_OWNER',
                          row_version=row_version+1,updated_at=?
                        WHERE source_id=? AND payload_ref=?
                          AND state NOT IN ('completed','dead')
                        """,
                        (timestamp(), loser["source_id"], payload_ref),
                    )
                    connection.execute(
                        "DELETE FROM discovered_link WHERE source_id=? AND url=?",
                        (loser["source_id"], loser["url"]),
                    )
                if owner["url"] != canonical_url:
                    # A queued job still names the pre-canonical URL. Retire it
                    # and let the normal reservation path create one job for the
                    # canonical owner on the next tick.
                    connection.execute(
                        """
                        UPDATE job SET state='dead',lease_owner=NULL,
                          lease_expires_at=NULL,last_error_code='URL_CANONICALIZED',
                          row_version=row_version+1,updated_at=?
                        WHERE source_id=? AND payload_ref=?
                          AND state NOT IN ('completed','dead')
                        """,
                        (
                            timestamp(),
                            owner["source_id"],
                            f"registered-link:{owner['url']}",
                        ),
                    )
                    reset_queued = owner["state"] == "queued"
                    connection.execute(
                        """
                        UPDATE discovered_link
                        SET url=?,url_sha256=?,
                          state=CASE WHEN ? THEN 'pending' ELSE state END,
                          queue_mode=CASE WHEN ? THEN NULL ELSE queue_mode END,
                          job_id=CASE WHEN ? THEN NULL ELSE job_id END,
                          row_version=row_version+1
                        WHERE source_id=? AND url=?
                        """,
                        (
                            canonical_url,
                            hashlib.sha256(canonical_url.encode()).hexdigest(),
                            reset_queued,
                            reset_queued,
                            reset_queued,
                            owner["source_id"],
                            owner["url"],
                        ),
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_discovered_url_global
                ON discovered_link(url_sha256)
                """
            )

    def _migrate_live_evidence_to_non_retention(self) -> None:
        """Erase legacy live spans and all evidence-derived persistence hashes.

        Synthetic fixture snapshots remain inspectable.  Every other row is
        migrated transactionally so an interrupted startup cannot leave a
        partially protected database.
        """

        placeholder_hash = hashlib.sha256(
            LIVE_EVIDENCE_PLACEHOLDER.encode("utf-8")
        ).hexdigest()
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.content_key, s.source_id, s.source_snapshot_id,
                       s.district_id, s.disease, s.assertion, s.processing_state,
                       s.language, s.extractor_version, s.evidence_text,
                       s.evidence_start, s.evidence_end, s.content_sha256,
                       s.redaction_state,
                       p.content_sha256 AS document_sha256,
                       p.final_url AS canonical_url
                FROM signal s
                LEFT JOIN source_snapshot p ON p.id=s.source_snapshot_id
                WHERE COALESCE(p.is_fixture, 0)=0
                ORDER BY s.id
                """
            ).fetchall()
            legacy_rows = []
            for row in rows:
                target_item = dict(row)
                target_item["content_sha256"] = placeholder_hash
                already_protected = (
                    row["evidence_text"] == LIVE_EVIDENCE_PLACEHOLDER
                    and row["evidence_start"] == 0
                    and row["evidence_end"] == len(LIVE_EVIDENCE_PLACEHOLDER)
                    and row["content_sha256"] == placeholder_hash
                    and row["redaction_state"] == LIVE_EVIDENCE_REDACTION_STATE
                    and row["content_key"] == self._signal_content_key(target_item)
                )
                if not already_protected:
                    legacy_rows.append(row)
            if not legacy_rows:
                return

            # Content keys formerly included the source-span digest. Move all
            # live keys out of the unique namespace first, then rebuild them
            # from the fixed placeholder and structured metadata only.
            for row in legacy_rows:
                connection.execute(
                    "UPDATE signal SET content_key=? WHERE id=?",
                    (f"privacy-migration-{uuid.uuid4().hex}", row["id"]),
                )
            reserved = {
                str(row["content_key"])
                for row in connection.execute(
                    "SELECT content_key FROM signal"
                ).fetchall()
            }
            for row in legacy_rows:
                item = dict(row)
                item["content_sha256"] = placeholder_hash
                content_key = self._signal_content_key(item)
                duplicate = 1
                while content_key in reserved:
                    content_key = hashlib.sha256(
                        f"{self._signal_content_key(item)}\x1fduplicate:{duplicate}".encode()
                    ).hexdigest()
                    duplicate += 1
                reserved.add(content_key)
                connection.execute(
                    """
                    UPDATE signal
                    SET content_key=?, evidence_text=?, evidence_start=0,
                        evidence_end=?, content_sha256=?, redaction_state=?
                    WHERE id=?
                    """,
                    (
                        content_key,
                        LIVE_EVIDENCE_PLACEHOLDER,
                        len(LIVE_EVIDENCE_PLACEHOLDER),
                        placeholder_hash,
                        LIVE_EVIDENCE_REDACTION_STATE,
                        row["id"],
                    ),
                )

    @staticmethod
    def _sqlite_path(database_url: str) -> str:
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            return ":memory:"
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            scheme = database_url.split(":", 1)[0]
            raise ValueError(
                f"DATABASE_URL scheme {scheme!r} is unsupported; use sqlite:///... or "
                "postgresql[+psycopg]://..."
            )
        path = database_url[len(prefix) :]
        return path or ":memory:"

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with self._lock:
            if self.backend == "postgresql":
                with self._raw_connection.transaction():
                    yield self._connection
            else:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    yield self._connection
                except Exception:
                    self._connection.rollback()
                    raise
                else:
                    self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._raw_connection.close()

    def _seed_sources(self) -> None:
        configured = list(load_registered_sources())
        with self.transaction() as connection:
            # Preserve historical foreign keys but retire routes removed from the
            # checked-in registry. The following upserts reactivate every current
            # route in the same transaction.
            connection.execute(
                """
                UPDATE source_registry
                SET enabled=0, policy_state='retired_not_in_current_registry'
                """
            )
            for source in configured:
                connection.execute(
                    """
                    INSERT INTO source_registry(
                      source_id,name,canonical_url,language,content_type,policy_state
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(source_id) DO UPDATE SET
                      name=excluded.name, canonical_url=excluded.canonical_url,
                      language=excluded.language, content_type=excluded.content_type,
                      policy_state=excluded.policy_state, enabled=excluded.enabled
                    """,
                    (
                        source["source_id"],
                        source["name"],
                        source["canonical_url"],
                        source["language"],
                        source["content_type"],
                        source["policy_state"],
                    ),
                )
                connection.execute(
                    "UPDATE source_registry SET enabled=? WHERE source_id=?",
                    (source["enabled"], source["source_id"]),
                )

    def ready(self) -> bool:
        with self._lock:
            try:
                return self._connection.execute("SELECT 1 AS ready").fetchone()["ready"] == 1
            except Exception:  # noqa: BLE001 - readiness must not leak driver errors
                if self.backend != "postgresql":
                    return False
                try:
                    self._raw_connection.close()
                except Exception:  # noqa: BLE001, S110 - broken transports may not close
                    pass
                try:
                    self._replace_postgres_connection()
                    return (
                        self._connection.execute("SELECT 1 AS ready").fetchone()["ready"]
                        == 1
                    )
                except Exception:  # noqa: BLE001 - typed unready is the public contract
                    return False

    def list_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM source_registry
                WHERE policy_state != 'retired_not_in_current_registry'
                ORDER BY source_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_source_collection(
        self, source_id: str, *, succeeded: bool, error_code: str | None = None
    ) -> None:
        now = timestamp()
        with self.transaction() as connection:
            if succeeded:
                updated = connection.execute(
                    """
                    UPDATE source_registry
                    SET last_success_at=?, last_error_code=NULL WHERE source_id=?
                    """,
                    (now, source_id),
                )
            else:
                updated = connection.execute(
                    "UPDATE source_registry SET last_error_code=? WHERE source_id=?",
                    (error_code or "collection_failed", source_id),
                )
            if updated.rowcount != 1:
                raise RepositoryNotFound(source_id)

    def register_discovered_links(
        self,
        *,
        source_id: str,
        links: list[dict[str, Any]],
    ) -> int:
        """Persist every eligible link without resetting its progress state."""

        now = timestamp()
        inserted = 0
        with self.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM source_registry WHERE source_id=? AND enabled=1",
                    (source_id,),
                ).fetchone()
                is None
            ):
                raise RepositoryNotFound(f"registered source {source_id!r} not found")
            for priority_rank, link in enumerate(links):
                url = canonicalize_discovered_url(str(link["url"]))
                url_sha256 = hashlib.sha256(url.encode()).hexdigest()
                existing = connection.execute(
                    "SELECT source_id,url FROM discovered_link WHERE url_sha256=?",
                    (url_sha256,),
                ).fetchone()
                if existing is not None:
                    if existing["url"] != url:
                        raise RepositoryConflict(
                            "DISCOVERED_URL_HASH_COLLISION",
                            "two canonical URLs produced the same digest",
                        )
                    connection.execute(
                        """
                        UPDATE discovered_link SET last_seen_at=?,row_version=row_version+1
                        WHERE url_sha256=?
                        """,
                        (now, url_sha256),
                    )
                    continue
                inserted_row = connection.execute(
                    """
                    INSERT INTO discovered_link(
                      source_id,url,url_sha256,label_sha256,content_hint,priority_rank,
                      state,first_seen_at,last_seen_at
                    ) VALUES(?,?,?,?,?,?,'pending',?,?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        source_id,
                        url,
                        url_sha256,
                        hashlib.sha256(str(link.get("label", "")).encode()).hexdigest(),
                        str(link.get("content_hint", "text/html")),
                        priority_rank,
                        now,
                        now,
                    ),
                )
                inserted += int(inserted_row.rowcount == 1)
        return inserted

    def reserve_discovered_links(
        self,
        *,
        source_id: str,
        limit: int,
        approved_content_sha256s: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        """Atomically reserve the next pending links for durable job creation."""

        if limit < 1:
            return []
        reserved: list[dict[str, Any]] = []
        with self.transaction() as connection:
            query = """
                SELECT * FROM discovered_link
                WHERE source_id=? AND (
                  state='pending'
            """
            parameters: list[Any] = [source_id]
            if approved_content_sha256s:
                placeholders = ",".join("?" for _ in approved_content_sha256s)
                query += (
                    " OR (state='pending_approval' "
                    f"AND observed_content_sha256 IN ({placeholders}))"
                )
                parameters.extend(sorted(approved_content_sha256s))
            query += ") ORDER BY priority_rank, first_seen_at, url LIMIT ?"
            parameters.append(limit)
            rows = connection.execute(query, parameters).fetchall()
            for row in rows:
                queue_mode = (
                    "approved" if row["state"] == "pending_approval" else "discovery"
                )
                updated = connection.execute(
                    """
                    UPDATE discovered_link
                    SET state='queued', queue_mode=?, row_version=row_version+1
                    WHERE source_id=? AND url=? AND state=? AND row_version=?
                    """,
                    (
                        queue_mode,
                        source_id,
                        row["url"],
                        row["state"],
                        row["row_version"],
                    ),
                )
                if updated.rowcount == 1:
                    reserved.append(
                        dict(
                            connection.execute(
                                "SELECT * FROM discovered_link WHERE source_id=? AND url=?",
                                (source_id, row["url"]),
                            ).fetchone()
                        )
                    )
        return reserved

    def enqueue_reserved_discovered_link(
        self, *, source_id: str, url: str
    ) -> tuple[dict[str, Any], bool]:
        """Create and bind a fetch job in the same transaction as queue state."""

        now = timestamp()
        with self.transaction() as connection:
            link = connection.execute(
                """
                SELECT * FROM discovered_link
                WHERE source_id=? AND url=? AND state='queued'
                """,
                (source_id, url),
            ).fetchone()
            if link is None:
                raise RepositoryConflict(
                    "DISCOVERED_LINK_NOT_RESERVED", "discovered link is not reserved"
                )
            queue_mode = str(link["queue_mode"] or "discovery")
            approval_suffix = (
                f":{link['observed_content_sha256']}" if queue_mode == "approved" else ""
            )
            idempotency_key = (
                f"discovered-link:{source_id}:{link['url_sha256']}:{queue_mode}"
                f"{approval_suffix}"
            )
            existing = connection.execute(
                "SELECT * FROM job WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["source_id"] != source_id or existing["payload_hash"] != link[
                    "url_sha256"
                ]:
                    raise RepositoryConflict(
                        "IDEMPOTENCY_KEY_REUSED",
                        "discovered-link key belongs to different work",
                    )
                discovered_state = {
                    "completed": "fetched",
                    "dead": "failed",
                }.get(existing["state"], "queued")
                connection.execute(
                    """
                    UPDATE discovered_link
                    SET state=?, queue_mode=NULL, job_id=?, source_snapshot_id=?,
                      fetched_at=CASE
                        WHEN ?='fetched' THEN COALESCE(fetched_at, ?)
                        ELSE fetched_at
                      END,
                      last_error_code=?, row_version=row_version+1
                    WHERE source_id=? AND url=?
                    """,
                    (
                        discovered_state,
                        existing["id"],
                        existing["source_snapshot_id"],
                        discovered_state,
                        now,
                        existing["last_error_code"],
                        source_id,
                        url,
                    ),
                )
                return dict(existing), True
            job_id = f"job_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO job(
                  id,kind,source_id,payload_ref,payload_hash,idempotency_key,state,
                  available_at,created_at,updated_at
                ) VALUES(?,'fetch',?,?,?,?,'queued',?,?,?)
                """,
                (
                    job_id,
                    source_id,
                    f"registered-link:{url}",
                    link["url_sha256"],
                    idempotency_key,
                    now,
                    now,
                    now,
                ),
            )
            updated = connection.execute(
                """
                UPDATE discovered_link SET job_id=?, row_version=row_version+1
                WHERE source_id=? AND url=? AND state='queued'
                """,
                (job_id, source_id, url),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict(
                    "DISCOVERED_LINK_NOT_RESERVED", "discovered link is not reserved"
                )
            self._audit(
                connection,
                "job_enqueued",
                "job",
                job_id,
                {"source_id": source_id, "discovered_url_sha256": link["url_sha256"]},
            )
            job = connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
            return dict(job), False

    def release_discovered_link(self, *, source_id: str, url: str) -> None:
        """Release a reservation when no durable job could be created."""

        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE discovered_link
                SET state=CASE
                      WHEN queue_mode='approved' THEN 'pending_approval'
                      ELSE 'pending'
                    END,
                    queue_mode=NULL, job_id=NULL, row_version=row_version+1
                WHERE source_id=? AND url=? AND state='queued' AND job_id IS NULL
                """,
                (source_id, url),
            )

    def list_discovered_links(self, source_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM discovered_link"
        parameters: tuple[Any, ...] = ()
        if source_id is not None:
            query += " WHERE source_id=?"
            parameters = (source_id,)
        query += " ORDER BY source_id, priority_rank, first_seen_at, url"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_pending_pdf_approvals(
        self, *, include_inspection_url: bool = False
    ) -> list[dict[str, Any]]:
        """Return the non-content fields needed for explicit PDF promotion.

        The original URL and anchor label are intentionally absent.  They may
        contain personal data and are not required to approve a fetched byte
        sequence by its observed digest.
        """

        selected_url = ", url AS inspection_url" if include_inspection_url else ""
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT source_id, url_sha256, observed_content_sha256,
                       source_snapshot_id, first_seen_at, last_seen_at, state
                       {selected_url}
                FROM discovered_link
                WHERE state='pending_approval'
                  AND content_hint='application/pdf'
                  AND observed_content_sha256 IS NOT NULL
                ORDER BY source_id, first_seen_at, url_sha256
                """  # noqa: S608 - selected fragment is a fixed server-side column
            ).fetchall()
        return [dict(row) for row in rows]

    def record_sensitive_operator_read(
        self,
        *,
        operator_id: str,
        resource: str,
        item_count: int,
    ) -> None:
        with self.transaction() as connection:
            self._audit(
                connection,
                "sensitive_operator_read",
                "operator_view",
                hashlib.sha256(resource.encode()).hexdigest()[:24],
                {
                    "operator_id": operator_id,
                    "resource": resource,
                    "item_count": item_count,
                },
            )

    def list_signals(
        self,
        *,
        district_id: str | None = None,
        disease: str | None = None,
        language: str | None = None,
        assertion: str | None = None,
        retrieved_from: str | None = None,
        retrieved_to: str | None = None,
        fixture_mode: str = "all",
        public_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if fixture_mode not in {"all", "live_only", "fixture_only"}:
            raise ValueError("fixture_mode must be all, live_only or fixture_only")
        query = """
            SELECT s.*, t.state AS review_task_state, d.decision AS review_decision,
                   p.requested_url, p.final_url, p.access_path,
                   p.archive_timestamp, p.is_fixture,
                   p.content_sha256 AS snapshot_content_sha256,
                   r.canonical_url AS registered_source_url
            FROM signal s
            LEFT JOIN source_snapshot p ON p.id=s.source_snapshot_id
            LEFT JOIN source_registry r ON r.source_id=s.source_id
            LEFT JOIN review_task t ON t.signal_id=s.id
            LEFT JOIN review_decision d ON d.id=t.current_decision_id
        """
        parameters: list[Any] = []
        conditions: list[str] = []
        # Signals on a privacy, language or entity-linkage hold are internal
        # review material.  Public callers must opt into this database-level
        # release boundary instead of attempting to filter after retrieval.
        if public_only:
            conditions.append("s.processing_state = 'active_direct'")
            conditions.append(
                "COALESCE(d.decision, '') NOT IN ('rejected', 'duplicate')"
            )
        if district_id:
            conditions.append("s.district_id = ?")
            parameters.append(district_id)
        if disease:
            conditions.append("s.disease = ?")
            parameters.append(disease)
        if language:
            conditions.append("s.language = ?")
            parameters.append(language)
        if assertion:
            conditions.append("s.assertion = ?")
            parameters.append(assertion)
        if retrieved_from:
            conditions.append("s.retrieved_at >= ?")
            parameters.append(retrieved_from)
        if retrieved_to:
            conditions.append("s.retrieved_at <= ?")
            parameters.append(retrieved_to)
        if fixture_mode == "live_only":
            conditions.append("COALESCE(p.is_fixture, 0) = 0")
        elif fixture_mode == "fixture_only":
            conditions.append("p.is_fixture = 1")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY s.created_at DESC, s.id DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def has_live_signals(self) -> bool:
        """Return true only after a public-eligible non-fixture signal was persisted."""

        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1
                FROM signal s
                JOIN source_snapshot p ON p.id=s.source_snapshot_id
                LEFT JOIN review_task t ON t.signal_id=s.id
                LEFT JOIN review_decision d ON d.id=t.current_decision_id
                WHERE COALESCE(p.is_fixture, 0)=0
                  AND s.processing_state='active_direct'
                  AND COALESCE(d.decision, '') NOT IN ('rejected', 'duplicate')
                LIMIT 1
                """
            ).fetchone()
        return row is not None

    def job_backlog_counts(self) -> dict[str, int]:
        """Return non-sensitive durable queue depth for operations/readiness views."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT state, COUNT(*) AS count
                FROM job
                GROUP BY state
                """
            ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        for state in ("queued", "running", "retry_wait", "completed", "dead"):
            counts.setdefault(state, 0)
        counts["actionable"] = (
            counts["queued"] + counts["running"] + counts["retry_wait"]
        )
        return counts

    def aggregate_signal_counts(
        self,
        *,
        disease: str | None = None,
        language: str | None = None,
        assertion: str | None = None,
        retrieved_from: str | None = None,
        retrieved_to: str | None = None,
        fixture_mode: str = "all",
    ) -> list[dict[str, Any]]:
        if fixture_mode not in {"all", "live_only", "fixture_only"}:
            raise ValueError("fixture_mode must be all, live_only or fixture_only")
        conditions = [
            "s.district_id IS NOT NULL",
            "s.processing_state = 'active_direct'",
            "COALESCE(d.decision, '') NOT IN ('rejected', 'duplicate')",
        ]
        parameters: list[Any] = []
        for column, value in (
            ("s.disease", disease),
            ("s.language", language),
            ("s.assertion", assertion),
        ):
            if value:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        if retrieved_from:
            conditions.append("s.retrieved_at >= ?")
            parameters.append(retrieved_from)
        if retrieved_to:
            conditions.append("s.retrieved_at <= ?")
            parameters.append(retrieved_to)
        if fixture_mode == "live_only":
            conditions.append("COALESCE(p.is_fixture, 0) = 0")
        elif fixture_mode == "fixture_only":
            conditions.append("p.is_fixture = 1")
        query = f"""
            SELECT s.district_id, COUNT(*) AS published_signal_count,
                   MIN(s.retrieved_at) AS first_retrieved_at,
                   MAX(s.retrieved_at) AS last_retrieved_at
            FROM signal s
            LEFT JOIN source_snapshot p ON p.id=s.source_snapshot_id
            LEFT JOIN review_task t ON t.signal_id=s.id
            LEFT JOIN review_decision d ON d.id=t.current_decision_id
            WHERE {' AND '.join(conditions)}
            GROUP BY s.district_id ORDER BY s.district_id
        """  # noqa: S608 - all fragments come from fixed server-side fields
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_review_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT r.*, s.source_id, s.source_snapshot_id, s.district_id,
                       s.disease, s.assertion, s.evidence_text, s.evidence_start,
                       s.evidence_end, s.content_sha256, s.retrieved_at,
                       s.processing_state, s.redaction_state, s.language,
                       s.extractor_version, p.requested_url AS inspection_url,
                       p.final_url AS final_inspection_url,
                       p.content_sha256 AS snapshot_content_sha256,
                       p.access_path, p.archive_timestamp, p.archive_digest,
                       registry.canonical_url AS registered_source_url
                FROM review_task r
                JOIN signal s ON s.id = r.signal_id
                LEFT JOIN source_snapshot p ON p.id = s.source_snapshot_id
                LEFT JOIN source_registry registry ON registry.source_id=s.source_id
                ORDER BY r.created_at DESC, r.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_verified_events(
        self,
        limit: int = 100,
        *,
        district_id: str | None = None,
        disease: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["t.current_decision_id = v.decision_id"]
        parameters: list[Any] = []
        if district_id:
            conditions.append("v.district_id = ?")
            parameters.append(district_id)
        if disease:
            conditions.append("v.disease = ?")
            parameters.append(disease)
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT v.* FROM verified_event v
                JOIN review_task t ON t.id = v.task_id
                WHERE {' AND '.join(conditions)}
                ORDER BY v.created_at DESC, v.id DESC LIMIT ?
                """,  # noqa: S608 - conditions contain fixed column expressions only
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["event"] = json.loads(item.pop("event_json"))
            result.append(item)
        return result

    def list_catalogue_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM official_catalogue_event ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["event"] = json.loads(item.pop("event_json"))
            result.append(item)
        return result

    def upsert_catalogue_events(
        self,
        *,
        source_id: str,
        source_snapshot_id: str,
        events: list[dict[str, Any]],
    ) -> int:
        """Persist positive-only official catalogue rows with stable provenance."""

        now = timestamp()
        created = 0
        with self.transaction() as connection:
            for event in events:
                outbreak_id = str(event["outbreak_id"])
                seed = f"{source_id}\x1f{source_snapshot_id}\x1f{outbreak_id}".encode()
                event_id = f"catalogue_{hashlib.sha256(seed).hexdigest()[:24]}"
                result = connection.execute(
                    """
                    INSERT INTO official_catalogue_event(
                      id,source_id,source_snapshot_id,district_id,disease,event_json,created_at
                    ) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        event_id,
                        source_id,
                        source_snapshot_id,
                        event.get("district_id"),
                        event.get("disease"),
                        json.dumps(event, sort_keys=True, separators=(",", ":")),
                        now,
                    ),
                )
                created += int(result.rowcount == 1)
            if created:
                self._audit(
                    connection,
                    "catalogue_rows_ingested",
                    "source_snapshot",
                    source_snapshot_id,
                    {"source_id": source_id, "created": created},
                )
        return created

    @staticmethod
    def _job(row: Any | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def enqueue_job(
        self,
        *,
        source_id: str,
        kind: str,
        payload_ref: str | None,
        payload_hash: str,
        idempotency_key: str,
        allow_disabled_source: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        now = timestamp()
        with self.transaction() as connection:
            if (
                connection.execute(
                    """
                    SELECT 1 FROM source_registry
                    WHERE source_id=? AND (enabled=1 OR ?=1)
                    """,
                    (source_id, int(allow_disabled_source)),
                ).fetchone()
                is None
            ):
                raise RepositoryNotFound(f"registered source {source_id!r} not found")
            existing = connection.execute(
                "SELECT * FROM job WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash or existing["source_id"] != source_id:
                    raise RepositoryConflict(
                        "IDEMPOTENCY_KEY_REUSED",
                        "idempotency key was already used with a different request",
                    )
                return dict(existing), True
            job_id = f"job_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO job(
                  id,kind,source_id,payload_ref,payload_hash,idempotency_key,state,
                  available_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,'queued',?,?,?)
                """,
                (
                    job_id,
                    kind,
                    source_id,
                    payload_ref,
                    payload_hash,
                    idempotency_key,
                    now,
                    now,
                    now,
                ),
            )
            self._audit(connection, "job_enqueued", "job", job_id, {"source_id": source_id})
            return dict(
                connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
            ), False

    def claim_job(
        self,
        *,
        owner: str,
        lease_seconds: int,
        job_id: str | None = None,
        kind: str | None = None,
        payload_prefix: str | None = None,
    ) -> dict[str, Any] | None:
        now_dt = utc_now()
        now = timestamp(now_dt)
        lease_expires = timestamp(now_dt + timedelta(seconds=lease_seconds))
        with self.transaction() as connection:
            query = """
                SELECT * FROM job
                WHERE (
                  (state IN ('queued','retry_wait') AND available_at <= ?)
                  OR (state='running' AND lease_expires_at < ?)
                )
            """
            parameters: list[Any] = [now, now]
            if job_id is not None:
                query += " AND id=?"
                parameters.append(job_id)
            if kind is not None:
                query += " AND kind=?"
                parameters.append(kind)
            if payload_prefix is not None:
                query += " AND payload_ref LIKE ?"
                parameters.append(f"{payload_prefix}%")
            query += " ORDER BY created_at, id LIMIT 1"
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                return None
            previous_version = row["row_version"]
            updated = connection.execute(
                """
                UPDATE job SET state='running', attempt=attempt+1,
                  lease_owner=?, lease_expires_at=?, fencing_token=fencing_token+1,
                  row_version=row_version+1, updated_at=?
                WHERE id=? AND row_version=? AND (
                  (state IN ('queued','retry_wait') AND available_at <= ?)
                  OR (state='running' AND lease_expires_at < ?)
                )
                """,
                (owner, lease_expires, now, row["id"], previous_version, now, now),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict("CLAIM_RACE", "job was claimed by another worker")
            claimed = connection.execute("SELECT * FROM job WHERE id=?", (row["id"],)).fetchone()
            attempt_id = f"attempt_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO job_attempt(id,job_id,attempt,owner,fencing_token,started_at)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    attempt_id,
                    claimed["id"],
                    claimed["attempt"],
                    owner,
                    claimed["fencing_token"],
                    now,
                ),
            )
            self._audit(
                connection,
                "job_claimed",
                "job",
                claimed["id"],
                {"owner": owner, "fencing_token": claimed["fencing_token"]},
            )
            return dict(claimed)

    def complete_job(
        self,
        *,
        job_id: str,
        owner: str,
        fencing_token: int,
        idempotency_key: str,
        receipt: SourceReceiptInput | None,
        signals: list[RedactedSignalInput],
        link_disposition: str = "fetched",
    ) -> tuple[dict[str, Any], bool, list[str]]:
        if link_disposition not in {"fetched", "pending_approval"}:
            raise ValueError("link_disposition must be fetched or pending_approval")
        now_dt = utc_now()
        now = timestamp(now_dt)
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise RepositoryNotFound(job_id)
            if row["state"] == "completed":
                if row["completion_idempotency_key"] == idempotency_key:
                    replay_task_ids = [
                        item["id"]
                        for item in connection.execute(
                            """
                            SELECT r.id FROM review_task r
                            JOIN signal s ON s.id=r.signal_id
                            WHERE s.source_snapshot_id = ? ORDER BY r.id
                            """,
                            (row["source_snapshot_id"],),
                        ).fetchall()
                    ]
                    return dict(row), True, replay_task_ids
                raise RepositoryConflict("JOB_ALREADY_COMPLETED", "job is already completed")
            self._assert_current_lease(row, owner, fencing_token, now_dt)
            trusted_fixture = self._trusted_fixture_receipt(receipt, job_kind=row["kind"])
            prepared = [
                self._validate_signal(
                    signal,
                    row["source_id"],
                    is_trusted_fixture=trusted_fixture,
                )
                for signal in signals
            ]
            snapshot_id = self._store_snapshot(
                connection,
                job=row,
                receipt=receipt,
                created_at=now,
            )
            document_sha256 = receipt.sha256 if receipt is not None else row["payload_hash"]
            canonical_url = (
                canonicalize_discovered_url(receipt.final_url)
                if receipt is not None
                else str(row["payload_ref"] or "")
            )
            for signal in prepared:
                supplied_snapshot = signal["source_snapshot_id"]
                if not supplied_snapshot.startswith("job:") and supplied_snapshot != snapshot_id:
                    raise RepositoryConflict(
                        "SNAPSHOT_MISMATCH",
                        "signal snapshot does not match the completion receipt",
                    )
                # Logical evidence identity is stable across repeated retrieval
                # vintages of unchanged bytes. The snapshot remains append-only
                # provenance, but cannot manufacture another signal by itself.
                signal["document_sha256"] = document_sha256
                signal["canonical_url"] = canonical_url
                signal["content_key"] = self._signal_content_key(signal)
                if not trusted_fixture:
                    signal["signal_id"] = f"sig_{signal['content_key'][:32]}"
            updated = connection.execute(
                """
                UPDATE job SET state='completed', completion_idempotency_key=?,
                  source_snapshot_id=?,
                  lease_owner=NULL, lease_expires_at=NULL, row_version=row_version+1,
                  updated_at=?
                WHERE id=? AND state='running' AND lease_owner=? AND fencing_token=?
                  AND lease_expires_at >= ?
                """,
                (idempotency_key, snapshot_id, now, job_id, owner, fencing_token, now),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict("STALE_FENCING_TOKEN", "lease changed before completion")
            payload_ref = str(row["payload_ref"] or "")
            if payload_ref.startswith("registered-link:"):
                link_url = payload_ref.removeprefix("registered-link:")
                observed_sha256 = receipt.sha256 if receipt is not None else row["payload_hash"]
                connection.execute(
                    """
                    UPDATE discovered_link
                    SET state=?, queue_mode=NULL, source_snapshot_id=?,
                      observed_content_sha256=?,
                      fetched_at=CASE WHEN ?='fetched' THEN ? ELSE NULL END,
                      last_error_code=NULL, row_version=row_version+1
                    WHERE source_id=? AND url=? AND state='queued'
                    """,
                    (
                        link_disposition,
                        snapshot_id,
                        observed_sha256,
                        link_disposition,
                        now,
                        row["source_id"],
                        link_url,
                    ),
                )
            task_ids: list[str] = []
            for signal in prepared:
                signal_id = signal["signal_id"] or f"sig_{uuid.uuid4().hex}"
                signal_snapshot_id = signal["source_snapshot_id"]
                if signal_snapshot_id.startswith("job:"):
                    signal_snapshot_id = snapshot_id
                connection.execute(
                    """
                    INSERT INTO signal(
                      id,content_key,source_id,source_snapshot_id,district_id,disease,
                      assertion,evidence_text,evidence_start,evidence_end,content_sha256,
                      retrieved_at,event_review_eligible,processing_state,redaction_state,
                      language,extractor_version,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(content_key) DO NOTHING
                    """,
                    (
                        signal_id,
                        signal["content_key"],
                        signal["source_id"],
                        signal_snapshot_id,
                        signal["district_id"],
                        signal["disease"],
                        signal["assertion"],
                        signal["evidence_text"],
                        signal["evidence_start"],
                        signal["evidence_end"],
                        signal["content_sha256"],
                        signal["retrieved_at"],
                        int(bool(signal["event_review_eligible"])),
                        signal["processing_state"],
                        signal["redaction_state"],
                        signal["language"],
                        signal["extractor_version"],
                        now,
                    ),
                )
                stored = connection.execute(
                    """
                    SELECT id,assertion,event_review_eligible,processing_state
                    FROM signal WHERE content_key=?
                    """,
                    (signal["content_key"],),
                ).fetchone()
                # Non-affirmed/non-current statements remain evidence but cannot become events.
                if stored["assertion"] == "affirmed":
                    task_kind = (
                        "event_verification"
                        if bool(stored["event_review_eligible"])
                        else "quality_hold"
                    )
                    task_id = self._ensure_review_task(
                        connection, stored["id"], now, task_kind=task_kind
                    )
                    task_ids.append(task_id)
            connection.execute(
                """
                UPDATE job_attempt SET finished_at=?, outcome='completed'
                WHERE job_id=? AND fencing_token=?
                """,
                (now, job_id, fencing_token),
            )
            self._audit(
                connection,
                "job_completed",
                "job",
                job_id,
                {"owner": owner, "fencing_token": fencing_token, "signal_count": len(prepared)},
            )
            completed = connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
            return dict(completed), False, task_ids

    @staticmethod
    def _store_snapshot(
        connection: Any,
        *,
        job: Any,
        receipt: SourceReceiptInput | None,
        created_at: str,
    ) -> str:
        if receipt is not None:
            item = receipt.model_dump()
            if item["source_id"] != job["source_id"]:
                raise RepositoryConflict(
                    "SOURCE_MISMATCH",
                    "receipt source does not match the registered job source",
                )
            try:
                parse_timestamp(item["retrieved_at"])
            except ValueError as exc:
                raise RepositoryConflict(
                    "INVALID_RETRIEVAL_TIME", "receipt retrieved_at must be RFC3339"
                ) from exc
            snapshot_id = item["source_snapshot_id"]
        else:
            snapshot_id = f"snapshot_{job['id']}"
            payload_ref = str(job["payload_ref"] or "unspecified")
            item = {
                "source_id": job["source_id"],
                "requested_url": f"urn:registered-object:{payload_ref}",
                "final_url": f"urn:registered-object:{payload_ref}",
                "retrieved_at": created_at,
                "status_code": 200,
                "content_type": "application/octet-stream",
                "byte_length": 0,
                "sha256": job["payload_hash"],
                "access_path": "trusted_collector_without_receipt",
                "archive_timestamp": None,
                "archive_digest": None,
                "fallback_reason": None,
                "is_fixture": False,
            }
        connection.execute(
            """
            INSERT INTO source_snapshot(
              id,source_id,requested_url,final_url,retrieved_at,status_code,
              content_type,byte_length,content_sha256,access_path,archive_timestamp,
              archive_digest,fallback_reason,is_fixture,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                snapshot_id,
                item["source_id"],
                item["requested_url"],
                item["final_url"],
                item["retrieved_at"],
                item["status_code"],
                item["content_type"],
                item["byte_length"],
                item["sha256"],
                item["access_path"],
                item.get("archive_timestamp"),
                item.get("archive_digest"),
                item.get("fallback_reason"),
                int(bool(item.get("is_fixture", False))),
                created_at,
            ),
        )
        stored = connection.execute(
            "SELECT source_id,content_sha256,is_fixture FROM source_snapshot WHERE id=?",
            (snapshot_id,),
        ).fetchone()
        if (
            stored is None
            or stored["source_id"] != item["source_id"]
            or stored["content_sha256"] != item["sha256"]
            or bool(stored["is_fixture"]) != bool(item.get("is_fixture", False))
        ):
            raise RepositoryConflict(
                "SNAPSHOT_ID_REUSED",
                "snapshot id already belongs to different source bytes",
            )
        return snapshot_id

    def fail_job(
        self,
        *,
        job_id: str,
        owner: str,
        fencing_token: int,
        reason_code: str,
        retryable: bool,
    ) -> dict[str, Any]:
        now_dt = utc_now()
        now = timestamp(now_dt)
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise RepositoryNotFound(job_id)
            self._assert_current_lease(row, owner, fencing_token, now_dt)
            can_retry = retryable and row["attempt"] < 3
            state = "retry_wait" if can_retry else "dead"
            delay = min(300, 2 ** max(1, row["attempt"])) if can_retry else 0
            available_at = timestamp(now_dt + timedelta(seconds=delay))
            updated = connection.execute(
                """
                UPDATE job SET state=?, available_at=?, lease_owner=NULL,
                  lease_expires_at=NULL, row_version=row_version+1,
                  last_error_code=?, updated_at=?
                WHERE id=? AND state='running' AND lease_owner=? AND fencing_token=?
                  AND lease_expires_at >= ?
                """,
                (state, available_at, reason_code, now, job_id, owner, fencing_token, now),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict(
                    "STALE_FENCING_TOKEN", "lease changed before failure commit"
                )
            payload_ref = str(row["payload_ref"] or "")
            if state == "dead" and payload_ref.startswith("registered-link:"):
                link_url = payload_ref.removeprefix("registered-link:")
                connection.execute(
                    """
                UPDATE discovered_link
                    SET state='failed', queue_mode=NULL, last_error_code=?,
                      row_version=row_version+1
                    WHERE source_id=? AND url=? AND state='queued'
                    """,
                    (reason_code, row["source_id"], link_url),
                )
            connection.execute(
                """
                UPDATE job_attempt SET finished_at=?, outcome=?, error_code=?
                WHERE job_id=? AND fencing_token=?
                """,
                (now, state, reason_code, job_id, fencing_token),
            )
            self._audit(
                connection,
                "job_failed",
                "job",
                job_id,
                {"reason_code": reason_code, "retryable": can_retry},
            )
            return dict(connection.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone())

    @staticmethod
    def _assert_current_lease(row: Any, owner: str, fencing_token: int, now: datetime) -> None:
        if row["state"] != "running" or row["lease_owner"] != owner:
            raise RepositoryConflict("STALE_FENCING_TOKEN", "worker does not own the active lease")
        if row["fencing_token"] != fencing_token:
            raise RepositoryConflict("STALE_FENCING_TOKEN", "fencing token is stale")
        if not row["lease_expires_at"] or parse_timestamp(row["lease_expires_at"]) < now:
            raise RepositoryConflict("LEASE_EXPIRED", "job lease has expired")

    @staticmethod
    def _trusted_fixture_receipt(
        receipt: SourceReceiptInput | None, *, job_kind: str
    ) -> bool:
        if receipt is None or not receipt.is_fixture:
            return False
        trusted = (
            job_kind == "replay"
            and receipt.access_path == "bundled_hash_verified_fixture"
            and receipt.requested_url.startswith("fixture://bundled/")
            and receipt.final_url.startswith("fixture://bundled/")
        )
        if not trusted:
            raise RepositoryConflict(
                "UNTRUSTED_FIXTURE_RECEIPT",
                "only bundled hash-verified replay receipts may retain evidence text",
            )
        return True

    @staticmethod
    def _signal_content_key(item: dict[str, Any]) -> str:
        document_identity = item.get("document_sha256") or item.get(
            "source_snapshot_id"
        )
        key_material = "\x1f".join(
            (
                str(item.get("source_id") or ""),
                str(item.get("canonical_url") or ""),
                str(document_identity or ""),
                str(item.get("district_id") or ""),
                str(item.get("disease") or ""),
                str(item.get("assertion") or ""),
                str(item.get("processing_state") or ""),
                str(item.get("language") or ""),
                str(item.get("extractor_version") or ""),
            )
        )
        return hashlib.sha256(key_material.encode()).hexdigest()

    @classmethod
    def _validate_signal(
        cls,
        signal: RedactedSignalInput,
        expected_source: str,
        *,
        is_trusted_fixture: bool = False,
    ) -> dict[str, Any]:
        item = signal.model_dump()
        if item["source_id"] != expected_source:
            raise RepositoryConflict(
                "SOURCE_MISMATCH", "completion signal source does not match registered job source"
            )
        if is_trusted_fixture:
            evidence_hash = hashlib.sha256(item["evidence_text"].encode("utf-8")).hexdigest()
            if item["redaction_state"] != "heuristic_unvalidated":
                raise RepositoryConflict(
                    "INVALID_FIXTURE_REDACTION_STATE",
                    "inspectable bundled fixtures must declare heuristic_unvalidated",
                )
            if item["content_sha256"] != evidence_hash:
                raise RepositoryConflict(
                    "FIXTURE_EVIDENCE_HASH_MISMATCH",
                    "fixture evidence hash does not match its retained synthetic span",
                )
        else:
            # This is the final persistence boundary.  Collector extraction may
            # inspect live source text in memory, but authenticated callers also
            # cannot bypass non-retention by submitting their own signal body.
            item["evidence_text"] = LIVE_EVIDENCE_PLACEHOLDER
            item["evidence_start"] = 0
            item["evidence_end"] = len(LIVE_EVIDENCE_PLACEHOLDER)
            item["content_sha256"] = hashlib.sha256(
                LIVE_EVIDENCE_PLACEHOLDER.encode("utf-8")
            ).hexdigest()
            item["redaction_state"] = LIVE_EVIDENCE_REDACTION_STATE
        if (
            item["evidence_start"] >= item["evidence_end"]
            or item["evidence_end"] - item["evidence_start"]
            != len(item["evidence_text"])
        ):
            raise RepositoryConflict(
                "INVALID_EVIDENCE_OFFSET",
                "evidence offsets must match the bounded canonical-redacted span length",
            )
        try:
            parse_timestamp(item["retrieved_at"])
        except ValueError as exc:
            raise RepositoryConflict(
                "INVALID_RETRIEVAL_TIME", "retrieved_at must be RFC3339"
            ) from exc
        item["content_key"] = cls._signal_content_key(item)
        if not is_trusted_fixture:
            # Never persist a caller-provided identifier: it may itself contain
            # a person name or be derived from the discarded source span.
            item["signal_id"] = f"sig_{item['content_key'][:32]}"
        return item

    @staticmethod
    def _ensure_review_task(
        connection: Any, signal_id: str, now: str, *, task_kind: str
    ) -> str:
        existing = connection.execute(
            "SELECT id FROM review_task WHERE signal_id=?", (signal_id,)
        ).fetchone()
        if existing:
            return existing["id"]
        task_id = f"task_{uuid.uuid4().hex}"
        connection.execute(
            """
            INSERT INTO review_task(id,signal_id,task_kind,state,created_at,updated_at)
            VALUES(?,?,?,'open',?,?)
            """,
            (task_id, signal_id, task_kind, now, now),
        )
        connection.execute(
            """
            INSERT INTO review_transition(
              id,task_id,from_state,to_state,actor_id,reason_code,row_version,created_at
            ) VALUES(?,?,NULL,'open','system',?,0,?)
            """,
            (
                f"transition_{uuid.uuid4().hex}",
                task_id,
                "SIGNAL_REQUIRES_REVIEW"
                if task_kind == "event_verification"
                else "SIGNAL_REQUIRES_QUALITY_HOLD",
                now,
            ),
        )
        return task_id

    def claim_review_task(
        self,
        *,
        task_id: str,
        reviewer_id: str,
        expected_row_version: int,
        lease_seconds: int,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        now_dt = utc_now()
        now = timestamp(now_dt)
        expires = timestamp(now_dt + timedelta(seconds=lease_seconds))
        scope = f"review_claim:{task_id}"
        with self.transaction() as connection:
            replay = connection.execute(
                "SELECT resource_id FROM mutation_idempotency WHERE scope=? AND idempotency_key=?",
                (scope, idempotency_key),
            ).fetchone()
            if replay:
                row = connection.execute(
                    "SELECT * FROM review_task WHERE id=?", (task_id,)
                ).fetchone()
                return dict(row), True
            row = connection.execute("SELECT * FROM review_task WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise RepositoryNotFound(task_id)
            eligible = row["state"] == "open" or (
                row["state"] == "claimed"
                and row["claim_expires_at"]
                and parse_timestamp(row["claim_expires_at"]) < now_dt
            )
            if not eligible:
                raise RepositoryConflict("TASK_NOT_CLAIMABLE", "review task is not claimable")
            if row["row_version"] != expected_row_version:
                raise RepositoryConflict("STALE_ROW_VERSION", "review task version changed")
            updated = connection.execute(
                """
                UPDATE review_task SET state='claimed', claimed_by=?, claim_expires_at=?,
                  row_version=row_version+1, updated_at=?
                WHERE id=? AND row_version=? AND (
                  state='open' OR (state='claimed' AND claim_expires_at < ?)
                )
                """,
                (reviewer_id, expires, now, task_id, expected_row_version, now),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict("CLAIM_RACE", "review task was claimed concurrently")
            current = connection.execute(
                "SELECT * FROM review_task WHERE id=?", (task_id,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO review_transition(
                  id,task_id,from_state,to_state,actor_id,reason_code,row_version,created_at
                ) VALUES(?,?,?,'claimed',?,'REVIEWER_CLAIMED',?,?)
                """,
                (
                    f"transition_{uuid.uuid4().hex}",
                    task_id,
                    row["state"],
                    reviewer_id,
                    current["row_version"],
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO mutation_idempotency VALUES(?,?,?,?)",
                (scope, idempotency_key, task_id, now),
            )
            self._audit(
                connection, "review_claimed", "review_task", task_id, {"reviewer": reviewer_id}
            )
            return dict(current), False

    def decide_review_task(
        self,
        *,
        task_id: str,
        reviewer_id: str,
        expected_row_version: int,
        decision: str,
        rationale: str,
        supersedes_decision_id: str | None,
        event: dict[str, Any] | None,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        now = timestamp()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM review_decision WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing:
                if existing["task_id"] != task_id:
                    raise RepositoryConflict(
                        "IDEMPOTENCY_KEY_REUSED", "decision key belongs to another task"
                    )
                return self._decision_dict(existing), True
            task = connection.execute("SELECT * FROM review_task WHERE id=?", (task_id,)).fetchone()
            if task is None:
                raise RepositoryNotFound(task_id)
            if decision == "verified" and task["task_kind"] != "event_verification":
                raise RepositoryConflict(
                    "QUALITY_HOLD_NOT_EVENT_ELIGIBLE",
                    "quality/language/privacy holds cannot be verified as events",
                )
            signal = connection.execute(
                """
                SELECT s.* FROM signal s JOIN review_task t ON t.signal_id=s.id
                WHERE t.id=?
                """,
                (task_id,),
            ).fetchone()
            if signal is None:
                raise RepositoryNotFound(f"signal for {task_id}")
            if decision == "verified":
                if event is None:
                    raise RepositoryConflict(
                        "VERIFIED_EVENT_REQUIRED",
                        "verified decisions require a typed event payload",
                    )
                if (
                    event.get("district_id") != signal["district_id"]
                    or event.get("disease") != signal["disease"]
                ):
                    raise RepositoryConflict(
                        "EVENT_SCOPE_MISMATCH",
                        "verified event scope must equal the reviewed signal scope",
                    )
            elif event is not None:
                raise RepositoryConflict(
                    "EVENT_NOT_ALLOWED",
                    "only verified decisions may carry an event payload",
                )
            if task["row_version"] != expected_row_version:
                raise RepositoryConflict("STALE_ROW_VERSION", "review task version changed")
            current_decision = task["current_decision_id"]
            if current_decision:
                if supersedes_decision_id != current_decision:
                    raise RepositoryConflict(
                        "SUPERSEDES_REQUIRED",
                        "a correction must explicitly supersede the current decision",
                    )
            elif supersedes_decision_id is not None:
                raise RepositoryConflict("INVALID_SUPERSEDES", "there is no decision to supersede")
            if not current_decision:
                if task["state"] != "claimed" or task["claimed_by"] != reviewer_id:
                    raise RepositoryConflict(
                        "TASK_NOT_OWNED", "reviewer must hold the task before deciding"
                    )
                if (
                    task["claim_expires_at"]
                    and parse_timestamp(task["claim_expires_at"]) < utc_now()
                ):
                    raise RepositoryConflict("REVIEW_LEASE_EXPIRED", "review claim has expired")
            decision_id = f"decision_{uuid.uuid4().hex}"
            event_json = json.dumps(event, sort_keys=True, separators=(",", ":")) if event else None
            connection.execute(
                """
                INSERT INTO review_decision(
                  id,task_id,reviewer_id,decision,rationale,supersedes_id,
                  idempotency_key,event_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision_id,
                    task_id,
                    reviewer_id,
                    decision,
                    rationale,
                    supersedes_decision_id,
                    idempotency_key,
                    event_json,
                    now,
                ),
            )
            updated = connection.execute(
                """
                UPDATE review_task SET state='decided', claimed_by=NULL,
                  claim_expires_at=NULL, current_decision_id=?, row_version=row_version+1,
                  updated_at=? WHERE id=? AND row_version=?
                """,
                (decision_id, now, task_id, expected_row_version),
            )
            if updated.rowcount != 1:
                raise RepositoryConflict("STALE_ROW_VERSION", "task changed before decision commit")
            new_task = connection.execute(
                "SELECT * FROM review_task WHERE id=?", (task_id,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO review_transition(
                  id,task_id,from_state,to_state,actor_id,reason_code,row_version,created_at
                ) VALUES(?,?,?,'decided',?,?,?,?)
                """,
                (
                    f"transition_{uuid.uuid4().hex}",
                    task_id,
                    task["state"],
                    reviewer_id,
                    f"DECISION_{decision.upper()}",
                    new_task["row_version"],
                    now,
                ),
            )
            if decision == "verified":
                assert event is not None  # narrowed and validated above
                payload = {
                    "district_id": event["district_id"],
                    "disease": event["disease"],
                    "status": "reviewer_verified_public_source_event",
                }
                connection.execute(
                    """
                    INSERT INTO verified_event(
                      id,task_id,decision_id,district_id,disease,event_json,created_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        f"event_{uuid.uuid4().hex}",
                        task_id,
                        decision_id,
                        payload["district_id"],
                        payload["disease"],
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        now,
                    ),
                )
            self._audit(
                connection,
                "review_decided",
                "review_task",
                task_id,
                {
                    "decision_id": decision_id,
                    "decision": decision,
                    "supersedes": supersedes_decision_id,
                },
            )
            stored = connection.execute(
                "SELECT * FROM review_decision WHERE id=?", (decision_id,)
            ).fetchone()
            return self._decision_dict(stored), False

    @staticmethod
    def _decision_dict(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["event"] = json.loads(item.pop("event_json")) if item["event_json"] else None
        return item

    def replay_demo_fixtures(self) -> dict[str, Any]:
        """Replay all twelve hash-pinned fixtures through extraction and durable jobs."""

        from workers.ingestion.fixture_pack import load_fixture_pack

        pack = load_fixture_pack()
        signal_count_before = len(self.list_signals())
        task_count_before = len(self.list_review_tasks(limit=500))
        catalogue_count_before = len(self.list_catalogue_events(limit=500))
        for item in pack.items:
            key = f"fixture:{item.receipt.sha256}"
            job, _ = self.enqueue_job(
                source_id=item.source_id,
                kind="replay",
                payload_ref=f"fixture:{item.fixture_name}",
                payload_hash=item.receipt.sha256,
                idempotency_key=key,
                allow_disabled_source=True,
            )
            if job["state"] == "completed":
                continue
            owner = "fixture-replay-worker"
            claimed = self.claim_job(owner=owner, lease_seconds=300, job_id=job["id"])
            if claimed is None:
                raise RepositoryConflict("FIXTURE_JOB_UNAVAILABLE", job["id"])
            self.complete_job(
                job_id=job["id"],
                owner=owner,
                fencing_token=claimed["fencing_token"],
                idempotency_key=f"complete:{key}",
                receipt=item.receipt,
                signals=[item.signal],
            )

        catalogue_receipt = pack.catalogue_receipt
        catalogue_job, _ = self.enqueue_job(
            source_id=catalogue_receipt.source_id,
            kind="replay",
            payload_ref="fixture:12_idsp_weekly_pdf_surrogate.json",
            payload_hash=catalogue_receipt.sha256,
            idempotency_key=f"fixture:{catalogue_receipt.sha256}",
            allow_disabled_source=True,
        )
        if catalogue_job["state"] != "completed":
            owner = "fixture-catalogue-worker"
            claimed = self.claim_job(
                owner=owner, lease_seconds=300, job_id=catalogue_job["id"]
            )
            if claimed is None:
                raise RepositoryConflict("FIXTURE_JOB_UNAVAILABLE", catalogue_job["id"])
            self.complete_job(
                job_id=catalogue_job["id"],
                owner=owner,
                fencing_token=claimed["fencing_token"],
                idempotency_key=f"complete:fixture:{catalogue_receipt.sha256}",
                receipt=catalogue_receipt,
                signals=[],
            )
        self._upsert_fixture_catalogue(pack)
        signal_count_after = len(self.list_signals())
        task_count_after = len(self.list_review_tasks(limit=500))
        catalogue_count_after = len(self.list_catalogue_events(limit=500))
        return {
            "fixture_set": pack.pack_id,
            "manifest_sha256": pack.manifest_sha256,
            "created_signals": signal_count_after - signal_count_before,
            "created_review_tasks": task_count_after - task_count_before,
            "created_catalogue_events": catalogue_count_after - catalogue_count_before,
            "total_signals": signal_count_after,
        }

    def _upsert_fixture_catalogue(self, pack: Any) -> None:
        now = timestamp(datetime(2026, 7, 21, 12, 0, tzinfo=UTC))
        with self.transaction() as connection:
            for row in pack.catalogue_rows:
                event_digest = hashlib.sha256(row.outbreak_id.encode()).hexdigest()[:20]
                event_id = f"fixture_catalogue_{event_digest}"
                payload = {
                    "outbreak_id": row.outbreak_id,
                    "year": row.year,
                    "week": row.week,
                    "district_code": row.district_code,
                    "authority_status": row.authority_status,
                    "positive_only_catalogue": True,
                    "is_synthetic_fixture": True,
                }
                connection.execute(
                    """
                    INSERT INTO official_catalogue_event(
                      id,source_id,source_snapshot_id,district_id,disease,event_json,created_at
                    ) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        event_id,
                        "idsp_weekly_outbreaks",
                        pack.catalogue_receipt.source_snapshot_id,
                        "OD-DIST-angul" if row.district_code == "ANU" else None,
                        None,
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        now,
                    ),
                )

    @staticmethod
    def _audit(
        connection: Any,
        event_type: str,
        entity_type: str,
        entity_id: str,
        detail: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_event(
              event_id,event_type,entity_type,entity_id,detail_json,created_at
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                f"audit_{uuid.uuid4().hex}",
                event_type,
                entity_type,
                entity_id,
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
                timestamp(),
            ),
        )


def default_database_url() -> str:
    configured = os.getenv("DATABASE_URL")
    require_postgres = os.getenv("REQUIRE_POSTGRES", "false").casefold() == "true"
    if require_postgres and not configured:
        raise RuntimeError("REQUIRE_POSTGRES=true but DATABASE_URL is not configured")
    if require_postgres and configured and not configured.startswith(
        ("postgresql://", "postgres://", "postgresql+psycopg://")
    ):
        raise RuntimeError("REQUIRE_POSTGRES=true rejects a non-PostgreSQL DATABASE_URL")
    return configured or "sqlite:////tmp/odisha-health-hub.db"
