#!/usr/bin/env python3
"""Keep the free demo database active and optionally back it up to R2."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import boto3
import psycopg


def verified_postgres_url(database_url: str) -> str:
    """Require authenticated TLS for every public managed-Postgres connection."""

    normalised = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    parsed = urlsplit(normalised)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("backup requires a PostgreSQL DATABASE_URL")
    parameters = parse_qs(parsed.query, keep_blank_values=True)
    if parameters.get("sslmode") != ["verify-full"]:
        raise ValueError("DATABASE_URL must set sslmode=verify-full")
    if parameters.get("sslrootcert") != ["system"]:
        raise ValueError("DATABASE_URL must set sslrootcert=system")
    return normalised


def postgres_environment(database_url: str) -> dict[str, str]:
    parsed = urlsplit(verified_postgres_url(database_url))
    hostname = parsed.hostname
    if hostname is None:  # defensive: verified_postgres_url already enforces this
        raise ValueError("backup requires a PostgreSQL hostname")
    return {
        "PGHOST": hostname,
        "PGPORT": str(parsed.port or 5432),
        "PGUSER": unquote(parsed.username or ""),
        "PGPASSWORD": unquote(parsed.password or ""),
        "PGDATABASE": unquote(parsed.path.lstrip("/")),
        "PGSSLMODE": "verify-full",
        "PGSSLROOTCERT": "system",
    }


def validated_r2_configuration(values: dict[str, str]) -> dict[str, str]:
    required = {
        "endpoint_url": values["R2_ENDPOINT_URL"],
        "aws_access_key_id": values["R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": values["R2_SECRET_ACCESS_KEY"],
        "bucket": values["R2_BUCKET"],
    }
    endpoint = urlsplit(required["endpoint_url"])
    expected_host = re.fullmatch(
        r"[0-9a-f]{32}\.r2\.cloudflarestorage\.com",
        (endpoint.hostname or "").casefold(),
    )
    if (
        endpoint.scheme != "https"
        or not expected_host
        or endpoint.username
        or endpoint.password
        or endpoint.port is not None
        or endpoint.path not in {"", "/"}
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("R2_ENDPOINT_URL must be the account-scoped Cloudflare R2 HTTPS origin")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", required["bucket"]):
        raise ValueError("R2_BUCKET is not a valid fixed bucket name")
    return required


def upload_to_r2(path: Path, key: str) -> str:
    required = validated_r2_configuration(dict(os.environ))
    client = boto3.client(
        "s3",
        endpoint_url=required["endpoint_url"],
        aws_access_key_id=required["aws_access_key_id"],
        aws_secret_access_key=required["aws_secret_access_key"],
        region_name="auto",
    )
    client.upload_file(
        str(path),
        required["bucket"],
        key,
        ExtraArgs={"ContentType": "application/gzip"},
    )
    return f"r2://{required['bucket']}/{key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(json.dumps({"database": "not_configured", "backup": "not_attempted"}))
        return
    try:
        psycopg_url = verified_postgres_url(database_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    with psycopg.connect(psycopg_url, connect_timeout=15) as connection:
        value = connection.execute("SELECT 1").fetchone()
        if value != (1,):
            raise SystemExit("database keepalive failed")
    result: dict[str, object] = {"database": "reachable", "backup": "not_requested"}
    if args.backup:
        pg_dump = shutil.which("pg_dump")
        if not pg_dump:
            raise SystemExit("pg_dump is required for --backup")
        now = datetime.now(UTC)
        with tempfile.TemporaryDirectory(prefix="odisha-health-backup-") as directory:
            dump = Path(directory) / "demo.dump"
            compressed = Path(directory) / "demo.dump.gz"
            environment = os.environ.copy()
            environment.update(postgres_environment(database_url))
            subprocess.run(  # noqa: S603 - executable resolved; arguments are fixed
                [
                    pg_dump,
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    "--file",
                    str(dump),
                ],
                env=environment,
                check=True,
                timeout=300,
            )
            with dump.open("rb") as source, gzip.open(compressed, "wb", compresslevel=9) as target:
                shutil.copyfileobj(source, target)
            key = f"database-backups/{now:%Y/%m/%d}/odisha-health-{now:%H%M%SZ}.dump.gz"
            result["backup"] = upload_to_r2(compressed, key)
            result["compressed_bytes"] = compressed.stat().st_size
    print(json.dumps(result))


if __name__ == "__main__":
    main()
