#!/usr/bin/env python
"""Download every local NLP model into ``models/`` and write a manifest.

Idempotent: Hugging Face snapshots are content-addressed, so a second run
re-verifies the files already on disk instead of re-downloading them.  Nothing
in the API imports this script; the runtime only ever reads what is on disk.

    .venv/bin/python scripts/fetch_models.py            # everything
    .venv/bin/python scripts/fetch_models.py --only translate_en_indic
    .venv/bin/python scripts/fetch_models.py --check    # report, download nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.nlp import models  # noqa: E402


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def fetch(spec: models.ModelSpec, *, check_only: bool) -> dict[str, object]:
    target = models.download_directory(spec.key)
    missing = models.missing_files(spec.key)
    record: dict[str, object] = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "licence": spec.licence,
        "purpose": spec.purpose,
        "local_directory": str(target),
        "runtime_path": str(models.runtime_path(spec.key)),
    }
    if not missing:
        size = _directory_size(target)
        print(f"  [present] {spec.key:<22} {_human(size):>9}  {spec.repo_id}")
        record.update({"state": "present", "bytes_on_disk": size})
        return record
    if check_only:
        print(f"  [missing] {spec.key:<22} {'-':>9}  {spec.repo_id} -> {list(missing)}")
        record.update({"state": "missing", "bytes_on_disk": 0, "missing_files": list(missing)})
        return record

    from huggingface_hub import snapshot_download  # noqa: PLC0415 - optional dependency

    print(
        f"  [fetch  ] {spec.key:<22} {_human(spec.approximate_bytes):>9}  {spec.repo_id}",
        flush=True,
    )
    started = time.monotonic()
    # `local_dir=` makes huggingface_hub create a second, deeply nested
    # `.cache/huggingface/download/...incomplete` path below the destination.
    # That exceeds the legacy Windows MAX_PATH limit for IndicTrans2 even when
    # the checkout itself is reasonably short. Download into a short, isolated
    # cache and copy the resolved snapshot into the runtime directory instead.
    cache_root = models.models_directory() / ".download-cache" / spec.key
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(cache_root),
                allow_patterns=list(spec.allow_patterns),
            )
        )
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, target, dirs_exist_ok=True)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)
    elapsed = time.monotonic() - started
    size = _directory_size(target)
    still_missing = models.missing_files(spec.key)
    state = "downloaded" if not still_missing else "incomplete"
    print(f"  [{state:<8}] {spec.key:<22} {_human(size):>9}  in {elapsed:.1f}s", flush=True)
    record.update(
        {
            "state": state,
            "bytes_on_disk": size,
            "seconds": round(elapsed, 1),
            "missing_files": list(still_missing),
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[], help="model key to fetch")
    parser.add_argument("--check", action="store_true", help="report state, download nothing")
    arguments = parser.parse_args(argv)

    root = models.models_directory()
    root.mkdir(parents=True, exist_ok=True)
    specifications = list(models.iter_specifications())
    requested = set(arguments.only)
    selected = [
        spec for spec in specifications if not requested or spec.key in requested
    ]
    if not selected:
        print(f"no model matches {arguments.only}", file=sys.stderr)
        return 2

    print(f"models directory: {root}")
    # The manifest is an inventory of the complete runtime, even when --only is
    # used to download one component. Previously --only silently replaced the
    # manifest with a one-model partial view.
    selected_keys = {spec.key for spec in selected}
    records = [
        fetch(
            spec,
            check_only=arguments.check or spec.key not in selected_keys,
        )
        for spec in specifications
    ]
    total = sum(int(str(record.get("bytes_on_disk", 0) or 0)) for record in records)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models_directory": str(root),
        "total_bytes_on_disk": total,
        "models": records,
    }
    manifest_path = root / "manifest.json"
    if not arguments.check:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"manifest: {manifest_path}")
    print(f"total on disk: {_human(total)}")
    incomplete = [record for record in records if record["state"] in {"missing", "incomplete"}]
    return 1 if incomplete and not arguments.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
