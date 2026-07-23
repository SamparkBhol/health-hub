#!/usr/bin/env python3
"""Measure what each NLP component costs in resident memory.

The deployment figures come from here. They decide which host can serve which
languages, so they are measured rather than estimated -- a model's
file size is a poor predictor of its resident cost, and for the translator the
*peak* during load is nearly two and a half times the steady state.

    uv run python scripts/measure_memory.py                 # every component
    uv run python scripts/measure_memory.py translation     # just one

Each component is measured in a fresh subprocess, because a component that has
already been loaded cannot be un-measured: allocator arenas and import side
effects do not unwind, so measuring two components in one process attributes the
first one's residue to the second.

To confirm a component actually fits a host rather than merely reporting a small
number, run it under a real cap:

    docker run --rm --memory=1g -e ODISHA_TRANSLATION_RESIDENT=1 \\
      -v "$PWD/models:/app/models:ro" --entrypoint python ok-api:latest \\
      /app/scripts/measure_memory.py translation
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = ("api", "translation", "embeddings", "llm")


def _memory_kb(field: str) -> float:
    """VmRSS (resident now) or VmHWM (the high-water mark), in MB."""

    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def _probe(component: str) -> dict[str, object]:
    """Run inside the child process. Loads one component and reports its cost."""

    import time

    sys.path.insert(0, str(ROOT))
    baseline = _memory_kb("VmRSS")
    started = time.monotonic()
    detail = ""

    if component == "api":
        from services.api.main import create_app

        create_app("sqlite:///:memory:")
        detail = "FastAPI app with every data layer resident, no models"

    elif component == "translation":
        from packages.nlp import translate

        baseline = _memory_kb("VmRSS")
        first = translate.translate("Dengue cases are rising in Cuttack.", "en", "or")
        after_one = _memory_kb("VmRSS")
        peak_one = _memory_kb("VmHWM")
        second = translate.translate("ମ୍ୟାଲେରିଆ ବଢ଼ୁଛି ।", "or", "en")
        detail = (
            f"one direction: steady {after_one:.0f} MB, peak {peak_one:.0f} MB; "
            f"en->or {first.state}, or->en {second.state}; "
            f"resident limit {translate._resident_engine_limit()}"
        )

    elif component == "embeddings":
        from packages.nlp import retrieval

        baseline = _memory_kb("VmRSS")
        vector = retrieval.embed(["malaria cases are rising"], kind="query")
        detail = f"embedding shape {getattr(vector, 'shape', None)}"

    elif component == "llm":
        from packages.nlp import answer

        baseline = _memory_kb("VmRSS")
        model = answer._model()
        detail = "loaded" if model is not None else "unavailable (model not on disk)"
        # The GGUF is mmapped, so this RSS is page-cache backed and evictable
        # under pressure rather than anonymous. It still counts against a cgroup
        # limit; it is simply reclaimable instead of fatal.
        maps = Path("/proc/self/maps").read_text()
        detail += f", mmapped={'.gguf' in maps}"

    else:  # pragma: no cover - guarded by the caller
        raise SystemExit(f"unknown component: {component}")

    return {
        "component": component,
        "baseline_rss_mb": round(baseline, 1),
        "steady_rss_mb": round(_memory_kb("VmRSS"), 1),
        "peak_rss_mb": round(_memory_kb("VmHWM"), 1),
        "load_seconds": round(time.monotonic() - started, 2),
        "detail": detail,
    }


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--probe":
        print(json.dumps(_probe(sys.argv[2])))
        return 0

    wanted = sys.argv[1:] or list(COMPONENTS)
    unknown = [name for name in wanted if name not in COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown component(s) {unknown}; choose from {COMPONENTS}")

    rows = []
    for component in wanted:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(Path(__file__).resolve()), "--probe", component],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            rows.append(
                {
                    "component": component,
                    "error": (result.stderr or "").strip().splitlines()[-1:] or ["failed"],
                }
            )
            continue
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))

    print(f"{'component':<13}{'baseline':>10}{'steady':>10}{'peak':>10}{'load s':>9}  detail")
    for row in rows:
        if "error" in row:
            print(f"{row['component']:<13}{'--':>10}{'--':>10}{'--':>10}{'--':>9}  {row['error']}")
            continue
        print(
            f"{row['component']:<13}"
            f"{row['baseline_rss_mb']:>10.1f}"
            f"{row['steady_rss_mb']:>10.1f}"
            f"{row['peak_rss_mb']:>10.1f}"
            f"{row['load_seconds']:>9.2f}  {row['detail']}"
        )
    print("\nAll figures are MB of resident memory read from /proc/self/status.")
    print("Peak (VmHWM), not steady state, is what a container memory cap kills on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
