"""Local model registry for the on-device NLP stack.

Every model used by the assistant is an ungated Hugging Face artefact that is
downloaded into a gitignored ``models/`` directory by ``scripts/fetch_models.py``.
Nothing here downloads anything implicitly: library code only ever *reads* what
is already on disk and reports a typed unavailable state otherwise, so an API
process never blocks on a multi-gigabyte network fetch during a request.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIRECTORY = REPOSITORY_ROOT / "models"

ModelKey = Literal[
    "translate_en_indic",
    "translate_indic_en",
    "embed_multilingual",
    "llm_grounded_answer",
]

NlpMode = Literal["auto", "off"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A single downloadable artefact and the files that prove it is complete."""

    key: str
    repo_id: str
    purpose: str
    licence: str
    approximate_bytes: int
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    #: Path (relative to the download directory) handed to the runtime loader.
    runtime_path: str

    @property
    def local_directory_name(self) -> str:
        return self.repo_id.split("/")[-1]


MANIFEST: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="translate_en_indic",
        repo_id="adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
        purpose="English -> Hindi/Odia translation (IndicTrans2 distilled 200M, CTranslate2)",
        licence="MIT",
        approximate_bytes=847_151_318,
        allow_patterns=("en-indic-200m-ct2/ctranslate2_model/*",),
        required_files=(
            "en-indic-200m-ct2/ctranslate2_model/model.bin",
            "en-indic-200m-ct2/ctranslate2_model/vocab/model.SRC",
            "en-indic-200m-ct2/ctranslate2_model/vocab/model.TGT",
        ),
        runtime_path="en-indic-200m-ct2/ctranslate2_model",
    ),
    ModelSpec(
        key="translate_indic_en",
        repo_id="adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
        purpose="Hindi/Odia -> English translation (IndicTrans2 distilled 200M, CTranslate2)",
        licence="MIT",
        approximate_bytes=847_167_702,
        allow_patterns=("indic-en-200m-ct2/ctranslate2_model/*",),
        required_files=(
            "indic-en-200m-ct2/ctranslate2_model/model.bin",
            "indic-en-200m-ct2/ctranslate2_model/vocab/model.SRC",
            "indic-en-200m-ct2/ctranslate2_model/vocab/model.TGT",
        ),
        runtime_path="indic-en-200m-ct2/ctranslate2_model",
    ),
    ModelSpec(
        key="embed_multilingual",
        repo_id="intfloat/multilingual-e5-small",
        purpose="Cross-lingual sentence embeddings for semantic evidence retrieval (ONNX)",
        licence="MIT",
        approximate_bytes=470_000_000,
        allow_patterns=(
            "onnx/model.onnx",
            "onnx/tokenizer.json",
            "onnx/tokenizer_config.json",
            "onnx/special_tokens_map.json",
            "onnx/config.json",
            "1_Pooling/config.json",
        ),
        required_files=("onnx/model.onnx", "onnx/tokenizer.json"),
        runtime_path="onnx",
    ),
    ModelSpec(
        key="llm_grounded_answer",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        purpose="Grounded generative answering over retrieved evidence (llama.cpp, Q4_K_M)",
        # Unlike the 3B checkpoint, Qwen2.5-1.5B is Apache-2.0 and can be used in
        # the employer/enterprise profile without a non-commercial restriction.
        licence="Apache-2.0",
        approximate_bytes=1_050_000_000,
        allow_patterns=("qwen2.5-1.5b-instruct-q4_k_m.gguf", "LICENSE", "README.md"),
        required_files=("qwen2.5-1.5b-instruct-q4_k_m.gguf", "LICENSE"),
        runtime_path="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
)

_BY_KEY = {spec.key: spec for spec in MANIFEST}


def models_directory() -> Path:
    """Root directory that holds every downloaded artefact."""

    override = os.environ.get("ODISHA_MODELS_DIR", "").strip()
    return Path(override).expanduser() if override else DEFAULT_MODELS_DIRECTORY


def nlp_mode() -> NlpMode:
    """``auto`` uses whatever is on disk; ``off`` disables every model path."""

    value = os.environ.get("ODISHA_NLP_MODE", "auto").strip().lower()
    return "off" if value in {"off", "0", "false", "disabled"} else "auto"


def inference_threads() -> int:
    """Threads for the small models (translation, embeddings).

    Deliberately smaller than the generator's pool: all three runtimes live in
    one API process, and oversubscribing the CPU made generation collapse from
    15 tok/s to 0.4 tok/s in a thread-count probe on this host.
    """

    raw = os.environ.get("ODISHA_NLP_THREADS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, min(4, os.cpu_count() or 1))


def generation_threads() -> int:
    raw = os.environ.get("ODISHA_LLM_THREADS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(1, min(8, os.cpu_count() or 1))


@functools.lru_cache(maxsize=32)
def runtime_importable(*modules: str) -> bool:
    """True when every optional runtime package is installed.

    Model files on disk are not enough: a deployment that skipped the ``nlp``
    extra has the weights but not the runtimes, and must degrade rather than
    raise ImportError inside a request.
    """

    try:
        return all(importlib.util.find_spec(module) is not None for module in modules)
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def specification(key: str) -> ModelSpec:
    try:
        return _BY_KEY[key]
    except KeyError as error:  # pragma: no cover - programming error
        raise KeyError(f"unknown model key: {key}") from error


def download_directory(key: str) -> Path:
    return models_directory() / specification(key).local_directory_name


def runtime_path(key: str) -> Path:
    """Path handed to the runtime loader (a directory or a single file)."""

    if key == "llm_grounded_answer":
        override = os.environ.get("ODISHA_LLM_MODEL_PATH", "").strip()
        if override:
            return Path(override).expanduser()
    return download_directory(key) / specification(key).runtime_path


def missing_files(key: str) -> tuple[str, ...]:
    if key == "llm_grounded_answer" and os.environ.get("ODISHA_LLM_MODEL_PATH", "").strip():
        override = runtime_path(key)
        return () if override.is_file() else (str(override),)
    base = download_directory(key)
    return tuple(name for name in specification(key).required_files if not (base / name).exists())


def is_available(key: str) -> bool:
    """True when the model is enabled and every required file is on disk."""

    if nlp_mode() == "off":
        return False
    return not missing_files(key)


def iter_specifications() -> Iterator[ModelSpec]:
    yield from MANIFEST


def status() -> dict[str, dict[str, object]]:
    """Machine-readable readiness of every model, for /healthz style surfaces."""

    return {
        spec.key: {
            "repo_id": spec.repo_id,
            "licence": spec.licence,
            "purpose": spec.purpose,
            "approximate_bytes": spec.approximate_bytes,
            "path": str(runtime_path(spec.key)),
            "available": is_available(spec.key),
            "missing_files": list(missing_files(spec.key)),
            "mode": nlp_mode(),
        }
        for spec in MANIFEST
    }
