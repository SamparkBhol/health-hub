"""Cross-lingual semantic retrieval over collected evidence records.

``intfloat/multilingual-e5-small`` is executed with ONNX Runtime on CPU, which
keeps the dependency footprint to a ~470 MB graph and a tokenizer, with no
PyTorch.  E5 requires the ``query:`` / ``passage:`` prefixes, and it is trained
so that a question in English lands next to a passage in Odia or Hindi, which is
exactly the retrieval behaviour this platform needs.

Vectors are cached in a plain SQLite file keyed by the SHA-256 of the encoded
string, so a record is embedded once per model and never again -- no pgvector,
no external index service.  When the model is not on disk the retriever falls
back to a transparent lexical overlap score and reports that in its state.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from . import models

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
else:  # numpy ships with the optional `nlp` extra; the API must import without it.
    try:
        import numpy as np
    except ImportError:
        np = None

EMBEDDING_DIMENSION = 384
_MAXIMUM_TOKENS = 512
_MODEL_TAG = "multilingual-e5-small-onnx"

_LOCK = threading.Lock()
_ENCODER: _OnnxEncoder | None = None

RetrievalState = Literal[
    "semantic_cross_lingual",
    "lexical_fallback_model_unavailable",
    "no_records",
]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A retrievable unit of evidence: an id, the text, and its provenance row."""

    record_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedRecord:
    record: EvidenceRecord
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    ranked: tuple[RankedRecord, ...]
    state: RetrievalState
    model: str
    query_language: str
    considered: int
    #: What was actually encoded for each record.  ``state`` names the encoder;
    #: this names the corpus, because a cross-lingual encoder over strings that
    #: carry no content ranks nothing and must not be reported as if it did.
    document_basis: str = "record_text_as_supplied"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "model": self.model,
            "query_language": self.query_language,
            "considered": self.considered,
            "document_basis": self.document_basis,
            "ranked": [
                {
                    "record_id": item.record.record_id,
                    "score": round(item.score, 6),
                    "rank": item.rank,
                }
                for item in self.ranked
            ],
        }


def vector_cache_path() -> Path:
    import os  # noqa: PLC0415 - local to keep module import cheap

    override = os.environ.get("ODISHA_VECTOR_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return models.REPOSITORY_ROOT / "runtime" / "nlp_vectors.sqlite3"


class _VectorCache:
    """SQLite-backed store of normalised float32 vectors keyed by text digest."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def key(model: str, text: str) -> str:
        digest = hashlib.sha256(f"{model}\x1f{text}".encode()).hexdigest()
        return digest

    def read(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        found: dict[str, np.ndarray] = {}
        with self._connect() as connection:
            for start in range(0, len(keys), 500):
                window = keys[start : start + 500]
                placeholders = ",".join("?" * len(window))
                rows = connection.execute(
                    f"SELECT cache_key, vector FROM embedding WHERE cache_key IN ({placeholders})",  # noqa: S608 - placeholders only
                    tuple(window),
                ).fetchall()
                for cache_key, blob in rows:
                    found[cache_key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def write(self, entries: Iterable[tuple[str, np.ndarray]]) -> None:
        payload = [
            (key, _MODEL_TAG, int(vector.shape[0]), vector.astype(np.float32).tobytes())
            for key, vector in entries
        ]
        if not payload:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embedding (cache_key, model, dimension, vector)"
                " VALUES (?, ?, ?, ?)",
                payload,
            )


class _OnnxEncoder:
    def __init__(self) -> None:
        import onnxruntime  # noqa: PLC0415 - optional heavy dependency
        from tokenizers import Tokenizer  # noqa: PLC0415 - optional heavy dependency

        directory = models.runtime_path("embed_multilingual")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = models.inference_threads()
        options.inter_op_num_threads = 1
        # Spin-waiting ONNX worker threads would starve llama.cpp, which shares
        # the same CPU in this process.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=_MAXIMUM_TOKENS)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")  # noqa: S106 - tokenizer pad symbol

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(list(texts))
        identifiers = np.array([item.ids for item in encoded], dtype=np.int64)
        mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        feed: dict[str, np.ndarray] = {"input_ids": identifiers, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(identifiers)
        hidden = self.session.run(None, feed)[0]
        expanded = mask[..., None].astype(np.float32)
        pooled = (hidden * expanded).sum(axis=1) / np.clip(expanded.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)


def _encoder() -> _OnnxEncoder | None:
    global _ENCODER
    if not available():
        return None
    with _LOCK:
        if _ENCODER is None:
            _ENCODER = _OnnxEncoder()
        return _ENCODER


def reset_encoder() -> None:
    global _ENCODER
    with _LOCK:
        _ENCODER = None


def available() -> bool:
    return (
        np is not None
        and models.runtime_importable("onnxruntime", "tokenizers")
        and models.is_available("embed_multilingual")
    )


def embed(texts: Sequence[str], *, kind: str = "passage") -> np.ndarray | None:
    """Embed texts with the E5 prefix convention, using the on-disk vector cache."""

    encoder = _encoder()
    if encoder is None:
        return None
    prefix = "query: " if kind == "query" else "passage: "
    prepared = [prefix + (text or "").strip() for text in texts]
    keys = [_VectorCache.key(_MODEL_TAG, text) for text in prepared]
    # The cache is an optimisation, never a dependency: a read-only or corrupt
    # cache file degrades to plain re-encoding instead of failing the request.
    cache: _VectorCache | None
    try:
        cache = _VectorCache(vector_cache_path())
        known = cache.read(keys)
    except (OSError, sqlite3.Error):
        cache, known = None, {}
    missing = [index for index, key in enumerate(keys) if key not in known]
    if missing:
        fresh = encoder.encode([prepared[index] for index in missing])
        if cache is not None:
            with contextlib.suppress(OSError, sqlite3.Error):
                cache.write(
                    (keys[index], fresh[position]) for position, index in enumerate(missing)
                )
        for position, index in enumerate(missing):
            known[keys[index]] = fresh[position]
    return np.stack([known[key] for key in keys]).astype(np.float32)


_WORD = re.compile(r"\w+", re.UNICODE)


def _lexical_scores(query: str, records: Sequence[EvidenceRecord]) -> list[float]:
    terms = {token.casefold() for token in _WORD.findall(query)}
    scores: list[float] = []
    for record in records:
        tokens = {token.casefold() for token in _WORD.findall(record.text)}
        overlap = len(terms & tokens)
        scores.append(overlap / max(len(terms), 1))
    return scores


def rank(
    query: str,
    records: Sequence[EvidenceRecord],
    *,
    top_k: int = 5,
    minimum_score: float = 0.0,
    document_basis: str = "record_text_as_supplied",
) -> RetrievalResult:
    """Rank records against the query, cross-lingually when the model is present.

    ``document_basis`` is passed through to the result so the caller can declare
    what the embedded strings actually are; retrieval itself cannot know whether
    it was handed source text or a structured description of it.
    """

    from .translate import detect_language  # noqa: PLC0415 - avoid import cycle

    language = detect_language(query)
    if not records:
        return RetrievalResult(
            ranked=(),
            state="no_records",
            model=_MODEL_TAG if available() else "none",
            query_language=language,
            considered=0,
            document_basis=document_basis,
        )
    document_vectors = embed([record.text for record in records], kind="passage")
    query_vector = None if document_vectors is None else embed([query], kind="query")
    if document_vectors is None or query_vector is None:
        scores = _lexical_scores(query, records)
        state: RetrievalState = "lexical_fallback_model_unavailable"
        model_name = "lexical_overlap"
    else:
        scores = (document_vectors @ query_vector[0]).astype(float).tolist()
        state = "semantic_cross_lingual"
        model_name = _MODEL_TAG
    order = sorted(range(len(records)), key=lambda index: scores[index], reverse=True)
    ranked = tuple(
        RankedRecord(record=records[index], score=float(scores[index]), rank=position + 1)
        for position, index in enumerate(order[:top_k])
        if scores[index] >= minimum_score
    )
    return RetrievalResult(
        ranked=ranked,
        state=state,
        model=model_name,
        query_language=language,
        considered=len(records),
        document_basis=document_basis,
    )


def status() -> dict[str, Any]:
    return {
        "available": available(),
        "model": _MODEL_TAG,
        "dimension": EMBEDDING_DIMENSION,
        "vector_cache": str(vector_cache_path()),
        "loaded": _ENCODER is not None,
    }
