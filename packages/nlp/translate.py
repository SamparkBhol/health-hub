"""English <-> Hindi <-> Odia machine translation on CPU.

The engine is IndicTrans2 (distilled 200M) converted to CTranslate2 and served
int8 on CPU.  Two facts about IndicTrans2 drive the whole module:

* the *indic-en* direction expects the source already normalised into the
  Devanagari script, so Odia input is transliterated ``or -> hi`` first;
* the *en-indic* direction emits Odia in that same Devanagari normalisation, so
  the hypothesis is transliterated back ``hi -> or`` before it is returned.

Script conversion is a deterministic Unicode mapping from ``indic-nlp-library``;
it is not a second translation step.  Hindi <-> Odia has no direct distilled
checkpoint, so it pivots through English and says so in ``engine``.

Nothing here downloads a model.  If the artefacts are absent the call returns a
typed ``translation_unavailable_source_language_only`` result and the caller
keeps the source-language string, which is the documented platform behaviour.
"""

from __future__ import annotations

import gc
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import models

LanguageCode = Literal["en", "hi", "or"]
SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en", "hi", "or")

TranslationState = Literal[
    "translated",
    "identity",
    "translation_unavailable_source_language_only",
    "unsupported_language_pair",
    "empty_input",
    "translation_rejected_corrupt_output",
]

_FLORES_TAG: dict[str, str] = {"en": "eng_Latn", "hi": "hin_Deva", "or": "ory_Orya"}

# Odia YYA is a Unicode composition exclusion, so NFC will not compose the
# nukta pair the transliterator emits.  Map it explicitly to the letter Odia
# readers actually expect (ସ୍ୱାସ୍ଥ୍ଯ଼ -> ସ୍ୱାସ୍ଥ୍ୟ).
_ODIA_NUKTA_REPAIRS = (("ଯ଼", "ୟ"),)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])[\s ]+")
_MAXIMUM_SENTENCE_CHARACTERS = 400
# `XX<n>XX` is the one sentinel shape that survived both decoders verbatim in a
# probe of four candidates, so a protected proper noun rides through translation
# untouched and is restored from the gazetteer afterwards.
#
# The decoder does not always return the sentinel cleanly: it fuses a stray
# symbol onto the closing marker ("XX1XX®"), and restoring only the marker left
# that symbol welded to the proper noun -- the served Odia headline named the
# district as "କନ୍ଧମାଳ®".  Trailing symbol junk is therefore absorbed into the
# sentinel match.  Sentence punctuation is deliberately *not* absorbed: a danda,
# comma or full stop after a district name is real text, and eating it would
# silently damage the sentence to hide a cosmetic defect.
# The junk run must end the token: "XX1XX°C" is a protected number carrying a
# real unit, so nothing is absorbed there, while "XX1XX®" before a space is.
_SENTINEL_TRAILING_JUNK = r"[^\w\s.,;:!?()\[\]{}'\"/%…।॥–—‘’“”-]"
_SENTINEL_PATTERN = re.compile(
    rf"X\s*X\s*(\d{{1,3}})\s*X\s*X(?:{_SENTINEL_TRAILING_JUNK})*(?!\w)",
    re.IGNORECASE,
)
# A decoder that loops on a sentinel emits a long run of Xs.  That output is
# rejected rather than shipped as a translation.
_SENTINEL_CORRUPTION = re.compile(r"X{6,}", re.IGNORECASE)
# ... and a decoder that *reads* the sentinel aloud spells it in the target
# script ("ଏକ୍ସ", "एक्स"), which is just as unusable.
_SENTINEL_SPELLED_OUT = ("ଏକ୍ସ", "एक्स", "ଏକ୍ସ୍", "एक्स्")
_LATIN_TERM = re.compile(r"^[A-Za-z][A-Za-z '\-]*$")
_ODIA_RANGE = range(0x0B00, 0x0B80)
_DEVANAGARI_RANGE = range(0x0900, 0x0980)

_ENGINE_LOCK = threading.Lock()
_ENGINES: dict[str, _CtranslateEngine] = {}


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """A translation attempt and the typed reason it produced this text."""

    text: str
    source_language: str
    target_language: str
    state: TranslationState
    engine: str
    reason_code: str | None = None
    #: Glossary terms that did not survive the decoder and could not be restored.
    unresolved_terms: tuple[str, ...] = ()

    @property
    def translated(self) -> bool:
        return self.state in {"translated", "identity"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "state": self.state,
            "engine": self.engine,
            "reason_code": self.reason_code,
            "unresolved_terms": list(self.unresolved_terms),
        }


class _CtranslateEngine:
    """A loaded CTranslate2 translator plus its two SentencePiece vocabularies."""

    def __init__(self, key: str) -> None:
        import ctranslate2  # noqa: PLC0415 - optional heavy dependency
        import sentencepiece  # noqa: PLC0415 - optional heavy dependency

        directory = models.runtime_path(key)
        threads = models.inference_threads()
        self.key = key
        self.translator = ctranslate2.Translator(
            str(directory),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=threads,
        )
        self.source_pieces = sentencepiece.SentencePieceProcessor(
            str(directory / "vocab" / "model.SRC")
        )
        self.target_pieces = sentencepiece.SentencePieceProcessor(
            str(directory / "vocab" / "model.TGT")
        )

    def translate(
        self, sentences: Sequence[str], *, source_tag: str, target_tag: str
    ) -> list[str]:
        batch = [
            [source_tag, target_tag] + self.source_pieces.encode(sentence, out_type=str)
            for sentence in sentences
        ]
        results = self.translator.translate_batch(
            batch,
            beam_size=4,
            max_batch_size=8,
            max_decoding_length=256,
            replace_unknowns=True,
        )
        return [
            self.target_pieces.decode(result.hypotheses[0]).strip() for result in results
        ]


def _release_freed_arenas() -> None:
    """Return the loader's freed heap to the OS.

    The checked-in CTranslate2 weights are float32 and ``compute_type="int8"``
    quantises them while loading, so each engine briefly allocates the full
    precision tensors and then frees them.  glibc keeps those arenas for reuse,
    which leaves roughly 300 MB of the process resident but unused after both
    directions are loaded -- measured 923 MB before this call and 624 MB after.
    ``malloc_trim`` is glibc-only and advisory; failing to find it is not an
    error, it just means the memory stays with the allocator.
    """

    try:
        import ctypes  # noqa: PLC0415 - only needed on the model-loading path

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # musl, or a libc without malloc_trim
        return


def _resident_engine_limit() -> int:
    """How many translation directions may stay loaded at once.

    Two, so a conversation that goes Odia in and Odia out never pays a reload.
    A memory-capped host sets ``ODISHA_TRANSLATION_RESIDENT=1``: loading the
    second direction while the first is resident peaks at roughly 1.2 GB, but
    evicting first peaks at roughly 0.9 GB, which is the difference between
    fitting a 1 GB container and being OOM-killed by it.  The cost is a reload
    of about one second whenever the direction flips.
    """

    import os  # noqa: PLC0415 - read per call so a test can flip it

    raw = os.environ.get("ODISHA_TRANSLATION_RESIDENT", "").strip()
    if not raw.isdigit() or int(raw) < 1:
        return 2
    return int(raw)


def _engine(key: str) -> _CtranslateEngine | None:
    """Load a translator once per process; loading takes seconds, requests do not."""

    if not models.is_available(key):
        return None
    with _ENGINE_LOCK:
        engine = _ENGINES.get(key)
        if engine is not None:
            return engine
        # Evict before allocating, never after: the point of the limit is to keep
        # the old and new weights from being resident simultaneously, and freeing
        # afterwards would already have hit the peak that kills the container.
        while len(_ENGINES) >= _resident_engine_limit():
            _ENGINES.pop(next(iter(_ENGINES)))
            gc.collect()
            _release_freed_arenas()
        engine = _CtranslateEngine(key)
        _ENGINES[key] = engine
        _release_freed_arenas()
        return engine


def reset_engines() -> None:
    """Drop the cached translators (used by tests that flip model availability)."""

    with _ENGINE_LOCK:
        _ENGINES.clear()


def available() -> bool:
    return (
        models.runtime_importable("ctranslate2", "sentencepiece", "indicnlp")
        and models.is_available("translate_en_indic")
        and models.is_available("translate_indic_en")
    )


def status() -> dict[str, Any]:
    return {
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "available": available(),
        "loaded_engines": sorted(_ENGINES),
        "models": {
            key: models.status()[key]
            for key in ("translate_en_indic", "translate_indic_en")
        },
    }


def detect_language(text: str) -> str:
    """Script-proportional detection reusing the ingestion router (en/hi/or/mixed/und)."""

    from workers.ingestion.language import route_unicode  # noqa: PLC0415 - avoid cycle

    return str(route_unicode(text).value)


def _script_counts(text: str) -> tuple[int, int]:
    odia = sum(1 for character in text if ord(character) in _ODIA_RANGE)
    devanagari = sum(1 for character in text if ord(character) in _DEVANAGARI_RANGE)
    return odia, devanagari


def to_odia_script(text: str) -> str:
    """Convert Devanagari-normalised IndicTrans2 Odia output into Odia script."""

    odia, devanagari = _script_counts(text)
    if devanagari == 0 or odia > devanagari:
        return text
    from indicnlp.transliterate.unicode_transliterate import (  # noqa: PLC0415
        UnicodeIndicTransliterator,
    )

    converted = str(UnicodeIndicTransliterator.transliterate(text, "hi", "or"))
    for source, replacement in _ODIA_NUKTA_REPAIRS:
        converted = converted.replace(source, replacement)
    return unicodedata.normalize("NFC", converted)


def to_devanagari_script(text: str) -> str:
    """Normalise Odia source text into Devanagari, as IndicTrans2 indic-en expects."""

    odia, devanagari = _script_counts(text)
    if odia == 0:
        return text
    from indicnlp.transliterate.unicode_transliterate import (  # noqa: PLC0415
        UnicodeIndicTransliterator,
    )

    return str(UnicodeIndicTransliterator.transliterate(text, "or", "hi"))


def split_sentences(text: str) -> list[str]:
    """Split on Latin and Indic sentence terminators, then hard-cap long fragments."""

    pieces: list[str] = []
    for candidate in _SENTENCE_BOUNDARY.split(text.strip()):
        sentence = candidate.strip()
        if not sentence:
            continue
        while len(sentence) > _MAXIMUM_SENTENCE_CHARACTERS:
            window = sentence[:_MAXIMUM_SENTENCE_CHARACTERS]
            cut = window.rfind(" ")
            if cut <= 0:
                cut = _MAXIMUM_SENTENCE_CHARACTERS
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _substitute_outside_sentinels(
    pattern: re.Pattern[str], replacement: str, text: str
) -> str:
    """Apply ``pattern`` only to the parts of ``text`` that are not already a sentinel.

    Numeric literals are protected with the same machinery as district names and
    are matched without a word boundary, so an unguarded pass rewrote the digits
    *inside* an earlier sentinel: "Kandhamal" became ``XX1XX``, then the literal
    "1" turned that into ``XXXX12XXXX``.  Both protections were then lost in
    decoding, which is how a protected district name reached the reader mangled.
    """

    pieces: list[str] = []
    position = 0
    for sentinel in _SENTINEL_PATTERN.finditer(text):
        pieces.append(pattern.sub(replacement, text[position : sentinel.start()]))
        pieces.append(sentinel.group(0))
        position = sentinel.end()
    pieces.append(pattern.sub(replacement, text[position:]))
    return "".join(pieces)


def protect_terms(text: str, glossary: Mapping[str, str]) -> tuple[str, dict[int, str]]:
    """Replace glossary keys with inert sentinels; return the map to restore them."""

    if not glossary:
        return text, {}
    ordered = sorted(glossary, key=len, reverse=True)
    replacements: dict[int, str] = {}
    protected = text
    for term in ordered:
        if not term.strip():
            continue
        pattern = (
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if _LATIN_TERM.match(term)
            else re.compile(re.escape(term))
        )
        index = len(replacements) + 1
        if index > 99:  # pragma: no cover - defensive cap
            break
        candidate = _substitute_outside_sentinels(pattern, f"XX{index}XX", protected)
        if candidate == protected:
            continue
        replacements[index] = glossary[term]
        protected = candidate
    return protected, replacements


def restore_terms(text: str, replacements: Mapping[int, str]) -> tuple[str, tuple[str, ...]]:
    """Put protected terms back, reporting any sentinel the decoder swallowed."""

    if not replacements:
        return text, ()
    seen: set[int] = set()

    def _substitute(match: re.Match[str]) -> str:
        index = int(match.group(1))
        seen.add(index)
        return replacements.get(index, match.group(0))

    restored = _SENTINEL_PATTERN.sub(_substitute, text)
    missing = tuple(
        replacements[index] for index in sorted(replacements) if index not in seen
    )
    return restored, missing


def _sentinel_damaged(text: str) -> bool:
    """True when the decoder looped on or transliterated a protection sentinel."""

    if _SENTINEL_CORRUPTION.search(text):
        return True
    return any(marker in text for marker in _SENTINEL_SPELLED_OUT)


def _run(sentences: list[str], *, source: str, target: str) -> list[str] | None:
    """Translate a prepared sentence list, pivoting through English for hi<->or."""

    if source == "en" or target == "en":
        return _translate_direct(sentences, source_language=source, target_language=target)
    english = _translate_direct(sentences, source_language=source, target_language="en")
    if english is None:
        return None
    return _translate_direct(english, source_language="en", target_language=target)


def _unavailable(text: str, source: str, target: str) -> TranslationResult:
    return TranslationResult(
        text=text,
        source_language=source,
        target_language=target,
        state="translation_unavailable_source_language_only",
        engine="none",
        reason_code="TRANSLATION_MODEL_NOT_DOWNLOADED",
    )


def _translate_direct(
    sentences: list[str], *, source_language: str, target_language: str
) -> list[str] | None:
    """One model hop: en->{hi,or} or {hi,or}->en."""

    if source_language == "en":
        engine = _engine("translate_en_indic")
    else:
        engine = _engine("translate_indic_en")
    if engine is None:
        return None
    prepared = (
        [to_devanagari_script(sentence) for sentence in sentences]
        if source_language == "or"
        else list(sentences)
    )
    hypotheses = engine.translate(
        prepared,
        source_tag=_FLORES_TAG[source_language],
        target_tag=_FLORES_TAG[target_language],
    )
    if target_language == "or":
        hypotheses = [to_odia_script(hypothesis) for hypothesis in hypotheses]
    return hypotheses


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    *,
    glossary: Mapping[str, str] | None = None,
    protect_districts: bool = True,
) -> TranslationResult:
    """Translate ``text`` between en/hi/or, pivoting hi<->or through English.

    District names are protected by default: neural MT otherwise renames rare
    Indian proper nouns, and a district-level health answer that renames the
    district is worse than useless.
    """

    source = (source_lang or "").strip().lower()
    target = (target_lang or "").strip().lower()
    if source in {"", "auto", "und", "mixed"}:
        detected = detect_language(text)
        source = detected if detected in SUPPORTED_LANGUAGES else "en"
    if source not in SUPPORTED_LANGUAGES or target not in SUPPORTED_LANGUAGES:
        return TranslationResult(
            text=text,
            source_language=source,
            target_language=target,
            state="unsupported_language_pair",
            engine="none",
            reason_code="LANGUAGE_PAIR_NOT_SUPPORTED",
        )
    if not text.strip():
        return TranslationResult(
            text=text,
            source_language=source,
            target_language=target,
            state="empty_input",
            engine="none",
        )
    if source == target:
        return TranslationResult(
            text=text,
            source_language=source,
            target_language=target,
            state="identity",
            engine="none",
        )
    if not available():
        return _unavailable(text, source, target)

    # Odia and Devanagari digits are rewritten by the decoder rather than carried
    # through -- live testing produced 11,146 -> 11,986 and dropped 789 entirely.
    # Fold them to ASCII first, then protect every numeric literal with the same
    # sentinel machinery used for district names, so a case count survives verbatim.
    text = _ascii_digits(text)
    terms: dict[str, str] = {}
    if protect_districts:
        from .glossary import district_terms  # noqa: PLC0415 - avoid import cycle

        terms.update(district_terms(source, target))
    if glossary:
        terms.update(glossary)
    terms.update({literal: literal for literal in _NUMERIC_LITERAL.findall(text)})
    protected, replacements = protect_terms(text, terms)
    engine = (
        f"indictrans2-ct2-int8:{source}->{target}"
        if source == "en" or target == "en"
        # hi <-> or: pivot through English, which is the only pair the distilled
        # checkpoints cover.  The pivot is named in `engine` so a reviewer can
        # see that two hops of error are possible.
        else f"indictrans2-ct2-int8:{source}->en->{target}"
    )

    hypotheses = _run(split_sentences(protected), source=source, target=target)
    if hypotheses is None:
        return _unavailable(text, source, target)
    restored, unresolved = restore_terms(" ".join(hypotheses).strip(), replacements)

    if replacements and (unresolved or _sentinel_damaged(restored)):
        # A sentinel was dropped or spelled out by the decoder, so the protected
        # spelling cannot be trusted.  Retry once with no protection and record
        # that the guarantee was lost rather than shipping mangled tokens.
        plain = _run(split_sentences(text), source=source, target=target)
        if plain is not None and not _sentinel_damaged(" ".join(plain)):
            return TranslationResult(
                text=" ".join(plain).strip(),
                source_language=source,
                target_language=target,
                state="translated",
                engine=f"{engine}+unprotected_retry",
                reason_code="PROTECTED_TERM_LOST_IN_DECODING",
                unresolved_terms=tuple(replacements.values()),
            )
        return TranslationResult(
            text=text,
            source_language=source,
            target_language=target,
            state="translation_rejected_corrupt_output",
            engine=engine,
            reason_code="SENTINEL_CORRUPTION_DETECTED",
            unresolved_terms=unresolved or tuple(replacements.values()),
        )
    final = _collapse_degenerate_runs(restored)
    # A number that changed value or vanished is worse than a refusal: the caller
    # cannot see it happened, and a case count is exactly what a reviewer reads.
    # Compared as a multiset, because Odia and Hindi word order legitimately moves
    # a figure within the sentence -- only its value and presence must survive.
    if sorted(_NUMERIC_LITERAL.findall(text)) != sorted(_NUMERIC_LITERAL.findall(final)):
        return TranslationResult(
            text=text,
            source_language=source,
            target_language=target,
            state="translation_rejected_corrupt_output",
            engine=engine,
            reason_code="NUMERIC_LITERAL_ALTERED_IN_DECODING",
            unresolved_terms=tuple(_NUMERIC_LITERAL.findall(text)),
        )
    return TranslationResult(
        text=final,
        source_language=source,
        target_language=target,
        state="translated",
        engine=engine,
        unresolved_terms=unresolved,
    )


#: A decoder that loses the plot emits the same short token dozens of times. Seen
#: live on Odia output when the English answer carried underscore-heavy source
#: identifiers: the sentence started correctly then ran on with "| | | | |".
_DEGENERATE_RUN = re.compile(r"(?:(\S{1,3})(?:\s+\1){3,})")

#: A number as a reader would write it, including grouping and decimals.
_NUMERIC_LITERAL = re.compile(r"\d[\d,.]*\d|\d")

#: Odia (U+0B66-U+0B6F) and Devanagari (U+0966-U+096F) digits to ASCII.
_INDIC_DIGITS = str.maketrans(
    "".join(chr(code) for code in range(0x0B66, 0x0B70))
    + "".join(chr(code) for code in range(0x0966, 0x0970)),
    "0123456789" * 2,
)


def _ascii_digits(text: str) -> str:
    """Fold Odia and Devanagari digits to ASCII so a count survives decoding."""

    return text.translate(_INDIC_DIGITS)


def _collapse_degenerate_runs(text: str) -> str:
    """Collapse a repeated-token run down to a single occurrence.

    Truncating the tail outright would silently drop real content when the run
    sits mid-sentence, so the run is reduced rather than removed. Legitimate Indic
    text does not repeat the same one-to-three character token four times running.
    """

    collapsed = _DEGENERATE_RUN.sub(lambda match: match.group(1), text)
    return re.sub(r"\s{2,}", " ", collapsed).strip()


def translate_batch(
    texts: Sequence[str],
    source_lang: str,
    target_lang: str,
) -> list[TranslationResult]:
    """Translate many strings; identical inputs are translated once."""

    unique: dict[str, TranslationResult] = {}
    results: list[TranslationResult] = []
    for text in texts:
        cached = unique.get(text)
        if cached is None:
            cached = translate(text, source_lang, target_lang)
            unique[text] = cached
        results.append(cached)
    return results
