#!/usr/bin/env python3
"""Rewrite the IndicTrans2 CTranslate2 weights as int8 on disk.

The published checkpoints store float32 weights -- 847 MB per direction -- and
``packages/nlp/translate.py`` asks CTranslate2 for ``compute_type="int8"``, so
the full-precision tensors are read and quantised *during* load and then thrown
away.  The resident engine is 367 MB either way; the difference is the transient.
Loading one direction from the float32 file touches 856 MB to get there, and a
container capped below that is OOM-killed before it ever reaches steady state.

Quantising once, offline, removes the transient:

    disk        847 MB  ->  214 MB   per direction
    load peak   856 MB  ->  464 MB
    load time   0.77 s  ->  0.27 s
    steady RSS  367 MB  ->  367 MB   (unchanged -- it was already int8 in memory)

This utility is retained for translation-only deployments with strict memory
caps. The complete public deployment uses the original weights on a 12 GB host.

    uv run python scripts/quantize_translation_models.py --check
    uv run python scripts/quantize_translation_models.py --output models-int8
    uv run python scripts/quantize_translation_models.py --in-place

CTranslate2 has no API for this.  Its converters all ingest a *foreign*
checkpoint (Fairseq, Marian, Transformers) and call ``ModelSpec.optimize()``
before writing; nothing in the package reads back a ``model.bin``.  Re-converting
from the original IndicTrans2 checkpoint would mean downloading it again, so this
reads and rewrites the container format directly.  The quantisation arithmetic
below is not a reimplementation of the idea -- it is the exact branch from
``ctranslate2.specs.model_spec.LayerSpec._quantize``, and ``--check`` proves the
result is right by comparing beam scores, which are bitwise identical when the
on-disk int8 weights equal the ones CTranslate2 would have computed itself.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path
from typing import BinaryIO, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.nlp import models  # noqa: E402

#: Container revision this script understands. CTranslate2 bumps it when the
#: layout changes; refusing an unknown version is safer than writing a file that
#: loads but is subtly wrong.
BINARY_VERSION = 6
#: Wire order of the dtype tag. The index is what lands in the file.
TYPE_IDS = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
NUMPY_DTYPES: dict[str, type[np.generic]] = {
    "float32": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "float16": np.float16,
}
TRANSLATION_KEYS = ("translate_en_indic", "translate_indic_en")
SIDECAR_FILES = ("config.json", "source_vocabulary.json", "target_vocabulary.json")


class Model(NamedTuple):
    spec_name: str
    revision: int
    variables: list[tuple[str, np.ndarray]]
    aliases: list[tuple[str, str]]


def _read_string(handle: BinaryIO) -> str:
    (length,) = struct.unpack("H", handle.read(2))
    return handle.read(length)[:-1].decode("utf-8")


def _write_string(handle: BinaryIO, value: str) -> None:
    handle.write(struct.pack("H", len(value) + 1))
    handle.write(value.encode("utf-8"))
    handle.write(struct.pack("B", 0))


def read_model(path: Path) -> Model:
    with path.open("rb") as handle:
        (version,) = struct.unpack("I", handle.read(4))
        if version != BINARY_VERSION:
            raise SystemExit(
                f"{path} is CTranslate2 binary version {version}; this script "
                f"only understands {BINARY_VERSION}. Re-check against the "
                "installed ctranslate2 before touching the weights."
            )
        spec_name = _read_string(handle)
        (revision,) = struct.unpack("I", handle.read(4))
        (count,) = struct.unpack("I", handle.read(4))
        variables: list[tuple[str, np.ndarray]] = []
        for _ in range(count):
            name = _read_string(handle)
            (rank,) = struct.unpack("B", handle.read(1))
            shape = [struct.unpack("I", handle.read(4))[0] for _ in range(rank)]
            (type_id,) = struct.unpack("B", handle.read(1))
            (num_bytes,) = struct.unpack("I", handle.read(4))
            dtype_name = TYPE_IDS[type_id]
            payload = handle.read(num_bytes)
            if dtype_name not in NUMPY_DTYPES:
                raise SystemExit(f"{name}: dtype {dtype_name} has no numpy equivalent")
            variables.append(
                (name, np.frombuffer(payload, dtype=NUMPY_DTYPES[dtype_name]).reshape(shape))
            )
        (alias_count,) = struct.unpack("I", handle.read(4))
        aliases = [(_read_string(handle), _read_string(handle)) for _ in range(alias_count)]
        if handle.read():
            raise SystemExit(f"{path}: unexpected trailing bytes")
    return Model(spec_name, revision, variables, aliases)


def write_model(path: Path, model: Model) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("I", BINARY_VERSION))
        _write_string(handle, model.spec_name)
        handle.write(struct.pack("I", model.revision))
        handle.write(struct.pack("I", len(model.variables)))
        for name, array in model.variables:
            _write_string(handle, name)
            handle.write(struct.pack("B", len(array.shape)))
            for dimension in array.shape:
                handle.write(struct.pack("I", dimension))
            handle.write(struct.pack("B", TYPE_IDS.index(array.dtype.name)))
            handle.write(struct.pack("I", array.nbytes))
            handle.write(array.tobytes())
        handle.write(struct.pack("I", len(model.aliases)))
        for alias, target in model.aliases:
            _write_string(handle, alias)
            _write_string(handle, target)


def quantize(model: Model) -> tuple[Model, int]:
    """Per-output-row symmetric int8, byte-for-byte as CTranslate2 does it.

    A variable is quantisable exactly when its spec declares a ``_scale``
    companion, which in this architecture means the ``weight`` leaf of a linear,
    convolution or embedding layer. Quantising anything else -- layer-norm gains,
    biases, positional tables -- would produce a file CTranslate2 loads and then
    computes wrongly from, so the test is deliberately narrow.
    """

    output: list[tuple[str, np.ndarray]] = []
    quantized = 0
    for name, array in model.variables:
        leaf = name.split("/")[-1]
        if leaf != "weight" or array.dtype.name != "float32" or array.ndim < 2:
            output.append((name, array))
            continue
        value = array.astype(np.float32, copy=True)
        original_shape = value.shape if value.ndim == 3 else None
        if original_shape is not None:
            value = value.reshape(value.shape[0], -1)
        amax = np.amax(np.absolute(value), axis=1)
        # A row of exact zeros has no scale; CTranslate2 substitutes 127 so the
        # reciprocal is 1 and the row round-trips as zeros rather than NaN.
        amax[amax == 0] = 127.0
        scale = (127.0 / amax).astype(np.float32)
        value *= np.expand_dims(scale, 1)
        weights = np.rint(value).astype(np.int8)
        if original_shape is not None:
            weights = weights.reshape(original_shape)
        output.append((name, weights))
        output.append((f"{name}_scale", scale))
        quantized += 1
    # CTranslate2 serialises variables sorted by name; a different order loads
    # but pairs weights with the wrong scales.
    output.sort(key=lambda item: item[0])
    return Model(model.spec_name, model.revision, output, model.aliases), quantized


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def convert(source: Path, destination: Path) -> tuple[int, int, int]:
    model = read_model(source / "model.bin")
    # Not "does any tensor have dtype int8" -- the container stores enum-valued
    # config flags (activation, rotary_interleave, alibi) as int8 too, and 84 of
    # them are present in a stock float32 checkpoint. The unambiguous marker of a
    # quantised file is the presence of the "_scale" companions this script emits.
    if any(name.endswith("_scale") for name, _ in model.variables):
        raise SystemExit(f"{source} is already int8; nothing to do")
    before = _directory_bytes(source)
    quantized_model, count = quantize(model)
    destination.mkdir(parents=True, exist_ok=True)
    write_model(destination / "model.bin", quantized_model)
    for name in SIDECAR_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    if (source / "vocab").is_dir():
        shutil.copytree(source / "vocab", destination / "vocab", dirs_exist_ok=True)
    return count, before, _directory_bytes(destination)


def check() -> int:
    """Translate through whatever is on disk and report the engine and result."""

    from packages.nlp import translate

    probes = (
        ("Dengue cases are rising in Cuttack district this week.", "en", "or"),
        ("Kalahandi reported 14,399 cases at an API of 8.08.", "en", "hi"),
        ("ଆସନ୍ତା ମାସରେ ଖୋର୍ଦ୍ଧାରେ ମ୍ୟାଲେରିଆ ବଢ଼ିପାରେ ।", "or", "en"),
    )
    failures = 0
    for key in TRANSLATION_KEYS:
        directory = models.runtime_path(key)
        weights = directory / "model.bin"
        state = "missing"
        if weights.is_file():
            model = read_model(weights)
            quantised = any(name.endswith("_scale") for name, _ in model.variables)
            state = "int8" if quantised else "float32"
            print(f"{key:22s} {state:8s} {weights.stat().st_size / 1e6:8.1f} MB  {directory}")
        else:
            print(f"{key:22s} {state}")
            failures += 1
    print()
    for text, source, target in probes:
        result = translate.translate(text, source, target)
        marker = "ok " if result.translated else "FAIL"
        print(f"  [{marker}] {source}->{target} ({result.state}) {result.text}")
        failures += 0 if result.translated else 1
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        help="write the int8 tree here, mirroring the models/ layout",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace the weights under models/ (the originals are re-downloadable "
        "with scripts/fetch_models.py, so no backup is kept)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is on disk and translate through it; no writes",
    )
    arguments = parser.parse_args()

    if arguments.check:
        return check()
    if not arguments.output and not arguments.in_place:
        parser.error("pass --output DIR, --in-place, or --check")

    total_before = total_after = 0
    for key in TRANSLATION_KEYS:
        source = models.runtime_path(key)
        if not (source / "model.bin").is_file():
            raise SystemExit(f"{key}: {source / 'model.bin'} is not present; run make models")
        if arguments.in_place:
            staging = source.with_name(source.name + ".int8")
            count, before, after = convert(source, staging)
            shutil.rmtree(source)
            staging.rename(source)
            destination = source
        else:
            relative = source.relative_to(models.models_directory())
            destination = arguments.output / relative
            count, before, after = convert(source, destination)
        total_before += before
        total_after += after
        print(
            f"{key:22s} {count:3d} tensors  "
            f"{before / 1e6:8.1f} MB -> {after / 1e6:7.1f} MB  {destination}"
        )
    print(
        f"\ntotal {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB "
        f"({total_before / max(total_after, 1):.2f}x smaller)"
    )
    print("Verify with: uv run python scripts/quantize_translation_models.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
