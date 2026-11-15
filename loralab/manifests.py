"""MANIFEST.json: what a mix is, in the same shape as the upstream manifests.

A manifest is the durable record. The fused safetensors file's whole-file sha256
is **not** reproducible (safetensors serialises `__metadata__` in HashMap
order), so identity lives in the converted AR/NAR sha256. The manifest carries
both, labelled, plus the exact commands that produced them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

SCHEMA = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment() -> dict:
    """Versions, without importing torch on a catalog-only machine."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("torch", "safetensors"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    out["loralab"] = __version__
    return out


def build(*, command: list[str], carrier: dict, donor: dict,
          arms: list[dict], converter: dict | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "kind": "mix",
        "tool": f"loralab {__version__}",
        "created_utc": _now(),
        "command": command,
        "carrier": carrier,
        "donor": donor,
        "converter": converter,
        "environment": environment(),
        "arms": arms,
        "identity": {
            "reproducible": "converted AR/NAR sha256",
            "not_reproducible": "fused whole-file sha256 (metadata map order)",
        },
    }


def write(path, doc: dict) -> Path:
    """Atomic: a reader never sees a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def write_sidecar(doc: dict, arm: dict, out_dir) -> Path:
    """One JSON next to each arm, so a single alpha is readable without the rest."""
    sidecar = dict(doc)
    sidecar["arms"] = [arm]
    sidecar["arm"] = {
        "alpha": arm.get("alpha"),
        "stem": arm.get("stem"),
        "manifest": "sidecar",
    }
    stem = arm.get("stem") or f"a{arm.get('alpha')}"
    return write(Path(out_dir) / f"{stem}.json", sidecar)


def read(path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
