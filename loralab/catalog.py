"""Catalog: parse safetensors headers, classify adapters, link lineage.

A safetensors file is `u64 header length` + JSON (tensor names, dtypes, shapes,
`__metadata__`); the tensor bytes follow and are never needed here. One `Range`
request per file, ~41 KB, instead of 133 MB.

Field names match the catalog produced by the project repo's `lora_toolkit.py`
on purpose — a `catalog.json` from either side is readable by the other.
"""
from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, gcs

AR_PREFIX = "text_encoders."
NAR_PREFIX = "diffusion_model."
MAX_HEADER = 100_000_000  # guards against a corrupt length prefix
CATALOG_SCHEMA = 1


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
def parse_header(read_range) -> tuple[dict, dict]:
    """`read_range(start, end)` -> (metadata, tensors). One code path for all transports."""
    raw = read_range(0, 7)
    if len(raw) != 8:
        raise ValueError(f"short length prefix ({len(raw)} bytes)")
    n = struct.unpack("<Q", raw)[0]
    if not 0 < n <= MAX_HEADER:
        raise ValueError(f"implausible header length {n}")
    doc = json.loads(read_range(8, 8 + n - 1))
    if not isinstance(doc, dict):
        raise ValueError("header is not a JSON object")
    meta = doc.pop("__metadata__", None) or {}
    return meta, doc


def header_from_gcs(bucket: str, obj: str, token: gcs.Token) -> tuple[dict, dict]:
    return parse_header(lambda s, e: gcs.read_range(bucket, obj, s, e, token))


def header_from_local(path) -> tuple[dict, dict]:
    path = Path(path)
    with open(path, "rb") as fh:
        def read(start, end):
            fh.seek(start)
            return fh.read(end - start + 1)
        return parse_header(read)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_or(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def triggers(meta: dict, top: int = 6) -> list[str]:
    """ai-toolkit `ss_tag_frequency` -> the tags the adapter was trained on."""
    groups = _json_or(meta.get("ss_tag_frequency"), {})
    if not isinstance(groups, dict):
        return []
    counts: dict[str, int] = {}
    for tags in groups.values():
        if isinstance(tags, dict):
            for tag, n in tags.items():
                counts[tag] = counts.get(tag, 0) + int(n or 0)
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def training(meta: dict) -> dict:
    info = _json_or(meta.get("training_info"), {})
    if not isinstance(info, dict):
        return {}
    return {k: info[k] for k in ("step", "epoch") if k in info}


def software(meta: dict) -> str | None:
    sw = _json_or(meta.get("software"), {})
    if not isinstance(sw, dict) or not sw.get("name"):
        return None
    return f"{sw['name']} {sw.get('version', '')}".strip()


def classify(meta: dict, tensors: dict) -> dict:
    """kind / branches / rank / dtype / tensor count, inferred from the file."""
    branches: dict[str, set[int]] = {}
    dtypes: set[str] = set()
    for key, spec in tensors.items():
        if isinstance(spec, dict) and spec.get("dtype"):
            dtypes.add(spec["dtype"])
        if ".lora_A" not in key:
            continue
        if key.startswith(AR_PREFIX):
            branch = "ar"
        elif key.startswith(NAR_PREFIX):
            branch = "nar"
        else:
            branch = "flat"  # converted audio.cpp layout: `layers.*`
        shape = spec.get("shape") or [0]
        branches.setdefault(branch, set()).add(int(shape[0]))

    if meta.get("converted_from"):
        kind = "converted"
        declared = meta.get("branch")
        if declared in ("ar", "nar"):
            branches = {declared: branches.get("flat") or branches.get(declared) or set()}
    elif meta.get("merge_tool"):
        kind = "merged"
    else:
        kind = "fused"

    rank = {b: sorted(rs) for b, rs in branches.items()}
    if kind == "converted" and meta.get("rank"):
        rank = {next(iter(rank), "ar"): [_as_int(meta["rank"])]}

    return {
        "kind": kind,
        "branches": sorted(branches),
        "rank": rank,
        "tensors": len(tensors),
        "dtype": sorted(d for d in dtypes if d),
    }


def make_entry(path: str, meta: dict, tensors: dict, *, size=None, md5=None,
               crc32c=None, updated=None) -> dict:
    """Assemble one catalog record. Pure: no I/O."""
    entry = {
        "name": Path(path).name,
        "path": path,
        "size_bytes": size,
        "md5": md5,
        "crc32c": crc32c,
        "updated": updated,
        "sha256": None,
        "local_path": None,
    }
    entry.update(classify(meta, tensors))
    entry["base_model"] = meta.get("ss_base_model_version")
    entry["output_name"] = meta.get("ss_output_name") or meta.get("name")
    entry["software"] = software(meta)
    entry["training"] = training(meta)
    entry["triggers"] = triggers(meta)

    if entry["kind"] == "converted":
        entry["source_file"] = meta.get("source_file")
    if entry["kind"] == "merged":
        entry["merge"] = {
            "tool": meta.get("merge_tool"),
            "kind": meta.get("merge_kind"),
            "alpha": _as_float(meta.get("merge_alpha")),
            "carrier_name": meta.get("merge_carrier_name"),
            "carrier_sha256": meta.get("merge_v2_sha256"),
            # `merge_quran` is the legacy field upstream tools wrote; a file
            # from either side stays readable.
            "donor_name": meta.get("merge_donor_name") or meta.get("merge_quran"),
            "donor_sha256": meta.get("merge_quran_sha256"),
            "out_ranks": {
                "ar": _as_int(meta.get("merge_ar_rank")),
                "nar": _as_int(meta.get("merge_nar_rank")),
            },
        }
    return entry


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def resolve_lineage(entries: list[dict]) -> None:
    """converted -> fused by `source_file`; merged -> carrier by sha, only if known."""
    fused_by_name = {e["name"]: e["path"] for e in entries
                     if e["kind"] in ("fused", "merged")}
    by_sha = {e["sha256"]: e["path"] for e in entries if e.get("sha256")}
    for entry in entries:
        lineage = {}
        if entry["kind"] == "converted" and entry.get("source_file"):
            lineage["fused_path"] = fused_by_name.get(entry["source_file"])
            lineage["resolved"] = lineage["fused_path"] is not None
        elif entry["kind"] == "merged" and entry.get("merge"):
            merge = entry["merge"]
            lineage["carrier_path"] = by_sha.get(merge.get("carrier_sha256"))
            lineage["donor_path"] = by_sha.get(merge.get("donor_sha256"))
            lineage["resolved"] = bool(lineage["carrier_path"] and lineage["donor_path"])
        if lineage:
            entry["lineage"] = lineage


def scan(cfg, token: gcs.Token, *, log=None, cancelled=None) -> dict:
    """Sweep every configured prefix. One bad file never sinks the run."""
    log = log or (lambda *_: None)
    entries, errors = [], []
    for rel in cfg.prefixes:
        if cancelled and cancelled():
            raise ScanCancelled()
        prefix = cfg.object_prefix(rel)
        try:
            objects = gcs.list_objects(cfg.bucket, prefix, token=token)
        except gcs.GcsError as exc:
            errors.append({"path": prefix, "error": str(exc)})
            log(f"  {rel:<32} listing failed: {exc}")
            continue
        log(f"  {rel:<32} {len(objects):>3} file(s)")
        for obj in objects:
            if cancelled and cancelled():
                raise ScanCancelled()
            rel_path = obj["name"][len(cfg.root) + 1:] if cfg.root else obj["name"]
            try:
                meta, tensors = header_from_gcs(cfg.bucket, obj["name"], token)
            except Exception as exc:  # noqa: BLE001 — one file must not stop the sweep
                errors.append({"path": rel_path, "error": f"{type(exc).__name__}: {exc}"})
                continue
            entries.append(make_entry(
                rel_path, meta, tensors,
                size=_as_int(obj.get("size")),
                md5=obj.get("md5Hash"),
                crc32c=obj.get("crc32c"),
                updated=obj.get("updated"),
            ))
    resolve_lineage(entries)
    return {
        "schema": CATALOG_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": f"loralab {__version__}",
        "scope": cfg.display_scope(),
        "prefixes": list(cfg.prefixes),
        "count": len(entries),
        "entries": entries,
        "errors": errors,
    }


class ScanCancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# storage / views
# ---------------------------------------------------------------------------
def save_catalog(path: Path, doc: dict) -> None:
    """Atomic: a reader never sees a half-written catalog."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def load_catalog(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def kinds(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    return dict(sorted(counts.items()))


def rank_label(entry: dict) -> str:
    parts = []
    for branch in entry.get("branches", []):
        ranks = (entry.get("rank") or {}).get(branch) or []
        parts.append(f"{branch}:{'+'.join(str(r) for r in ranks) or '?'}")
    return ", ".join(parts)


def note(entry: dict) -> str:
    if entry["kind"] == "merged":
        merge = entry.get("merge") or {}
        tool = (merge.get("tool") or "").replace(".py", "")
        return f"α={merge.get('alpha')} {tool}".strip()
    if entry["kind"] == "converted":
        return f"from {entry.get('source_file') or '?'}"
    tags = entry.get("triggers") or []
    return tags[0] if tags else (entry.get("output_name") or "")


def group(entries: list[dict]) -> list[dict]:
    """Group by the first two path segments, like the CLI table does."""
    buckets: dict[str, list[dict]] = {}
    for entry in entries:
        parts = entry["path"].split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        buckets.setdefault(key, []).append(entry)
    return [
        {"key": key, "entries": sorted(buckets[key], key=lambda e: e["name"])}
        for key in sorted(buckets)
    ]


def filter_entries(entries: list[dict], *, q: str = "", kind: str = "",
                   prefix: str = "") -> list[dict]:
    q = (q or "").strip().lower()
    out = []
    for entry in entries:
        if kind and entry["kind"] != kind:
            continue
        if prefix and not entry["path"].startswith(prefix):
            continue
        if q:
            haystack = " ".join([
                entry.get("name") or "", entry.get("path") or "",
                entry.get("output_name") or "", " ".join(entry.get("triggers") or []),
            ]).lower()
            if q not in haystack:
                continue
        out.append(entry)
    return out


def prefix_options(entries: list[dict]) -> list[str]:
    seen = set()
    for entry in entries:
        parts = entry["path"].split("/")
        if len(parts) > 2:
            seen.add("/".join(parts[:2]) + "/")
    return sorted(seen)
