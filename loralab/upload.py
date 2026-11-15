"""Upload a finished mix to GCS: preview first, create-only by default.

This is the only code in the app that writes to a bucket, and it is deliberately
boring:

* the destination is a bucket-relative prefix, resolved once and shown as full
  object names before anything is sent;
* `loras/` is refused unless the caller passes `--allow-production-prefix`;
* an existing key is a `412` unless `--overwrite` was explicit, and overwrite
  still pins the generation the preview saw;
* there is no delete path. The recommended IAM grant has no delete permission.

The manifest goes **last**, after every file it names, so a reader never sees a
manifest for a half-finished upload.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, gcs, manifests, merge

KIND_ORDER = {"converted": 0, "fused": 1, "sidecar": 2, "manifest": 3}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(RuntimeError):
    """A destination/preview problem with a message the user can act on."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stem(name: str) -> str:
    stem = Path(str(name or "")).name
    for suffix in (".safetensors", ".sft"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _safe_slug(value: str) -> str:
    slug = _SAFE.sub("-", value).strip("-.")
    return slug or "mix"


def normalize_prefix(prefix: str) -> str:
    """A bucket-relative prefix: no leading slash, no `.`/`..`, not empty."""
    raw = str(prefix or "").strip().replace("\\", "/")
    if not raw:
        raise UploadError(
            "the upload prefix is empty. Set `upload_prefix` in the config "
            "(default: _prototypes/mixes/), or pass --prefix."
        )
    if raw.startswith("/"):
        raise UploadError(
            f"upload prefix {prefix!r} is absolute. Prefixes are bucket-relative "
            "(the configured root is applied automatically)."
        )
    raw = raw.strip("/")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UploadError(
            f"upload prefix {prefix!r} contains an empty, '.' or '..' segment. "
            "Name a directory inside the bucket, not a path traversal."
        )
    return "/".join(parts)


def normalize_name(name: str) -> str:
    """A directory name under the prefix; allows nested relative paths."""
    raw = str(name or "").strip().replace("\\", "/")
    if not raw:
        raise UploadError("give the mix a destination directory name")
    if raw.startswith("/"):
        raise UploadError(
            f"destination name {name!r} is absolute; it is always relative to "
            "the upload prefix"
        )
    if any(part in ("", ".", "..") for part in raw.split("/")):
        raise UploadError(
            f"destination name {name!r} must be a relative path with no empty, "
            "'.' or '..' segments"
        )
    return raw


def is_production_prefix(prefix: str) -> bool:
    """True when any destination segment is the production `loras` library."""
    return any(part.lower() == "loras" for part in prefix.split("/"))


def mix_slug(manifest: dict) -> str:
    """The directory name for a mix: `carrier__donor`, or its explicit stem."""
    command = manifest.get("command") or []
    if "--stem" in command:
        try:
            stem = command[command.index("--stem") + 1]
        except IndexError:
            stem = ""
        if stem:
            return _safe_slug(_stem(stem))
    carrier = _stem((manifest.get("carrier") or {}).get("name") or "")
    donor = _stem((manifest.get("donor") or {}).get("name") or "")
    if carrier and donor:
        return _safe_slug(f"{carrier}__{donor}")
    if carrier:
        return _safe_slug(carrier)
    # Last resort: the first arm's stem, stripped of its alpha suffix.
    for arm in manifest.get("arms") or []:
        stem = _stem(arm.get("stem") or "")
        if stem:
            return _safe_slug(re.sub(r"__?a[0-9.]+$", "", stem))
    return "mix"


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------
def _source_path(item: dict, base: Path) -> Path:
    raw = item.get("path")
    if not raw:
        raise UploadError(f"manifest entry {item.get('name')!r} has no path")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path)


def _artifact(item: dict, kind: str, alpha, base: Path) -> dict:
    path = _source_path(item, base)
    if not path.is_file():
        raise UploadError(
            f"{kind} artifact for alpha {alpha} is missing: {path}. "
            "Rebuild the mix before uploading it."
        )
    return {
        "source": str(path),
        "name": item.get("name") or path.name,
        "alpha": alpha,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": item.get("sha256") or merge.sha256_file(path),
    }


def _collect(manifest_path: Path, manifest: dict, *, include_fused: bool) -> list[dict]:
    base = manifest_path.parent
    arms = manifest.get("arms") or []
    if not arms:
        raise UploadError("the manifest has no arms; nothing to upload")
    objects: list[dict] = []
    for arm in arms:
        alpha = arm.get("alpha")
        converted = arm.get("converted") or {}
        if converted.get("ar") and converted.get("nar"):
            objects.append(_artifact(converted["ar"], "converted", alpha, base))
            objects.append(_artifact(converted["nar"], "converted", alpha, base))
            if include_fused and arm.get("fused"):
                objects.append(_artifact(arm["fused"], "fused", alpha, base))
        elif arm.get("fused"):
            # --no-convert builds are still useful to preserve.
            objects.append(_artifact(arm["fused"], "fused", alpha, base))
        else:
            raise UploadError(
                f"alpha {alpha}: neither a converted pair nor a fused file is recorded"
            )
        out_dir = Path(str(arm.get("out_dir") or base)).expanduser()
        if not out_dir.is_absolute():
            out_dir = base / out_dir
        sidecar = out_dir / f"{arm.get('stem') or f'a{alpha}'}.json"
        if not sidecar.is_file():
            sidecar = base / sidecar.name
        if sidecar.is_file():
            objects.append({
                "source": str(sidecar),
                "name": sidecar.name,
                "alpha": alpha,
                "kind": "sidecar",
                "size_bytes": sidecar.stat().st_size,
                "sha256": merge.sha256_file(sidecar),
            })
    objects.append({
        "source": str(manifest_path),
        "name": manifest_path.name,
        "alpha": None,
        "kind": "manifest",
        "size_bytes": manifest_path.stat().st_size,
        "sha256": merge.sha256_file(manifest_path),
    })
    names = [o["name"] for o in objects]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise UploadError(
            "two artifacts would land at the same object name: "
            + ", ".join(duplicates)
            + ". Use a different --stem or upload one arm at a time."
        )
    # Artifacts first, sidecars next, manifest last.
    return sorted(objects, key=lambda o: (KIND_ORDER[o["kind"]], o["alpha"] or 0,
                                          o["name"]))


def plan_uploads(cfg, manifest_path, manifest: dict | None = None, *,
                 prefix: str | None = None, name: str | None = None,
                 include_fused: bool = False, overwrite: bool = False,
                 allow_production_prefix: bool = False, token=None,
                 check_existing: bool = True, job_id: str | None = None) -> dict:
    """Work out exactly what would be written, and what blocks the write."""
    manifest_path = Path(manifest_path)
    manifest = manifest or manifests.read(manifest_path)
    if not manifest:
        raise UploadError(f"no readable manifest at {manifest_path}")
    if manifest.get("kind") != "mix":
        raise UploadError(
            f"{manifest_path} is a {manifest.get('kind') or 'unknown'} manifest, "
            "not a mix"
        )
    if not cfg.bucket:
        raise UploadError(
            "no bucket is configured, so there is nowhere to upload. Set "
            "`bucket` in the config or pass --bucket."
        )
    base_prefix = normalize_prefix(prefix if prefix is not None else cfg.upload_prefix)
    directory = normalize_name(name if name is not None else mix_slug(manifest))
    destination = f"{base_prefix}/{directory}".strip("/")
    production = is_production_prefix(destination)

    objects = _collect(manifest_path, manifest, include_fused=include_fused)
    object_prefix = cfg.object_prefix(destination + "/")
    for item in objects:
        item["object"] = f"{object_prefix}{item['name']}"
        item["exists"] = False
        item["generation"] = None

    warnings: list[str] = []
    blockers: list[str] = []
    if production and not allow_production_prefix:
        blockers.append(
            f"destination {destination!r} is under the production `loras/` library. "
            "Pass --allow-production-prefix (or tick the box) to write there; the "
            "default destination is _prototypes/mixes/."
        )
    if not any(o["kind"] == "converted" for o in objects):
        warnings.append(
            "this mix has no converted pair; only the fused file (which is not "
            "byte-reproducible) will be uploaded"
        )
    if check_existing:
        if token is None:
            raise UploadError(
                "no credentials were resolved, so existing keys cannot be checked. "
                "Run `loralab doctor` first."
            )
        for item in objects:
            stored = gcs.stat_or_none(cfg.bucket, item["object"], token)
            if stored is not None:
                item["exists"] = True
                item["generation"] = stored.get("generation")
        existing = [o for o in objects if o["exists"]]
        if existing and not overwrite:
            blockers.append(
                f"{len(existing)} destination object(s) already exist. Tick "
                "'overwrite' (or pass --overwrite) to replace them deliberately."
            )
    total = sum(o["size_bytes"] for o in objects)
    command = ["loralab", "upload"]
    if job_id:
        command += ["--job", job_id]
    else:
        command += ["--dir", str(manifest_path.parent)]
    command += ["--prefix", base_prefix, "--name", directory]
    command.append("--include-fused" if include_fused else "--no-include-fused")
    if overwrite:
        command.append("--overwrite")
    if allow_production_prefix:
        command.append("--allow-production-prefix")
    return {
        "bucket": cfg.bucket,
        "root": cfg.root,
        "prefix": base_prefix,
        "directory": directory,
        "destination": destination,
        "object_prefix": object_prefix,
        "display": f"gs://{cfg.bucket}/{object_prefix}",
        "production": production,
        "manifest": str(manifest_path),
        "job_id": job_id,
        "objects": objects,
        "count": len(objects),
        "total_bytes": total,
        "total_mb": round(total / 1e6, 1),
        "existing": sum(1 for o in objects if o["exists"]),
        "warnings": warnings,
        "blockers": blockers,
        "overwrite": overwrite,
        "include_fused": include_fused,
        "allow_production_prefix": allow_production_prefix,
        "command": command,
    }


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def apply_plan(cfg, plan: dict, *, token=None, overwrite: bool | None = None,
               log=None, record_path=None) -> dict:
    """Execute a preview. Every gate is re-applied here, not trusted from the UI."""
    log = log or (lambda *_: None)
    token = token or gcs.resolve_token()
    overwrite = plan.get("overwrite") if overwrite is None else overwrite
    if plan.get("blockers"):
        raise UploadError("; ".join(plan["blockers"]))
    objects = []
    uploaded = 0
    for item in plan["objects"]:
        try:
            result = gcs.upload_object(
                cfg.bucket, item["object"], item["source"], token=token,
                overwrite=bool(overwrite), generation=item.get("generation"),
                log=log,
            )
        except gcs.GcsError as exc:
            raise UploadError(str(exc)) from exc
        uploaded += result["size"]
        objects.append({
            **{k: item.get(k) for k in ("name", "object", "kind", "alpha",
                                        "size_bytes", "sha256")},
            "generation": result.get("generation"),
            "md5_verified": result.get("md5_verified", False),
            "crc32c": result.get("crc32c"),
            "updated": result.get("updated"),
        })
        log(f"  ok: {item['object']} ({result['size'] / 1e6:.1f} MB"
            + (", md5 verified" if result.get("md5_verified") else "") + ")")
    record = {
        "schema": 1,
        "kind": "upload",
        "tool": f"loralab {__version__}",
        "created_utc": _now(),
        "bucket": cfg.bucket,
        "root": cfg.root,
        "destination": plan["destination"],
        "object_prefix": plan["object_prefix"],
        "display": plan["display"],
        "manifest": plan["manifest"],
        "command": plan["command"],
        "overwrite": bool(overwrite),
        "objects": objects,
        "count": len(objects),
        "total_bytes": uploaded,
    }
    if record_path:
        manifests.write(record_path, record)
    return record


def dedupe_key(cfg, manifest_path: Path, *, prefix: str, name: str,
               include_fused: bool, overwrite: bool,
               allow_production_prefix: bool) -> str:
    payload = {
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha": merge.sha256_file(manifest_path),
        "bucket": cfg.bucket,
        "root": cfg.root,
        "prefix": prefix,
        "name": name,
        "include_fused": bool(include_fused),
        "overwrite": bool(overwrite),
        "allow_production_prefix": bool(allow_production_prefix),
        "version": __version__,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return "upload:" + hashlib.sha1(blob).hexdigest()


def run(ctx, cfg, params: dict, *, token=None) -> dict:
    """A web upload job. Params name the source mix job and the destination."""
    source_job = params.get("job")
    if not source_job:
        raise UploadError("upload needs the source mix job id")
    manifest_path = Path(cfg.jobs_dir) / source_job / "MANIFEST.json"
    plan = plan_uploads(
        cfg, manifest_path,
        prefix=params.get("prefix"),
        name=params.get("name"),
        include_fused=bool(params.get("include_fused")),
        overwrite=bool(params.get("overwrite")),
        allow_production_prefix=bool(params.get("allow_production_prefix")),
        token=token, job_id=source_job,
    )
    ctx.log(f"upload: {plan['count']} object(s) · {plan['total_mb']} MB · "
            f"{plan['display']}")
    for warning in plan["warnings"]:
        ctx.log(f"warning: {warning}")
    for blocker in plan["blockers"]:
        raise UploadError(blocker)
    if ctx.cancelled():
        raise UploadError("cancel requested before upload")
    record = apply_plan(
        cfg, plan, token=token, overwrite=bool(params.get("overwrite")),
        log=ctx.log, record_path=Path(ctx.dir) / "upload.json",
    )
    ctx.log(f"uploaded {record['count']} object(s), "
            f"{record['total_bytes'] / 1e6:.1f} MB")
    return {
        "count": record["count"],
        "total_bytes": record["total_bytes"],
        "destination": plan["destination"],
        "display": plan["display"],
        "record": str(Path(ctx.dir) / "upload.json"),
    }


def describe(plan: dict) -> str:
    """A terminal-friendly preview, used by `loralab upload`."""
    lines = [
        f"destination: {plan['display']}",
        f"objects:     {plan['count']} · {plan['total_mb']} MB",
    ]
    for item in plan["objects"]:
        mark = "!" if item["exists"] else "+"
        lines.append(
            f"  {mark} {item['object'].split(plan['object_prefix'], 1)[-1]}"
            f"  {item['size_bytes'] / 1e6:.1f} MB  {item['sha256'][:12]}…"
            + ("  (exists)" if item["exists"] else "")
        )
    for warning in plan["warnings"]:
        lines.append(f"warning: {warning}")
    for blocker in plan["blockers"]:
        lines.append(f"BLOCKED: {blocker}")
    return "\n".join(lines)
