"""The mix pipeline: resolve inputs → merge each α → convert → manifest.

This is the one implementation behind both `loralab mix` and the web job. The
web page is a client of the same function, which is why the job page can show
the equivalent command line and it is not a lie.

Checkpointing is per α: the manifest is rewritten after every arm, so a crash
leaves the arms that finished readable instead of nothing.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import __version__, catalog as cat, convert as convertmod, gcs
from . import manifests, merge

ELIGIBLE_KINDS = ("fused", "merged")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def parse_alphas(value) -> list[float]:
    """Accept '0.1, 0.2', ['0.1', '0.2'], or a single number."""
    if value is None or value == "":
        raise merge.MergeError("give at least one alpha, e.g. --alpha 0.2")
    if isinstance(value, (list, tuple)):
        raw = [part for item in value for part in str(item).replace(",", " ").split()]
    else:
        raw = str(value).replace(",", " ").split()
    if not raw:
        raise merge.MergeError("give at least one alpha, e.g. --alpha 0.2")
    alphas = []
    for part in raw:
        alpha = merge.check_alpha(part)
        if alpha not in alphas:
            alphas.append(alpha)
    return alphas


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:g}"


def _slug(name: str) -> str:
    stem = Path(name).name
    for suffix in (".safetensors", ".sft"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _ranks(source: dict) -> dict[str, int]:
    """Normalise a catalog/local record's rank map to {branch: int}."""
    out = {}
    for branch, values in (source.get("rank") or {}).items():
        numbers = [int(v) for v in values if v is not None]
        if numbers:
            out[branch] = max(numbers)
    return out


def find_entry(doc: dict | None, spec: str) -> dict | None:
    if not doc:
        return None
    entries = doc.get("entries") or []
    exact = [e for e in entries if e.get("path") == spec]
    if exact:
        return exact[0]
    by_name = [e for e in entries if e.get("name") == spec]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise merge.MergeError(
            f"{spec} is ambiguous: {len(by_name)} adapters share that name. "
            "Pass the full catalog path."
        )
    return None


def inspect(spec: str, doc: dict | None, cfg=None) -> dict:
    """Describe an input without downloading or hashing it. Used by previews."""
    path = Path(spec).expanduser()
    if path.is_file():
        try:
            meta, tensors = cat.header_from_local(path)
            info = cat.classify(meta, tensors)
        except Exception as exc:  # noqa: BLE001
            raise merge.MergeError(
                f"{path.name}: not a readable safetensors file ({exc})"
            ) from exc
        return {
            "name": path.name,
            "path": str(path),
            "source": "local",
            "object": None,
            "entry_path": None,
            "size_bytes": path.stat().st_size,
            "kind": info["kind"],
            "rank": info["rank"],
        }
    entry = find_entry(doc, spec)
    if entry is None:
        raise merge.MergeError(
            f"input not found: {spec}. Pass a local .safetensors path or a path "
            "from the catalog (`loralab catalog` lists them)."
        )
    return {
        "name": entry["name"],
        "path": None,
        "source": "gcs",
        "object": entry["path"],
        "entry_path": entry["path"],
        "size_bytes": entry.get("size_bytes") or 0,
        "kind": entry["kind"],
        "rank": entry.get("rank") or {},
        "sha256": entry.get("sha256"),
        "crc32c": entry.get("crc32c"),
    }


def _cache_dest(cfg, source: dict) -> Path:
    key = source.get("crc32c") or source.get("entry_path") or source["name"]
    digest = hashlib.sha1(str(key).encode()).hexdigest()[:12]
    return Path(cfg.data_dir) / "cache" / "src" / f"{digest}-{Path(source['name']).name}"


def resolve_source(spec: str, cfg, doc: dict | None, *, token=None,
                   log=None) -> dict:
    """Stage one input locally (downloading from GCS when needed) and hash it."""
    log = log or (lambda *_: None)
    source = inspect(spec, doc, cfg)
    if source["source"] == "local":
        path = Path(source["path"])
        log(f"stage: local {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    else:
        dest = _cache_dest(cfg, source)
        expected_size = source.get("size_bytes") or 0
        download = True
        if dest.is_file() and (not expected_size or dest.stat().st_size == expected_size):
            digest = merge.sha256_file(dest)
            if source.get("sha256") and digest != source["sha256"]:
                log(f"stage: cached {dest.name} does not match the catalog sha256; "
                    "re-downloading")
                dest.unlink()
            else:
                log(f"stage: cached {dest.name}")
                source["sha256"] = digest
                download = False
        if download:
            if not cfg.bucket:
                raise merge.MergeError(
                    "no bucket configured, so a catalog path cannot be staged. "
                    "Pass local .safetensors paths, or configure the bucket."
                )
            object_name = (f"{cfg.root}/{source['entry_path']}" if cfg.root
                           else source["entry_path"])
            log(f"stage: gs://{cfg.bucket}/{object_name}")
            got = gcs.download_object(cfg.bucket, object_name, dest, token=token, log=log)
            if expected_size and got["size"] != expected_size:
                raise merge.MergeError(
                    f"{source['name']}: downloaded {got['size']} bytes, catalog "
                    f"says {expected_size}; the object changed since the scan. "
                    "Rescan and try again."
                )
            source["sha256"] = got["sha256"]
        path = dest

    try:
        meta, tensors = cat.header_from_local(path)
        info = cat.classify(meta, tensors)
    except Exception as exc:  # noqa: BLE001 — a corrupt download must be actionable
        raise merge.MergeError(
            f"{source['name']}: not a readable safetensors file ({exc})"
        ) from exc
    if source.get("entry_path") and info["kind"] != source["kind"]:
        log(f"  note: {source['name']} is {info['kind']} in the file, "
            f"{source['kind']} in the catalog")
    source.update({
        "path": str(path),
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "kind": info["kind"],
        "rank": info["rank"],
        "sha256": source.get("sha256") or merge.sha256_file(path),
    })
    return source


# ---------------------------------------------------------------------------
# preview / plan
# ---------------------------------------------------------------------------
def _free_bytes(cfg) -> int:
    target = Path(cfg.data_dir)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(str(target if target.exists() else "/")).free


def estimate(alphas, carrier: dict, donor: dict, *, convert: bool = True) -> dict:
    """A deliberately conservative size forecast, shown before the click."""
    fused = (carrier.get("size_bytes") or 0) + (donor.get("size_bytes") or 0)
    pair = int(fused * 1.2) if convert else 0
    per_arm = fused + pair
    return {
        "fused_mb": round(fused / 1e6, 1),
        "converted_pair_mb": round(pair / 1e6, 1),
        "per_arm_mb": round(per_arm / 1e6, 1),
        "total_mb": round(per_arm * len(alphas) / 1e6, 1),
        "arms": len(alphas),
    }


def preview(carrier: dict, donor: dict, alphas: list[float], cfg, *,
            convert: bool = True) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    for role, source in (("carrier", carrier), ("donor", donor)):
        if source.get("kind") not in ELIGIBLE_KINDS:
            issues.append(
                f"{role} {source['name']} is {source.get('kind') or 'unknown'}; "
                "only fused or merged adapters can be mixed (a converted file "
                "has no .alpha key and a different tensor layout)."
            )
    c_ranks, d_ranks = _ranks(carrier), _ranks(donor)
    for branch in d_ranks:
        if branch not in c_ranks:
            issues.append(
                f"donor has a {branch} branch the carrier does not; a merge "
                "needs matching projections."
            )
    out_ranks = {branch: rank + d_ranks.get(branch, 0) for branch, rank in c_ranks.items()}
    target = max(out_ranks.values()) if out_ranks else None
    if target and out_ranks and len(set(out_ranks.values())) > 1:
        short = sorted(b for b, r in out_ranks.items() if r != target)
        warnings.append(
            f"{', '.join(short)} stays at rank "
            f"{', '.join(str(out_ranks[b]) for b in short)} and is zero-padded "
            f"to rank {target}; the converter requires one uniform rank. "
            "Padding is a numerical no-op."
        )
    if "ar" not in c_ranks:
        issues.append("carrier has no AR (text_encoders) branch")
    sizes = estimate(alphas, carrier, donor, convert=convert)
    free = _free_bytes(cfg)
    if free < sizes["total_mb"] * 1e6 * 1.5:
        issues.append(
            f"not enough disk: the run needs about {sizes['total_mb']:.0f} MB "
            f"plus headroom, and {free / 1e9:.1f} GB is free."
        )
    if convert and not cfg.bucket:
        cached = convertmod.cached_path(cfg)
        if not cached.is_file():
            warnings.append(
                "no bucket is configured, so the converter cannot be downloaded. "
                "Pass --converter PATH, or build with --no-convert."
            )
    return {
        "issues": issues,
        "warnings": warnings,
        "out_ranks": out_ranks,
        "target_rank": target,
        "donor_branches": sorted(d_ranks),
        "sizes": sizes,
        "disk_free_gb": round(free / 1e9, 1),
    }


# ---------------------------------------------------------------------------
# params, command, dedupe
# ---------------------------------------------------------------------------
def build_params(carrier: str, donor: str, alphas, *, convert: bool = True,
                 stem: str = "") -> dict:
    params = {
        "carrier": carrier,
        "donor": donor,
        "alpha": ",".join(f"{a:g}" for a in parse_alphas(alphas)),
        "convert": bool(convert),
    }
    if stem:
        params["stem"] = stem
    return params


def command(params: dict) -> list[str]:
    cmd = ["loralab", "mix", "--carrier", str(params["carrier"]),
           "--donor", str(params["donor"]),
           "--alpha", str(params.get("alpha", ""))]
    if params.get("stem"):
        cmd += ["--stem", str(params["stem"])]
    cmd.append("--convert" if params.get("convert", True) else "--no-convert")
    return cmd


def command_text(params: dict) -> str:
    return " ".join(
        part if " " not in part else json.dumps(part) for part in command(params)
    )


def dedupe_key(params: dict) -> str:
    payload = {
        "carrier": params.get("carrier"),
        "donor": params.get("donor"),
        "alpha": sorted(parse_alphas(params.get("alpha"))),
        "convert": bool(params.get("convert", True)),
        "stem": params.get("stem") or "",
        "version": __version__,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return "mix:" + hashlib.sha1(blob).hexdigest()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
class ConsoleContext:
    """A stand-in for a job context when `loralab mix` runs in a terminal."""

    def __init__(self, directory, params: dict, log=print):
        self.id = "cli"
        self.dir = Path(directory)
        self.params = params
        self._log = log

    def log(self, message: str = "") -> None:
        self._log(message)

    def cancelled(self) -> bool:
        return False


def run(ctx, cfg, params: dict, *, token=None, converter_path=None) -> dict:
    alphas = parse_alphas(params.get("alpha"))
    if not params.get("carrier") or not params.get("donor"):
        raise merge.MergeError("mix needs both --carrier and --donor")
    doc = cat.load_catalog(cfg.catalog_path)
    convert = bool(params.get("convert", True))
    env = merge.environment()
    ctx.log(f"mix: {len(alphas)} arm(s) · torch {env['torch']} · "
            f"safetensors {env['safetensors']} · loralab {__version__}")
    ctx.log(f"command: {command_text(params)}")

    carrier = resolve_source(str(params["carrier"]), cfg, doc, token=token, log=ctx.log)
    donor = resolve_source(str(params["donor"]), cfg, doc, token=token, log=ctx.log)
    plan = preview(carrier, donor, alphas, cfg, convert=convert)
    if plan["issues"]:
        raise merge.MergeError("; ".join(plan["issues"]))
    for warning in plan["warnings"]:
        ctx.log(f"warning: {warning}")
    ctx.log(f"plan: ranks {plan['out_ranks']} · about {plan['sizes']['total_mb']} MB "
            f"for {len(alphas)} arm(s) · {plan['disk_free_gb']} GB free")

    converter = None
    if convert:
        converter = convertmod.resolve(cfg, path=converter_path, token=token, log=ctx.log)
        ctx.log(f"converter: {converter['path']} (sha256 {converter['sha256']})")

    stem_base = params.get("stem") or f"{_slug(carrier['name'])}__{_slug(donor['name'])}"
    out_root = Path(ctx.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    arms: list[dict] = []
    doc_manifest = manifests.build(
        command=command(params),
        carrier={k: carrier.get(k) for k in ("name", "path", "sha256", "size_bytes",
                                             "kind", "rank", "entry_path")},
        donor={k: donor.get(k) for k in ("name", "path", "sha256", "size_bytes",
                                         "kind", "rank", "entry_path")},
        arms=arms,
        converter=({**{k: converter[k] for k in ("sha256", "pinned", "source")},
                    "path": str(converter["path"])} if converter else None),
    )
    manifest_path = out_root / "MANIFEST.json"

    for alpha in alphas:
        if ctx.cancelled():
            ctx.log("cancel requested before the next arm")
            break
        tag = alpha_tag(alpha)
        arm_dir = out_root / tag
        # At alpha=0 the fused file keeps the carrier's basename: that is what
        # makes the converted pair byte-identical to a converted carrier.
        fused_name = (carrier["name"] if alpha == 0.0
                      else f"{stem_base}__{tag}.safetensors")
        fused_path = arm_dir / fused_name
        ctx.log(f"arm {tag}: merging {donor['name']} into {carrier['name']}")
        result = merge.merge_files(
            carrier["path"], donor["path"], alpha, fused_path,
            carrier_sha=carrier["sha256"], donor_sha=donor["sha256"],
            log=ctx.log,
        )
        arm = {
            "alpha": alpha,
            "stem": Path(fused_name).stem,
            "out_dir": str(arm_dir),
            "fused": result["fused"],
            "stats": result["stats"],
            "verify": result["verify"],
            "elapsed_s": result["elapsed_s"],
        }
        if converter:
            converted = convertmod.run(
                converter["path"], fused_path, arm_dir,
                stem=Path(fused_name).stem, log=ctx.log,
            )
            arm["converted"] = converted["outputs"]
            arm["converter"] = {
                "command": converted["command"],
                "byte_verified": converted["byte_verified"],
            }
        arms.append(arm)
        manifests.write(manifest_path, doc_manifest)
        manifests.write_sidecar(doc_manifest, arm, arm_dir)
        if arm.get("converted"):
            ctx.log("  converted: " + ", ".join(
                f"{b} {v['name']} ({v['sha256'][:12]}…)"
                for b, v in arm["converted"].items()))
        ctx.log(f"  record: {manifest_path}")

    if not arms:
        raise merge.MergeError("no arm was built")
    return {
        "arms": len(arms),
        "alphas": [arm["alpha"] for arm in arms],
        "manifest": str(manifest_path),
        "out": str(out_root),
        "converted": bool(convert),
    }
