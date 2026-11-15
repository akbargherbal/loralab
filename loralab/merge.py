"""Rank-concatenation merge of two ai-toolkit fused YuE2 LoRAs.

The merge keeps the LoRA in low-rank (A, B) form — the dense (out x in) weight
delta is never materialised. Per projection, with A = down and B = up:

    A_mix = cat([A_carrier,        A_donor], dim=0)     # rank_c + rank_d
    B_mix = cat([B_carrier, α ·    B_donor], dim=1)

α is folded into the donor's **B** because ai-toolkit applies its scale to the
up-projection (`up(down(x)) * scale`). At α = 0 the donor contributes nothing
and the output is the carrier, tensor for tensor.

The conventions are the ones documented upstream in
`maqamrock-yue2-lora-finetuning/docs/PRON_LORA_MERGE.md`; this is a
re-implementation, not an import. Two consequences are load-bearing:

* an AR-only donor merges only the AR branch; NAR is copied through, and is
  zero-padded to the merged rank because the converter refuses mixed ranks
  (padding is a numerical no-op);
* an input must be a `fused` or `merged` adapter. A `converted` audio.cpp file
  is a different layout (`layers.*`, no `.alpha`) and is rejected.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

from . import __version__
from .catalog import AR_PREFIX, NAR_PREFIX

BRANCHES = (("ar", AR_PREFIX), ("nar", NAR_PREFIX))
ALPHA_MAX = 1.0
DELTA_TOLERANCE = 5e-3  # relative; bf16 rounding of the scaled donor is ~5e-4


class MergeError(RuntimeError):
    """A structural or numerical problem, with a message the user can act on."""


# ---------------------------------------------------------------------------
# environment / file helpers
# ---------------------------------------------------------------------------
def environment() -> dict:
    """Versions of the merge engine, or an actionable error if it is absent."""
    try:
        import safetensors
        import torch
    except Exception as exc:  # noqa: BLE001 — the message is the point
        raise MergeError(
            "mixing needs torch and safetensors. Install them with:\n"
            '  python3 -m pip install -e ".[merge]" --extra-index-url '
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return {"torch": torch.__version__, "safetensors": safetensors.__version__}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        return dict(fh.metadata() or {})


def load_tensors(path) -> dict:
    environment()
    from safetensors.torch import load_file

    path = Path(path)
    if not path.is_file():
        raise MergeError(f"input not found: {path}")
    try:
        return load_file(str(path))
    except Exception as exc:  # noqa: BLE001
        raise MergeError(f"{path.name}: not a readable safetensors file ({exc})") from exc


def save_tensors(tensors: dict, path, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {str(k): str(v) for k, v in (metadata or {}).items() if v is not None}
    try:
        save_file(tensors, str(path), metadata=meta)
    except Exception as exc:  # noqa: BLE001
        raise MergeError(f"cannot write {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# key grammar
# ---------------------------------------------------------------------------
def _stem(key: str) -> str:
    return key[: -len(".weight")] if key.endswith(".weight") else key


def _ab(key: str):
    """'A' / 'B' for a LoRA down/up tensor, else None."""
    stem = _stem(key)
    if stem.endswith(".lora_A"):
        return "A"
    if stem.endswith(".lora_B"):
        return "B"
    return None


def _proj(key: str) -> str:
    """Projection base, e.g. `diffusion_model...mlp.down_proj`."""
    stem = _stem(key)
    for suffix in (".lora_A", ".lora_B", ".alpha"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _key_set(tensors: dict) -> set[str]:
    # ai-toolkit drops `.alpha` on save; a stray one is validated separately and
    # never merged.
    return {k for k in tensors if not k.endswith(".alpha")}


def _partner(tensors: dict, base: str):
    return tensors.get(base + ".lora_B.weight", tensors.get(base + ".lora_B"))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def check_alpha(value) -> float:
    """α is a multiplier in [0, 1] — not a percentage, and never 30."""
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        raise MergeError(f"alpha must be a number, got {value!r}") from None
    if not math.isfinite(alpha):
        raise MergeError(f"alpha must be finite, got {alpha}")
    if alpha < 0.0 or alpha > ALPHA_MAX:
        raise MergeError(
            f"alpha {alpha} is outside [0, {ALPHA_MAX:g}]. alpha scales a "
            "multiplier (0 = carrier only, 1 = full donor); it is not a "
            "percentage."
        )
    return alpha


def _check_alpha_keys(tensors: dict, label: str) -> None:
    """A saved `.alpha` is only valid when it equals the rank."""
    ranks = _rank_of_a(tensors, label)
    for key, value in tensors.items():
        if not key.endswith(".alpha"):
            continue
        rank = ranks.get(_proj(key))
        got = float(value.reshape(-1)[0]) if value.numel() else float("nan")
        if rank is None or not math.isfinite(got) or abs(got - rank) > 1e-6 * max(1.0, rank):
            raise MergeError(
                f"{label}: alpha key {key}={got} does not equal rank {rank}; "
                "the converter applies no alpha/rank factor, so this file "
                "would merge with the wrong scale."
            )


def _rank_of_a(tensors: dict, label: str) -> dict[str, int]:
    """{projection: rank} for every A tensor, checking its B partner."""
    ranks: dict[str, int] = {}
    for key, tensor in tensors.items():
        if _ab(key) != "A":
            continue
        if tensor.ndim != 2:
            raise MergeError(f"{label}: {key} is not rank-2 (shape {tuple(tensor.shape)})")
        base = _proj(key)
        partner = _partner(tensors, base)
        if partner is None:
            raise MergeError(f"{label}: {key} has no lora_B partner")
        if partner.ndim != 2 or partner.shape[1] != tensor.shape[0]:
            raise MergeError(
                f"{label}: {base} rank mismatch A{tuple(tensor.shape)} "
                f"B{tuple(partner.shape)}"
            )
        ranks[base] = int(tensor.shape[0])
    return ranks


def _branch_ranks(ranks: dict[str, int], label: str) -> dict[str, int]:
    """One rank per branch, or a hard error on a mixed-rank input."""
    out: dict[str, int] = {}
    for name, prefix in BRANCHES:
        values = {r for base, r in ranks.items() if base.startswith(prefix)}
        if len(values) > 1:
            raise MergeError(f"{label}: {name} branch has mixed ranks {sorted(values)}")
        if values:
            out[name] = values.pop()
    return out


def validate(carrier: dict, donor: dict) -> dict:
    """Hard-fail on any structural surprise; return what the inputs are."""
    if not carrier:
        raise MergeError("carrier contains no tensors")
    if not donor:
        raise MergeError("donor contains no tensors")

    c_keys, d_keys = _key_set(carrier), _key_set(donor)
    unknown = sorted(k for k in c_keys
                     if not (k.startswith(AR_PREFIX) or k.startswith(NAR_PREFIX)))
    if unknown:
        raise MergeError(
            "carrier has keys under neither text_encoders (AR) nor "
            f"diffusion_model (NAR): {unknown[:5]}"
        )
    for label, keys in (("carrier", c_keys), ("donor", d_keys)):
        stray = sorted(k for k in keys if not (k.startswith(AR_PREFIX)
                                               or k.startswith(NAR_PREFIX)))
        if stray:
            raise MergeError(f"{label} has unexpected keys: {stray[:5]}")

    extras = sorted(d_keys - c_keys)
    if extras:
        raise MergeError(
            f"donor has {len(extras)} key(s) the carrier lacks (a merge needs "
            f"identical projections): {extras[:5]}"
        )

    donor_branches = []
    for name, prefix in BRANCHES:
        c_branch = {k for k in c_keys if k.startswith(prefix)}
        d_branch = {k for k in d_keys if k.startswith(prefix)}
        if d_branch and d_branch != c_branch:
            missing = sorted(c_branch - d_branch)
            raise MergeError(
                f"donor is missing {len(missing)} {name} key(s) the carrier has: "
                f"{missing[:5]}"
            )
        if d_branch:
            donor_branches.append(name)
    if not donor_branches:
        raise MergeError("donor shares no keys with the carrier")
    if "ar" not in {n for n, p in BRANCHES if any(k.startswith(p) for k in c_keys)}:
        raise MergeError("carrier has no text_encoders (AR) branch")

    _check_alpha_keys(carrier, "carrier")
    _check_alpha_keys(donor, "donor")

    c_ranks = _rank_of_a(carrier, "carrier")
    d_ranks = _rank_of_a(donor, "donor")
    if not c_ranks:
        raise MergeError("carrier has no lora_A/lora_B pairs")

    # Same projection geometry: only the rank dimension may differ.
    for base, d_rank in d_ranks.items():
        c_a = carrier.get(base + ".lora_A.weight", carrier.get(base + ".lora_A"))
        d_a = donor.get(base + ".lora_A.weight", donor.get(base + ".lora_A"))
        c_b = _partner(carrier, base)
        d_b = _partner(donor, base)
        if c_a.shape[1] != d_a.shape[1] or c_b.shape[0] != d_b.shape[0]:
            raise MergeError(
                f"{base}: input/output dimensions differ "
                f"(carrier A{tuple(c_a.shape)} B{tuple(c_b.shape)}, "
                f"donor A{tuple(d_a.shape)} B{tuple(d_b.shape)})"
            )

    return {
        "carrier_ranks": _branch_ranks(c_ranks, "carrier"),
        "donor_ranks": _branch_ranks(d_ranks, "donor"),
        "donor_branches": donor_branches,
        "tensors": len(c_keys),
    }


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def merge(carrier: dict, donor: dict, alpha, *, pad: bool = True
          ) -> tuple[dict, dict]:
    """Return (merged_tensors, stats). Pure: no I/O, no dtype outside the carrier's."""
    alpha = check_alpha(alpha)
    info = validate(carrier, donor)
    keep = alpha != 0.0

    merged: dict = {}
    merged_counts = {"ar": 0, "nar": 0}
    copied_counts = {"ar": 0, "nar": 0}
    for key, base_tensor in carrier.items():
        if key.endswith(".alpha"):
            continue
        branch = "ar" if key.startswith(AR_PREFIX) else "nar"
        side = _ab(key)
        donor_tensor = donor.get(key)
        if side is None:
            raise MergeError(f"unexpected key {key}")

        if not keep:
            # The free control arm: a clone makes the α=0 identity check real.
            merged[key] = base_tensor.clone()
        elif donor_tensor is None:
            # The donor has no such branch (AR-only donor, NAR copied through).
            merged[key] = base_tensor.clone()
        elif side == "A":
            block = donor_tensor.to(base_tensor.dtype)
            merged[key] = torch_cat([base_tensor, block], dim=0)
        else:
            block = (donor_tensor.to(_float32()) * alpha).to(base_tensor.dtype)
            merged[key] = torch_cat([base_tensor, block], dim=1)

        if donor_tensor is not None and keep:
            merged_counts[branch] += 1
        else:
            copied_counts[branch] += 1

    out_ranks = _branch_ranks(_rank_of_a(merged, "merged"), "merged")
    padded = {"ar": 0, "nar": 0}
    if keep and pad and len(set(out_ranks.values())) > 1:
        target = max(out_ranks.values())
        for key, tensor in list(merged.items()):
            if _ab(key) != "A":
                continue
            base = _proj(key)
            if tensor.shape[0] == target:
                continue
            pad_rows = torch_zeros(
                (target - tensor.shape[0], tensor.shape[1]), tensor.dtype)
            merged[key] = torch_cat([tensor, pad_rows], dim=0)
            partner_key = _partner_key(carrier, key)
            b = merged[partner_key]
            pad_cols = torch_zeros((b.shape[0], target - b.shape[1]), b.dtype)
            merged[partner_key] = torch_cat([b, pad_cols], dim=1)
            padded["ar" if base.startswith(AR_PREFIX) else "nar"] += 1
        out_ranks = {name: target for name in out_ranks}

    stats = {
        "carrier_ranks": info["carrier_ranks"],
        "donor_ranks": info["donor_ranks"],
        "donor_branches": info["donor_branches"],
        "out_ranks": out_ranks,
        "donor_kept": keep,
        "merged_tensors": merged_counts,
        "copied_tensors": copied_counts,
        "padded_tensors": padded,
        "target_rank": max(out_ranks.values()) if out_ranks else None,
    }
    return merged, stats


def _partner_key(carrier: dict, a_key: str) -> str:
    """The lora_B key that pairs with `a_key`, preserving the `.weight` style."""
    base = _proj(a_key)
    if base + ".lora_B.weight" in carrier:
        return base + ".lora_B.weight"
    return base + ".lora_B"


def _float32():
    import torch

    return torch.float32


def torch_cat(tensors, dim):
    import torch

    return torch.cat(tensors, dim=dim)


def torch_zeros(shape, dtype):
    import torch

    return torch.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def _projections(tensors: dict, sample: int | None) -> list[str]:
    bases = sorted({_proj(k) for k in tensors if _ab(k) == "A"})
    if sample and sample < len(bases):
        if sample == 1:
            picks = [0]
        else:
            picks = sorted({round(i * (len(bases) - 1) / (sample - 1))
                            for i in range(sample)})
        bases = [bases[i] for i in picks]
    return bases


def verify(carrier: dict, donor: dict, merged: dict, alpha, *,
           sample: int | None = 8) -> dict:
    """Measure the merge instead of asserting it.

    α = 0 compares every tensor against the carrier (`torch.equal`). α > 0
    compares the dense delta on a sample of projections:

        B_mix @ A_mix  ==  B_c @ A_c + α · B_d @ A_d
    """
    alpha = check_alpha(alpha)
    if _key_set(merged) != _key_set(carrier):
        raise MergeError("merged key set differs from the carrier")
    dtypes = {str(t.dtype) for t in merged.values()}
    report = {
        "kind": "identity" if alpha == 0.0 else "delta",
        "alpha": alpha,
        "dtype_preserved": all(merged[k].dtype == carrier[k].dtype
                               for k in _key_set(carrier)),
        "dtypes": sorted(dtypes),
    }
    if alpha == 0.0:
        mismatched = []
        for key in _key_set(carrier):
            a, b = carrier[key], merged[key]
            if a.shape != b.shape or not _torch_equal(a, b):
                mismatched.append(key)
        report.update({
            "tensors_checked": len(_key_set(carrier)),
            "mismatched": mismatched[:5],
            "max_abs_error": 0.0 if not mismatched else None,
            "passed": not mismatched,
        })
        return report

    bases = _projections(carrier, sample)
    worst_abs, worst_rel, failed = 0.0, 0.0, []
    for base in bases:
        c_a = _get(carrier, base, "A")
        c_b = _get(carrier, base, "B")
        m_a = _get(merged, base, "A")
        m_b = _get(merged, base, "B")
        ref = c_b.to(_float32()) @ c_a.to(_float32())
        d_a = _get(donor, base, "A")
        d_b = _get(donor, base, "B")
        if d_a is not None and d_b is not None:
            ref = ref + alpha * (d_b.to(_float32()) @ d_a.to(_float32()))
        got = m_b.to(_float32()) @ m_a.to(_float32())
        diff = (got - ref).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-12)
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / scale)
        if diff / scale > DELTA_TOLERANCE:
            failed.append(base)
    report.update({
        "projections_checked": len(bases),
        "max_abs_error": worst_abs,
        "max_rel_error": worst_rel,
        "tolerance": DELTA_TOLERANCE,
        "mismatched": failed[:5],
        "passed": not failed,
    })
    return report


def _get(tensors: dict, base: str, side: str):
    return tensors.get(f"{base}.lora_{side}.weight",
                       tensors.get(f"{base}.lora_{side}"))


def _torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def build_metadata(carrier_meta: dict, *, alpha: float, carrier_name: str,
                   carrier_sha: str | None, donor_name: str,
                   donor_sha: str | None, stats: dict) -> dict:
    """Metadata in the shape the catalog (and upstream manifests) already read."""
    meta = dict(carrier_meta)
    branches = "+".join(stats["donor_branches"]) or "none"
    meta.update({
        "merge_tool": "loralab-merge",
        "merge_tool_version": __version__,
        "merge_kind": f"rank_concat_{branches}",
        "merge_alpha": repr(float(alpha)),
        # Legacy keys: the upstream tools wrote merge_v2_*/merge_quran_*, and
        # the catalog reads them from files produced on either side.
        "merge_quran": donor_name,
        "merge_quran_sha256": donor_sha,
        "merge_v2_sha256": carrier_sha,
        "merge_carrier_name": carrier_name,
        "merge_donor_name": donor_name,
        "merge_ar_rank": str(stats["out_ranks"].get("ar")),
        "merge_nar_rank": str(stats["out_ranks"].get("nar")),
    })
    return meta


# ---------------------------------------------------------------------------
# one file in, one file out
# ---------------------------------------------------------------------------
def merge_files(carrier_path, donor_path, alpha, out_path, *,
                carrier_sha: str | None = None, donor_sha: str | None = None,
                extra_metadata: dict | None = None, sample: int | None = 8,
                log=None) -> dict:
    """Merge two files, verify, write, and return everything worth recording."""
    log = log or (lambda *_: None)
    carrier_path, donor_path, out_path = (Path(carrier_path), Path(donor_path),
                                         Path(out_path))
    started = time.perf_counter()
    carrier = load_tensors(carrier_path)
    donor = load_tensors(donor_path)
    merged, stats = merge(carrier, donor, alpha)
    report = verify(carrier, donor, merged, alpha, sample=sample)
    if not report["passed"]:
        raise MergeError(
            f"alpha={alpha}: merge verification failed "
            f"(max relative error {report.get('max_rel_error')}, "
            f"tolerance {report.get('tolerance')})"
        )
    carrier_sha = carrier_sha or sha256_file(carrier_path)
    donor_sha = donor_sha or sha256_file(donor_path)
    meta = build_metadata(
        read_metadata(carrier_path), alpha=float(alpha),
        carrier_name=carrier_path.name, carrier_sha=carrier_sha,
        donor_name=donor_path.name, donor_sha=donor_sha, stats=stats,
    )
    if extra_metadata:
        meta.update({k: v for k, v in extra_metadata.items() if v is not None})
    save_tensors(merged, out_path, meta)
    result = {
        "alpha": float(alpha),
        "stats": stats,
        "verify": report,
        "fused": {
            "path": str(out_path),
            "name": out_path.name,
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "ranks": stats["out_ranks"],
        },
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    log(f"  a{float(alpha):g}: rank {stats['out_ranks']} · "
        f"{result['fused']['size_bytes'] / 1e6:.0f} MB · "
        f"{'identical to carrier' if report['kind'] == 'identity' else 'max |δ| ' + format(report['max_abs_error'], '.1e')}")
    return result
