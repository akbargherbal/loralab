"""The published-artifact gate: rebuild `v3 × arabic_instrumental` and match it.

The arithmetic is pinned by unit tests, but the real gate is bytes: the converted
AR/NAR pair from the published `mix_v3` directory must have exactly these sha256
values. They were read from the published
`.../arabic_instrumental_lora/mix_v3/MANIFEST.json` object in
`gs://akbar-december-2024-backup/OSTRIS_Arabic_Suno_Finetuning`.

A converted pair is reproducible only when the fused file keeps the arm's
basename and the converter stem matches; a different basename changes only the
embedded `source_file`. If you deliberately used another name, the gate reports
the mismatch and you should compare tensor sets, ranks and the measured δ
instead — never the fused file's whole-file sha.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import merge

REFERENCE = {
    0.1: {
        "ar": "d48109b3117bf7851390ac9d4de86a45c5edd99cc5d17ddb97481e5c77ef64a0",
        "nar": "88f00bcb1b60cb7638eff07d7f1a7ff1d19705e987a138bf82e0c36e84ea5b58",
        "ar_bytes": 139501720,
        "nar_bytes": 139503288,
    },
    0.2: {
        "ar": "820f1ce90ab64965606a5a57451b2dcf255072ab4b9c7be182bc14b80bd86dd6",
        "nar": "f9738e214361c02428f6baa01327ba42ce1208236d33868fd6c3d3a7b584225f",
        "ar_bytes": 139501720,
        "nar_bytes": 139503288,
    },
}
REFERENCE_INPUTS = {
    "carrier": "37dba84a7fff301b159a30322bd6d631d34c6a85f1ef8667df969d6c7d39a37b",
    "donor": "cf6d44bbf4c03b28b2511e03c985e1d49894f6d4b2508a23a8f2bda3182e3584",
}


def _arms(doc: dict) -> list[dict]:
    """Support this app's list-of-arms shape and the published dict shape."""
    arms = doc.get("arms") or []
    if isinstance(arms, list):
        return arms
    out = []
    for key, arm in arms.items():
        out.append({
            "alpha": arm.get("alpha", key),
            "converted": {
                "ar": {"sha256": arm.get("ar"), "size_bytes": arm.get("ar_bytes")},
                "nar": {"sha256": arm.get("nar"), "size_bytes": arm.get("nar_bytes")},
            },
        })
    return out


def _find(arms: list[dict], alpha: float) -> dict | None:
    for arm in arms:
        try:
            if abs(float(arm.get("alpha")) - alpha) < 1e-9:
                return arm
        except (TypeError, ValueError):
            continue
    return None


def check(doc: dict, *, base_dir=None) -> dict:
    """Compare a manifest (and optionally its files) with the published pair."""
    arms = _arms(doc)
    results = []
    errors = []
    if not arms:
        errors.append("manifest has no arms")
    for alpha, expected in REFERENCE.items():
        arm = _find(arms, alpha)
        if arm is None:
            errors.append(f"alpha {alpha:g}: arm is missing")
            continue
        converted = arm.get("converted") or {}
        row = {"alpha": alpha, "ar": None, "nar": None, "passed": True}
        for branch in ("ar", "nar"):
            item = converted.get(branch) or {}
            actual = item.get("sha256")
            row[branch] = actual
            if actual != expected[branch]:
                row["passed"] = False
                errors.append(
                    f"alpha {alpha:g} {branch}: sha256 {actual or '(missing)'} "
                    f"!= published {expected[branch]}"
                )
            if base_dir and item.get("path"):
                path = Path(item["path"])
                if not path.is_absolute():
                    path = Path(base_dir) / path
                if not path.is_file():
                    row["passed"] = False
                    errors.append(f"alpha {alpha:g} {branch}: file is missing: {path}")
                    continue
                actual_sha = merge.sha256_file(path)
                expected_size = expected[f"{branch}_bytes"]
                if path.stat().st_size != expected_size:
                    row["passed"] = False
                    errors.append(
                        f"alpha {alpha:g} {branch}: {path.stat().st_size} bytes "
                        f"!= published {expected_size}"
                    )
                if actual_sha != expected[branch]:
                    row["passed"] = False
                    errors.append(
                        f"alpha {alpha:g} {branch}: file sha256 {actual_sha} != "
                        f"published {expected[branch]} (if the basename differs, "
                        "compare tensors/ranks/δ instead)"
                    )
        verify = arm.get("verify")
        if verify:
            if not verify.get("passed", True):
                row["passed"] = False
                errors.append(f"alpha {alpha:g}: the recorded δ check failed")
            if verify.get("alpha") is not None and verify["alpha"] != alpha:
                row["passed"] = False
                errors.append(
                    f"alpha {alpha:g}: the manifest records {verify['alpha']!r}; "
                    "alpha must not round-trip through float32"
                )
        results.append(row)

    inputs = {}
    for role, expected in REFERENCE_INPUTS.items():
        actual = (doc.get(role) or {}).get("sha256")
        if actual:
            inputs[role] = {"expected": expected, "actual": actual,
                            "passed": actual == expected}
            if actual != expected:
                errors.append(
                    f"{role}: sha256 {actual} != published input {expected}"
                )
    return {
        "passed": not errors,
        "arms": results,
        "inputs": inputs,
        "errors": errors,
    }


def check_file(path, *, base_dir=None) -> dict:
    path = Path(path)
    doc = json.loads(path.read_text())
    report = check(doc, base_dir=base_dir if base_dir is not None else path.parent)
    report["manifest"] = str(path)
    return report


def format_report(report: dict) -> str:
    lines = [f"manifest: {report.get('manifest') or '(in memory)'}"]
    for row in report["arms"]:
        mark = "ok  " if row["passed"] else "FAIL"
        lines.append(f"  {mark} a{row['alpha']:g}  ar {str(row.get('ar'))[:12]}…  "
                     f"nar {str(row.get('nar'))[:12]}…")
    for role, item in report["inputs"].items():
        lines.append(f"  {'ok  ' if item['passed'] else 'FAIL'} {role}: "
                     f"{item['actual'][:12]}…")
    if report["passed"]:
        lines.append("conformance: PASS — converted AR/NAR match the published pair")
    else:
        lines.append("conformance: FAIL")
        for error in report["errors"]:
            lines.append(f"  - {error}")
    return "\n".join(lines)
