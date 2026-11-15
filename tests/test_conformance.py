"""The conformance gate: embedded published hashes, shapes, and failures."""
from __future__ import annotations

import json

from loralab import conformance


def arm(alpha, ar=None, nar=None, *, path=None, verify=True):
    ref = conformance.REFERENCE[alpha]
    out = {
        "alpha": alpha,
        "converted": {
            "ar": {"sha256": ar if ar is not None else ref["ar"],
                   "size_bytes": ref["ar_bytes"]},
            "nar": {"sha256": nar if nar is not None else ref["nar"],
                    "size_bytes": ref["nar_bytes"]},
        },
    }
    if path:
        out["converted"]["ar"]["path"] = str(path)
    if verify:
        out["verify"] = {"kind": "delta", "passed": True, "alpha": alpha}
    return out


def doc(*, alphas=(0.1, 0.2), carrier=None, donor=None, arms=None):
    return {
        "kind": "mix",
        "carrier": {"name": "v3", "sha256": carrier
                    or conformance.REFERENCE_INPUTS["carrier"]},
        "donor": {"name": "instr", "sha256": donor
                  or conformance.REFERENCE_INPUTS["donor"]},
        "arms": arms if arms is not None else [arm(a) for a in alphas],
    }


def test_reference_has_the_two_published_arms():
    assert set(conformance.REFERENCE) == {0.1, 0.2}
    for item in conformance.REFERENCE.values():
        assert len(item["ar"]) == 64 and len(item["nar"]) == 64


def test_a_matching_manifest_passes():
    report = conformance.check(doc())
    assert report["passed"] is True
    assert report["errors"] == []
    assert [row["alpha"] for row in report["arms"]] == [0.1, 0.2]
    assert all(row["passed"] for row in report["arms"])


def test_a_changed_sha_fails_and_names_the_arm():
    bad = doc(arms=[arm(0.1, ar="0" * 64), arm(0.2)])
    report = conformance.check(bad)
    assert report["passed"] is False
    assert any("alpha 0.1 ar" in error for error in report["errors"])
    assert not report["arms"][0]["passed"]
    assert report["arms"][1]["passed"]


def test_a_missing_arm_fails():
    report = conformance.check(doc(alphas=(0.1,)))
    assert report["passed"] is False
    assert any("alpha 0.2: arm is missing" in e for e in report["errors"])


def test_a_missing_converted_pair_fails():
    bad = doc(arms=[{"alpha": 0.1, "converted": {}}, arm(0.2)])
    report = conformance.check(bad)
    assert report["passed"] is False
    assert any("(missing)" in error for error in report["errors"])


def test_wrong_input_hashes_fail():
    report = conformance.check(doc(carrier="1" * 64, donor="2" * 64))
    assert report["passed"] is False
    assert not report["inputs"]["carrier"]["passed"]
    assert not report["inputs"]["donor"]["passed"]


def test_the_published_dict_shape_is_accepted():
    published = {
        "arms": {
            "0.1": {"alpha": 0.1, "ar": conformance.REFERENCE[0.1]["ar"],
                    "nar": conformance.REFERENCE[0.1]["nar"],
                    "ar_bytes": conformance.REFERENCE[0.1]["ar_bytes"],
                    "nar_bytes": conformance.REFERENCE[0.1]["nar_bytes"]},
            "0.2": {"alpha": 0.2, "ar": conformance.REFERENCE[0.2]["ar"],
                    "nar": conformance.REFERENCE[0.2]["nar"],
                    "ar_bytes": conformance.REFERENCE[0.2]["ar_bytes"],
                    "nar_bytes": conformance.REFERENCE[0.2]["nar_bytes"]},
        }
    }
    report = conformance.check(published)
    assert report["passed"] is True


def test_a_float32_alpha_is_not_0_1():
    bad = doc(arms=[arm(0.1), arm(0.2)])
    bad["arms"][0]["verify"]["alpha"] = 0.10000000149011612
    report = conformance.check(bad)
    assert report["passed"] is False
    assert any("float32" in error for error in report["errors"])


def test_a_recorded_delta_failure_fails():
    bad = doc()
    bad["arms"][0]["verify"]["passed"] = False
    report = conformance.check(bad)
    assert report["passed"] is False
    assert any("δ check failed" in error for error in report["errors"])


def test_a_manifest_without_arms_fails():
    report = conformance.check({"arms": []})
    assert report["passed"] is False
    assert "manifest has no arms" in report["errors"]


def test_check_file_hashes_the_files_it_names(tmp_path):
    path = tmp_path / "ar.safetensors"
    path.write_bytes(b"not the published file")
    bad = doc(arms=[arm(0.1, path=path), arm(0.2)])
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps(bad))
    report = conformance.check_file(manifest)
    assert report["passed"] is False
    assert report["manifest"] == str(manifest)
    assert any("file sha256" in error for error in report["errors"])
    assert any("bytes" in error for error in report["errors"])


def test_check_file_reports_a_missing_file(tmp_path):
    bad = doc(arms=[arm(0.1, path=tmp_path / "gone.safetensors"), arm(0.2)])
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps(bad))
    report = conformance.check_file(manifest)
    assert report["passed"] is False
    assert any("file is missing" in error for error in report["errors"])


def test_format_report_shows_pass_and_fail():
    passed = conformance.format_report(conformance.check(doc()))
    assert "conformance: PASS" in passed
    assert "a0.1" in passed and "a0.2" in passed
    failed = conformance.format_report(conformance.check(
        doc(arms=[arm(0.1, ar="0" * 64), arm(0.2)])))
    assert "conformance: FAIL" in failed
    assert "  - alpha 0.1 ar" in failed
