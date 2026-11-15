"""MANIFEST.json: shape, atomic writes, per-arm sidecars, tolerant reads."""
from __future__ import annotations

import json

from loralab import manifests


def test_environment_reports_versions_and_tolerates_missing_packages(monkeypatch):
    import importlib.metadata as md

    def fake_version(name):
        if name == "torch":
            raise md.PackageNotFoundError(name)
        return "9.9"

    monkeypatch.setattr(md, "version", fake_version)
    env = manifests.environment()
    assert env["torch"] is None
    assert env["safetensors"] == "9.9"
    assert env["loralab"]


def test_build_and_write_are_stable_and_atomic(tmp_path):
    doc = manifests.build(
        command=["loralab", "mix"], carrier={"name": "c"}, donor={"name": "d"},
        arms=[{"alpha": 0.1}], converter={"path": "/x.py"},
    )
    assert doc["schema"] == manifests.SCHEMA
    assert doc["kind"] == "mix"
    assert doc["identity"]["reproducible"] == "converted AR/NAR sha256"
    path = manifests.write(tmp_path / "MANIFEST.json", doc)
    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text())["carrier"]["name"] == "c"


def test_sidecar_carries_one_arm(tmp_path):
    doc = manifests.build(command=[], carrier={}, donor={}, arms=[])
    arm = {"alpha": 0.2, "stem": "run_a0.2", "verify": {"passed": True}}
    path = manifests.write_sidecar(doc, arm, tmp_path)
    assert path.name == "run_a0.2.json"
    sidecar = json.loads(path.read_text())
    assert sidecar["arms"] == [arm]
    assert sidecar["arm"]["alpha"] == 0.2


def test_sidecar_falls_back_to_the_alpha_for_a_name(tmp_path):
    path = manifests.write_sidecar({"arms": []}, {"alpha": 0.3}, tmp_path)
    assert path.name == "a0.3.json"


def test_read_is_tolerant_of_missing_and_corrupt_files(tmp_path):
    assert manifests.read(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert manifests.read(bad) is None
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}')
    assert manifests.read(good) == {"a": 1}
