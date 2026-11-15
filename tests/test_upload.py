"""Uploads: prefix discipline, create-only previews, and the write itself."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from loralab import gcs, manifests, upload
from loralab.config import Config


def make_mix(tmp_path, *, alphas=(0.1,), command=None, sidecars=True,
             converted=True, carrier="carrier.safetensors",
             donor="donor.safetensors", stem=None):
    root = tmp_path / "mix"
    root.mkdir()
    arms = []
    for alpha in alphas:
        arm_dir = root / f"a{alpha:g}"
        arm_dir.mkdir()
        tag = f"{stem or 'carrier__donor'}__a{alpha:g}" if stem is None \
            else f"{stem}_a{alpha:g}"
        fused = arm_dir / f"{tag}.safetensors"
        fused.write_bytes(b"fused-" + str(alpha).encode())
        arm = {
            "alpha": alpha,
            "stem": tag,
            "out_dir": str(arm_dir),
            "fused": {"path": str(fused), "name": fused.name,
                      "size_bytes": fused.stat().st_size,
                      "sha256": "f" * 64},
            "verify": {"kind": "delta", "passed": True, "alpha": alpha},
        }
        if converted:
            ar = arm_dir / f"{tag}_ar.safetensors"
            nar = arm_dir / f"{tag}_nar.safetensors"
            ar.write_bytes(b"ar")
            nar.write_bytes(b"nar")
            arm["converted"] = {
                "ar": {"path": str(ar), "name": ar.name, "sha256": "a" * 64},
                "nar": {"path": str(nar), "name": nar.name, "sha256": "n" * 64},
            }
        arms.append(arm)
        if sidecars:
            manifests.write_sidecar({}, arm, arm_dir)
    doc = manifests.build(
        command=command or ["loralab", "mix", "--alpha", "0.1"],
        carrier={"name": carrier, "sha256": "c" * 64},
        donor={"name": donor, "sha256": "d" * 64},
        arms=arms,
    )
    path = root / "MANIFEST.json"
    manifests.write(path, doc)
    return path, doc


@pytest.fixture
def upload_cfg(tmp_path):
    return Config(bucket="bucket", root="ROOT", prefixes=("loras/",),
                  data_dir=tmp_path / "data",
                  config_path=tmp_path / "config.toml")


def plan(cfg, path, doc=None, **kwargs):
    kwargs.setdefault("check_existing", False)
    return upload.plan_uploads(cfg, path, doc, **kwargs)


# ---------------------------------------------------------------------------
# destination validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["", "  ", "/abs", "a/../b", "a//b", "./a"])
def test_prefix_rejects_unsafe_values(value):
    with pytest.raises(upload.UploadError):
        upload.normalize_prefix(value)


def test_prefix_strips_trailing_slashes_and_normalises_backslashes():
    assert upload.normalize_prefix("_prototypes/mixes/") == "_prototypes/mixes"
    assert upload.normalize_prefix("a\\b") == "a/b"


@pytest.mark.parametrize("value", ["", "/abs", "../x", "a/../b", "a//b"])
def test_name_rejects_escape_attempts(value):
    with pytest.raises(upload.UploadError):
        upload.normalize_name(value)


def test_name_allows_nested_relative_directories():
    assert upload.normalize_name("v3/0.1") == "v3/0.1"


def test_production_detection_is_any_segment():
    assert upload.is_production_prefix("loras/x")
    assert upload.is_production_prefix("mixes/LoRas/x")
    assert not upload.is_production_prefix("_prototypes/mixes")


def test_slug_defaults_to_carrier_donor(tmp_path):
    path, doc = make_mix(tmp_path)
    assert upload.mix_slug(doc) == "carrier__donor"


def test_slug_uses_an_explicit_stem(tmp_path):
    path, doc = make_mix(tmp_path, stem="v3instr",
                         command=["loralab", "mix", "--stem", "v3instr_a0.1"])
    assert upload.mix_slug(doc) == "v3instr_a0.1"


def test_slug_falls_back_to_the_arm_stem(tmp_path):
    doc = {"carrier": {}, "donor": {},
           "arms": [{"alpha": 0.1, "stem": "a__b__a0.1"}]}
    assert upload.mix_slug(doc) == "a__b"


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------
def test_preview_names_every_object_and_puts_the_manifest_last(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path, alphas=(0.1, 0.2))
    preview = plan(upload_cfg, path, doc)
    assert preview["bucket"] == "bucket"
    assert preview["destination"] == "_prototypes/mixes/carrier__donor"
    assert preview["display"] == (
        "gs://bucket/ROOT/_prototypes/mixes/carrier__donor/")
    names = [o["name"] for o in preview["objects"]]
    assert names[-1] == "MANIFEST.json"
    assert "carrier__donor__a0.1_ar.safetensors" in names
    assert "carrier__donor__a0.2_nar.safetensors" in names
    assert all(o["object"].startswith("ROOT/_prototypes/mixes/carrier__donor/")
               for o in preview["objects"])
    assert preview["count"] == len(names) == 7   # 2 arms × (ar,nar,sidecar) + manifest
    assert preview["total_bytes"] > 0
    assert preview["blockers"] == [] and preview["warnings"] == []
    assert "--no-include-fused" in preview["command"]


def test_preview_can_include_the_fused_file(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc, include_fused=True)
    assert any(o["kind"] == "fused" for o in preview["objects"])
    assert "--include-fused" in preview["command"]


def test_preview_refuses_loras_without_an_explicit_flag(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc, prefix="loras/mixes")
    assert preview["production"] is True
    assert preview["blockers"] and "production" in preview["blockers"][0]
    allowed = plan(upload_cfg, path, doc, prefix="loras/mixes",
                   allow_production_prefix=True)
    assert allowed["blockers"] == []


def test_preview_requires_an_overwrite_for_existing_keys(tmp_path, upload_cfg,
                                                         monkeypatch):
    path, doc = make_mix(tmp_path)
    monkeypatch.setattr(gcs, "stat_or_none",
                        lambda *a: {"generation": "17"})
    blocked = plan(upload_cfg, path, doc, check_existing=True, token=object())
    assert blocked["existing"] == blocked["count"]
    assert "already exist" in blocked["blockers"][0]
    assert all(o["generation"] == "17" for o in blocked["objects"])
    allowed = plan(upload_cfg, path, doc, check_existing=True, token=object(),
                   overwrite=True)
    assert allowed["blockers"] == []


def test_preview_without_a_token_explains_why_it_cannot_check(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    with pytest.raises(upload.UploadError, match="credentials"):
        plan(upload_cfg, path, doc, check_existing=True, token=None)


def test_preview_rejects_a_bucketless_config(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    cfg = Config(bucket="", root="ROOT", data_dir=upload_cfg.data_dir,
                 config_path=upload_cfg.config_path)
    with pytest.raises(upload.UploadError, match="bucket"):
        plan(cfg, path, doc)


def test_preview_rejects_a_non_mix_manifest(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    doc["kind"] = "catalog"
    manifests.write(path, doc)
    with pytest.raises(upload.UploadError, match="not a mix"):
        plan(upload_cfg, path, doc)


def test_preview_rejects_a_missing_artifact(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    Path(doc["arms"][0]["converted"]["ar"]["path"]).unlink()
    with pytest.raises(upload.UploadError, match="missing"):
        plan(upload_cfg, path, doc)


def test_preview_rejects_colliding_object_names(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path, alphas=(0.1,))
    doc["arms"].append({**doc["arms"][0]})
    with pytest.raises(upload.UploadError, match="same object name"):
        plan(upload_cfg, path, doc)


def test_preview_rejects_a_manifest_without_arms(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    doc["arms"] = []
    with pytest.raises(upload.UploadError, match="no arms"):
        plan(upload_cfg, path, doc)


def test_fused_only_mix_warns_but_is_allowed(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path, converted=False)
    preview = plan(upload_cfg, path, doc)
    assert preview["warnings"] and "no converted pair" in preview["warnings"][0]
    assert preview["blockers"] == []


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_upload(monkeypatch):
    calls = []

    def fake(bucket, obj, source, token=None, *, overwrite=False,
             generation=None, log=None):
        calls.append({"bucket": bucket, "object": obj, "overwrite": overwrite,
                      "generation": generation})
        return {"name": obj, "size": Path(source).stat().st_size,
                "sha256": "x" * 64, "md5_verified": True,
                "generation": "99", "crc32c": "AAAAAA==",
                "updated": "2026-11-15T00:00:00Z"}

    monkeypatch.setattr(gcs, "upload_object", fake)
    monkeypatch.setattr(gcs, "stat_or_none", lambda *a: None)
    monkeypatch.setattr(gcs, "resolve_token", lambda *a, **k: gcs.Token("test"))
    return calls


def test_apply_writes_every_object_and_records_it(tmp_path, upload_cfg,
                                                  fake_upload):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc)
    record_path = tmp_path / "upload.json"
    record = upload.apply_plan(upload_cfg, preview, token=gcs.Token("test"),
                               record_path=record_path)
    assert record["count"] == preview["count"]
    assert record["total_bytes"] == sum(o["size_bytes"] for o in preview["objects"])
    assert len(fake_upload) == preview["count"]
    assert all(c["generation"] is None for c in fake_upload)
    saved = json.loads(record_path.read_text())
    assert saved["kind"] == "upload"
    assert saved["objects"][-1]["name"] == "MANIFEST.json"
    assert saved["display"] == preview["display"]


def test_apply_pins_the_generation_on_overwrite(tmp_path, upload_cfg,
                                                fake_upload):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc, overwrite=True)
    for item in preview["objects"]:
        item["exists"] = True
        item["generation"] = "123"
    upload.apply_plan(upload_cfg, preview, token=gcs.Token("test"))
    assert all(c["overwrite"] is True and c["generation"] == "123"
               for c in fake_upload)


def test_apply_refuses_a_blocked_plan(tmp_path, upload_cfg, fake_upload):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc, prefix="loras/mixes")
    with pytest.raises(upload.UploadError, match="production"):
        upload.apply_plan(upload_cfg, preview, token=gcs.Token("test"))
    assert fake_upload == []


def test_apply_turns_a_transport_error_into_an_upload_error(tmp_path, upload_cfg,
                                                            monkeypatch):
    path, doc = make_mix(tmp_path)
    preview = plan(upload_cfg, path, doc)

    def boom(*a, **k):
        raise gcs.GcsError("HTTP 412 — object exists")

    monkeypatch.setattr(gcs, "upload_object", boom)
    monkeypatch.setattr(gcs, "resolve_token", lambda *a, **k: gcs.Token("test"))
    with pytest.raises(upload.UploadError, match="412"):
        upload.apply_plan(upload_cfg, preview, token=gcs.Token("test"))


# ---------------------------------------------------------------------------
# details
# ---------------------------------------------------------------------------
def test_dedupe_key_changes_when_the_destination_changes(tmp_path, upload_cfg):
    path, _ = make_mix(tmp_path)
    one = upload.dedupe_key(upload_cfg, path, prefix="_prototypes/mixes",
                            name="x", include_fused=False, overwrite=False,
                            allow_production_prefix=False)
    two = upload.dedupe_key(upload_cfg, path, prefix="_prototypes/mixes",
                            name="y", include_fused=False, overwrite=False,
                            allow_production_prefix=False)
    assert one.startswith("upload:") and one != two


def test_describe_marks_existing_objects_and_blockers(tmp_path, upload_cfg,
                                                      monkeypatch):
    path, doc = make_mix(tmp_path)
    monkeypatch.setattr(gcs, "stat_or_none",
                        lambda *a: {"generation": "1"})
    preview = plan(upload_cfg, path, doc, check_existing=True, token=object())
    text = upload.describe(preview)
    assert "destination:" in text and "exists" in text
    assert "BLOCKED:" in text and "already exist" in text


def test_run_uses_the_mix_job_directory(tmp_path, upload_cfg, fake_upload):
    path, doc = make_mix(tmp_path)
    jobs_dir = upload_cfg.jobs_dir / "mixjob1"
    jobs_dir.mkdir(parents=True)
    manifests.write(jobs_dir / "MANIFEST.json", doc)

    class Ctx:
        def __init__(self):
            self.dir = upload_cfg.data_dir / "jobs" / "uploadjob1"
            self.dir.mkdir(parents=True)
            self.lines = []

        def log(self, message):
            self.lines.append(message)

        def cancelled(self):
            return False

    ctx = Ctx()
    result = upload.run(ctx, upload_cfg, {"job": "mixjob1"},
                        token=gcs.Token("test"))
    assert result["count"] == len(doc["arms"]) * 3 + 1
    assert (ctx.dir / "upload.json").is_file()
    assert any("uploaded" in line for line in ctx.lines)


def test_run_without_a_source_job_is_an_error(tmp_path, upload_cfg):
    with pytest.raises(upload.UploadError, match="source mix job"):
        upload.run(object(), upload_cfg, {}, token=gcs.Token("test"))


def test_slug_ignores_a_dangling_stem_flag(tmp_path):
    doc = {"command": ["loralab", "mix", "--stem"],
           "carrier": {"name": "c.safetensors"},
           "donor": {"name": "d.safetensors"}}
    assert upload.mix_slug(doc) == "c__d"


def test_slug_falls_back_to_mix_without_names_or_arms():
    assert upload.mix_slug({"carrier": {}, "donor": {}, "arms": []}) == "mix"


def test_preview_reads_the_manifest_from_disk(tmp_path, upload_cfg):
    path, _ = make_mix(tmp_path)
    preview = upload.plan_uploads(upload_cfg, path, check_existing=False)
    assert preview["count"] == 4


def test_preview_rejects_a_missing_manifest(tmp_path, upload_cfg):
    with pytest.raises(upload.UploadError, match="no readable manifest"):
        plan(upload_cfg, tmp_path / "gone.json", check_existing=False)


def test_preview_requires_a_path_for_every_artifact(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    doc["arms"][0]["converted"]["ar"].pop("path")
    with pytest.raises(upload.UploadError, match="has no path"):
        plan(upload_cfg, path, doc)


def test_preview_rejects_an_arm_with_neither_artifact(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path)
    doc["arms"][0].pop("converted")
    doc["arms"][0].pop("fused")
    with pytest.raises(upload.UploadError, match="neither a converted pair"):
        plan(upload_cfg, path, doc)


def test_preview_resolves_a_relative_out_dir_for_the_sidecar(tmp_path,
                                                            upload_cfg):
    path, doc = make_mix(tmp_path)
    doc["arms"][0]["out_dir"] = "a0.1"
    preview = plan(upload_cfg, path, doc)
    assert any(o["kind"] == "sidecar" for o in preview["objects"])


def test_describe_includes_warnings(tmp_path, upload_cfg):
    path, doc = make_mix(tmp_path, converted=False)
    text = upload.describe(plan(upload_cfg, path, doc))
    assert "warning: this mix has no converted pair" in text


def test_run_logs_warnings_for_a_fused_only_mix(tmp_path, upload_cfg,
                                                fake_upload):
    path, doc = make_mix(tmp_path, converted=False)
    jobs_dir = upload_cfg.jobs_dir / "mixjob1"
    jobs_dir.mkdir(parents=True)
    manifests.write(jobs_dir / "MANIFEST.json", doc)

    class Ctx:
        def __init__(self):
            self.dir = upload_cfg.data_dir / "jobs" / "uploadjob1"
            self.dir.mkdir(parents=True)
            self.lines = []

        def log(self, message):
            self.lines.append(message)

        def cancelled(self):
            return False

    ctx = Ctx()
    upload.run(ctx, upload_cfg, {"job": "mixjob1"}, token=gcs.Token("test"))
    assert any("warning: this mix has no converted pair" in line
               for line in ctx.lines)


def test_run_refuses_a_blocked_plan(tmp_path, upload_cfg, fake_upload):
    path, doc = make_mix(tmp_path)
    jobs_dir = upload_cfg.jobs_dir / "mixjob1"
    jobs_dir.mkdir(parents=True)
    manifests.write(jobs_dir / "MANIFEST.json", doc)

    class Ctx:
        dir = tmp_path / "upload"
        lines = []

        def log(self, message):
            self.lines.append(message)

        def cancelled(self):
            return False

    with pytest.raises(upload.UploadError, match="production"):
        upload.run(Ctx(), upload_cfg,
                   {"job": "mixjob1", "prefix": "loras/mixes"},
                   token=gcs.Token("test"))


def test_run_honours_cancel_before_upload(tmp_path, upload_cfg, fake_upload):
    path, doc = make_mix(tmp_path)
    jobs_dir = upload_cfg.jobs_dir / "mixjob1"
    jobs_dir.mkdir(parents=True)
    manifests.write(jobs_dir / "MANIFEST.json", doc)

    class Ctx:
        dir = tmp_path / "upload"
        lines = []

        def log(self, message):
            self.lines.append(message)

        def cancelled(self):
            return True

    with pytest.raises(upload.UploadError, match="cancel requested"):
        upload.run(Ctx(), upload_cfg, {"job": "mixjob1"},
                   token=gcs.Token("test"))
