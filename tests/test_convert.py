"""The converter runner: pinning, caching, subprocess behaviour, outputs."""
from __future__ import annotations

import dataclasses

import pytest

from loralab import convert, gcs
from loralab import merge

pytest.importorskip("torch")


def test_run_writes_and_reports_the_pair(tmp_path, fake_converter, weights_factory):
    fused = weights_factory("merged.safetensors")
    script = fake_converter()
    logs = []
    result = convert.run(script, fused, tmp_path / "out", log=logs.append)
    assert result["byte_verified"] is True
    assert result["stem"] == "merged"
    assert set(result["outputs"]) == {"ar", "nar"}
    ar = result["outputs"]["ar"]
    assert ar["name"] == "merged_ar.safetensors"
    assert ar["rank"] == 4
    assert ar["size_bytes"] > 0
    assert ar["sha256"] == merge.sha256_file(ar["path"])
    assert result["command"][1] == str(script)
    assert any("byte-level verification" in line for line in result["stdout_tail"])
    assert any("converter chatter" in line for line in logs)


def test_run_rejects_a_missing_fused_input(tmp_path, fake_converter):
    with pytest.raises(convert.ConvertError, match="fused input not found"):
        convert.run(fake_converter(), tmp_path / "nope.safetensors", tmp_path)


def test_run_fails_on_a_nonzero_exit(tmp_path, fake_converter, weights_factory):
    with pytest.raises(convert.ConvertError, match="exited 7"):
        convert.run(fake_converter(exit_code=7), weights_factory("x.safetensors"),
                    tmp_path / "out")


def test_run_fails_when_a_branch_is_missing(tmp_path, fake_converter,
                                            weights_factory):
    with pytest.raises(convert.ConvertError, match="wrote no _nar"):
        convert.run(fake_converter(write_nar=False),
                    weights_factory("x.safetensors"), tmp_path / "out")


def test_run_warns_when_the_verification_line_is_absent(tmp_path, fake_converter,
                                                        weights_factory):
    logs = []
    result = convert.run(fake_converter(verify=False),
                         weights_factory("x.safetensors"), tmp_path / "out",
                         log=logs.append)
    assert result["byte_verified"] is False
    assert any("WARNING" in line for line in logs)


def test_run_finds_outputs_written_under_another_stem(tmp_path, fake_converter,
                                                      weights_factory):
    result = convert.run(fake_converter(stem="other"),
                         weights_factory("x.safetensors"), tmp_path / "out")
    assert result["outputs"]["ar"]["name"] == "other_ar.safetensors"
    assert result["outputs"]["nar"]["name"] == "other_nar.safetensors"


def test_run_times_out_and_kills_the_converter(tmp_path, fake_converter,
                                               weights_factory):
    with pytest.raises(convert.ConvertError, match="did not finish"):
        convert.run(fake_converter(sleep=5), weights_factory("x.safetensors"),
                    tmp_path / "out", timeout=0.2)


def test_run_reports_an_unstartable_converter(tmp_path, weights_factory, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(convert.subprocess, "Popen", boom)
    with pytest.raises(convert.ConvertError, match="cannot start"):
        convert.run(tmp_path / "absent.py", weights_factory("x.safetensors"),
                    tmp_path / "out")


def test_resolve_prefers_an_explicit_local_path(cfg, fake_converter):
    script = fake_converter()
    got = convert.resolve(cfg, path=script)
    assert got["source"] == "local"
    assert got["path"] == script
    assert got["sha256"] == merge.sha256_file(script)
    assert got["pinned"] is False


def test_resolve_rejects_a_local_path_with_the_wrong_pin(cfg, fake_converter):
    cfg = dataclasses.replace(cfg, converter_sha256="0" * 64)
    with pytest.raises(convert.ConvertError, match="mismatch"):
        convert.resolve(cfg, path=fake_converter())


def test_resolve_accepts_a_local_path_with_the_right_pin(cfg, fake_converter):
    script = fake_converter()
    cfg = dataclasses.replace(cfg, converter_sha256=merge.sha256_file(script))
    got = convert.resolve(cfg, path=script)
    assert got["pinned"] is True


def test_resolve_reports_a_missing_explicit_path(cfg, tmp_path):
    with pytest.raises(convert.ConvertError, match="converter not found"):
        convert.resolve(cfg, path=tmp_path / "absent.py")


def test_resolve_downloads_pins_then_reuses_the_cache(cfg, fake_converter,
                                                      monkeypatch):
    script = fake_converter()
    calls = []

    def fake_download(bucket, obj, dest, token=None, log=None):
        calls.append((bucket, obj, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(script.read_bytes())
        if log:
            log("downloaded")
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    logs = []
    first = convert.resolve(cfg, token=gcs.Token("test"), log=logs.append)
    assert first["source"] == "download"
    assert first["pinned"] is False
    pin = convert.cached_path(cfg).with_suffix(".py.sha256")
    assert pin.is_file()
    assert pin.read_text().strip() == first["sha256"]
    assert any("pinned sha256" in line for line in logs)

    second = convert.resolve(cfg, token=gcs.Token("test"))
    assert second["source"] == "cache"
    assert second["pinned"] is True
    assert len(calls) == 1
    assert calls[0][0] == "demo-bucket"


def test_resolve_uses_an_explicit_pin_without_writing_one(cfg, fake_converter,
                                                          monkeypatch):
    script = fake_converter()
    cfg = dataclasses.replace(cfg, converter_sha256=merge.sha256_file(script))

    def fake_download(bucket, obj, dest, token=None, log=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(script.read_bytes())
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    got = convert.resolve(cfg, token=gcs.Token("test"))
    assert got["pinned"] is True
    assert not convert.cached_path(cfg).with_suffix(".py.sha256").is_file()


def test_resolve_rejects_a_tampered_cache(cfg, fake_converter):
    script = fake_converter()
    cached = convert.cached_path(cfg)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(script.read_bytes())
    cached.with_suffix(".py.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(convert.ConvertError, match="cached converter"):
        convert.resolve(cfg)


def test_resolve_uses_a_cached_file_without_a_pin(cfg, fake_converter):
    cached = convert.cached_path(cfg)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(fake_converter().read_bytes())
    got = convert.resolve(cfg)
    assert got["source"] == "cache"
    assert got["pinned"] is False


def test_resolve_explains_when_there_is_no_bucket_and_no_cache(cfg):
    cfg = dataclasses.replace(cfg, bucket="")
    with pytest.raises(convert.ConvertError, match="no bucket configured"):
        convert.resolve(cfg)


def test_cached_path_hangs_off_the_data_dir(cfg):
    assert convert.cached_path(cfg) == (
        cfg.data_dir / "cache" / "tools" / "convert_aitoolkit_yue2_lora.py")
    assert convert.cached_path(cfg, "other.py").name == "other.py"
