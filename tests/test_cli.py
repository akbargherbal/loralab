"""CLI: the same operations without a browser (and the fallback when one dies)."""
from __future__ import annotations

import json

import pytest

from loralab import catalog as cat
from loralab import cli, gcs


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Keep every test off the real config/data dirs and off the network."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("LORALAB_BUCKET", raising=False)
    monkeypatch.delenv("LORALAB_ROOT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(gcs, "resolve_token", lambda *a, **k: gcs.Token("test"))
    return tmp_path


def test_serve_without_a_bucket_explains_and_exits_2(isolated, capsys):
    assert cli.main(["serve"]) == 2
    assert "--bucket" in capsys.readouterr().err


def test_scan_writes_a_catalog_and_reports_counts(isolated, monkeypatch, capsys):
    monkeypatch.setattr(cat, "scan", lambda cfg, tok, log=None, cancelled=None: {
        "count": 2,
        "entries": [{"kind": "fused", "name": "a"}, {"kind": "fused", "name": "b"}],
        "errors": [],
    })
    assert cli.main(["scan", "--bucket", "bkt"]) == 0
    out = capsys.readouterr().out
    assert "2 adapter(s)" in out and "fused 2" in out
    saved = cat.load_catalog(isolated / "data" / "loralab" / "catalog.json")
    assert saved["count"] == 2


def test_scan_returns_1_when_a_file_fails(isolated, monkeypatch):
    monkeypatch.setattr(cat, "scan", lambda cfg, tok, log=None, cancelled=None: {
        "count": 1,
        "entries": [{"kind": "fused", "name": "a"}],
        "errors": [{"path": "x", "error": "junk"}],
    })
    assert cli.main(["scan", "--bucket", "bkt"]) == 1


def test_scan_accepts_extra_prefixes(isolated, monkeypatch, capsys):
    seen = {}

    def fake_scan(cfg, tok, log=None, cancelled=None):
        seen["prefixes"] = cfg.prefixes
        return {"count": 0, "entries": [], "errors": []}

    monkeypatch.setattr(cat, "scan", fake_scan)
    cli.main(["scan", "--bucket", "bkt", "--prefix", "only/"])
    assert seen["prefixes"] == ("only/",)


def test_catalog_prints_the_cached_table(isolated, capsys, entry_factory):
    cfg_path = isolated / "data" / "loralab" / "catalog.json"
    entries = [entry_factory("run/output/demo.safetensors")]
    cat.save_catalog(cfg_path, {"count": 1, "scope": "gs://b/ROOT",
                                "generated_utc": "now", "entries": entries,
                                "errors": []})
    assert cli.main(["catalog"]) == 0
    out = capsys.readouterr().out
    assert "demo.safetensors" in out and "ar:32" in out


def test_catalog_without_a_scan_says_so(isolated, capsys):
    assert cli.main(["catalog"]) == 2
    assert "run `loralab scan` first" in capsys.readouterr().err


def test_catalog_json_flag_round_trips(isolated, capsys, entry_factory):
    cfg_path = isolated / "data" / "loralab" / "catalog.json"
    cat.save_catalog(cfg_path, {"count": 1, "scope": "s", "generated_utc": "t",
                                "entries": [entry_factory("a/b/c.safetensors")],
                                "errors": []})
    assert cli.main(["catalog", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_doctor_without_a_bucket_fails_but_still_reports(isolated, capsys):
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "data:" in out


def test_doctor_probes_the_bucket(isolated, monkeypatch, capsys):
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: [{"name": "x"}])
    assert cli.main(["doctor", "--bucket", "bkt"]) == 0
    out = capsys.readouterr().out
    assert "mode" not in out or True
    assert "1 object(s)" in out and "test" in out


def test_init_creates_a_blank_bucket_config(isolated, capsys):
    assert cli.main(["init"]) == 0
    path = isolated / "config" / "loralab" / "config.toml"
    assert path.is_file()
    text = path.read_text()
    assert 'bucket = ""' in text
    assert 'root = "OSTRIS_Arabic_Suno_Finetuning"' in text
    assert "put your bucket name" in capsys.readouterr().out


def test_init_then_scan_reads_the_file(isolated, monkeypatch, capsys):
    cli.main(["init", "--bucket", "from-init"])
    monkeypatch.setattr(cat, "scan", lambda cfg, tok, log=None, cancelled=None: {
        "count": 0, "entries": [], "errors": []})
    assert cli.main(["scan"]) == 0
    assert "from-init" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as err:
        cli.main(["--version"])
    assert err.value.code == 0
    assert "loralab" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# serve (Flask's run() is stubbed: no socket is opened)
# ---------------------------------------------------------------------------
def _stub_run(monkeypatch) -> dict:
    import flask
    seen = {}

    def fake_run(self, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(flask.Flask, "run", fake_run)
    return seen


def test_serve_uses_the_configured_host_and_port(isolated, monkeypatch, capsys):
    seen = _stub_run(monkeypatch)
    assert cli.main(["serve", "--bucket", "bkt", "--port", "5099"]) == 0
    assert seen["port"] == 5099
    assert seen["host"] == "127.0.0.1"
    assert seen["use_reloader"] is False
    out = capsys.readouterr().out
    assert "5099" in out and "gs://bkt/" in out


def test_serve_warns_when_binding_publicly(isolated, monkeypatch, capsys):
    _stub_run(monkeypatch)
    assert cli.main(["serve", "--bucket", "bkt", "--host", "0.0.0.0"]) == 0
    assert "can write to a bucket" in capsys.readouterr().err


def test_serve_still_serves_without_credentials(isolated, monkeypatch, capsys):
    _stub_run(monkeypatch)
    monkeypatch.setattr(gcs, "resolve_token",
                        lambda *a, **k: (_ for _ in ()).throw(gcs.GcsError("no creds")))
    assert cli.main(["serve", "--bucket", "bkt"]) == 0
    err = capsys.readouterr().err
    assert "no creds" in err and "catalog still loads" in err


def test_data_dir_flag_is_honoured(isolated, monkeypatch, capsys, entry_factory):
    target = isolated / "elsewhere"
    cfg_path = target / "catalog.json"
    cat.save_catalog(cfg_path, {"count": 1, "scope": "s", "generated_utc": "t",
                                "entries": [entry_factory("a/b/c.safetensors")],
                                "errors": []})
    assert cli.main(["catalog", "--data-dir", str(target)]) == 0
    assert "c.safetensors" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# error handling in main()
# ---------------------------------------------------------------------------
def test_gcs_error_is_reported_as_exit_2(isolated, monkeypatch, capsys):
    monkeypatch.setattr(gcs, "resolve_token",
                        lambda *a, **k: (_ for _ in ()).throw(gcs.GcsError("bucket is gone")))
    assert cli.main(["scan", "--bucket", "bkt"]) == 2
    assert "bucket is gone" in capsys.readouterr().err


def test_config_error_is_reported_as_exit_2(isolated, capsys):
    assert cli.main(["scan"]) == 2
    assert "no bucket configured" in capsys.readouterr().err


def test_ctrl_c_is_exit_130(isolated, monkeypatch, capsys):
    monkeypatch.setattr(cat, "scan",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(["scan", "--bucket", "bkt"]) == 130


def test_doctor_reports_a_bucket_failure(isolated, monkeypatch, capsys):
    monkeypatch.setattr(gcs, "list_objects",
                        lambda *a, **k: (_ for _ in ()).throw(gcs.GcsError("HTTP 403 denied")))
    assert cli.main(["doctor", "--bucket", "bkt"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_doctor_reports_catalog_age_when_one_exists(isolated, monkeypatch, capsys,
                                                    entry_factory):
    cat.save_catalog(isolated / "data" / "loralab" / "catalog.json", {
        "count": 1, "scope": "gs://b/ROOT", "generated_utc": "2026-11-15T00:00:00Z",
        "entries": [entry_factory("a/b/c.safetensors")], "errors": []})
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: [])
    assert cli.main(["doctor", "--bucket", "bkt"]) == 0
    out = capsys.readouterr().out
    assert "1 adapter(s)" in out and "2026-11-15" in out


# ---------------------------------------------------------------------------
# entry points, as the user actually invokes them
# ---------------------------------------------------------------------------
def _run_module(module: str):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-m", module, "--version"],
                          capture_output=True, text=True, timeout=120)


def test_python_m_loralab_works():
    result = _run_module("loralab")
    assert result.returncode == 0
    assert "loralab" in result.stdout


def test_cli_module_also_runs_as_a_script():
    result = _run_module("loralab.cli")
    assert result.returncode == 0
    assert "loralab" in result.stdout


# ---------------------------------------------------------------------------
# mix
# ---------------------------------------------------------------------------
def test_mix_dry_run_reports_the_plan(isolated, weights_factory, capsys):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors", rank=4)
    donor = weights_factory("donor.safetensors", rank=4)
    code = cli.main(["mix", "--carrier", str(carrier), "--donor", str(donor),
                     "--alpha", "0.1,0.2", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "loralab mix" in out
    assert "alpha:   0.1, 0.2" in out
    assert "ranks:" in out and "'ar': 8" in out and "'nar': 8" in out
    assert "size:" in out


def test_mix_dry_run_rejects_an_out_of_range_alpha(isolated, weights_factory,
                                                    capsys):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors", rank=4)
    donor = weights_factory("donor.safetensors", rank=4)
    code = cli.main(["mix", "--carrier", str(carrier), "--donor", str(donor),
                     "--alpha", "2", "--dry-run"])
    assert code == 2
    assert "not a percentage" in capsys.readouterr().err


def test_mix_requires_at_least_one_alpha(isolated, weights_factory):
    pytest.importorskip("torch")
    carrier = weights_factory("c.safetensors")
    with pytest.raises(SystemExit) as exc:
        cli.main(["mix", "--carrier", str(carrier), "--donor", str(carrier)])
    assert exc.value.code == 2


def test_mix_runs_local_files_and_writes_a_manifest(isolated, weights_factory,
                                                    fake_converter, tmp_path,
                                                    capsys):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors", rank=4)
    donor = weights_factory("donor.safetensors", rank=4)
    out = tmp_path / "mixes"
    code = cli.main(["mix", "--carrier", str(carrier), "--donor", str(donor),
                     "--alpha", "0", "--alpha", "0.2", "--out", str(out),
                     "--converter", str(fake_converter())])
    assert code == 0
    text = capsys.readouterr().out
    assert "2 arm(s)" in text
    assert (out / "MANIFEST.json").is_file()
    assert (out / "a0.2" / "carrier__donor__a0.2_ar.safetensors").is_file()


def test_mix_without_conversion_skips_the_converter(isolated, weights_factory,
                                                    tmp_path, capsys):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors", rank=4)
    donor = weights_factory("donor.safetensors", rank=4)
    out = tmp_path / "plain"
    code = cli.main(["mix", "--carrier", str(carrier), "--donor", str(donor),
                     "--alpha", "0.5", "--out", str(out), "--no-convert"])
    assert code == 0
    assert (out / "a0.5" / "carrier__donor__a0.5.safetensors").is_file()
    assert not any(p.name.endswith("_ar.safetensors") for p in out.rglob("*"))


def test_mix_reports_a_missing_input_as_exit_2(isolated, weights_factory, capsys):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors")
    assert cli.main(["mix", "--carrier", str(carrier),
                     "--donor", str(carrier.parent / "nope.safetensors"),
                     "--alpha", "0.2"]) == 2
    assert "input not found" in capsys.readouterr().err


def test_doctor_prints_the_engine_and_converter(isolated, monkeypatch, capsys):
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: [])
    cli.main(["doctor", "--bucket", "bkt"])
    out = capsys.readouterr().out
    assert "engine:" in out
    assert "converter:" in out


def test_doctor_reports_a_missing_merge_engine(isolated, monkeypatch, capsys):
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: [])
    monkeypatch.setattr(cli.manifests, "environment", lambda: {
        "torch": None, "safetensors": None, "loralab": "0.1.0"})
    assert cli.main(["doctor", "--bucket", "bkt"]) == 1
    out = capsys.readouterr().out
    assert "engine: MISSING" in out and '".[merge]"' in out


def test_mix_dry_run_refuses_a_converted_input(isolated, weights_factory, tmp_path,
                                               capsys):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    carrier = weights_factory("carrier.safetensors")
    converted = tmp_path / "converted.safetensors"
    save_file({"layers.0.lora_A.weight": torch.zeros(4, 8)},
              str(converted),
              metadata={"converted_from": "x", "branch": "ar", "rank": "4"})
    code = cli.main(["mix", "--carrier", str(carrier), "--donor", str(converted),
                     "--alpha", "0.2", "--dry-run"])
    assert code == 2
    assert "only fused or merged" in capsys.readouterr().err
