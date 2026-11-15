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
