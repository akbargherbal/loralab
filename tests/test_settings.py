"""Settings: atomic config writes, live reload, and the cache panel."""
from __future__ import annotations

import tomllib

import pytest

from loralab import gcs
from loralab.config import (ConfigError, Config, config_values, load_config,
                            validate_values, write_config)
from loralab.jobs import Jobs
from loralab.web.app import create_app


def values(**overrides):
    base = {
        "bucket": "new-bucket",
        "root": "ROOT/",
        "prefixes": "loras/\nmixes/",
        "converter_object": "tools/convert.py",
        "converter_sha256": "",
        "upload_prefix": "_prototypes/mixes/",
        "host": "127.0.0.1",
        "port": "5001",
        "data_dir": "/tmp/loralab-data",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# config writing
# ---------------------------------------------------------------------------
def test_validate_normalises_paths_prefixes_and_port():
    out = validate_values(values(prefixes="/a/\n\n b/ ", port="8080"))
    assert out["root"] == "ROOT"
    assert out["upload_prefix"] == "_prototypes/mixes"
    assert out["prefixes"] == ["a/", "b/"]
    assert out["port"] == 8080


@pytest.mark.parametrize("bad, message", [
    ({"upload_prefix": ""}, "upload_prefix"),
    ({"upload_prefix": "../escape"}, "upload_prefix"),
    ({"prefixes": ""}, "prefixes"),
    ({"prefixes": "a/../b"}, "invalid prefix"),
    ({"port": "nope"}, "port must be a number"),
    ({"port": "0"}, "between 1 and 65535"),
])
def test_validate_rejects_bad_values(bad, message):
    with pytest.raises(ConfigError, match=message):
        validate_values(values(**bad))


def test_write_config_round_trips_through_load(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, values(prefixes="loras/\nmixes/", port="5002"))
    doc = tomllib.loads(path.read_text())
    assert doc["bucket"] == "new-bucket"
    assert doc["prefixes"] == ["loras/", "mixes/"]
    assert doc["port"] == 5002
    loaded = load_config(config_path=path, environ={})
    assert loaded.bucket == "new-bucket"
    assert loaded.upload_prefix == "_prototypes/mixes"
    assert loaded.config_from_file is True
    assert loaded.port == 5002


def test_write_config_is_atomic_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, values())
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_config_escapes_quotes(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, values(bucket='a"b\\c'))
    loaded = load_config(config_path=path, environ={})
    assert loaded.bucket == 'a"b\\c'


def test_config_values_is_the_full_effective_set(tmp_path):
    cfg = Config(bucket="b", data_dir=tmp_path, config_path=tmp_path / "c.toml")
    out = config_values(cfg)
    assert out["bucket"] == "b"
    assert out["data_dir"] == str(tmp_path)
    assert out["port"] == 5000


# ---------------------------------------------------------------------------
# web settings
# ---------------------------------------------------------------------------
@pytest.fixture
def settings_app(cfg, monkeypatch):
    for name in ("LORALAB_BUCKET", "LORALAB_ROOT", "LORALAB_PORT",
                 "LORALAB_PREFIXES", "LORALAB_UPLOAD_PREFIX",
                 "LORALAB_CONVERTER_SHA256", "LORALAB_HOST",
                 "LORALAB_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    jobs = Jobs(cfg.jobs_dir)
    return create_app(cfg, jobs, token=gcs.Token("test"))


@pytest.fixture
def settings_client(settings_app):
    return settings_app.test_client()


def test_settings_page_shows_the_effective_config(settings_client, cfg):
    body = settings_client.get("/settings").get_data(as_text=True)
    assert "Settings" in body
    assert f'value="{cfg.bucket}"' in body
    assert "demo-bucket" in body
    assert "loras/" in body
    assert cfg.upload_prefix in body
    assert "auth: test" in body


def test_settings_save_writes_and_reloads(settings_client, cfg):
    response = settings_client.post("/settings", data=values(
        data_dir=str(cfg.data_dir), port="5009"))
    assert response.status_code == 303
    assert response.headers["Location"].endswith("/settings?saved=1")
    loaded = load_config(config_path=cfg.config_path, environ={})
    assert loaded.bucket == "new-bucket"
    assert loaded.root == "ROOT"
    assert loaded.prefixes == ("loras/", "mixes/")
    assert loaded.port == 5009
    assert loaded.config_from_file is True
    body = settings_client.get("/settings?saved=1").get_data(as_text=True)
    assert "Saved." in body


def test_settings_save_rejects_a_bad_port_and_writes_nothing(settings_client, cfg):
    response = settings_client.post("/settings", data=values(
        data_dir=str(cfg.data_dir), port="99999"))
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert "Not saved" in body and "between 1 and 65535" in body
    assert not cfg.config_path.exists()


def test_settings_lists_environment_overrides(settings_app, cfg, monkeypatch):
    monkeypatch.setenv("LORALAB_BUCKET", "from-env")
    body = settings_app.test_client().get("/settings").get_data(as_text=True)
    assert "Environment overrides present" in body
    assert "LORALAB_BUCKET" in body


def test_cache_panel_lists_and_sums_cached_files(settings_client, cfg):
    src = cfg.cache_dir / "src" / "bucket"
    src.mkdir(parents=True)
    (src / "carrier.safetensors").write_bytes(b"x" * 10)
    tools = cfg.cache_dir / "tools"
    tools.mkdir(parents=True)
    (tools / "convert.py").write_bytes(b"y" * 5)
    body = settings_client.get("/settings").get_data(as_text=True)
    assert "carrier.safetensors" in body
    assert "convert.py" in body
    assert "0.0 MB" in body          # both files are tiny
    assert "free disk" in body


def test_cache_report_counts_only_files(settings_app, cfg):
    src = cfg.cache_dir / "src" / "bucket"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "one").write_bytes(b"1234")
    from loralab.web.app import _cache_report
    report = _cache_report(cfg)
    sources = next(d for d in report["dirs"] if d["name"] == "sources")
    assert sources["count"] == 1
    assert sources["bytes"] == 4
    assert sources["files"][0]["name"] == "bucket/nested/one"


def test_settings_route_is_in_the_nav(settings_client):
    body = settings_client.get("/catalog").get_data(as_text=True)
    assert 'href="/settings"' in body
