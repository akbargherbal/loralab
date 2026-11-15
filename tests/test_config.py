import os
from pathlib import Path

import pytest

from loralab.config import (Config, ConfigError, load_config, write_default_config,
                            DEFAULT_ROOT, DEFAULT_PREFIXES)


def test_defaults_have_no_bucket_but_everything_after_it(tmp_path):
    cfg = load_config(config_path=tmp_path / "absent.toml", environ={})
    assert cfg.bucket == ""
    assert cfg.root == DEFAULT_ROOT
    assert cfg.prefixes == tuple(DEFAULT_PREFIXES)
    assert cfg.converter_object.startswith("audiocpp_inference/converter/")
    assert cfg.upload_prefix.startswith("_prototypes/")
    assert cfg.host == "127.0.0.1"
    assert cfg.config_from_file is False


def test_require_bucket_explains_the_three_ways(tmp_path):
    cfg = load_config(config_path=tmp_path / "absent.toml", environ={})
    with pytest.raises(ConfigError) as err:
        cfg.require_bucket()
    assert "--bucket" in str(err.value) and "LORALAB_BUCKET" in str(err.value)


def test_file_then_env_then_cli(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('bucket = "from-file"\nroot = "FILE_ROOT"\nport = 7000\n')
    cfg = load_config(config_path=path, environ={"LORALAB_BUCKET": "from-env"},
                      cli={"port": 9000})
    assert cfg.bucket == "from-env"      # env beats file
    assert cfg.root == "FILE_ROOT"       # file survives when env is silent
    assert cfg.port == 9000              # cli beats file
    assert cfg.config_from_file is True


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('bucket = "b"\nbuckets = "typo"\n')
    with pytest.raises(ConfigError, match="buckets"):
        load_config(config_path=path, environ={})


def test_init_writes_a_working_file_with_a_blank_bucket(tmp_path):
    path = tmp_path / "nested" / "config.toml"
    write_default_config(path)
    cfg = load_config(config_path=path, environ={})
    assert cfg.bucket == ""
    assert cfg.root == DEFAULT_ROOT
    with pytest.raises(ConfigError) as err:
        cfg.require_bucket()
    assert "--bucket" in str(err.value)


def test_init_refuses_to_clobber(tmp_path):
    path = tmp_path / "config.toml"
    write_default_config(path)
    with pytest.raises(ConfigError, match="already exists"):
        write_default_config(path)


def test_empty_root_is_legal(tmp_path):
    cfg = Config(bucket="b", root="", data_dir=tmp_path, config_path=tmp_path / "c.toml")
    assert cfg.object_prefix("loras/") == "loras/"
    assert cfg.display_scope() == "gs://b"


def test_env_prefixes_and_data_dir(tmp_path):
    cfg = load_config(
        config_path=tmp_path / "absent.toml",
        environ={"LORALAB_PREFIXES": "a/, b/", "LORALAB_DATA_DIR": str(tmp_path / "d")},
    )
    assert cfg.prefixes == ("a/", "b/")
    assert cfg.data_dir == tmp_path / "d"


def test_corrupt_config_is_reported_not_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("bucket = [\n")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(config_path=path, environ={})


def test_port_and_data_dir_from_env(tmp_path):
    cfg = load_config(config_path=tmp_path / "a.toml",
                      environ={"LORALAB_PORT": "8001",
                               "LORALAB_DATA_DIR": str(tmp_path / "dd")})
    assert cfg.port == 8001
    assert cfg.data_dir == tmp_path / "dd"


def test_data_dir_and_prefixes_from_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(f'bucket = "b"\ndata_dir = "{tmp_path}/from-file"\n'
                    'prefixes = ["one/", "two/"]\n')
    cfg = load_config(config_path=path, environ={})
    assert cfg.data_dir == tmp_path / "from-file"
    assert cfg.prefixes == ("one/", "two/")


def test_derived_paths_hang_off_the_data_dir(tmp_path):
    cfg = Config(bucket="b", data_dir=tmp_path, config_path=tmp_path / "c.toml")
    assert cfg.catalog_path == tmp_path / "catalog.json"
    assert cfg.jobs_dir == tmp_path / "jobs"
    assert cfg.cache_dir == tmp_path / "cache"
