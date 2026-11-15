"""Configuration: defaults, file, env, CLI — in that order of increasing authority.

The bucket is intentionally **blank** in the defaults. Everything after the bucket
(root, prefixes, upload prefix) is filled with working values for this project's
layout, so the first run needs exactly one thing from the user:

    loralab serve --bucket my-bucket

or `bucket = "..."` in `~/.config/loralab/config.toml`.
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

APP = "loralab"


class ConfigError(RuntimeError):
    """Bad or missing configuration, with a message the user can act on."""


DEFAULT_ROOT = "OSTRIS_Arabic_Suno_Finetuning"
DEFAULT_PREFIXES = [
    "loras/",
    "v3_arabmaqamrock_lora/",
    "arabic_instrumental_lora/",
    "quran_ahh_r32/",
    "akbar_arabic_rock_lora/",
]
DEFAULT_CONVERTER = "audiocpp_inference/converter/convert_aitoolkit_yue2_lora.py"
DEFAULT_UPLOAD_PREFIX = "_prototypes/mixes/"


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP / "config.toml"


def default_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP


@dataclass(frozen=True)
class Config:
    bucket: str = ""
    root: str = DEFAULT_ROOT
    prefixes: tuple[str, ...] = tuple(DEFAULT_PREFIXES)
    converter_object: str = DEFAULT_CONVERTER
    converter_sha256: str = ""      # empty = trust-on-first-use pin (see convert.py)
    upload_prefix: str = DEFAULT_UPLOAD_PREFIX
    host: str = "127.0.0.1"
    port: int = 5000
    data_dir: Path = field(default_factory=default_data_dir)
    config_path: Path = field(default_factory=default_config_path)
    config_from_file: bool = False

    # -- derived -----------------------------------------------------------
    def object_prefix(self, rel: str) -> str:
        """Bucket-object prefix for a bucket-relative prefix ('' root is legal)."""
        rel = rel.lstrip("/")
        return f"{self.root}/{rel}" if self.root else rel

    def display_scope(self) -> str:
        if not self.bucket:
            return "(no bucket configured)"
        return f"gs://{self.bucket}/{self.root}" if self.root else f"gs://{self.bucket}"

    def require_bucket(self) -> str:
        if not self.bucket:
            raise ConfigError(
                "no bucket configured. Pass one of:\n"
                "  loralab serve --bucket <BUCKET>\n"
                "  LORALAB_BUCKET=<BUCKET> loralab serve\n"
                f"  echo 'bucket = \"<BUCKET>\"' >> {self.config_path}"
            )
        return self.bucket

    # -- paths -------------------------------------------------------------
    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.json"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


KNOWN_KEYS = {
    "bucket", "root", "prefixes", "converter_object", "converter_sha256",
    "upload_prefix", "host", "port", "data_dir",
}


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    unknown = sorted(set(data) - KNOWN_KEYS)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {', '.join(unknown)}; "
            f"known: {', '.join(sorted(KNOWN_KEYS))}"
        )
    return data


def _env_overrides(environ) -> dict:
    out = {}
    for key, env in (
        ("bucket", "LORALAB_BUCKET"),
        ("root", "LORALAB_ROOT"),
        ("converter_sha256", "LORALAB_CONVERTER_SHA256"),
        ("upload_prefix", "LORALAB_UPLOAD_PREFIX"),
        ("host", "LORALAB_HOST"),
    ):
        value = environ.get(env)
        if value is not None:
            out[key] = value
    if environ.get("LORALAB_PORT"):
        out["port"] = int(environ["LORALAB_PORT"])
    if environ.get("LORALAB_DATA_DIR"):
        out["data_dir"] = Path(environ["LORALAB_DATA_DIR"]).expanduser()
    if environ.get("LORALAB_PREFIXES"):
        out["prefixes"] = tuple(
            p.strip() for p in environ["LORALAB_PREFIXES"].split(",") if p.strip()
        )
    return out


def load_config(*, config_path: Path | None = None, cli: dict | None = None,
                environ=None) -> Config:
    """File -> env -> CLI. Only keys that were actually supplied are replaced."""
    environ = os.environ if environ is None else environ
    path = Path(config_path).expanduser() if config_path else default_config_path()

    file_values = _read_toml(path)
    if "data_dir" in file_values:
        file_values["data_dir"] = Path(file_values["data_dir"]).expanduser()
    if "prefixes" in file_values:
        file_values["prefixes"] = tuple(file_values["prefixes"])

    cfg = Config(**file_values, config_path=path, config_from_file=path.is_file())
    cfg = replace(cfg, **{
        k: v for k, v in _env_overrides(environ).items() if v is not None
    })
    if cli:
        cfg = replace(cfg, **{k: v for k, v in cli.items() if v is not None})
    if isinstance(cfg.data_dir, str):
        cfg = replace(cfg, data_dir=Path(cfg.data_dir).expanduser())
    return cfg


def config_values(cfg: "Config") -> dict:
    """The effective config as plain TOML values (what Settings saves)."""
    return {
        "bucket": cfg.bucket,
        "root": cfg.root,
        "prefixes": list(cfg.prefixes),
        "converter_object": cfg.converter_object,
        "converter_sha256": cfg.converter_sha256,
        "upload_prefix": cfg.upload_prefix,
        "host": cfg.host,
        "port": int(cfg.port),
        "data_dir": str(cfg.data_dir),
    }


def validate_values(values: dict) -> dict:
    """Validate and normalise values before they reach a config file."""
    out = dict(values)
    for key in ("bucket", "root", "converter_object", "upload_prefix", "host"):
        out[key] = str(out.get(key, "") or "").strip()
    out["root"] = out["root"].strip("/")
    out["upload_prefix"] = out["upload_prefix"].strip("/")
    if not out["upload_prefix"]:
        raise ConfigError("upload_prefix cannot be empty")
    if any(part in ("", ".", "..") for part in out["upload_prefix"].split("/")):
        raise ConfigError("upload_prefix must be a relative path with no '..'")
    prefixes = out.get("prefixes") or []
    if isinstance(prefixes, str):
        prefixes = prefixes.replace(",", "\n").splitlines()
    cleaned = []
    for prefix in prefixes:
        prefix = str(prefix).strip().lstrip("/")
        if not prefix:
            continue
        if any(part in ("", ".", "..") for part in prefix.rstrip("/").split("/")):
            raise ConfigError(f"invalid prefix {prefix!r}")
        cleaned.append(prefix)
    if not cleaned:
        raise ConfigError("prefixes cannot be empty; list at least one catalog prefix")
    out["prefixes"] = cleaned
    try:
        port = int(out.get("port", 5000))
    except (TypeError, ValueError):
        raise ConfigError("port must be a number") from None
    if not (1 <= port <= 65535):
        raise ConfigError("port must be between 1 and 65535")
    out["port"] = port
    out["data_dir"] = str(Path(out.get("data_dir") or default_data_dir()).expanduser())
    return out


def write_config(path: Path, values: dict) -> Path:
    """Write every known setting to `path`, atomically. Used by `loralab init`
    and the Settings page. Environment variables still win at load time."""
    path = Path(path).expanduser()
    values = validate_values(values)
    prefixes = ",\n  ".join(json.dumps(p) for p in values["prefixes"])
    lines = [
        "# loralab configuration",
        "# Written by `loralab init` or the Settings page; LORALAB_* overrides this file.",
        f"bucket = {json.dumps(values['bucket'])}",
        f"root = {json.dumps(values['root'])}",
        f"prefixes = [\n  {prefixes},\n]",
        f"converter_object = {json.dumps(values['converter_object'])}",
        f"converter_sha256 = {json.dumps(values['converter_sha256'])}",
        f"upload_prefix = {json.dumps(values['upload_prefix'])}",
        f"host = {json.dumps(values['host'])}",
        f"port = {values['port']}",
        f"data_dir = {json.dumps(values['data_dir'])}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)
    return path


def write_default_config(path: Path, *, bucket: str = "") -> Path:
    """Create a config file with working defaults and a blank bucket."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ConfigError(f"{path} already exists")
    prefixes = ",\n  ".join(f'"{p}"' for p in DEFAULT_PREFIXES)
    path.write_text(
        "# loralab configuration\n"
        "# Only `bucket` is required.\n"
        f'bucket = "{bucket}"\n'
        f'root = "{DEFAULT_ROOT}"\n'
        f"prefixes = [\n  {prefixes},\n]\n"
        f'converter_object = "{DEFAULT_CONVERTER}"\n'
        'converter_sha256 = ""\n'
        f'upload_prefix = "{DEFAULT_UPLOAD_PREFIX}"\n'
        'host = "127.0.0.1"\n'
        "port = 5000\n"
    )
    return path
