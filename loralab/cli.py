"""Command line: serve | scan | doctor | catalog."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__, catalog as cat, gcs
from .config import Config, ConfigError, load_config, write_default_config
from .jobs import Jobs


def _cfg(args) -> Config:
    cli = {
        "bucket": getattr(args, "bucket", None),
        "root": getattr(args, "root", None),
        "host": getattr(args, "host", None),
        "port": getattr(args, "port", None),
    }
    if getattr(args, "data_dir", None):
        cli["data_dir"] = Path(args.data_dir).expanduser()
    return load_config(config_path=args.config, cli=cli)


# ---------------------------------------------------------------------------
def cmd_serve(args) -> int:
    cfg = _cfg(args)
    try:
        cfg.require_bucket()
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    token, auth_error = None, None
    try:
        token = gcs.resolve_token()
        print(f"auth: {token.mode}")
    except gcs.GcsError as exc:
        auth_error = str(exc)
        print(f"WARNING: {exc}", file=sys.stderr)
        print("The catalog still loads; scanning needs credentials.", file=sys.stderr)

    from .web.app import create_app

    jobs = Jobs(cfg.jobs_dir)
    app = create_app(cfg, jobs, token=token, auth_error=auth_error)

    if cfg.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {cfg.host} — this app can write to a bucket. "
              f"Prefer 127.0.0.1 and a tunnel.", file=sys.stderr)

    print(f"loralab {__version__} · {cfg.display_scope()} · data {cfg.data_dir}")
    print(f"serving on http://{cfg.host}:{cfg.port}")
    app.run(host=cfg.host, port=cfg.port, debug=False, use_reloader=False)
    return 0


def cmd_scan(args) -> int:
    cfg = _cfg(args)
    cfg.require_bucket()
    prefixes = tuple(args.prefix) if args.prefix else cfg.prefixes
    cfg = type(cfg)(**{**cfg.__dict__, "prefixes": prefixes})
    token = gcs.resolve_token()
    print(f"auth: {token.mode} · {cfg.display_scope()}")
    print(f"prefixes: {', '.join(prefixes)}")

    doc = cat.scan(cfg, token, log=print)
    cat.save_catalog(cfg.catalog_path, doc)

    print(f"\n{doc['count']} adapter(s): "
          + ", ".join(f"{k} {v}" for k, v in cat.kinds(doc["entries"]).items()))
    if doc["errors"]:
        print(f"{len(doc['errors'])} file(s) failed to parse:")
        for err in doc["errors"][:10]:
            print(f"  {err['path']}: {err['error']}")
    print(f"catalog: {cfg.catalog_path}")
    return 0 if not doc["errors"] else 1


def cmd_catalog(args) -> int:
    cfg = _cfg(args)
    doc = cat.load_catalog(cfg.catalog_path)
    if not doc:
        print(f"no catalog at {cfg.catalog_path}; run `loralab scan` first", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return 0
    print(f"{doc['count']} adapter(s) · {doc['scope']} · generated {doc['generated_utc']}")
    for group in cat.group(doc["entries"]):
        print(f"\n{group['key']}  ({len(group['entries'])})")
        for entry in group["entries"]:
            mb = (entry.get("size_bytes") or 0) / 1048576
            print(f"  {entry['name'][:52]:<52} {entry['kind']:<10} "
                  f"{cat.rank_label(entry)[:18]:<18} {mb:>7.1f}  {cat.note(entry)[:40]}")
    return 0


def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    ok = True
    print(f"loralab {__version__}")
    print(f"config: {cfg.config_path}{'' if cfg.config_from_file else ' (absent)'}")
    print(f"data:   {cfg.data_dir}")
    print(f"scope:  {cfg.display_scope()}")

    if not cfg.bucket:
        print("bucket: MISSING — `loralab serve --bucket <name>` or set it in the config")
        ok = False
    else:
        try:
            token = gcs.resolve_token()
            print(f"auth:   {token.mode}")
            prefix = cfg.object_prefix(cfg.prefixes[0])
            objects = gcs.list_objects(cfg.bucket, prefix, token=token)
            print(f"bucket: ok — {len(objects)} object(s) under {prefix}")
        except (gcs.GcsError, ConfigError) as exc:
            print(f"bucket: FAIL — {exc}")
            ok = False

    doc = cat.load_catalog(cfg.catalog_path)
    if doc:
        print(f"catalog: {doc['count']} adapter(s), {len(doc['errors'])} error(s), "
              f"generated {doc['generated_utc']}")
    else:
        print("catalog: absent — run `loralab scan`")

    target = cfg.data_dir if cfg.data_dir.exists() else Path("/")
    print(f"disk:   {shutil.disk_usage(str(target)).free / 1e9:.1f} GB free")
    return 0 if ok else 1


def cmd_init(args) -> int:
    path = write_default_config(Path(args.config) if args.config else _default_config(),
                                bucket=args.bucket or "")
    print(f"wrote {path}")
    if not args.bucket:
        print("next: put your bucket name in it, or run `loralab serve --bucket <name>`")
    return 0


def _default_config() -> Path:
    from .config import default_config_path
    return default_config_path()


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="loralab",
        description="Catalog YuE2 LoRA adapters (header-only) and mix them at an alpha.",
    )
    ap.add_argument("--version", action="version", version=f"loralab {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--config", type=Path, default=None,
                       help="config.toml (default: ~/.config/loralab/config.toml)")
        p.add_argument("--bucket", default=None,
                       help="GCS bucket (required; not baked into the defaults)")
        p.add_argument("--root", default=None,
                       help="prefix inside the bucket (default: OSTRIS_Arabic_Suno_Finetuning)")
        p.add_argument("--data-dir", default=None,
                       help="catalog/jobs/cache location (default: ~/.local/share/loralab)")

    s = sub.add_parser("serve", help="run the local web app")
    add_common(s)
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    sc = sub.add_parser("scan", help="sweep the bucket -> catalog.json")
    add_common(sc)
    sc.add_argument("--prefix", action="append", default=None,
                    help="bucket-relative prefix (repeatable); default: all configured")
    sc.set_defaults(func=cmd_scan)

    ca = sub.add_parser("catalog", help="print the cached catalog")
    add_common(ca)
    ca.add_argument("--json", action="store_true")
    ca.set_defaults(func=cmd_catalog)

    d = sub.add_parser("doctor", help="check auth, bucket, catalog and disk")
    add_common(d)
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("init", help="write a config file with working defaults")
    i.add_argument("--config", type=Path, default=None)
    i.add_argument("--bucket", default=None)
    i.set_defaults(func=cmd_init)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except gcs.GcsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
