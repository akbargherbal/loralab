"""Web layer: catalog and adapter detail. Read-only in P1."""
from __future__ import annotations

import json
import platform
import shutil

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   url_for)

from .. import __version__, catalog as cat, gcs
from ..config import Config

FILTER_KINDS = ("fused", "merged", "converted")


def create_app(cfg: Config, jobs, *, scanner=None, token=None, auth_error=None) -> Flask:
    """Build the app. `scanner` is injectable so tests need no network."""
    app = Flask(__name__)
    app.config["LORALAB_CFG"] = cfg
    app.config["LORALAB_JOBS"] = jobs

    def current_catalog():
        return cat.load_catalog(cfg.catalog_path)

    def default_scanner(ctx):
        tok = token or gcs.resolve_token()
        doc = cat.scan(cfg, tok, log=ctx.log, cancelled=ctx.cancelled)
        cat.save_catalog(cfg.catalog_path, doc)
        ctx.log(f"catalog: {doc['count']} adapter(s), {len(doc['errors'])} error(s)")
        return {"count": doc["count"], "errors": len(doc["errors"])}

    scan_fn = scanner or default_scanner

    @app.context_processor
    def inject_globals():
        return {
            "cfg": cfg,
            "auth_mode": getattr(token, "mode", None) or ("unknown" if auth_error else "—"),
            "auth_error": auth_error,
            "version": __version__,
        }

    @app.get("/")
    def index():
        return redirect(url_for("catalog_view"))

    @app.get("/catalog")
    def catalog_view():
        doc = current_catalog()
        if not doc:
            return render_template("empty.html", jobs=jobs.list(limit=5))
        entries = doc.get("entries", [])
        q = request.args.get("q", "")
        kind = request.args.get("kind", "")
        prefix = request.args.get("prefix", "")
        shown = cat.filter_entries(entries, q=q, kind=kind, prefix=prefix)
        return render_template(
            "catalog.html",
            doc=doc,
            groups=cat.group(shown),
            shown=len(shown),
            total=len(entries),
            kinds=cat.kinds(entries),
            prefixes=cat.prefix_options(entries),
            filter_kinds=FILTER_KINDS,
            q=q, kind=kind, prefix=prefix,
            rank_label=cat.rank_label,
            note=cat.note,
            jobs=jobs.list(limit=5),
        )

    @app.get("/catalog/<path:name>")
    def detail(name: str):
        doc = current_catalog()
        if not doc:
            abort(404)
        entry = next((e for e in doc.get("entries", []) if e["path"] == name), None)
        if entry is None:
            abort(404)
        return render_template(
            "detail.html",
            entry=entry,
            doc=doc,
            rank_label=cat.rank_label,
            raw=json.dumps(entry, indent=2, ensure_ascii=False),
            siblings=[e for e in doc.get("entries", [])
                      if e["kind"] == "converted" and e.get("source_file") == entry["name"]],
        )

    @app.post("/catalog/scan")
    def scan_now():
        job = jobs.submit(
            "scan",
            {"scope": cfg.display_scope(), "prefixes": list(cfg.prefixes)},
            scan_fn,
            dedupe_key=f"scan:{cfg.bucket}:{cfg.root}:{','.join(cfg.prefixes)}",
        )
        return redirect(url_for("job_view", job_id=job["id"]))

    @app.get("/jobs")
    def jobs_view():
        return render_template("jobs.html", jobs=jobs.list(limit=50))

    @app.get("/jobs/<job_id>")
    def job_view(job_id: str):
        job = jobs.get(job_id)
        if not job:
            abort(404)
        return render_template("job.html", job=job, jobs_list=jobs.list(limit=10))

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        job = jobs.get(job_id)
        if not job:
            abort(404)
        offset = int(request.args.get("log_offset", 0))
        text, next_offset = jobs.log_text(job_id, offset)
        return jsonify({
            "state": job.get("state"),
            "error": job.get("error"),
            "elapsed_s": job.get("elapsed_s"),
            "log": text,
            "next_offset": next_offset,
        })

    @app.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        jobs.cancel(job_id)
        return redirect(url_for("job_view", job_id=job_id))

    @app.get("/doctor")
    def doctor():
        checks = []
        if auth_error:
            checks.append(("auth", False, str(auth_error)))
        elif token is not None:
            checks.append(("auth", True, f"mode: {token.mode}"))
        else:
            checks.append(("auth", None, "not probed"))

        checks.append(("catalog", cfg.catalog_path.is_file(),
                       f"{cfg.catalog_path}"
                       + ("" if cfg.catalog_path.is_file() else " (absent — run a scan)")))

        if cfg.bucket and token is not None:
            try:
                listed = gcs.list_objects(cfg.bucket, cfg.object_prefix(cfg.prefixes[0]),
                                          token=token)
                checks.append(("bucket", True,
                               f"{cfg.display_scope()} — {len(listed)} object(s) under "
                               f"{cfg.prefixes[0]}"))
            except gcs.GcsError as exc:
                checks.append(("bucket", False, str(exc)))
        else:
            checks.append(("bucket", False, "no bucket configured"))

        usage = shutil.disk_usage(str(cfg.data_dir if cfg.data_dir.exists() else "/"))
        checks.append(("disk", True, f"{usage.free / 1e9:.1f} GB free"))
        checks.append(("python", True, platform.python_version()))
        checks.append(("data dir", True, str(cfg.data_dir)))
        return render_template("doctor.html", checks=checks,
                               cfg=cfg, catalog=current_catalog())

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("notfound.html"), 404

    return app
