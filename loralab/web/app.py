"""Web layer: catalog, adapter detail, and the mixer."""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, url_for)

from .. import __version__, catalog as cat, convert as convertmod, gcs
from .. import manifests, merge, mixjob, upload as uploadmod
from ..config import (Config, ConfigError, config_values, load_config,
                      write_config)

FILTER_KINDS = ("fused", "merged", "converted")


def _checked(value) -> bool:
    return str(value or "").lower() in ("1", "true", "yes", "on")


def _cache_report(cfg: Config, limit: int = 50) -> dict:
    """What is actually on disk under cache/src and cache/tools."""
    dirs = []
    total = 0
    for label, path in (("sources", cfg.cache_dir / "src"),
                        ("tools", cfg.cache_dir / "tools")):
        files = []
        if path.is_dir():
            for item in sorted(path.rglob("*"), key=lambda p: p.stat().st_mtime,
                               reverse=True):
                if not item.is_file():
                    continue
                stat = item.stat()
                total += stat.st_size
                files.append({
                    "name": str(item.relative_to(path)),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                })
        dirs.append({
            "name": label,
            "path": str(path),
            "count": len(files),
            "bytes": sum(f["size_bytes"] for f in files),
            "files": files[:limit],
        })
    usage = shutil.disk_usage(str(cfg.data_dir if cfg.data_dir.exists() else "/"))
    return {"dirs": dirs, "total_bytes": total, "free_bytes": usage.free}


def create_app(cfg: Config, jobs, *, scanner=None, token=None, auth_error=None,
               mixer=None, uploader=None) -> Flask:
    """Build the app. `scanner`/`mixer`/`uploader` are injectable for tests."""
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

    def default_mixer(ctx):
        return mixjob.run(ctx, cfg, ctx.params, token=token)

    mix_fn = mixer or default_mixer

    def default_uploader(ctx):
        return uploadmod.run(ctx, cfg, ctx.params, token=token)

    upload_fn = uploader or default_uploader

    def mix_state(wire, *, default_convert: bool):
        """Read the form/query, inspect inputs, and build a preview if possible."""
        doc = current_catalog()
        entries = (doc or {}).get("entries", [])
        eligible = [e for e in entries if e.get("kind") in mixjob.ELIGIBLE_KINDS]
        carrier = (wire.get("carrier") or "").strip()
        donor = (wire.get("donor") or "").strip()
        alpha_text = (wire.get("alpha") or "0.1, 0.2").strip()
        stem = (wire.get("stem") or "").strip()
        raw_convert = wire.get("convert")
        convert = default_convert if raw_convert is None else \
            str(raw_convert).lower() not in ("0", "", "false", "off")
        errors: list[str] = []
        plan = None
        try:
            alphas = mixjob.parse_alphas(alpha_text)
        except merge.MergeError as exc:
            alphas = []
            errors.append(str(exc))
        if carrier and donor and alphas:
            try:
                c = mixjob.inspect(carrier, doc, cfg)
                d = mixjob.inspect(donor, doc, cfg)
                plan = mixjob.preview(c, d, alphas, cfg, convert=convert)
                errors.extend(plan["issues"])
            except merge.MergeError as exc:
                errors.append(str(exc))
        return {
            "doc": doc, "eligible": eligible, "carrier": carrier,
            "donor": donor, "alpha_text": alpha_text, "stem": stem,
            "convert": convert, "errors": errors, "plan": plan,
        }

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

    @app.get("/mix/new")
    def mix_new():
        state = mix_state(request.args, default_convert=True)
        return render_template("mix.html", rank_label=cat.rank_label, **state)

    @app.post("/mix")
    def mix_submit():
        state = mix_state(request.form, default_convert=False)
        if state["errors"]:
            return render_template("mix.html", rank_label=cat.rank_label,
                                   **state), 409
        params = mixjob.build_params(
            state["carrier"], state["donor"], state["alpha_text"],
            convert=state["convert"], stem=state["stem"],
        )
        job = jobs.submit("mix", params, mix_fn,
                          dedupe_key=mixjob.dedupe_key(params))
        return redirect(url_for("job_view", job_id=job["id"]), code=303)

    def job_dir(job_id: str):
        """Only a real job id, and only its own directory."""
        if not job_id.isalnum() or jobs.get(job_id) is None:
            abort(404)
        return (cfg.jobs_dir / job_id).resolve()

    def upload_state(job_id: str, wire):
        """Preview an upload from a finished mix job. Never writes anything."""
        job = jobs.get(job_id)
        if not job or job.get("kind") != "mix":
            abort(404)
        manifest_path = cfg.jobs_dir / job_id / "MANIFEST.json"
        manifest = manifests.read(manifest_path)
        if not manifest:
            abort(404)
        prefix = (wire.get("prefix") or "").strip() or cfg.upload_prefix
        name = (wire.get("name") or "").strip()
        state = {
            "job": job,
            "manifest": manifest,
            "prefix": prefix,
            "name": name,
            "include_fused": _checked(wire.get("include_fused")),
            "overwrite": _checked(wire.get("overwrite")),
            "allow_production_prefix": _checked(wire.get("allow_production_prefix")),
            "errors": [],
            "plan": None,
        }
        try:
            state["plan"] = uploadmod.plan_uploads(
                cfg, manifest_path, manifest,
                prefix=prefix, name=name or None,
                include_fused=state["include_fused"],
                overwrite=state["overwrite"],
                allow_production_prefix=state["allow_production_prefix"],
                token=token, check_existing=token is not None, job_id=job_id,
            )
            state["name"] = state["plan"]["directory"]
        except uploadmod.UploadError as exc:
            state["errors"].append(str(exc))
        except gcs.GcsError as exc:
            state["errors"].append(f"cannot check the destination: {exc}")
        if token is None:
            state["errors"].append(
                "no credentials were resolved, so existing keys cannot be "
                "checked and upload is disabled. Run `loralab doctor`."
            )
        return state

    @app.get("/mix/<job_id>/upload")
    def upload_preview(job_id: str):
        state = upload_state(job_id, request.args)
        return render_template("upload.html", **state)

    @app.post("/mix/<job_id>/upload")
    def upload_submit(job_id: str):
        state = upload_state(job_id, request.form)
        if state["errors"] or not state["plan"]:
            return render_template("upload.html", **state), 409
        if not _checked(request.form.get("confirm")):
            state["errors"].append(
                "Review the destination object names above, then tick the "
                "confirmation box before uploading."
            )
            return render_template("upload.html", **state), 409
        if state["plan"]["blockers"]:
            return render_template("upload.html", **state), 409
        params = {
            "job": job_id,
            "prefix": state["plan"]["prefix"],
            "name": state["plan"]["directory"],
            "include_fused": state["include_fused"],
            "overwrite": state["overwrite"],
            "allow_production_prefix": state["allow_production_prefix"],
        }
        key = uploadmod.dedupe_key(
            cfg, Path(state["plan"]["manifest"]), prefix=params["prefix"],
            name=params["name"], include_fused=params["include_fused"],
            overwrite=params["overwrite"],
            allow_production_prefix=params["allow_production_prefix"],
        )
        job = jobs.submit("upload", params, upload_fn, dedupe_key=key)
        return redirect(url_for("job_view", job_id=job["id"]), code=303)

    @app.get("/mix/<job_id>/manifest.json")
    def mix_manifest(job_id: str):
        path = job_dir(job_id) / "MANIFEST.json"
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="application/json", as_attachment=True,
                         download_name=f"{job_id}-MANIFEST.json")

    @app.get("/mix/<job_id>/artifact/<path:rel>")
    def mix_artifact(job_id: str, rel: str):
        base = job_dir(job_id)
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            abort(404)
        doc = manifests.read(base / "MANIFEST.json") or {}
        allowed = set()
        for arm in doc.get("arms", []):
            entries = [arm.get("fused") or {}]
            entries += list((arm.get("converted") or {}).values())
            for item in entries:
                if item.get("path"):
                    try:
                        allowed.add(str(Path(item["path"]).resolve().relative_to(base)))
                    except ValueError:
                        continue
        if rel not in allowed or not target.is_file():
            abort(404)
        return send_file(target, as_attachment=True)

    @app.get("/jobs")
    def jobs_view():
        return render_template("jobs.html", jobs=jobs.list(limit=50))

    @app.get("/jobs/<job_id>")
    def job_view(job_id: str):
        if not job_id.isalnum():
            abort(404)
        job = jobs.get(job_id)
        if not job:
            abort(404)
        manifest = None
        snippet = None
        upload_record = manifests.read(cfg.jobs_dir / job_id / "upload.json")
        if job.get("kind") == "mix":
            manifest = manifests.read(cfg.jobs_dir / job_id / "MANIFEST.json")
            loras = {}
            for arm in (manifest or {}).get("arms", []):
                converted = arm.get("converted") or {}
                if not (converted.get("ar") and converted.get("nar")):
                    continue
                key = (arm.get("stem") or f"a{arm.get('alpha')}").replace(".", "p")
                loras[key] = {"ar": converted["ar"]["path"],
                              "nar": converted["nar"]["path"]}
            if loras:
                snippet = json.dumps({"loras": loras}, indent=2)
        return render_template("job.html", job=job, manifest=manifest,
                               snippet=snippet, upload_record=upload_record,
                               jobs_list=jobs.list(limit=10))

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        if not job_id.isalnum():
            abort(404)
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
        if not job_id.isalnum():
            abort(404)
        jobs.cancel(job_id)
        return redirect(url_for("job_view", job_id=job_id))

    def settings_context(values, *, saved=False, error=None):
        return dict(
            values=values,
            cache=_cache_report(cfg),
            saved=saved,
            error=error,
            cfg=cfg,
            auth_mode=getattr(token, "mode", None) or ("unknown" if auth_error else "—"),
            auth_error=auth_error,
            env_overrides=sorted(k for k in os.environ if k.startswith("LORALAB_")),
        )

    @app.get("/settings")
    def settings():
        return render_template(
            "settings.html",
            **settings_context(config_values(cfg), saved=request.args.get("saved") == "1"),
        )

    @app.post("/settings")
    def save_settings():
        nonlocal cfg
        values = {
            "bucket": request.form.get("bucket", ""),
            "root": request.form.get("root", ""),
            "prefixes": request.form.get("prefixes", ""),
            "converter_object": request.form.get("converter_object", ""),
            "converter_sha256": request.form.get("converter_sha256", ""),
            "upload_prefix": request.form.get("upload_prefix", ""),
            "host": request.form.get("host", ""),
            "port": request.form.get("port", ""),
            "data_dir": str(cfg.data_dir),
        }
        try:
            write_config(cfg.config_path, values)
            cfg = load_config(config_path=cfg.config_path)
            app.config["LORALAB_CFG"] = cfg
        except (ConfigError, OSError) as exc:
            return render_template(
                "settings.html",
                **settings_context(
                    {**values, "prefixes": [
                        p.strip() for p in
                        str(values["prefixes"]).replace(",", "\n").splitlines()
                        if p.strip()
                    ]},
                    error=str(exc),
                ),
            ), 400
        return redirect(url_for("settings", saved=1), code=303)

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
        env = manifests.environment()
        missing = [name for name in ("torch", "safetensors") if not env.get(name)]
        checks.append(("merge engine", not missing,
                       "missing " + ", ".join(missing) + ' — pip install -e ".[merge]"'
                       if missing else
                       f"torch {env['torch']} · safetensors {env['safetensors']}"))
        cached = convertmod.cached_path(cfg)
        pin_file = cached.with_suffix(cached.suffix + ".sha256")
        pin = cfg.converter_sha256 or (pin_file.read_text().strip()
                                       if pin_file.is_file() else "")
        checks.append(("converter", None,
                       f"cached {cached}" + (f" · pin {pin}" if pin else " · no pin")
                       if cached.is_file() else
                       ("not cached — will download " + cfg.converter_object
                        if cfg.bucket else "not cached and no bucket configured")))
        checks.append(("upload", None,
                       f"destination {cfg.upload_prefix} · create permission is "
                       "only probed by an actual preview/upload"))
        checks.append(("python", True, platform.python_version()))
        checks.append(("data dir", True, str(cfg.data_dir)))
        return render_template("doctor.html", checks=checks,
                               cfg=cfg, catalog=current_catalog())

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("notfound.html"), 404

    return app
