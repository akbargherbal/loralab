"""Web layer: catalog, adapter detail, and the mixer."""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, url_for)

from .. import __version__, catalog as cat, convert as convertmod, gcs
from .. import manifests, merge, mixjob
from ..config import Config

FILTER_KINDS = ("fused", "merged", "converted")


def create_app(cfg: Config, jobs, *, scanner=None, token=None, auth_error=None,
               mixer=None) -> Flask:
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

    def default_mixer(ctx):
        return mixjob.run(ctx, cfg, ctx.params, token=token)

    mix_fn = mixer or default_mixer

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
                               snippet=snippet, jobs_list=jobs.list(limit=10))

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
        checks.append(("python", True, platform.python_version()))
        checks.append(("data dir", True, str(cfg.data_dir)))
        return render_template("doctor.html", checks=checks,
                               cfg=cfg, catalog=current_catalog())

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("notfound.html"), 404

    return app
