"""The web surface: routes, filters, the scan job, and the empty state."""
from __future__ import annotations

import time

import pytest

from loralab import catalog as cat
from loralab.jobs import Jobs
from loralab.web.app import create_app


def wait_for(jobs: Jobs, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {jobs.get(job_id)}")


@pytest.fixture
def app(cfg, entry_factory):
    entries = [
        entry_factory("run/output/alpha.safetensors"),
        entry_factory("run/convert/alpha_ar.safetensors", kind="converted",
                      meta={"source_file": "alpha.safetensors"}),
        entry_factory("run/merge/alpha_a0.2.safetensors", kind="merged"),
        entry_factory("loras/audio_cpp/style/beta_ar.safetensors", kind="converted"),
    ]
    cat.resolve_lineage(entries)
    cat.save_catalog(cfg.catalog_path, {
        "schema": 1, "generated_utc": "2026-11-15T00:00:00Z", "tool": "loralab 0.1.0",
        "scope": "gs://demo-bucket/ROOT", "prefixes": list(cfg.prefixes),
        "count": len(entries), "entries": entries, "errors": [],
    })
    jobs = Jobs(cfg.jobs_dir)
    return create_app(cfg, jobs, scanner=lambda ctx: {"count": 0, "errors": 0})


@pytest.fixture
def client(app):
    return app.test_client()


def test_catalog_lists_every_adapter(client):
    page = client.get("/catalog")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "alpha.safetensors" in body
    assert "4 adapters" in body
    assert "gs://demo-bucket/ROOT" in body


def test_root_redirects_to_the_catalog(client):
    assert client.get("/").status_code == 302
    assert "/catalog" in client.get("/").headers["Location"]


def test_detail_shows_provenance_and_lineage(client):
    page = client.get("/catalog/run/merge/alpha_a0.2.safetensors")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "Merge provenance" in body and "0.2" in body
    assert "lineage" in body.lower()


def test_converted_detail_lists_the_fused_source(client):
    page = client.get("/catalog/run/convert/alpha_ar.safetensors")
    body = page.get_data(as_text=True)
    assert "run/output/alpha.safetensors" in body


def test_unknown_adapter_is_404(client):
    assert client.get("/catalog/nope.safetensors").status_code == 404


def test_filters_narrow_the_table(client):
    body = client.get("/catalog?kind=merged").get_data(as_text=True)
    assert "alpha_a0.2.safetensors" in body
    assert "beta_ar.safetensors" not in body
    body = client.get("/catalog?q=beta").get_data(as_text=True)
    assert "beta_ar.safetensors" in body
    assert "alpha_a0.2.safetensors" not in body


def test_scan_posts_a_job_and_redirects(client, app):
    response = client.post("/catalog/scan")
    assert response.status_code == 302
    job_id = response.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(app.config["LORALAB_JOBS"], job_id)
    assert job["state"] == "done"
    assert job["result"] == {"count": 0, "errors": 0}
    assert "scan: queued" in app.config["LORALAB_JOBS"].log_text(job_id)[0]


def test_job_api_exposes_state_and_incremental_log(client, app):
    response = client.post("/catalog/scan")
    job_id = response.headers["Location"].rstrip("/").split("/")[-1]
    wait_for(app.config["LORALAB_JOBS"], job_id)
    payload = client.get(f"/api/jobs/{job_id}").get_json()
    assert payload["state"] == "done"
    assert "queued" in payload["log"]
    tail = client.get(f"/api/jobs/{job_id}?log_offset={payload['next_offset']}").get_json()
    assert tail["log"] == ""


def test_job_page_and_jobs_list_render(client, app):
    response = client.post("/catalog/scan")
    job_id = response.headers["Location"].rstrip("/").split("/")[-1]
    wait_for(app.config["LORALAB_JOBS"], job_id)
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert "loralab scan" in client.get(f"/jobs/{job_id}").get_data(as_text=True)
    assert job_id in client.get("/jobs").get_data(as_text=True)


def test_doctor_reports_without_a_token(cfg, entry_factory):
    jobs = Jobs(cfg.jobs_dir)
    app = create_app(cfg, jobs, token=None, auth_error="no credentials")
    page = app.test_client().get("/doctor")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "no credentials" in body
    assert "demo-bucket" in body


def test_empty_catalog_offers_a_scan(cfg):
    jobs = Jobs(cfg.jobs_dir)
    app = create_app(cfg, jobs, scanner=lambda ctx: {"count": 0, "errors": 0})
    body = app.test_client().get("/catalog").get_data(as_text=True)
    assert "No catalog yet" in body
    assert "Rescan" not in body        # the empty state has its own button
    assert "Scan gs://demo-bucket/ROOT" in body


def test_stale_running_job_is_failed_on_start(cfg):
    """A process that dies mid-job must not leave a zombie in the UI."""
    import json
    job_dir = cfg.jobs_dir / "deadbeef"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps(
        {"id": "deadbeef", "kind": "scan", "state": "running",
         "created_utc": "2026-11-15T00:00:00Z"}))
    jobs = Jobs(cfg.jobs_dir)
    job = jobs.get("deadbeef")
    assert job["state"] == "failed"
    assert job["error"] == "engine restarted"


def test_cancel_stops_a_running_job(cfg):
    import threading

    started = threading.Event()

    def slow(ctx):
        started.set()
        for _ in range(500):
            if ctx.cancelled():
                return {"cancelled": True}
            time.sleep(0.01)
        return {"cancelled": False}

    jobs = Jobs(cfg.jobs_dir)
    job = jobs.submit("slow", {}, slow)
    assert started.wait(2.0)
    assert jobs.cancel(job["id"]) is True
    final = wait_for(jobs, job["id"])
    assert final["state"] == "cancelled"
    assert jobs.cancel(job["id"]) is False   # already terminal


def test_doctor_probes_the_bucket_when_it_has_a_token(cfg, monkeypatch):
    from loralab import gcs
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: [{"name": "x"}, {"name": "y"}])
    app = create_app(cfg, Jobs(cfg.jobs_dir), token=gcs.Token("test"))
    body = app.test_client().get("/doctor").get_data(as_text=True)
    assert "2 object(s)" in body
    assert "mode: test" in body


def test_cancel_route_flags_a_running_job(client, app):
    import threading
    gate = threading.Event()

    def slow(ctx):
        gate.wait(5)
        return {}

    jobs = app.config["LORALAB_JOBS"]
    job = jobs.submit("scan", {}, slow)
    assert jobs.cancel(job["id"]) is True          # direct
    gate.set()
    wait_for(jobs, job["id"])

    gate.clear()
    job2 = jobs.submit("scan", {}, slow)
    response = client.post(f"/jobs/{job2['id']}/cancel")
    assert response.status_code == 302
    assert (cfg_dir := app.config["LORALAB_CFG"].jobs_dir / job2["id"] / "cancel").exists()
    gate.set()
    assert wait_for(jobs, job2["id"])["state"] == "cancelled"


def test_doctor_reports_a_failing_bucket(cfg, monkeypatch):
    from loralab import gcs
    monkeypatch.setattr(gcs, "list_objects", lambda *a, **k: (_ for _ in ()).throw(
        gcs.GcsError("HTTP 403 — the credentials lack storage.objects.list/get")))
    app = create_app(cfg, Jobs(cfg.jobs_dir), token=gcs.Token("test"))
    body = app.test_client().get("/doctor").get_data(as_text=True)
    assert "403" in body and "fail" in body


def test_doctor_without_a_probe_says_so(cfg):
    app = create_app(cfg, Jobs(cfg.jobs_dir))
    assert "not probed" in app.test_client().get("/doctor").get_data(as_text=True)


def test_detail_404_when_there_is_no_catalog_at_all(cfg):
    app = create_app(cfg, Jobs(cfg.jobs_dir))
    assert app.test_client().get("/catalog/anything.safetensors").status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_default_scanner_writes_the_catalog(cfg, monkeypatch):
    """The production wiring: no injected scanner, real code path, faked transport."""
    from loralab import catalog as cat
    from loralab import gcs

    monkeypatch.setattr(gcs, "resolve_token", lambda: gcs.Token("test"))
    monkeypatch.setattr(cat, "scan", lambda cfg_, tok, log=None, cancelled=None: {
        "count": 1, "entries": [], "errors": []})

    app = create_app(cfg, Jobs(cfg.jobs_dir))
    response = app.test_client().post("/catalog/scan")
    job_id = response.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(app.config["LORALAB_JOBS"], job_id)
    assert job["state"] == "done"
    assert job["result"] == {"count": 1, "errors": 0}
    assert cfg.catalog_path.is_file()
