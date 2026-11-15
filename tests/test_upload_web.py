"""The upload surface: preview, explicit confirmation, guards, and the job."""
from __future__ import annotations

import json
import time

import pytest

from loralab import catalog as cat
from loralab import gcs, manifests, mixjob
from loralab.jobs import Jobs
from loralab.web.app import create_app


def wait_for(jobs, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {jobs.get(job_id)}")


def fake_mixer(ctx):
    arm_dir = ctx.dir / "a0.1"
    arm_dir.mkdir(parents=True, exist_ok=True)
    fused = arm_dir / "carrier__donor__a0.1.safetensors"
    fused.write_bytes(b"fused")
    ar = arm_dir / "carrier__donor__a0.1_ar.safetensors"
    ar.write_bytes(b"ar")
    nar = arm_dir / "carrier__donor__a0.1_nar.safetensors"
    nar.write_bytes(b"nar")
    doc = manifests.build(
        command=mixjob.command(ctx.params),
        carrier={"name": "carrier.safetensors"},
        donor={"name": "donor.safetensors"},
        arms=[{
            "alpha": 0.1,
            "stem": "carrier__donor__a0.1",
            "out_dir": str(arm_dir),
            "fused": {"path": str(fused), "name": fused.name,
                      "size_bytes": 5, "sha256": "a" * 64,
                      "ranks": {"ar": 8, "nar": 8}},
            "verify": {"kind": "delta", "passed": True, "alpha": 0.1,
                       "max_abs_error": 1e-7, "max_rel_error": 1e-7},
            "converted": {
                "ar": {"path": str(ar), "name": ar.name, "size_bytes": 2,
                       "sha256": "b" * 64, "rank": 8},
                "nar": {"path": str(nar), "name": nar.name, "size_bytes": 3,
                        "sha256": "c" * 64, "rank": 8},
            },
        }],
    )
    manifests.write(ctx.dir / "MANIFEST.json", doc)
    return {"arms": 1, "alphas": [0.1],
            "manifest": str(ctx.dir / "MANIFEST.json"),
            "out": str(ctx.dir), "converted": True}


@pytest.fixture
def uploads(monkeypatch):
    calls = []

    def fake_upload(bucket, obj, source, token=None, *, overwrite=False,
                    generation=None, log=None):
        calls.append({"object": obj, "overwrite": overwrite,
                      "generation": generation})
        return {"name": obj, "size": 2, "sha256": "x" * 64,
                "md5_verified": True, "generation": "42",
                "crc32c": "AAAAAA==", "updated": "2026-11-15T00:00:00Z"}

    monkeypatch.setattr(gcs, "upload_object", fake_upload)
    monkeypatch.setattr(gcs, "stat_or_none", lambda *a: None)
    return calls


@pytest.fixture
def app(cfg, entry_factory, uploads):
    entries = [entry_factory("run/carrier.safetensors"),
               entry_factory("run/merged.safetensors", kind="merged")]
    cat.save_catalog(cfg.catalog_path, {
        "schema": 1, "generated_utc": "2026-11-15T00:00:00Z",
        "tool": "loralab 0.1.0", "scope": "gs://demo-bucket/ROOT",
        "prefixes": list(cfg.prefixes), "count": len(entries),
        "entries": entries, "errors": [],
    })
    jobs = Jobs(cfg.jobs_dir)
    return create_app(cfg, jobs, mixer=fake_mixer, token=gcs.Token("test"))


@pytest.fixture
def client(app):
    return app.test_client()


def make_mix_job(client, jobs):
    response = client.post("/mix", data={
        "carrier": "run/carrier.safetensors",
        "donor": "run/merged.safetensors",
        "alpha": "0.1", "convert": "1",
    })
    assert response.status_code == 303
    job_id = response.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(jobs, job_id)
    assert job["state"] == "done", job
    return job_id


def test_upload_preview_names_the_destination_and_every_object(client, app):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    body = client.get(f"/mix/{job_id}/upload").get_data(as_text=True)
    assert "Upload mix" in body
    assert "_prototypes/mixes" in body
    assert "carrier__donor" in body
    assert "carrier__donor__a0.1_ar.safetensors" in body
    assert "MANIFEST.json" in body
    assert "_prototypes default" in body
    assert "I reviewed these 3 object name(s)" in body
    assert "loralab upload --job" in body


def test_upload_link_appears_on_a_finished_mix(client, app):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    body = client.get(f"/jobs/{job_id}").get_data(as_text=True)
    assert f"/mix/{job_id}/upload" in body


def test_upload_preview_blocks_the_production_prefix(client, app):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    body = client.get(f"/mix/{job_id}/upload?prefix=loras/mixes"
                      ).get_data(as_text=True)
    assert "Blocked" in body
    assert "production" in body
    assert "under loras/" in body


def test_production_prefix_needs_its_own_tick(client, app):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    body = client.get(
        f"/mix/{job_id}/upload?prefix=loras/mixes&allow_production_prefix=1"
    ).get_data(as_text=True)
    assert "Blocked" not in body


def test_upload_post_without_confirmation_is_rejected(client, app):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    response = client.post(f"/mix/{job_id}/upload", data={"prefix": ""})
    assert response.status_code == 409
    assert "tick the confirmation box" in response.get_data(as_text=True)
    assert all(j["kind"] != "upload" for j in jobs.list())


def test_upload_post_queues_the_job_and_shows_the_record(client, app, uploads):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    response = client.post(f"/mix/{job_id}/upload", data={
        "prefix": "_prototypes/mixes", "name": "carrier__donor", "confirm": "1",
    })
    assert response.status_code == 303
    upload_id = response.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(jobs, upload_id)
    assert job["state"] == "done", job
    assert len(uploads) == 3
    assert uploads[-1]["object"].endswith("/MANIFEST.json")
    record = json.loads((app.config["LORALAB_CFG"].jobs_dir / upload_id
                         / "upload.json").read_text())
    assert record["count"] == 3
    body = client.get(f"/jobs/{upload_id}").get_data(as_text=True)
    assert "Upload" in body
    assert "gs://demo-bucket/ROOT/_prototypes/mixes/carrier__donor/" in body
    assert "md5 verified" in body
    assert "no delete path" in body


def test_upload_post_requires_overwrite_when_a_key_exists(client, app, monkeypatch):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    monkeypatch.setattr(gcs, "stat_or_none",
                        lambda *a: {"generation": "7"})
    response = client.post(f"/mix/{job_id}/upload", data={
        "prefix": "_prototypes/mixes", "confirm": "1",
    })
    assert response.status_code == 409
    assert "already exist" in response.get_data(as_text=True)
    assert all(j["kind"] != "upload" for j in jobs.list())


def test_upload_overwrite_pins_the_seen_generation(client, app, uploads,
                                                   monkeypatch):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)
    monkeypatch.setattr(gcs, "stat_or_none",
                        lambda *a: {"generation": "7"})
    response = client.post(f"/mix/{job_id}/upload", data={
        "prefix": "_prototypes/mixes", "confirm": "1", "overwrite": "1",
    })
    assert response.status_code == 303
    upload_id = response.headers["Location"].rstrip("/").split("/")[-1]
    assert wait_for(jobs, upload_id)["state"] == "done"
    assert all(c["generation"] == "7" for c in uploads)
    assert all(c["overwrite"] is True for c in uploads)


def test_upload_for_an_unknown_job_is_404(client):
    assert client.get("/mix/nope/upload").status_code == 404
    assert client.post("/mix/nope/upload", data={"confirm": "1"}).status_code == 404


def test_settings_page_is_reachable(client):
    assert client.get("/settings").status_code == 200
    assert client.get("/settings").get_data(as_text=True).count("Cache") >= 1


def test_upload_preview_reports_an_auth_failure_without_500(client, app,
                                                           monkeypatch):
    jobs = app.config["LORALAB_JOBS"]
    job_id = make_mix_job(client, jobs)

    def deny(*_a):
        raise gcs.GcsError("HTTP 403 denied", 403)

    monkeypatch.setattr(gcs, "stat_or_none", deny)
    response = client.get(f"/mix/{job_id}/upload")
    assert response.status_code == 200
    assert "HTTP 403 denied" in response.get_data(as_text=True)
