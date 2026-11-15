"""The mixer's web surface: eligibility, preview, 409s, the job and its files."""
from __future__ import annotations

import json
import time

import pytest

from loralab import catalog as cat
from loralab import manifests, mixjob
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
    """Write a manifest and the files it names, inside the job directory."""
    arm_dir = ctx.dir / "a0.1"
    arm_dir.mkdir(parents=True, exist_ok=True)
    fused = arm_dir / "carrier__donor__a0.1.safetensors"
    fused.write_bytes(b"fused")
    ar = arm_dir / "carrier__donor__a0.1_ar.safetensors"
    ar.write_bytes(b"ar")
    nar = arm_dir / "carrier__donor__a0.1_nar.safetensors"
    nar.write_bytes(b"nar")
    (ctx.dir / "not-in-the-manifest.txt").write_text("secret")
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
            "stats": {},
            "verify": {"kind": "delta", "passed": True,
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
    return {"arms": 1, "alphas": [0.1], "manifest": str(ctx.dir / "MANIFEST.json"),
            "out": str(ctx.dir), "converted": True}


@pytest.fixture
def mix_app(cfg, entry_factory):
    entries = [
        entry_factory("run/carrier.safetensors"),
        entry_factory("run/merged.safetensors", kind="merged"),
        entry_factory("run/converted_ar.safetensors", kind="converted"),
    ]
    cat.resolve_lineage(entries)
    cat.save_catalog(cfg.catalog_path, {
        "schema": 1, "generated_utc": "2026-11-15T00:00:00Z",
        "tool": "loralab 0.1.0", "scope": "gs://demo-bucket/ROOT",
        "prefixes": list(cfg.prefixes), "count": len(entries),
        "entries": entries, "errors": [],
    })
    jobs = Jobs(cfg.jobs_dir)
    return create_app(cfg, jobs, mixer=fake_mixer,
                      scanner=lambda ctx: {"count": 0, "errors": 0})


@pytest.fixture
def mix_client(mix_app):
    return mix_app.test_client()


def submit(client, **overrides):
    data = {"carrier": "run/carrier.safetensors",
            "donor": "run/merged.safetensors",
            "alpha": "0.1", "convert": "1"}
    data.update(overrides)
    return client.post("/mix", data=data)


def test_mix_page_offers_only_fused_and_merged_inputs(mix_client):
    body = mix_client.get("/mix/new").get_data(as_text=True)
    assert "Mix two adapters" in body
    assert 'value="run/carrier.safetensors"' in body
    assert 'value="run/merged.safetensors"' in body
    assert 'value="run/converted_ar.safetensors"' not in body


def test_mix_preview_sums_the_ranks(mix_client):
    body = mix_client.get(
        "/mix/new?carrier=run/carrier.safetensors&donor=run/merged.safetensors"
        "&alpha=0.1").get_data(as_text=True)
    assert "Preview" in body
    assert "ar:96" in body          # carrier rank 32 + merged rank 64
    assert "uniform rank" in body
    assert "MB" in body


def test_mix_preview_reports_an_out_of_range_alpha(mix_client):
    body = mix_client.get(
        "/mix/new?carrier=run/carrier.safetensors&donor=run/merged.safetensors"
        "&alpha=30").get_data(as_text=True)
    assert "not a percentage" in body


def test_mix_preview_reports_a_converted_input(mix_client):
    body = mix_client.get(
        "/mix/new?carrier=run/converted_ar.safetensors"
        "&donor=run/merged.safetensors&alpha=0.1").get_data(as_text=True)
    assert "only fused or merged" in body


def test_mix_post_with_a_bad_alpha_is_a_409(mix_client):
    page = submit(mix_client, alpha="9")
    assert page.status_code == 409
    assert "not a percentage" in page.get_data(as_text=True)


def test_mix_post_with_a_converted_donor_is_a_409(mix_client):
    page = submit(mix_client, donor="run/converted_ar.safetensors")
    assert page.status_code == 409
    assert "only fused or merged" in page.get_data(as_text=True)


def test_mix_post_with_an_unknown_adapter_is_a_409(mix_client):
    page = submit(mix_client, carrier="run/does-not-exist.safetensors")
    assert page.status_code == 409
    assert "input not found" in page.get_data(as_text=True)


def test_mix_post_submits_a_job_and_the_page_shows_the_result(mix_app, mix_client):
    page = submit(mix_client)
    assert page.status_code == 303
    job_id = page.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(mix_app.config["LORALAB_JOBS"], job_id)
    assert job["state"] == "done"
    assert job["params"]["carrier"] == "run/carrier.safetensors"
    assert job["params"]["convert"] is True

    body = mix_client.get(f"/jobs/{job_id}").get_data(as_text=True)
    assert "Result" in body
    assert "MANIFEST.json" in body
    assert "α 0.1" in body
    assert "max |δ|" in body
    assert "loralab mix" in body
    assert "--carrier" in body and "--convert" in body
    assert "Use it" in body
    assert "loras" in body and "_nar.safetensors" in body


def test_mix_post_without_conversion_records_the_choice(mix_app, mix_client):
    page = submit(mix_client, convert="0")
    job_id = page.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(mix_app.config["LORALAB_JOBS"], job_id)
    assert job["params"]["convert"] is False
    body = mix_client.get(f"/jobs/{job_id}").get_data(as_text=True)
    assert "--no-convert" in body


def test_manifest_download(mix_app, mix_client):
    page = submit(mix_client)
    job_id = page.headers["Location"].rstrip("/").split("/")[-1]
    wait_for(mix_app.config["LORALAB_JOBS"], job_id)
    response = mix_client.get(f"/mix/{job_id}/manifest.json")
    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert json.loads(response.get_data(as_text=True))["kind"] == "mix"


def test_manifest_download_is_404_when_absent(mix_client):
    assert mix_client.get("/mix/nosuchjob/manifest.json").status_code == 404
    assert mix_client.get("/mix/not-alnum/manifest.json").status_code == 404


def test_artifact_download_is_restricted_to_the_manifest(mix_app, mix_client):
    page = submit(mix_client)
    job_id = page.headers["Location"].rstrip("/").split("/")[-1]
    wait_for(mix_app.config["LORALAB_JOBS"], job_id)
    ok = mix_client.get(
        f"/mix/{job_id}/artifact/a0.1/carrier__donor__a0.1.safetensors")
    assert ok.status_code == 200 and ok.data == b"fused"
    assert mix_client.get(
        f"/mix/{job_id}/artifact/not-in-the-manifest.txt").status_code == 404
    assert mix_client.get(
        f"/mix/{job_id}/artifact/a0.1/nope.safetensors").status_code == 404
    assert mix_client.get(
        f"/mix/{job_id}/artifact/../../etc/passwd").status_code == 404


def test_detail_page_offers_mix_actions(mix_client):
    body = mix_client.get("/catalog/run/carrier.safetensors").get_data(as_text=True)
    assert "Use as carrier" in body and "Use as donor" in body
    converted = mix_client.get(
        "/catalog/run/converted_ar.safetensors").get_data(as_text=True)
    assert "Use as carrier" not in converted


def test_header_has_a_mix_tab(mix_client):
    assert 'href="/mix/new"' in mix_client.get("/catalog").get_data(as_text=True)


def test_doctor_reports_the_merge_engine_and_converter(mix_client):
    body = mix_client.get("/doctor").get_data(as_text=True)
    assert "merge engine" in body
    assert "converter" in body


def test_mix_page_without_a_catalog_says_to_scan(cfg):
    app = create_app(cfg, Jobs(cfg.jobs_dir), mixer=fake_mixer)
    body = app.test_client().get("/mix/new").get_data(as_text=True)
    assert "No catalog yet" in body
    assert "auth: —" in body


def test_mix_post_uses_the_real_default_mixer(cfg, weights_factory):
    pytest.importorskip("torch")
    carrier = weights_factory("carrier.safetensors", seed=1, rank=4)
    donor = weights_factory("donor.safetensors", seed=2, rank=4)
    app = create_app(cfg, Jobs(cfg.jobs_dir))
    client = app.test_client()
    page = client.post("/mix", data={
        "carrier": str(carrier), "donor": str(donor),
        "alpha": "0.2", "convert": "0",
    })
    assert page.status_code == 303
    job_id = page.headers["Location"].rstrip("/").split("/")[-1]
    job = wait_for(app.config["LORALAB_JOBS"], job_id, timeout=20)
    assert job["state"] == "done", job
    assert job["result"]["arms"] == 1
    assert job["result"]["converted"] is False
