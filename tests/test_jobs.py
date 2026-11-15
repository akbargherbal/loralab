"""Jobs: failure, cancel semantics, dedupe, crash reconciliation."""
from __future__ import annotations

import json
import threading
import time

import pytest

from loralab.jobs import Jobs


def wait_for(jobs, job_id, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_a_raising_job_becomes_failed_with_the_message(cfg):
    def boom(ctx):
        ctx.log("about to fail")
        raise RuntimeError("donor is not a fused adapter")

    jobs = Jobs(cfg.jobs_dir)
    job = jobs.submit("mix", {"alpha": 0.2}, boom)
    final = wait_for(jobs, job["id"])
    assert final["state"] == "failed"
    assert "donor is not a fused adapter" in final["error"]
    log = jobs.log_text(job["id"])[0]
    assert "about to fail" in log and "ERROR" in log


def test_a_cancelled_job_that_raises_is_cancelled_not_failed(cfg):
    started = threading.Event()

    def stop_early(ctx):
        started.set()
        while not ctx.cancelled():
            time.sleep(0.01)
        raise RuntimeError("stopped at a checkpoint")

    jobs = Jobs(cfg.jobs_dir)
    job = jobs.submit("scan", {}, stop_early)
    assert started.wait(2.0)
    assert jobs.cancel(job["id"])
    final = wait_for(jobs, job["id"])
    assert final["state"] == "cancelled"
    assert final["error"] is None


def test_dedupe_key_reuses_a_live_job_and_starts_a_new_one_after(cfg):
    gate = threading.Event()

    def slow(ctx):
        gate.wait(5)
        return {}

    jobs = Jobs(cfg.jobs_dir)
    first = jobs.submit("scan", {}, slow, dedupe_key="same")
    second = jobs.submit("scan", {}, slow, dedupe_key="same")
    assert second["id"] == first["id"]
    gate.set()
    wait_for(jobs, first["id"])

    third = jobs.submit("scan", {}, slow, dedupe_key="same")
    assert third["id"] != first["id"]
    gate.set()
    wait_for(jobs, third["id"])


def test_a_corrupt_job_file_is_skipped_not_fatal(cfg):
    bad = cfg.jobs_dir / "corrupt"
    bad.mkdir(parents=True)
    (bad / "job.json").write_text("{not json")
    jobs = Jobs(cfg.jobs_dir)          # reconcile must not raise
    assert jobs.get("corrupt") is None
    assert jobs.list() == []


def test_missing_log_reads_as_empty(cfg):
    jobs = Jobs(cfg.jobs_dir)
    assert jobs.log_text("nope") == ("", 0)


def test_cancelling_a_finished_job_is_a_no_op(cfg):
    jobs = Jobs(cfg.jobs_dir)
    job = jobs.submit("scan", {}, lambda ctx: {})
    wait_for(jobs, job["id"])
    assert jobs.cancel(job["id"]) is False
