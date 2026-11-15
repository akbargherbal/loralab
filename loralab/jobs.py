"""Jobs: on-disk state, one FIFO worker, cooperative cancel.

A scan is 27 s and a mix is seconds, so neither belongs inside a request. Every
job is a directory:

    <data>/jobs/<job_id>/
        job.json     state machine, atomically replaced on every transition
        log.txt      append-only; also the thing the UI polls

Nothing lives only in memory: if the process dies, `reconcile()` turns a stale
`running` job into `failed` on the next start rather than showing a zombie.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = ("done", "failed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Context:
    """What a job function gets: its params, a logger, and a cancel check."""

    def __init__(self, jobs: "Jobs", job_id: str, params: dict):
        self._jobs = jobs
        self.id = job_id
        self.params = params
        self.dir = jobs.root / job_id

    def log(self, message: str = "") -> None:
        self._jobs.append_log(self.id, message)

    def cancelled(self) -> bool:
        return self._jobs.cancel_requested(self.id)


class Jobs:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.reconcile()
        self._thread = threading.Thread(target=self._worker, name="loralab-worker",
                                        daemon=True)
        self._thread.start()

    # -- disk --------------------------------------------------------------
    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _write(self, job_id: str, job: dict) -> None:
        path = self._dir(job_id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2) + "\n")
        tmp.replace(path)

    def append_log(self, job_id: str, message: str) -> None:
        with open(self._dir(job_id) / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(message.rstrip("\n") + "\n")

    def reconcile(self) -> None:
        """Any job left `running` by a dead process becomes `failed`."""
        for job_file in self.root.glob("*/job.json"):
            try:
                job = json.loads(job_file.read_text())
            except (OSError, ValueError):
                continue
            if job.get("state") in ("queued", "running"):
                job["state"] = "failed"
                job["error"] = "engine restarted"
                job["ended_utc"] = _now()
                self._write(job_file.parent.name, job)

    def get(self, job_id: str) -> dict | None:
        path = self._dir(job_id) / "job.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def list(self, limit: int = 20) -> list[dict]:
        jobs = []
        for job_file in self.root.glob("*/job.json"):
            try:
                jobs.append(json.loads(job_file.read_text()))
            except (OSError, ValueError):
                continue
        jobs.sort(key=lambda j: j.get("created_utc", ""), reverse=True)
        return jobs[:limit]

    def log_text(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        path = self._dir(job_id) / "log.txt"
        if not path.is_file():
            return "", 0
        try:
            with open(path, encoding="utf-8") as fh:
                fh.seek(offset)
                chunk = fh.read()
                return chunk, fh.tell()
        except OSError:
            return "", offset

    # -- lifecycle ---------------------------------------------------------
    def cancel_requested(self, job_id: str) -> bool:
        return (self._dir(job_id) / "cancel").exists()

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.get("state") in TERMINAL:
            return False
        (self._dir(job_id) / "cancel").touch()
        self.append_log(job_id, "cancel requested")
        return True

    def submit(self, kind: str, params: dict, func, *, dedupe_key: str | None = None) -> dict:
        """Queue a job. `func(ctx)` runs on the worker; returns the job record."""
        with self._lock:
            if dedupe_key:
                for job in self.list(limit=50):
                    if (job.get("dedupe_key") == dedupe_key
                            and job.get("state") in ("queued", "running")):
                        return job
            job_id = uuid.uuid4().hex[:12]
            self._dir(job_id).mkdir(parents=True, exist_ok=True)
            job = {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "created_utc": _now(),
                "started_utc": None,
                "ended_utc": None,
                "params": params,
                "dedupe_key": dedupe_key,
                "error": None,
            }
            self._write(job_id, job)
            self.append_log(job_id, f"{kind}: queued {json.dumps(params, sort_keys=True)}")
            self._queue.put((job_id, func))
        return job

    def _worker(self) -> None:
        while True:
            job_id, func = self._queue.get()
            job = self.get(job_id)
            if not job or self.cancel_requested(job_id):
                if job:
                    job.update(state="cancelled", ended_utc=_now())
                    self._write(job_id, job)
                self._queue.task_done()
                continue
            job.update(state="running", started_utc=_now())
            self._write(job_id, job)
            started = time.perf_counter()
            try:
                result = func(Context(self, job_id, job.get("params") or {}))
                job = self.get(job_id) or job
                if self.cancel_requested(job_id):
                    job["state"] = "cancelled"
                else:
                    job["state"] = "done"
                    job["result"] = result
            except Exception as exc:  # noqa: BLE001 — surfaced in the UI
                job = self.get(job_id) or job
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                self.append_log(job_id, f"ERROR {job['error']}")
            finally:
                job["elapsed_s"] = round(time.perf_counter() - started, 2)
                job["ended_utc"] = _now()
                self._write(job_id, job)
                self._queue.task_done()
