"""Run the audio.cpp converter, with the version pinned and cached.

The converter is a separate object (`converter_object` in the config, by default
`audiocpp_inference/converter/convert_aitoolkit_yue2_lora.py` in the bucket). It
is downloaded once into `<data>/cache/tools/`, and the first download's sha256
is written next to it as a **trust-on-first-use pin**. Later runs refuse to
execute a file whose hash changed. Set `converter_sha256` explicitly to pin a
known artifact instead.

This is a re-implementation of the "download once, pin the sha256, cache it"
rule in `PLAN.md`; the converter itself is not vendored here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import gcs
from . import catalog as cat
from .merge import sha256_file


class ConvertError(RuntimeError):
    """The converter is missing, untrusted, or failed."""


def cached_path(cfg, name: str | None = None) -> Path:
    return Path(cfg.data_dir) / "cache" / "tools" / (name or Path(cfg.converter_object).name)


def _pin_path(script: Path) -> Path:
    return script.with_suffix(script.suffix + ".sha256")


def _verify(path: Path, expected: str, *, source: str) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != expected.strip().lower():
        raise ConvertError(
            f"{source}: sha256 mismatch for {path}.\n"
            f"  expected {expected.strip().lower()}\n"
            f"  actual   {actual}\n"
            "Refusing to run it. Delete the cached file to re-download, or fix "
            "converter_sha256 if the pin is wrong."
        )
    return actual


def resolve(cfg, *, path=None, token=None, log=None) -> dict:
    """Return {path, sha256, pinned, source}; download and pin if needed."""
    log = log or (lambda *_: None)
    expected = (cfg.converter_sha256 or "").strip()
    if path:
        script = Path(path).expanduser()
        if not script.is_file():
            raise ConvertError(f"converter not found: {script}")
        digest = _verify(script, expected, source="--converter")
        return {"path": script, "sha256": digest, "pinned": bool(expected),
                "source": "local"}

    script = cached_path(cfg)
    pin_file = _pin_path(script)
    if not expected and pin_file.is_file():
        expected = pin_file.read_text().strip()
    if script.is_file():
        digest = _verify(script, expected, source="cached converter")
        return {"path": script, "sha256": digest, "pinned": bool(expected),
                "source": "cache"}

    if not cfg.bucket:
        raise ConvertError(
            "no converter available and no bucket configured to download it "
            f"from.\n  Pass --converter /path/to/convert_aitoolkit_yue2_lora.py\n"
            f"  or configure a bucket (the object is {cfg.converter_object})."
        )
    object_name = f"{cfg.root}/{cfg.converter_object}" if cfg.root else cfg.converter_object
    log(f"converter: downloading gs://{cfg.bucket}/{object_name}")
    gcs.download_object(cfg.bucket, object_name, script, token=token, log=log)
    digest = _verify(script, expected, source="downloaded converter")
    if not expected:
        pin_file.write_text(digest + "\n")
        log(f"converter: pinned sha256 {digest} (first use)")
    return {"path": script, "sha256": digest, "pinned": bool(expected),
            "source": "download"}


def run(script, fused_path, out_dir, *, stem: str | None = None,
        log=None, timeout: float = 900.0) -> dict:
    """Convert one fused file. Returns the pair's paths, hashes and ranks."""
    import queue
    import threading
    import time

    log = log or (lambda *_: None)
    script, fused_path, out_dir = Path(script), Path(fused_path), Path(out_dir)
    if not fused_path.is_file():
        raise ConvertError(f"fused input not found: {fused_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or fused_path.stem

    command = [sys.executable, str(script), str(fused_path),
               "--out-dir", str(out_dir), "--stem", stem]
    log(f"convert: {' '.join(command)}")
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(out_dir),
        )
    except OSError as exc:
        raise ConvertError(f"cannot start the converter: {exc}") from exc

    chunks: queue.Queue = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                chunks.put(line.rstrip("\n"))
        finally:
            chunks.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            proc.kill()
            break
        try:
            line = chunks.get(timeout=min(remaining, 0.25))
        except queue.Empty:
            if proc.poll() is not None and not reader_thread.is_alive():
                break
            continue
        if line is None:
            break
        lines.append(line)
        log(f"  {line}")
        if len(lines) > 400:
            lines = lines[-400:]
    code = proc.wait()
    if timed_out:
        raise ConvertError(
            f"converter did not finish within {timeout:.0f}s and was killed"
        )

    if code != 0:
        tail = "\n".join(lines[-12:])
        raise ConvertError(f"converter exited {code}\n{tail}")

    outputs = {}
    for branch in ("ar", "nar"):
        expected = out_dir / f"{stem}_{branch}.safetensors"
        found = [expected] if expected.is_file() else sorted(
            out_dir.glob(f"*_{branch}.safetensors"))
        if not found:
            raise ConvertError(
                f"converter exited 0 but wrote no _{branch} file in {out_dir} "
                f"(contains: {sorted(p.name for p in out_dir.iterdir())[:8]})"
            )
        path = found[0]
        meta, tensors = cat.header_from_local(path)
        info = cat.classify(meta, tensors)
        outputs[branch] = {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rank": (info["rank"].get(branch) or [None])[0],
        }

    verified = any("byte-level verification" in line and "OK" in line for line in lines)
    if not verified:
        log("  WARNING: the converter did not print its byte-level verification line")
    return {
        "command": command,
        "stem": stem,
        "outputs": outputs,
        "byte_verified": verified,
        "stdout_tail": lines[-40:],
    }
