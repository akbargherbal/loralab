"""Google Cloud Storage transport: auth ladder, listing, byte ranges.

No SDK is required. Everything goes through the JSON API and the XML API with a
bearer token, because the only thing a catalog needs from a 133 MB safetensors
file is its ~41 KB header — one `Range` request per adapter.

Auth ladder (first rung that yields a token wins, and the winner is reported in
the UI so a failure is diagnosable):

  1. `GOOGLE_APPLICATION_CREDENTIALS` -> service-account JSON via google-auth
  2. Application Default Credentials  -> `gcloud auth application-default login`
  3. `gcloud auth print-access-token` -> the store gsutil uses

Each rung **falls through** on failure. That is deliberate: on a non-GCP box
`google.auth.default()` can resolve to compute-engine metadata credentials and
then fail with a 404, which looks like "the app is broken" if you don't fall
through.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://storage.googleapis.com"
SCOPES = ["https://www.googleapis.com/auth/devstorage.read_write"]
TOKEN_TTL = 3000  # seconds; nominal, refreshed on expiry or on a 401


class GcsError(RuntimeError):
    """Storage/auth problem with an actionable message.

    `status` is the HTTP status when there was one, so callers can distinguish
    "not found" from "not allowed" without parsing the message.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Token:
    """Cached access token plus the auth mode that produced it."""

    def __init__(self, mode: str, fetch=None, initial=None):
        self.mode = mode
        self._fetch = fetch
        self._value: str | None = None
        self._expiry = 0.0
        if initial:
            self._value, self._expiry = initial[0], time.time() + float(initial[1])

    def get(self, force: bool = False) -> str:
        if force or not self._value or time.time() > self._expiry - 60:
            if self._fetch is None:
                raise GcsError("token expired and no refresher is configured")
            value, ttl = self._fetch()
            if not value:
                raise GcsError(f"auth mode {self.mode!r} returned no token")
            self._value, self._expiry = value, time.time() + float(ttl or TOKEN_TTL)
        return self._value


# ---------------------------------------------------------------------------
# auth rungs
# ---------------------------------------------------------------------------
def google_token():
    """(token, ttl) from google-auth, or None if unavailable/unusable."""
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except Exception:
        return None
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        creds.refresh(Request())
        return (creds.token, TOKEN_TTL) if creds.token else None
    except Exception:
        return None


def gcloud_token():
    exe = shutil.which("gcloud")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "auth", "print-access-token"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    token = (out.stdout or "").strip()
    return (token, TOKEN_TTL) if out.returncode == 0 and token else None


def resolve_token(environ=None) -> Token:
    """Walk the ladder; return a token pinned to the rung that worked."""
    environ = os.environ if environ is None else environ
    service_account = bool(environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    rungs = [
        ("service-account" if service_account else "adc", google_token),
        ("gcloud", gcloud_token),
    ]
    tried = []
    for mode, fetch in rungs:
        got = fetch()
        tried.append(f"{mode}: {'ok' if got else 'no'}")
        if got:
            return Token(mode, fetch=fetch, initial=got)
    raise GcsError(
        "no GCS credentials found (" + ", ".join(tried) + "). Use one of:\n"
        "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json\n"
        "  gcloud auth application-default login\n"
        "  gcloud auth login"
    )


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def _open(url: str, token: Token, headers: dict | None = None):
    """GET with a bearer token, refreshing once on 401."""
    def attempt(force: bool):
        hdrs = {"Authorization": f"Bearer {token.get(force=force)}"}
        hdrs.update(headers or {})
        req = urllib.request.Request(url, headers=hdrs)
        return urllib.request.urlopen(req, timeout=60)

    try:
        return attempt(False)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            try:
                return attempt(True)
            except urllib.error.HTTPError as again:
                raise GcsError(_explain(again, url), again.code) from again
        raise GcsError(_explain(exc, url), exc.code) from exc


def _explain(exc: urllib.error.HTTPError, url: str,
             action: str = "list/get") -> str:
    where = urllib.parse.urlsplit(url).path.split("/")[-1] or url
    detail = ""
    try:
        body = exc.read()
        if body:
            detail = (json.loads(body).get("error") or {}).get("message") or ""
    except Exception:
        detail = ""
    suffix = f" — {detail}" if detail else ""
    if exc.code == 403:
        return (f"{where}: HTTP 403 — the credentials lack storage.objects.{action} "
                f"on this prefix{suffix}")
    if exc.code == 404:
        return (f"{where}: HTTP 404 — object or bucket not found (check bucket/root)"
                + suffix)
    if exc.code == 412:
        return (f"{where}: HTTP 412 — the object already exists (or changed while "
                f"uploading); pass the overwrite flag to replace it deliberately"
                + suffix)
    return f"{where}: HTTP {exc.code} {exc.reason}{suffix}"


def list_objects(bucket: str, prefix: str, suffix: str = ".safetensors",
                 token: Token | None = None) -> list[dict]:
    """Paginated listing: name, size, md5Hash, crc32c, updated. No downloads."""
    token = token or resolve_token()
    fields = "items(name,size,md5Hash,crc32c,updated),nextPageToken"
    page, out = None, []
    while True:
        params = {"prefix": prefix, "fields": fields, "maxResults": "200"}
        if page:
            params["pageToken"] = page
        url = f"{API}/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?" + \
            urllib.parse.urlencode(params)
        resp = _open(url, token)
        with resp:
            data = json.load(resp)
        out.extend(data.get("items", []))
        page = data.get("nextPageToken")
        if not page:
            break
    return [o for o in out if o["name"].endswith(suffix)]


def read_range(bucket: str, obj: str, start: int, end: int,
               token: Token) -> bytes:
    """Inclusive byte range of one object."""
    url = f"{API}/{urllib.parse.quote(bucket, safe='')}/{urllib.parse.quote(obj, safe='/')}"
    resp = _open(url, token, {"Range": f"bytes={start}-{end}"})
    with resp:
        return resp.read()


def stat_object(bucket: str, obj: str, token: Token,
                fields: str | None = None) -> dict:
    url = (f"{API}/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o/"
           f"{urllib.parse.quote(obj, safe='')}")
    if fields:
        url += "?" + urllib.parse.urlencode({"fields": fields})
    resp = _open(url, token)
    with resp:
        return json.load(resp)


def stat_or_none(bucket: str, obj: str, token: Token) -> dict | None:
    """Stat one object, treating 404 as "does not exist" and nothing else as absent."""
    try:
        return stat_object(bucket, obj, token)
    except GcsError as exc:
        if exc.status == 404:
            return None
        raise


def download_object(bucket: str, obj: str, dest, token: Token | None = None,
                    *, log=None, chunk: int = 1 << 22,
                    progress_bytes: int = 16 << 20) -> dict:
    """Stream one object to `dest` and return size + sha256.

    Writes to `<dest>.part` first, so a killed download never looks like a
    complete input. Every `progress_bytes` it logs progress, which is what makes
    a 133 MB staging step visible in the job log instead of a silent pause.
    """
    import hashlib

    token = token or resolve_token()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/{urllib.parse.quote(bucket, safe='')}/{urllib.parse.quote(obj, safe='/')}"
    digest = hashlib.sha256()
    total = 0
    mark = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(url, token) as resp, open(tmp, "wb") as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            digest.update(block)
            total += len(block)
            if log and total - mark >= progress_bytes:
                mark = total
                log(f"  {dest.name}: {total / 1e6:.0f} MB")
    tmp.replace(dest)
    return {"path": str(dest), "size": total, "sha256": digest.hexdigest()}


def upload_object(bucket: str, obj: str, source, token: Token | None = None, *,
                  overwrite: bool = False, generation: str | None = None,
                  log=None) -> dict:
    """Create one object from a local file, with a create-only precondition.

    The default is **create-only** (`ifGenerationMatch=0`): an existing key is a
    `412`, never a silent replacement. `overwrite=True` still pins the generation
    seen by the preview, so a file that changed between preview and upload fails
    instead of replacing the wrong bytes. No prompt can turn this into a delete:
    the app never sends a DELETE, and the recommended IAM role has no delete
    permission.
    """
    import base64
    import hashlib

    token = token or resolve_token()
    source = Path(source)
    if not source.is_file():
        raise GcsError(f"cannot upload {source}: file not found")
    size = source.stat().st_size
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with open(source, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
            md5.update(block)
    sha256 = sha.hexdigest()
    md5_b64 = base64.b64encode(md5.digest()).decode()
    sha_b64 = base64.b64encode(bytes.fromhex(sha256)).decode()

    params = {"uploadType": "media", "name": obj}
    # Create-only unless an overwrite was explicitly requested *and* the preview
    # saw the object. If it did not exist then, keep create-only so a race cannot
    # turn into an accidental replacement.
    params["ifGenerationMatch"] = generation if (overwrite and generation) else "0"
    url = (f"{API}/upload/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?"
           + urllib.parse.urlencode(params))

    def attempt(force: bool):
        headers = {
            "Authorization": f"Bearer {token.get(force=force)}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "X-Goog-Hash": f"sha256={sha_b64}",
        }
        # A fresh handle per attempt: a 401 retry must not reuse a consumed body.
        handle = open(source, "rb")
        req = urllib.request.Request(url, data=handle, headers=headers,
                                     method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except Exception:
            handle.close()
            raise
        # urlopen has finished sending the body by the time it returns.
        handle.close()
        return resp

    if log:
        log(f"upload: {obj} ({size / 1e6:.1f} MB)")
    try:
        resp = attempt(False)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            try:
                resp = attempt(True)
            except urllib.error.HTTPError as again:
                raise GcsError(_explain(again, url, "create"), again.code) from again
        else:
            raise GcsError(_explain(exc, url, "create"), exc.code) from exc
    with resp:
        try:
            meta = json.load(resp)
        except ValueError:
            meta = {}

    # Read back the object the service actually stored. GCS keeps md5/crc32c,
    # not sha256, so size + md5 are the server-side checks; sha256 stays the
    # local identity recorded in the manifest.
    stored = stat_object(
        bucket, obj, token,
        fields="name,size,generation,md5Hash,crc32c,updated",
    )
    if int(stored.get("size", -1)) != size:
        raise GcsError(
            f"{obj}: uploaded {size} bytes but the bucket reports "
            f"{stored.get('size')} — the upload did not land intact"
        )
    server_md5 = stored.get("md5Hash")
    md5_verified = False
    if server_md5:
        if server_md5 != md5_b64:
            raise GcsError(
                f"{obj}: bucket md5 does not match the local file; refusing "
                "to call this upload complete"
            )
        md5_verified = True
    return {
        "name": obj,
        "size": size,
        "sha256": sha256,
        "md5_verified": md5_verified,
        "generation": stored.get("generation") or meta.get("generation"),
        "crc32c": stored.get("crc32c"),
        "updated": stored.get("updated"),
    }
