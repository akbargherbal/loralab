"""Transport: fake HTTP, no network, no credentials."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

import pytest

from loralab import gcs


class FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, *_a):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _token(fetch=None):
    fetch = fetch or (lambda: ("tok", 3000))
    return gcs.Token("test", fetch=fetch)


@pytest.fixture
def fake_http(monkeypatch):
    """Route listing URLs and range URLs to in-memory handlers."""
    state = {"pages": {}, "raw": b"", "calls": [], "fail_first": None}

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        state["calls"].append(url)
        if state["fail_first"] is not None:
            code, state["fail_first"] = state["fail_first"], None
            raise urllib.error.HTTPError(url, code, "nope", {}, None)
        if "/storage/v1/b/" in url:
            token = (parse_qs(urlparse(url).query).get("pageToken") or [None])[0]
            return FakeResp(json.dumps(state["pages"][token]).encode())
        rng = req.headers.get("Range") or req.get_header("Range")
        start, end = (int(x) for x in rng.split("=")[1].split("-"))
        return FakeResp(state["raw"][start:end + 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return state


def test_list_objects_paginates_and_filters_suffix(fake_http):
    fake_http["pages"] = {
        None: {"items": [{"name": "ROOT/a/x.safetensors", "size": "1"},
                         {"name": "ROOT/a/x.json", "size": "2"}],
               "nextPageToken": "p2"},
        "p2": {"items": [{"name": "ROOT/b/y.safetensors", "size": "3"}]},
    }
    out = gcs.list_objects("b", "ROOT/", token=_token())
    assert [o["name"] for o in out] == ["ROOT/a/x.safetensors", "ROOT/b/y.safetensors"]


def test_list_objects_requests_the_identity_fields(fake_http):
    fake_http["pages"] = {None: {"items": []}}
    gcs.list_objects("b", "p/", token=_token())
    url = parse_qs(urlparse(fake_http["calls"][0]).query)
    assert "crc32c" in url["fields"][0] and "md5Hash" in url["fields"][0]


def test_read_range_asks_for_the_exact_bytes(fake_http):
    fake_http["raw"] = b"0123456789"
    assert gcs.read_range("b", "obj", 2, 5, _token()) == b"2345"
    assert "obj" in fake_http["calls"][0]


def test_401_refreshes_the_token_once(fake_http):
    fake_http["raw"] = b"0123456789"
    fake_http["fail_first"] = 401
    calls = []

    def fetch():
        calls.append(1)
        return ("tok", 3000)

    assert gcs.read_range("b", "obj", 0, 3, _token(fetch)) == b"0123"
    assert len(calls) == 2          # once for the request, once forced on 401


def test_non_401_is_raised_as_gcs_error(fake_http):
    fake_http["raw"] = b"x"
    fake_http["fail_first"] = 403
    with pytest.raises(gcs.GcsError, match="403"):
        gcs.read_range("b", "obj", 0, 0, _token())


def test_403_message_names_the_permission(fake_http):
    fake_http["raw"] = b"x"
    fake_http["fail_first"] = 403
    with pytest.raises(gcs.GcsError, match="objects.list/get"):
        gcs.list_objects("b", "p/", token=_token())


def test_token_is_cached_between_calls():
    calls = []

    def fetch():
        calls.append(1)
        return ("tok", 3000)

    token = gcs.Token("test", fetch=fetch)
    assert token.get() == "tok"
    assert token.get() == "tok"
    assert len(calls) == 1


def test_ladder_falls_through_when_google_auth_fails(monkeypatch):
    monkeypatch.setattr(gcs, "google_token", lambda: None)
    monkeypatch.setattr(gcs, "gcloud_token", lambda: ("gcloud-tok", 3000))
    token = gcs.resolve_token(environ={})
    assert token.mode == "gcloud"
    assert token.get() == "gcloud-tok"


def test_ladder_reports_service_account_mode(monkeypatch):
    monkeypatch.setattr(gcs, "google_token", lambda: ("sa-tok", 3000))
    token = gcs.resolve_token(environ={"GOOGLE_APPLICATION_CREDENTIALS": "/k.json"})
    assert token.mode == "service-account"
    assert token.get() == "sa-tok"


def test_ladder_explains_how_to_authenticate(monkeypatch):
    monkeypatch.setattr(gcs, "google_token", lambda: None)
    monkeypatch.setattr(gcs, "gcloud_token", lambda: None)
    with pytest.raises(gcs.GcsError) as err:
        gcs.resolve_token(environ={})
    message = str(err.value)
    assert "GOOGLE_APPLICATION_CREDENTIALS" in message
    assert "gcloud auth login" in message


# ---------------------------------------------------------------------------
# auth rungs
# ---------------------------------------------------------------------------
def test_gcloud_token_parses_stdout(monkeypatch):
    monkeypatch.setattr(gcs.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(gcs.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "tok\n"})())
    assert gcs.gcloud_token() == ("tok", gcs.TOKEN_TTL)


def test_gcloud_token_is_none_on_failure(monkeypatch):
    monkeypatch.setattr(gcs.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(gcs.subprocess, "run", lambda *a, **k: type(
        "R", (), {"returncode": 1, "stdout": ""})())
    assert gcs.gcloud_token() is None


def test_gcloud_token_is_none_without_the_binary(monkeypatch):
    monkeypatch.setattr(gcs.shutil, "which", lambda name: None)
    assert gcs.gcloud_token() is None


def test_gcloud_token_survives_a_subprocess_error(monkeypatch):
    monkeypatch.setattr(gcs.shutil, "which", lambda name: "/usr/bin/gcloud")

    def boom(*a, **k):
        raise OSError("gone")

    monkeypatch.setattr(gcs.subprocess, "run", boom)
    assert gcs.gcloud_token() is None


def test_google_token_is_none_when_the_library_is_absent(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] == "google":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert gcs.google_token() is None


def test_stat_object_reads_one_object(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return FakeResp(b'{"name": "o", "size": "1"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert gcs.stat_object("b", "o", _token())["name"] == "o"


def test_404_message_names_bucket_and_root(fake_http):
    fake_http["pages"] = {}
    fake_http["fail_first"] = 404
    with pytest.raises(gcs.GcsError, match="bucket/root"):
        gcs.list_objects("b", "p/", token=_token())


def test_401_twice_is_a_gcs_error_not_a_loop(monkeypatch):
    def always_401(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", always_401)
    calls = []

    def fetch():
        calls.append(1)
        return ("tok", 3000)

    with pytest.raises(gcs.GcsError, match="401"):
        gcs.read_range("b", "o", 0, 3, gcs.Token("test", fetch=fetch))
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# the google-auth rung (fake modules, no library, no network)
# ---------------------------------------------------------------------------
def _install_fake_google(monkeypatch, *, token="g-token", raises=None):
    import sys
    import types

    calls = {}

    class Creds:
        def __init__(self):
            self.token = token

        def refresh(self, request):
            calls["refreshed"] = True
            if raises:
                raise raises

    class Request:  # noqa: D401 — stands in for google.auth.transport.requests.Request
        pass

    google = types.ModuleType("google")
    auth = types.ModuleType("google.auth")
    auth.default = lambda scopes=None: (Creds(), "proj")
    transport = types.ModuleType("google.auth.transport")
    requests = types.ModuleType("google.auth.transport.requests")
    requests.Request = Request
    google.auth, auth.transport, transport.requests = auth, transport, requests

    for name, module in (("google", google), ("google.auth", auth),
                         ("google.auth.transport", transport),
                         ("google.auth.transport.requests", requests)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_google_token_refreshes_and_returns_the_token(monkeypatch):
    calls = _install_fake_google(monkeypatch)
    assert gcs.google_token() == ("g-token", gcs.TOKEN_TTL)
    assert calls["refreshed"] is True


def test_google_token_is_none_when_refresh_fails(monkeypatch):
    _install_fake_google(monkeypatch, raises=RuntimeError("metadata 404"))
    assert gcs.google_token() is None


def test_google_token_is_none_when_credentials_have_no_token(monkeypatch):
    _install_fake_google(monkeypatch, token="")
    assert gcs.google_token() is None


# ---------------------------------------------------------------------------
# defensive token states
# ---------------------------------------------------------------------------
def test_an_expired_token_without_a_refresher_says_so():
    token = gcs.Token("test")
    with pytest.raises(gcs.GcsError, match="no refresher"):
        token.get()


def test_a_refresher_that_returns_nothing_says_so():
    token = gcs.Token("test", fetch=lambda: ("", 0))
    with pytest.raises(gcs.GcsError, match="returned no token"):
        token.get()


# ---------------------------------------------------------------------------
# streaming download (mix staging)
# ---------------------------------------------------------------------------
class ChunkedResp:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def read(self, n=None):
        block = self.data[self.pos:self.pos + (n or len(self.data))]
        self.pos += len(block)
        return block

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_download_object_streams_hashes_and_logs_progress(tmp_path, monkeypatch):
    payload = b"0123456789"
    urls = []

    def fake_open(url, token, headers=None):
        urls.append(url)
        return ChunkedResp(payload)

    monkeypatch.setattr(gcs, "_open", fake_open)
    logs = []
    dest = tmp_path / "nested" / "obj.safetensors"
    got = gcs.download_object("b", "root/obj.safetensors", dest,
                              token=_token(), log=logs.append, chunk=4,
                              progress_bytes=4)
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".safetensors.part").exists()
    assert got["size"] == len(payload)
    assert got["sha256"] == (
        "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882")
    assert got["path"] == str(dest)
    assert logs and "obj.safetensors" in logs[0]
    assert "root%2Fobj.safetensors" in urls[0] or "root/obj.safetensors" in urls[0]


def test_download_object_without_a_token_resolves_one(tmp_path, monkeypatch):
    monkeypatch.setattr(gcs, "_open", lambda *a, **k: ChunkedResp(b"x"))
    monkeypatch.setattr(gcs, "resolve_token", lambda: _token())
    got = gcs.download_object("b", "obj", tmp_path / "x", token=None)
    assert got["size"] == 1


def test_download_object_surfaces_a_storage_error(tmp_path, monkeypatch):
    def deny(*_a, **_k):
        raise gcs.GcsError("HTTP 403 denied")

    monkeypatch.setattr(gcs, "_open", deny)
    with pytest.raises(gcs.GcsError, match="403"):
        gcs.download_object("b", "obj", tmp_path / "x", token=_token())


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
class UploadResp:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def read(self, *_a):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _sha_b64(path):
    import base64
    import hashlib
    return base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()


def _md5_b64(path):
    import base64
    import hashlib
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


@pytest.fixture
def upload_http(monkeypatch):
    """A fake JSON API: POST records the body, GET serves the stat response."""
    state = {"posts": [], "stat": None, "post_status": None}

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            body = req.data.read() if hasattr(req.data, "read") else req.data
            state["posts"].append({"url": req.full_url, "body": body,
                                   "headers": dict(req.headers)})
            if state["post_status"]:
                raise urllib.error.HTTPError(req.full_url, state["post_status"],
                                             "no", {}, None)
            return UploadResp(b'{"generation": "42"}')
        return UploadResp(json.dumps(state["stat"]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return state


def test_upload_object_creates_with_a_generation_zero_precondition(
        tmp_path, upload_http):
    path = tmp_path / "a.safetensors"
    path.write_bytes(b"payload")
    upload_http["stat"] = {"size": "7", "generation": "42",
                           "md5Hash": _md5_b64(path), "crc32c": "AAAAAA==",
                           "updated": "2026-11-15T00:00:00Z"}
    result = gcs.upload_object("b", "ROOT/a.safetensors", path, token=_token())
    assert result["size"] == 7 and result["md5_verified"] is True
    assert result["generation"] == "42"
    post = upload_http["posts"][0]
    query = parse_qs(urlparse(post["url"]).query)
    assert query["uploadType"] == ["media"]
    assert query["name"] == ["ROOT/a.safetensors"]
    assert query["ifGenerationMatch"] == ["0"]
    assert post["body"] == b"payload"
    assert post["headers"].get("X-goog-hash") == f"sha256={_sha_b64(path)}"


def test_upload_object_overwrite_pins_the_previewed_generation(
        tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"payload")
    upload_http["stat"] = {"size": "7", "md5Hash": _md5_b64(path)}
    gcs.upload_object("b", "o", path, token=_token(), overwrite=True,
                      generation="17")
    query = parse_qs(urlparse(upload_http["posts"][0]["url"]).query)
    assert query["ifGenerationMatch"] == ["17"]


def test_upload_object_overwrite_without_a_seen_generation_stays_create_only(
        tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"x")
    upload_http["stat"] = {"size": "1"}
    gcs.upload_object("b", "o", path, token=_token(), overwrite=True)
    query = parse_qs(urlparse(upload_http["posts"][0]["url"]).query)
    assert query["ifGenerationMatch"] == ["0"]


def test_upload_object_verifies_the_stored_size(tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"payload")
    upload_http["stat"] = {"size": "6"}
    with pytest.raises(gcs.GcsError, match="did not land intact"):
        gcs.upload_object("b", "o", path, token=_token())


def test_upload_object_rejects_a_different_server_hash(tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"payload")
    upload_http["stat"] = {"size": "7", "md5Hash": "not-it"}
    with pytest.raises(gcs.GcsError, match="md5 does not match"):
        gcs.upload_object("b", "o", path, token=_token())


def test_upload_object_412_explains_overwrite(tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"x")
    upload_http["post_status"] = 412
    with pytest.raises(gcs.GcsError, match="overwrite") as err:
        gcs.upload_object("b", "o", path, token=_token())
    assert err.value.status == 412


def test_upload_object_needs_a_local_file(tmp_path):
    with pytest.raises(gcs.GcsError, match="file not found"):
        gcs.upload_object("b", "o", tmp_path / "missing", token=_token())


def test_stat_or_none_treats_only_404_as_absent(monkeypatch):
    def absent(*_a, **_k):
        raise gcs.GcsError("404", 404)

    monkeypatch.setattr(gcs, "stat_object", absent)
    assert gcs.stat_or_none("b", "o", _token()) is None

    def denied(*_a, **_k):
        raise gcs.GcsError("403", 403)

    monkeypatch.setattr(gcs, "stat_object", denied)
    with pytest.raises(gcs.GcsError, match="403"):
        gcs.stat_or_none("b", "o", _token())


def test_upload_401_refreshes_and_retries_with_a_fresh_handle(tmp_path,
                                                             monkeypatch):
    import io
    path = tmp_path / "a"
    path.write_bytes(b"payload")
    calls = {"post": 0}

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            calls["post"] += 1
            if calls["post"] == 1:
                raise urllib.error.HTTPError(req.full_url, 401, "expired",
                                             {}, io.BytesIO(b"{}"))
            assert req.data.read() == b"payload"
            return UploadResp(b'{"generation": "5"}')
        return UploadResp(json.dumps({"size": "7", "md5Hash": _md5_b64(path)}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    fetches = []

    def fetch():
        fetches.append(1)
        return ("tok", 3000)

    result = gcs.upload_object("b", "o", path, token=_token(fetch))
    assert calls["post"] == 2
    assert len(fetches) == 2
    assert result["generation"] == "5" and result["md5_verified"] is True


def test_upload_tolerates_an_empty_post_body(tmp_path, monkeypatch):
    path = tmp_path / "a"
    path.write_bytes(b"x")

    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            return UploadResp(b"")
        return UploadResp(json.dumps({"size": "1"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = gcs.upload_object("b", "o", path, token=_token())
    assert result["size"] == 1 and result["md5_verified"] is False


def test_upload_logs_the_object_and_size(tmp_path, upload_http):
    path = tmp_path / "a"
    path.write_bytes(b"1234")
    upload_http["stat"] = {"size": "4", "md5Hash": _md5_b64(path)}
    lines = []
    gcs.upload_object("b", "o", path, token=_token(), log=lines.append)
    assert lines and "o (0.0 MB)" in lines[0]


def test_error_message_includes_the_service_detail(monkeypatch):
    import io

    def bad(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error": {"message": "Invalid field selection"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", bad)
    with pytest.raises(gcs.GcsError, match="Invalid field selection") as err:
        gcs.stat_object("b", "o", _token())
    assert err.value.status == 400
