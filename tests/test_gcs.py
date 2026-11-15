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
