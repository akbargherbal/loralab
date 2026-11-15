"""Header parsing, classification, lineage, filtering — no network."""
from __future__ import annotations

import json
import struct

import pytest

from loralab import catalog as cat
from loralab.config import Config
from tests.conftest import (converted_tensors, fused_tensors, st_bytes, tensor)


def parse(meta=None, tensors=None):
    blob = st_bytes(meta, tensors)
    return cat.parse_header(lambda s, e: blob[s:e + 1])


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
def test_header_roundtrip_returns_metadata_and_tensors():
    meta, tensors = parse({"ss_output_name": "demo"}, fused_tensors(rank=32))
    assert meta["ss_output_name"] == "demo"
    assert len(tensors) == 4


def test_short_prefix_is_an_error():
    with pytest.raises(ValueError, match="short length prefix"):
        cat.parse_header(lambda s, e: b"\x00\x00")


def test_implausible_length_is_an_error():
    with pytest.raises(ValueError, match="implausible"):
        cat.parse_header(lambda s, e: (b"\x00" * 8) if s == 0 else b"{}")


def test_header_without_metadata_is_fine():
    meta, tensors = parse(None, fused_tensors())
    assert meta == {} and len(tensors) == 4


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_fused_has_both_branches_and_ranks(entry_factory):
    entry = entry_factory("v3/output/f.safetensors")
    assert entry["kind"] == "fused"
    assert entry["branches"] == ["ar", "nar"]
    assert entry["rank"] == {"ar": [32], "nar": [32]}
    assert entry["base_model"] == "yue2"


def test_converted_uses_the_declared_branch(entry_factory):
    entry = entry_factory("x/demo_ar.safetensors", kind="converted")
    assert entry["kind"] == "converted"
    assert entry["branches"] == ["ar"]
    assert entry["rank"] == {"ar": [64]}
    assert entry["source_file"] == "demo.safetensors"


def test_merged_recovers_full_provenance(entry_factory):
    entry = entry_factory("x/m.safetensors", kind="merged")
    assert entry["kind"] == "merged"
    assert entry["merge"]["alpha"] == 0.2
    assert entry["merge"]["tool"] == "merge_quran_lora.py"
    assert entry["merge"]["out_ranks"] == {"ar": 64, "nar": 64}
    assert entry["rank"] == {"ar": [64], "nar": [64]}


def test_triggers_are_sorted_across_groups():
    meta = {"ss_tag_frequency": json.dumps({
        "1_a": {"rock": 5, "arabmaqamrock": 100},
        "2_b": {"arabic": 70},
    })}
    assert cat.triggers(meta) == ["arabmaqamrock", "arabic", "rock"]


def test_garbage_metadata_never_raises():
    assert cat.triggers({"ss_tag_frequency": "not json"}) == []
    assert cat.training({"training_info": "not json"}) == {}
    assert cat.software({"software": "not json"}) is None
    assert cat.software({}) is None


def test_training_and_software_are_parsed():
    meta = {"training_info": json.dumps({"step": 3000, "epoch": 11, "x": 1}),
            "software": json.dumps({"name": "ai-toolkit", "version": "0.13.16"})}
    assert cat.training(meta) == {"step": 3000, "epoch": 11}
    assert cat.software(meta) == "ai-toolkit 0.13.16"


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------
def test_converted_links_to_fused_by_source_file(entry_factory):
    fused = entry_factory("run/output/demo.safetensors")
    converted = entry_factory("run/convert/demo_ar.safetensors", kind="converted")
    entries = [fused, converted]
    cat.resolve_lineage(entries)
    assert converted["lineage"] == {"fused_path": fused["path"], "resolved": True}


def test_merged_link_stays_open_without_a_known_sha(entry_factory):
    entry = entry_factory("x/m.safetensors", kind="merged",
                          meta={"merge_v2_sha256": "a" * 64,
                                "merge_quran_sha256": "b" * 64})
    cat.resolve_lineage([entry])
    assert entry["lineage"]["resolved"] is False
    assert entry["lineage"]["carrier_path"] is None


def test_merged_link_resolves_when_the_sha_is_present(entry_factory):
    carrier = entry_factory("run/output/demo.safetensors")
    carrier["sha256"] = "a" * 64
    merged = entry_factory("x/m.safetensors", kind="merged",
                           meta={"merge_v2_sha256": "a" * 64,
                                 "merge_quran_sha256": "b" * 64})
    cat.resolve_lineage([carrier, merged])
    assert merged["lineage"]["carrier_path"] == carrier["path"]
    assert merged["lineage"]["resolved"] is False  # donor still unknown


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def test_group_uses_the_first_two_segments(entry_factory):
    entries = [entry_factory("run/output/a.safetensors"),
               entry_factory("run/output/b.safetensors"),
               entry_factory("loras/audio_cpp/style/c.safetensors", kind="converted")]
    groups = cat.group(entries)
    assert [g["key"] for g in groups] == ["loras/audio_cpp", "run/output"]
    assert len(groups[1]["entries"]) == 2


def test_filters_are_case_insensitive_and_combine(entry_factory):
    entries = [
        entry_factory("run/output/alpha.safetensors"),
        entry_factory("run/convert/beta_ar.safetensors", kind="converted"),
        entry_factory("run/merge/gamma.safetensors", kind="merged"),
    ]
    assert [e["name"] for e in cat.filter_entries(entries, q="BETA")] == ["beta_ar.safetensors"]
    assert [e["name"] for e in cat.filter_entries(entries, kind="merged")] == ["gamma.safetensors"]
    assert [e["name"] for e in cat.filter_entries(entries, prefix="run/convert/")] == \
        ["beta_ar.safetensors"]
    assert cat.filter_entries(entries, q="zzz") == []


def test_rank_label_and_note_read_naturally(entry_factory):
    merged = entry_factory("x/m.safetensors", kind="merged")
    assert cat.rank_label(merged) == "ar:64, nar:64"
    assert cat.note(merged).startswith("α=0.2")
    converted = entry_factory("x/c_ar.safetensors", kind="converted")
    assert cat.note(converted) == "from demo.safetensors"


def test_kinds_counts_and_prefix_options(entry_factory):
    entries = [entry_factory("a/b/x.safetensors"),
               entry_factory("a/c/y_ar.safetensors", kind="converted")]
    assert cat.kinds(entries) == {"converted": 1, "fused": 1}
    assert cat.prefix_options(entries) == ["a/b/", "a/c/"]


# ---------------------------------------------------------------------------
# catalog file
# ---------------------------------------------------------------------------
def test_catalog_roundtrip_is_atomic(tmp_path, entry_factory):
    path = tmp_path / "catalog.json"
    doc = {"count": 1, "entries": [entry_factory("a/b/x.safetensors")], "errors": []}
    cat.save_catalog(path, doc)
    assert cat.load_catalog(path)["count"] == 1
    assert not (tmp_path / "catalog.json.tmp").exists()


def test_missing_or_corrupt_catalog_reads_as_none(tmp_path):
    assert cat.load_catalog(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cat.load_catalog(bad) is None


# ---------------------------------------------------------------------------
# scan (fake transport)
# ---------------------------------------------------------------------------
def test_scan_collects_and_tolerates_one_bad_file(tmp_path, monkeypatch, entry_factory):
    good = entry_factory("run/output/ok.safetensors")
    cfg = Config(bucket="b", root="ROOT", prefixes=("run/",),
                 data_dir=tmp_path, config_path=tmp_path / "c.toml")

    monkeypatch.setattr(cat.gcs, "list_objects", lambda *a, **k: [
        {"name": "ROOT/run/output/ok.safetensors", "size": "10", "crc32c": "X",
         "updated": "t"},
        {"name": "ROOT/run/output/bad.safetensors", "size": "4", "crc32c": "Y",
         "updated": "t"},
    ])

    blob = st_bytes({"ss_base_model_version": "yue2"}, fused_tensors())

    def header_from_gcs(bucket, obj, token):
        if obj.endswith("bad.safetensors"):
            raise ValueError("junk header")
        return cat.parse_header(lambda s, e: blob[s:e + 1])

    monkeypatch.setattr(cat, "header_from_gcs", header_from_gcs)

    doc = cat.scan(cfg, token=object(), log=lambda *_: None)
    assert doc["count"] == 1
    assert doc["entries"][0]["path"] == "run/output/ok.safetensors"
    assert doc["entries"][0]["crc32c"] == "X"
    assert len(doc["errors"]) == 1 and "junk header" in doc["errors"][0]["error"]


def test_scan_records_listing_failures_per_prefix(tmp_path, monkeypatch):
    cfg = Config(bucket="b", root="ROOT", prefixes=("missing/",),
                 data_dir=tmp_path, config_path=tmp_path / "c.toml")

    def boom(*a, **k):
        raise cat.gcs.GcsError("HTTP 404 — bucket not found")

    monkeypatch.setattr(cat.gcs, "list_objects", boom)
    doc = cat.scan(cfg, token=object(), log=lambda *_: None)
    assert doc["count"] == 0
    assert "404" in doc["errors"][0]["error"]


# ---------------------------------------------------------------------------
# transports and defensive branches
# ---------------------------------------------------------------------------
def test_header_that_is_not_an_object_is_rejected():
    blob = struct.pack("<Q", 2) + b"[]"
    with pytest.raises(ValueError, match="not a JSON object"):
        cat.parse_header(lambda s, e: blob[s:e + 1])


def test_header_from_gcs_delegates_to_the_transport(monkeypatch):
    blob = st_bytes({"ss_output_name": "via-gcs"}, fused_tensors())
    monkeypatch.setattr(cat.gcs, "read_range",
                        lambda bucket, obj, s, e, token: blob[s:e + 1])
    meta, tensors = cat.header_from_gcs("b", "o.safetensors", object())
    assert meta["ss_output_name"] == "via-gcs"
    assert len(tensors) == 4


def test_header_from_local_reads_a_file(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(st_bytes({"ss_output_name": "via-file"}, fused_tensors()))
    meta, tensors = cat.header_from_local(path)
    assert meta["ss_output_name"] == "via-file"
    assert len(tensors) == 4


def test_metadata_that_is_not_an_object_is_tolerated():
    assert cat.triggers({"ss_tag_frequency": '["a", "b"]'}) == []
    assert cat.training({"training_info": '["a"]'}) == {}


def test_scan_stops_at_a_cancel_checkpoint(cfg, monkeypatch):
    monkeypatch.setattr(cat.gcs, "list_objects", lambda *a, **k: [])
    with pytest.raises(cat.ScanCancelled):
        cat.scan(cfg, token=object(), cancelled=lambda: True)


def test_scan_cancels_inside_a_prefix_too(cfg, monkeypatch):
    checks = {"n": 0}

    def cancel_on_second_call():
        checks["n"] += 1
        return checks["n"] > 1

    monkeypatch.setattr(cat.gcs, "list_objects", lambda *a, **k: [
        {"name": "ROOT/loras/x.safetensors", "size": "1", "crc32c": "A"}])
    monkeypatch.setattr(cat, "header_from_gcs", lambda *a: ({}, {}))
    with pytest.raises(cat.ScanCancelled):
        cat.scan(cfg, token=object(), cancelled=cancel_on_second_call)


def test_malformed_merge_numbers_fall_back_to_none(entry_factory):
    entry = entry_factory("x/m.safetensors", kind="merged",
                          meta={"merge_alpha": "half", "merge_ar_rank": "lots"})
    assert entry["merge"]["alpha"] is None
    assert entry["merge"]["out_ranks"]["ar"] is None
