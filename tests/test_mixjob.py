"""The mix pipeline: staging, previews, the alpha grid, manifests, cancel."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from loralab import catalog as cat
from loralab import convert as convertmod
from loralab import gcs, merge, mixjob


class Ctx:
    """A job context with a scripted cancel flag."""

    def __init__(self, directory, params=None, flags=None):
        self.id = "test"
        self.dir = directory
        self.params = params or {}
        self.flags = list(flags or [])
        self.lines = []

    def log(self, message=""):
        self.lines.append(message)

    def cancelled(self):
        return self.flags.pop(0) if self.flags else False


def catalog_doc(*entries):
    return {"schema": 1, "count": len(entries), "entries": list(entries)}


def entry_for(path, rel=None, *, size=None, sha=None, kind=None):
    meta, tensors = cat.header_from_local(path)
    e = cat.make_entry(rel or path.name, meta, tensors,
                       size=size if size is not None else path.stat().st_size,
                       crc32c="AAAAAA==")
    if sha:
        e["sha256"] = sha
    if kind:
        e["kind"] = kind
    return e


# ---------------------------------------------------------------------------
# params and alpha parsing
# ---------------------------------------------------------------------------
def test_parse_alphas_accepts_comma_space_list_and_number():
    assert mixjob.parse_alphas("0.1, 0.2") == [0.1, 0.2]
    assert mixjob.parse_alphas(["0.3", "0.1,0.3"]) == [0.3, 0.1]
    assert mixjob.parse_alphas(0.5) == [0.5]


@pytest.mark.parametrize("value", [None, "", " ", [], "0.1,2"])
def test_parse_alphas_rejects_empty_and_out_of_range(value):
    with pytest.raises(merge.MergeError):
        mixjob.parse_alphas(value)


def test_alpha_tag_and_slug():
    assert mixjob.alpha_tag(0.2) == "a0.2"
    assert mixjob.alpha_tag(1.0) == "a1"
    assert mixjob._slug("dir/qahh_a0.2.safetensors") == "qahh_a0.2"


def test_build_params_omits_an_empty_stem_and_normalises_alphas():
    params = mixjob.build_params("c.safetensors", "d.safetensors", ["0.10", "0.2"])
    assert params == {"carrier": "c.safetensors", "donor": "d.safetensors",
                      "alpha": "0.1,0.2", "convert": True}
    assert mixjob.build_params("c", "d", ["0"], stem="x")["stem"] == "x"


def test_command_and_command_text_match_the_cli():
    params = mixjob.build_params("a b.safetensors", "d.safetensors", ["0.1,0.2"],
                                 convert=False, stem="run")
    assert mixjob.command(params) == [
        "loralab", "mix", "--carrier", "a b.safetensors", "--donor",
        "d.safetensors", "--alpha", "0.1,0.2", "--stem", "run", "--no-convert",
    ]
    assert '"a b.safetensors"' in mixjob.command_text(params)


def test_dedupe_key_is_stable_and_sensitive_to_the_parameters():
    one = mixjob.build_params("c", "d", ["0.1", "0.2"])
    two = mixjob.build_params("c", "d", ["0.2", "0.1"])
    assert mixjob.dedupe_key(one) == mixjob.dedupe_key(two)
    assert mixjob.dedupe_key(one) != mixjob.dedupe_key(
        mixjob.build_params("c", "d", ["0.3"]))
    assert mixjob.dedupe_key(one) != mixjob.dedupe_key(
        mixjob.build_params("c", "d", ["0.1", "0.2"], convert=False))
    assert mixjob.dedupe_key(one).startswith("mix:")


# ---------------------------------------------------------------------------
# inspection and preview
# ---------------------------------------------------------------------------
def test_find_entry_matches_path_then_unique_name_and_rejects_ambiguity():
    doc = catalog_doc({"path": "a/x.safetensors", "name": "x.safetensors"},
                      {"path": "b/x.safetensors", "name": "x.safetensors"})
    assert mixjob.find_entry(doc, "a/x.safetensors")["path"] == "a/x.safetensors"
    with pytest.raises(merge.MergeError, match="ambiguous"):
        mixjob.find_entry(doc, "x.safetensors")
    assert mixjob.find_entry(catalog_doc(), "nope") is None
    assert mixjob.find_entry(None, "nope") is None
    assert mixjob.find_entry(catalog_doc({"path": "p", "name": "n"}), "n")["path"] == "p"


def test_inspect_reads_a_local_header_without_hashing(weights_factory):
    path = weights_factory("carrier.safetensors", rank=6)
    info = mixjob.inspect(str(path), None)
    assert info["source"] == "local"
    assert info["kind"] == "fused"
    assert mixjob._ranks(info) == {"ar": 6, "nar": 6}
    assert info["size_bytes"] == path.stat().st_size
    assert "sha256" not in info


def test_inspect_uses_a_catalog_entry_for_a_remote_path():
    entry = {"name": "x.safetensors", "path": "run/x.safetensors",
             "kind": "merged", "rank": {"ar": [64], "nar": [64]},
             "size_bytes": 10, "sha256": "abc", "crc32c": "c"}
    info = mixjob.inspect("run/x.safetensors", catalog_doc(entry))
    assert info["source"] == "gcs"
    assert info["object"] == "run/x.safetensors"
    assert info["kind"] == "merged"
    assert info["sha256"] == "abc"


def test_inspect_rejects_a_missing_or_garbage_input(tmp_path):
    with pytest.raises(merge.MergeError, match="input not found"):
        mixjob.inspect(str(tmp_path / "nope.safetensors"), None)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"junk")
    with pytest.raises(merge.MergeError, match="not a readable safetensors"):
        mixjob.inspect(str(bad), None)


def test_ranks_normalises_a_header_rank_map():
    assert mixjob._ranks({"rank": {"ar": [8], "nar": [4, 8]}}) == {"ar": 8, "nar": 8}
    assert mixjob._ranks({}) == {}


def test_estimate_scales_with_arms_and_conversion():
    carrier, donor = {"size_bytes": 100_000_000}, {"size_bytes": 50_000_000}
    got = mixjob.estimate([0.1, 0.2], carrier, donor)
    assert got["fused_mb"] == pytest.approx(150.0)
    assert got["converted_pair_mb"] == pytest.approx(180.0)
    assert got["per_arm_mb"] == pytest.approx(330.0)
    assert got["total_mb"] == pytest.approx(660.0)
    assert got["arms"] == 2
    plain = mixjob.estimate([0.1], carrier, donor, convert=False)
    assert plain["converted_pair_mb"] == 0


def test_preview_sums_ranks_and_accepts_two_fused_inputs(cfg):
    carrier = {"name": "c", "kind": "fused", "size_bytes": 100, "rank": {"ar": [32], "nar": [32]}}
    donor = {"name": "d", "kind": "merged", "size_bytes": 100, "rank": {"ar": [32], "nar": [32]}}
    plan = mixjob.preview(carrier, donor, [0.1], cfg, convert=True)
    assert plan["issues"] == []
    assert plan["out_ranks"] == {"ar": 64, "nar": 64}
    assert plan["target_rank"] == 64
    assert plan["donor_branches"] == ["ar", "nar"]
    assert plan["sizes"]["arms"] == 1


def test_preview_rejects_a_converted_input(cfg):
    carrier = {"name": "c", "kind": "converted", "size_bytes": 1, "rank": {"ar": [64]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1, "rank": {"ar": [32]}}
    plan = mixjob.preview(carrier, donor, [0.1], cfg)
    assert any("only fused or merged" in issue for issue in plan["issues"])


def test_preview_warns_about_zero_padding_an_ar_only_donor(cfg):
    carrier = {"name": "c", "kind": "fused", "size_bytes": 1, "rank": {"ar": [32], "nar": [32]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1, "rank": {"ar": [32]}}
    plan = mixjob.preview(carrier, donor, [0.2], cfg)
    assert plan["out_ranks"] == {"ar": 64, "nar": 32}
    assert any("zero-padded" in warning for warning in plan["warnings"])


def test_preview_rejects_a_donor_branch_the_carrier_lacks(cfg):
    carrier = {"name": "c", "kind": "fused", "size_bytes": 1, "rank": {"ar": [32]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1, "rank": {"nar": [32]}}
    plan = mixjob.preview(carrier, donor, [0.2], cfg)
    assert any("branch the carrier does not" in issue for issue in plan["issues"])


def test_preview_rejects_a_carrier_without_ar(cfg):
    carrier = {"name": "c", "kind": "fused", "size_bytes": 1, "rank": {"nar": [32]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1, "rank": {"nar": [32]}}
    plan = mixjob.preview(carrier, donor, [0.2], cfg)
    assert any("no AR" in issue for issue in plan["issues"])


def test_preview_checks_disk_headroom(cfg, monkeypatch):
    monkeypatch.setattr(mixjob, "_free_bytes", lambda _cfg: 1000)
    carrier = {"name": "c", "kind": "fused", "size_bytes": 1_000_000_000,
               "rank": {"ar": [4], "nar": [4]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1_000_000_000,
             "rank": {"ar": [4], "nar": [4]}}
    plan = mixjob.preview(carrier, donor, [0.2], cfg)
    assert any("not enough disk" in issue for issue in plan["issues"])


def test_preview_warns_when_the_converter_is_not_reachable(cfg):
    carrier = {"name": "c", "kind": "fused", "size_bytes": 1, "rank": {"ar": [4], "nar": [4]}}
    donor = {"name": "d", "kind": "fused", "size_bytes": 1, "rank": {"ar": [4], "nar": [4]}}
    no_bucket = type(cfg)(**{**cfg.__dict__, "bucket": ""})
    plan = mixjob.preview(carrier, donor, [0.2], no_bucket, convert=True)
    assert any("converter cannot be downloaded" in w for w in plan["warnings"])
    cached = convertmod.cached_path(no_bucket)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("x")
    assert not mixjob.preview(carrier, donor, [0.2], no_bucket)["warnings"]


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------
def test_resolve_source_hashes_a_local_file(cfg, weights_factory):
    path = weights_factory("local.safetensors")
    got = mixjob.resolve_source(str(path), cfg, None)
    assert got["source"] == "local"
    assert got["sha256"] == merge.sha256_file(path)
    assert got["kind"] == "fused"


def test_resolve_source_downloads_once_and_reuses_the_cache(cfg, weights_factory,
                                                          monkeypatch):
    source = weights_factory("remote.safetensors", rank=8)
    entry = entry_for(source, "run/remote.safetensors")
    doc = catalog_doc(entry)
    calls = []

    def fake_download(bucket, obj, dest, token=None, log=None):
        calls.append(obj)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    first = mixjob.resolve_source("run/remote.safetensors", cfg, doc)
    assert first["sha256"] == merge.sha256_file(source)
    assert first["kind"] == "fused"
    assert first["name"] == "remote.safetensors"      # not the cache-prefixed name
    assert calls == ["ROOT/run/remote.safetensors"]
    second = mixjob.resolve_source("run/remote.safetensors", cfg, doc)
    assert second["sha256"] == first["sha256"]
    assert len(calls) == 1


def test_resolve_source_rejects_a_changed_size(cfg, weights_factory, monkeypatch):
    source = weights_factory("remote.safetensors")
    entry = entry_for(source, "run/remote.safetensors", sha="f" * 64)
    doc = catalog_doc(entry)

    def short_download(bucket, obj, dest, token=None, log=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return {"size": 1, "sha256": "x"}

    monkeypatch.setattr(gcs, "download_object", short_download)
    with pytest.raises(merge.MergeError, match="object changed since the scan"):
        mixjob.resolve_source("run/remote.safetensors", cfg, doc)


def test_resolve_source_redownloads_a_cache_entry_that_does_not_match_the_catalog(
        cfg, weights_factory, monkeypatch):
    source = weights_factory("remote.safetensors")
    entry = entry_for(source, "run/remote.safetensors", sha=merge.sha256_file(source))
    doc = catalog_doc(entry)
    cached = mixjob._cache_dest(cfg, mixjob.inspect("run/remote.safetensors", doc, cfg))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"x" * entry["size_bytes"])
    logs = []
    calls = []

    def fake_download(bucket, obj, dest, token=None, log=None):
        calls.append(obj)
        dest.write_bytes(source.read_bytes())
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    got = mixjob.resolve_source("run/remote.safetensors", cfg, doc, log=logs.append)
    assert calls == ["ROOT/run/remote.safetensors"]
    assert got["sha256"] == merge.sha256_file(source)
    assert any("does not match the catalog sha256" in line for line in logs)


def test_resolve_source_cannot_stage_without_a_bucket(cfg, monkeypatch):
    entry = {"name": "x.safetensors", "path": "run/x.safetensors", "kind": "fused",
             "rank": {"ar": [4]}, "size_bytes": 10}
    doc = catalog_doc(entry)
    no_bucket = type(cfg)(**{**cfg.__dict__, "bucket": ""})
    with pytest.raises(merge.MergeError, match="no bucket configured"):
        mixjob.resolve_source("run/x.safetensors", no_bucket, doc)


def test_resolve_source_reports_a_corrupt_download(cfg, monkeypatch):
    entry = {"name": "x.safetensors", "path": "run/x.safetensors", "kind": "fused",
             "rank": {"ar": [4]}, "size_bytes": 4}
    doc = catalog_doc(entry)

    def junk_download(bucket, obj, dest, token=None, log=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"junk")
        return {"size": 4, "sha256": "x"}

    monkeypatch.setattr(gcs, "download_object", junk_download)
    with pytest.raises(merge.MergeError, match="not a readable safetensors"):
        mixjob.resolve_source("run/x.safetensors", cfg, doc)


def test_cache_destination_is_stable_and_prefixed(cfg):
    source = {"name": "x.safetensors", "crc32c": "abc", "entry_path": "a/x"}
    one = mixjob._cache_dest(cfg, source)
    two = mixjob._cache_dest(cfg, dict(source))
    assert one == two
    assert one.parent == cfg.data_dir / "cache" / "src"
    assert one.name.endswith("-x.safetensors")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def _inputs(weights_factory):
    return (weights_factory("carrier.safetensors", seed=1, rank=4),
            weights_factory("donor.safetensors", seed=2, rank=4))


def test_run_builds_a_grid_and_writes_manifests(cfg, weights_factory,
                                                fake_converter, tmp_path):
    carrier, donor = _inputs(weights_factory)
    params = mixjob.build_params(str(carrier), str(donor), ["0", "0.2"],
                                 convert=True)
    ctx = Ctx(tmp_path / "run", params)
    result = mixjob.run(ctx, cfg, params, converter_path=fake_converter())
    assert result["arms"] == 2
    assert result["alphas"] == [0.0, 0.2]
    assert result["converted"] is True
    manifest = json.loads((tmp_path / "run" / "MANIFEST.json").read_text())
    assert manifest["kind"] == "mix"
    assert manifest["command"][:2] == ["loralab", "mix"]
    assert manifest["environment"]["loralab"] == "0.1.0"
    assert [arm["alpha"] for arm in manifest["arms"]] == [0.0, 0.2]
    # alpha 0 keeps the carrier's basename so the converted pair is comparable
    assert manifest["arms"][0]["fused"]["name"] == "carrier.safetensors"
    assert manifest["arms"][0]["verify"]["kind"] == "identity"
    assert manifest["arms"][1]["fused"]["name"] == (
        "carrier__donor__a0.2.safetensors")
    assert manifest["arms"][1]["verify"]["passed"] is True
    assert manifest["converter"]["pinned"] is False
    assert manifest["arms"][1]["converted"]["ar"]["rank"] == 4
    assert (tmp_path / "run" / "a0.2"
            / "carrier__donor__a0.2.json").is_file()
    assert any("identical to carrier" in line for line in ctx.lines)
    assert any("converted:" in line for line in ctx.lines)


def test_run_without_conversion_produces_only_fused_files(cfg, weights_factory,
                                                          tmp_path):
    carrier, donor = _inputs(weights_factory)
    params = mixjob.build_params(str(carrier), str(donor), ["0.3"], convert=False)
    result = mixjob.run(Ctx(tmp_path / "run", params), cfg, params)
    assert result["converted"] is False
    manifest = json.loads((tmp_path / "run" / "MANIFEST.json").read_text())
    assert manifest["converter"] is None
    assert "converted" not in manifest["arms"][0]


def test_run_requires_both_inputs(cfg, tmp_path):
    with pytest.raises(merge.MergeError, match="needs both"):
        mixjob.run(Ctx(tmp_path), cfg, {"alpha": "0.1"})


def test_run_rejects_a_converted_input(cfg, weights_factory, tmp_path):
    from safetensors.torch import save_file

    carrier = weights_factory("carrier.safetensors")
    donor = tmp_path / "converted.safetensors"
    save_file(
        {"layers.0.lora_A.weight": torch.zeros(4, 8),
         "layers.0.lora_B.weight": torch.zeros(6, 4)},
        str(donor),
        metadata={"converted_from": "x", "branch": "ar", "rank": "4",
                  "source_file": "carrier.safetensors"},
    )
    params = mixjob.build_params(str(carrier), str(donor), ["0.2"], convert=False)
    with pytest.raises(merge.MergeError, match="only fused or merged"):
        mixjob.run(Ctx(tmp_path / "run", params), cfg, params)


def test_run_notes_when_the_catalog_and_the_file_disagree(cfg, weights_factory,
                                                         monkeypatch, tmp_path):
    carrier = weights_factory("carrier.safetensors")
    donor = weights_factory("donor.safetensors")
    doc = catalog_doc(entry_for(carrier, "run/carrier.safetensors"),
                      entry_for(donor, "run/donor.safetensors", kind="converted"))
    cat.save_catalog(cfg.catalog_path, doc)

    def fake_download(bucket, obj, dest, token=None, log=None):
        src = carrier if obj.endswith("carrier.safetensors") else donor
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    params = mixjob.build_params("run/carrier.safetensors",
                                 "run/donor.safetensors", ["0.2"], convert=False)
    ctx = Ctx(tmp_path / "run", params)
    mixjob.run(ctx, cfg, params)
    assert any("note: donor.safetensors is fused" in line for line in ctx.lines)


def test_run_stops_at_a_cancel_before_the_first_arm(cfg, weights_factory, tmp_path):
    carrier, donor = _inputs(weights_factory)
    params = mixjob.build_params(str(carrier), str(donor), ["0.2"], convert=False)
    ctx = Ctx(tmp_path / "run", params, flags=[True])
    with pytest.raises(merge.MergeError, match="no arm was built"):
        mixjob.run(ctx, cfg, params)
    assert any("cancel requested" in line for line in ctx.lines)


def test_run_finishes_the_arm_in_flight_then_stops(cfg, weights_factory, tmp_path):
    carrier, donor = _inputs(weights_factory)
    params = mixjob.build_params(str(carrier), str(donor), ["0.1", "0.2"],
                                 convert=False)
    ctx = Ctx(tmp_path / "run", params, flags=[False, True])
    result = mixjob.run(ctx, cfg, params)
    assert result["alphas"] == [0.1]


def test_run_fails_on_a_structural_problem(cfg, weights_factory, tmp_path):
    carrier = weights_factory("carrier.safetensors", rank=4)
    donor = weights_factory("donor.safetensors", rank=4, in_dim=9)
    params = mixjob.build_params(str(carrier), str(donor), ["0.2"], convert=False)
    with pytest.raises(merge.MergeError, match="dimensions differ"):
        mixjob.run(Ctx(tmp_path / "run", params), cfg, params)


def test_run_stages_a_catalog_input(cfg, weights_factory, fake_converter,
                                    tmp_path, monkeypatch):
    carrier = weights_factory("carrier.safetensors")
    donor = weights_factory("donor.safetensors")
    entry = entry_for(carrier, "run/carrier.safetensors")
    entry2 = entry_for(donor, "run/donor.safetensors")
    doc = catalog_doc(entry, entry2)
    cat.save_catalog(cfg.catalog_path, doc)

    def fake_download(bucket, obj, dest, token=None, log=None):
        src = carrier if obj.endswith("carrier.safetensors") else donor
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return {"size": dest.stat().st_size, "sha256": merge.sha256_file(dest)}

    monkeypatch.setattr(gcs, "download_object", fake_download)
    params = mixjob.build_params("run/carrier.safetensors",
                                 "run/donor.safetensors", ["0.2"], convert=True)
    result = mixjob.run(Ctx(tmp_path / "run", params), cfg, params,
                        converter_path=fake_converter())
    assert result["arms"] == 1
    assert (cfg.data_dir / "cache" / "src").is_dir()
