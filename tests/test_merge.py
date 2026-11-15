"""The merge itself: rank concatenation, the alpha convention, verification."""
from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from loralab import catalog as cat
from loralab import merge


def make(rank=4, seed=1, branches=("ar", "nar"), dtype=torch.float32,
         in_dim=8, out_dim=6, alpha=None, weight_suffix=True, bases=None):
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for branch in branches:
        prefix = cat.AR_PREFIX if branch == "ar" else cat.NAR_PREFIX
        base = f"{prefix}layers.0.attn.to_q"
        suffix = ".weight" if weight_suffix else ""
        tensors[f"{base}.lora_A{suffix}"] = torch.randn(
            rank, in_dim, generator=gen).to(dtype)
        tensors[f"{base}.lora_B{suffix}"] = torch.randn(
            out_dim, rank, generator=gen).to(dtype)
        if alpha is not None:
            tensors[f"{base}.alpha"] = torch.tensor([float(alpha)], dtype=dtype)
    for extra in bases or []:
        tensors[extra] = torch.zeros(1, 1)
    return tensors


def base_of(tensors):
    return next(merge._proj(k) for k in tensors if merge._ab(k) == "A")


# ---------------------------------------------------------------------------
# alpha validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, "0.25"])
def test_alpha_accepts_the_closed_unit_interval(value):
    assert merge.check_alpha(value) == pytest.approx(float(value))


@pytest.mark.parametrize("value", [-0.01, 1.01, 30, float("nan"), float("inf"),
                                   "abc", None])
def test_alpha_outside_the_interval_is_rejected(value):
    with pytest.raises(merge.MergeError):
        merge.check_alpha(value)


def test_alpha_message_says_it_is_not_a_percentage():
    with pytest.raises(merge.MergeError, match="not a percentage"):
        merge.check_alpha(30)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_empty_inputs_are_rejected():
    with pytest.raises(merge.MergeError, match="carrier contains no tensors"):
        merge.validate({}, make())
    with pytest.raises(merge.MergeError, match="donor contains no tensors"):
        merge.validate(make(), {})


def test_carrier_unknown_prefix_is_rejected():
    with pytest.raises(merge.MergeError, match="neither text_encoders"):
        merge.validate(make(bases=["other.x"]), make())


def test_donor_unknown_prefix_is_rejected():
    with pytest.raises(merge.MergeError, match="donor has unexpected keys"):
        merge.validate(make(), make(bases=["other.x"]))


def test_donor_must_not_have_keys_the_carrier_lacks():
    donor = make()
    donor["text_encoders.layers.9.extra.lora_A.weight"] = torch.zeros(2, 8)
    donor["text_encoders.layers.9.extra.lora_B.weight"] = torch.zeros(6, 2)
    with pytest.raises(merge.MergeError, match="carrier lacks"):
        merge.validate(make(), donor)


def test_donor_missing_a_branch_key_is_a_partial_branch():
    donor = make()
    key = next(k for k in donor if k.endswith("lora_B.weight"))
    del donor[key]
    with pytest.raises(merge.MergeError, match="missing 1 ar key"):
        merge.validate(make(), donor)


def test_donor_with_only_alpha_keys_shares_nothing():
    with pytest.raises(merge.MergeError, match="shares no keys"):
        merge.validate(make(), {"text_encoders.x.alpha": torch.zeros(1)})


def test_carrier_without_an_ar_branch_is_rejected():
    with pytest.raises(merge.MergeError, match="no text_encoders"):
        merge.validate(make(branches=("nar",)), make(branches=("nar",)))


def test_carrier_without_ab_pairs_is_rejected():
    carrier = {"text_encoders.x.foo": torch.zeros(1, 1)}
    with pytest.raises(merge.MergeError, match="no lora_A/lora_B pairs"):
        merge.validate(carrier, carrier)


def test_non_2d_a_tensor_is_rejected():
    carrier = make()
    base = base_of(carrier)
    carrier[f"{base}.lora_A.weight"] = torch.zeros(4)
    with pytest.raises(merge.MergeError, match="not rank-2"):
        merge.validate(carrier, make())


def test_a_without_b_is_rejected():
    carrier, donor = make(), make()
    base = base_of(carrier)
    del carrier[f"{base}.lora_B.weight"]
    del donor[f"{base}.lora_B.weight"]
    with pytest.raises(merge.MergeError, match="has no lora_B partner"):
        merge.validate(carrier, donor)


def test_a_b_rank_disagreement_is_rejected():
    carrier = make()
    base = base_of(carrier)
    carrier[f"{base}.lora_B.weight"] = torch.zeros(6, 99)
    with pytest.raises(merge.MergeError, match="rank mismatch"):
        merge.validate(carrier, make())


def test_donor_input_dimension_mismatch_is_rejected():
    with pytest.raises(merge.MergeError, match="input/output dimensions differ"):
        merge.validate(make(in_dim=8), make(in_dim=9))


def test_donor_output_dimension_mismatch_is_rejected():
    with pytest.raises(merge.MergeError, match="input/output dimensions differ"):
        merge.validate(make(out_dim=6), make(out_dim=7))


def test_mixed_ranks_inside_a_branch_are_rejected():
    carrier = make()
    carrier["text_encoders.layers.1.attn.to_k.lora_A.weight"] = torch.zeros(2, 8)
    carrier["text_encoders.layers.1.attn.to_k.lora_B.weight"] = torch.zeros(6, 2)
    donor = make()
    donor["text_encoders.layers.1.attn.to_k.lora_A.weight"] = torch.zeros(2, 8)
    donor["text_encoders.layers.1.attn.to_k.lora_B.weight"] = torch.zeros(6, 2)
    with pytest.raises(merge.MergeError, match="mixed ranks"):
        merge.validate(carrier, donor)


def test_a_stray_donor_alpha_that_is_not_the_rank_is_rejected():
    with pytest.raises(merge.MergeError, match="does not equal rank"):
        merge.validate(make(alpha=8), make())


def test_a_donor_alpha_equal_to_the_rank_is_accepted_and_not_merged():
    merged, stats = merge.merge(make(alpha=4), make(alpha=4), 0.5)
    assert not any(key.endswith(".alpha") for key in merged)
    assert stats["target_rank"] == 8


def test_an_empty_alpha_tensor_is_rejected():
    donor = make()
    base = base_of(donor)
    donor[f"{base}.alpha"] = torch.zeros(0)
    with pytest.raises(merge.MergeError, match="does not equal rank"):
        merge.validate(make(), donor)


def test_a_key_that_is_neither_a_nor_b_is_rejected_at_merge_time():
    carrier = make(bases=["text_encoders.weird"])
    donor = make(bases=["text_encoders.weird"])
    with pytest.raises(merge.MergeError, match="unexpected key"):
        merge.merge(carrier, donor, 0.5)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------
def test_rank_concatenation_and_alpha_in_b():
    carrier, donor = make(rank=4, seed=1), make(rank=4, seed=2)
    merged, stats = merge.merge(carrier, donor, 0.25)
    base = base_of(carrier)
    ca = carrier[f"{base}.lora_A.weight"]
    da = donor[f"{base}.lora_A.weight"]
    cb = carrier[f"{base}.lora_B.weight"]
    db = donor[f"{base}.lora_B.weight"]
    assert torch.equal(merged[f"{base}.lora_A.weight"], torch.cat([ca, da]))
    expected = torch.cat([cb, (db.float() * 0.25).to(cb.dtype)], dim=1)
    assert torch.allclose(merged[f"{base}.lora_B.weight"], expected)
    assert merged[f"{base}.lora_A.weight"].shape == (8, 8)
    assert merged[f"{base}.lora_B.weight"].shape == (6, 8)
    assert stats["out_ranks"] == {"ar": 8, "nar": 8}
    assert stats["merged_tensors"] == {"ar": 2, "nar": 2}
    assert stats["copied_tensors"] == {"ar": 0, "nar": 0}
    assert stats["donor_kept"] is True


def test_alpha_zero_is_the_carrier_tensor_for_tensor():
    carrier, donor = make(rank=4), make(rank=8)
    merged, stats = merge.merge(carrier, donor, 0)
    assert stats["out_ranks"] == {"ar": 4, "nar": 4}
    assert stats["donor_kept"] is False
    assert merged.keys() == carrier.keys()
    for key in carrier:
        assert torch.equal(merged[key], carrier[key])
        assert merged[key].data_ptr() != carrier[key].data_ptr()


def test_alpha_zero_does_not_pad_an_ar_only_donor():
    carrier = make(rank=4, branches=("ar",))
    nar = make(rank=6, branches=("nar",))
    carrier.update(nar)
    donor = make(rank=2, branches=("ar",))
    merged, stats = merge.merge(carrier, donor, 0)
    assert stats["out_ranks"] == {"ar": 4, "nar": 6}
    assert stats["padded_tensors"] == {"ar": 0, "nar": 0}
    for key in carrier:
        assert torch.equal(merged[key], carrier[key])


def test_ar_only_donor_copies_nar_and_zero_pads_it():
    carrier = make(rank=4, branches=("ar", "nar"))
    donor = make(rank=2, branches=("ar",))
    merged, stats = merge.merge(carrier, donor, 0.3)
    assert stats["out_ranks"] == {"ar": 6, "nar": 6}
    assert stats["padded_tensors"] == {"ar": 0, "nar": 1}
    assert stats["copied_tensors"] == {"ar": 0, "nar": 2}
    base = "diffusion_model.layers.0.attn.to_q"
    a = merged[f"{base}.lora_A.weight"]
    b = merged[f"{base}.lora_B.weight"]
    assert a.shape == (6, 8) and b.shape == (6, 6)
    assert torch.equal(a[:4], carrier[f"{base}.lora_A.weight"])
    assert torch.equal(a[4:], torch.zeros(2, 8))
    assert torch.equal(b[:, 4:], torch.zeros(6, 2))
    # The NAR delta is unchanged by zero padding.
    ref = (carrier[f"{base}.lora_B.weight"].float()
           @ carrier[f"{base}.lora_A.weight"].float())
    assert torch.allclose(b.float() @ a.float(), ref)


def test_nar_only_donor_is_symmetric():
    carrier = make(rank=4, branches=("ar", "nar"))
    donor = make(rank=2, branches=("nar",))
    merged, stats = merge.merge(carrier, donor, 0.5)
    assert stats["out_ranks"] == {"ar": 6, "nar": 6}
    base = "text_encoders.layers.0.attn.to_q"
    assert merged[f"{base}.lora_A.weight"].shape == (6, 8)
    assert torch.equal(merged[f"{base}.lora_A.weight"][4:], torch.zeros(2, 8))


def test_padding_can_be_switched_off():
    carrier = make(rank=4, branches=("ar", "nar"))
    donor = make(rank=2, branches=("ar",))
    _, stats = merge.merge(carrier, donor, 0.3, pad=False)
    assert stats["out_ranks"] == {"ar": 6, "nar": 4}
    assert stats["padded_tensors"] == {"ar": 0, "nar": 0}


def test_keys_without_the_weight_suffix_are_supported_and_padded():
    carrier = make(rank=4, branches=("ar", "nar"), weight_suffix=False)
    donor = make(rank=2, branches=("ar",), weight_suffix=False)
    merged, stats = merge.merge(carrier, donor, 0.5)
    assert stats["out_ranks"]["nar"] == 6
    assert "diffusion_model.layers.0.attn.to_q.lora_A" in merged
    assert "diffusion_model.layers.0.attn.to_q.lora_B" in merged
    assert merged["diffusion_model.layers.0.attn.to_q.lora_B"].shape[1] == 6


def test_carrier_dtype_is_preserved():
    carrier, donor = make(dtype=torch.float32), make(dtype=torch.float32)
    donor = {k: v.to(torch.bfloat16) for k, v in donor.items()}
    merged, _ = merge.merge(carrier, donor, 0.5)
    assert {str(t.dtype) for t in merged.values()} == {"torch.float32"}


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def test_identity_verification_passes_and_reports_every_tensor():
    carrier, donor = make(), make(rank=8)
    merged, _ = merge.merge(carrier, donor, 0)
    report = merge.verify(carrier, donor, merged, 0)
    assert report["kind"] == "identity"
    assert report["passed"] is True
    assert report["tensors_checked"] == len(carrier)
    assert report["max_abs_error"] == 0.0


def test_identity_verification_catches_a_changed_value():
    carrier, donor = make(), make()
    merged, _ = merge.merge(carrier, donor, 0)
    key = base_of(carrier) + ".lora_A.weight"
    merged[key] = merged[key] + 1
    report = merge.verify(carrier, donor, merged, 0)
    assert report["passed"] is False
    assert report["mismatched"] == [key]


def test_identity_verification_catches_a_changed_shape():
    carrier, donor = make(), make()
    merged, _ = merge.merge(carrier, donor, 0)
    key = base_of(carrier) + ".lora_B.weight"
    merged[key] = torch.zeros(1, 1)
    assert merge.verify(carrier, donor, merged, 0)["passed"] is False


def test_delta_verification_measures_the_agreement():
    carrier, donor = make(), make(seed=3)
    merged, _ = merge.merge(carrier, donor, 0.5)
    report = merge.verify(carrier, donor, merged, 0.5)
    assert report["kind"] == "delta"
    assert report["passed"] is True
    assert report["projections_checked"] == 2
    assert report["max_rel_error"] < 1e-4
    assert report["tolerance"] == merge.DELTA_TOLERANCE


def test_delta_verification_accepts_a_donor_with_a_missing_branch():
    carrier = make(rank=4, branches=("ar", "nar"))
    donor = make(rank=2, branches=("ar",))
    merged, _ = merge.merge(carrier, donor, 0.4)
    report = merge.verify(carrier, donor, merged, 0.4)
    assert report["passed"] is True
    assert report["projections_checked"] == 2


def test_delta_verification_catches_a_wrong_scale():
    carrier, donor = make(), make(seed=4)
    merged, _ = merge.merge(carrier, donor, 0.5)
    key = base_of(carrier) + ".lora_B.weight"
    merged[key] = merged[key] * 3
    report = merge.verify(carrier, donor, merged, 0.5)
    assert report["passed"] is False
    assert report["mismatched"]


def test_verification_rejects_a_changed_key_set():
    carrier, donor = make(), make()
    merged, _ = merge.merge(carrier, donor, 0.5)
    merged.pop(next(iter(merged)))
    with pytest.raises(merge.MergeError, match="key set differs"):
        merge.verify(carrier, donor, merged, 0.5)


def test_verification_sampling_and_dtype_report():
    carrier, donor = make(), make(seed=2)
    merged, _ = merge.merge(carrier, donor, 0.5)
    assert merge.verify(carrier, donor, merged, 0.5, sample=1)[
        "projections_checked"] == 1
    assert merge.verify(carrier, donor, merged, 0.5, sample=99)[
        "projections_checked"] == 2
    merged[next(iter(merged))] = merged[next(iter(merged))].double()
    assert merge.verify(carrier, donor, merged, 0.5)["dtype_preserved"] is False


def test_verification_samples_across_many_projections():
    carrier = make()
    donor = make(seed=2)
    for i in range(2, 6):
        for branch, prefix in (("ar", cat.AR_PREFIX), ("nar", cat.NAR_PREFIX)):
            base = f"{prefix}layers.{i}.attn.to_q"
            carrier[f"{base}.lora_A.weight"] = torch.randn(4, 8)
            carrier[f"{base}.lora_B.weight"] = torch.randn(6, 4)
            donor[f"{base}.lora_A.weight"] = torch.randn(4, 8)
            donor[f"{base}.lora_B.weight"] = torch.randn(6, 4)
    merged, _ = merge.merge(carrier, donor, 0.5)
    report = merge.verify(carrier, donor, merged, 0.5, sample=3)
    assert report["projections_checked"] == 3
    assert report["passed"] is True


# ---------------------------------------------------------------------------
# files, metadata, the environment
# ---------------------------------------------------------------------------
def test_merge_files_writes_provenance_the_catalog_reads(weights_factory):
    carrier = weights_factory("carrier.safetensors", seed=1)
    donor = weights_factory("donor.safetensors", seed=2)
    out = carrier.parent / "out" / "mix.safetensors"
    result = merge.merge_files(carrier, donor, 0.2, out)
    assert out.is_file()
    assert result["verify"]["passed"] is True
    assert result["fused"]["ranks"] == {"ar": 8, "nar": 8}
    meta, tensors = cat.header_from_local(out)
    entry = cat.make_entry("x/mix.safetensors", meta, tensors)
    assert entry["kind"] == "merged"
    assert entry["merge"]["alpha"] == pytest.approx(0.2)
    assert entry["merge"]["tool"] == "loralab-merge"
    assert entry["merge"]["carrier_name"] == "carrier.safetensors"
    assert entry["merge"]["donor_name"] == "donor.safetensors"
    assert entry["merge"]["out_ranks"] == {"ar": 8, "nar": 8}
    assert entry["merge"]["carrier_sha256"] == merge.sha256_file(carrier)
    assert entry["merge"]["donor_sha256"] == merge.sha256_file(donor)


def test_merge_files_alpha_zero_matches_the_carrier_bytes(weights_factory):
    carrier = weights_factory("carrier.safetensors", seed=1)
    donor = weights_factory("donor.safetensors", seed=2)
    out = carrier.parent / "a0.safetensors"
    result = merge.merge_files(carrier, donor, 0.0, out)
    assert result["verify"]["kind"] == "identity"
    from safetensors.torch import load_file
    written, original = load_file(str(out)), load_file(str(carrier))
    for key in original:
        assert torch.equal(written[key], original[key])


def test_merge_files_hashes_inputs_once_and_accepts_extra_metadata(weights_factory,
                                                                  monkeypatch):
    carrier = weights_factory("carrier.safetensors")
    donor = weights_factory("donor.safetensors")
    calls = []
    real = merge.sha256_file
    monkeypatch.setattr(merge, "sha256_file",
                        lambda path: (calls.append(path), real(path))[1])
    out = carrier.parent / "x.safetensors"
    result = merge.merge_files(carrier, donor, 0.1, out,
                               carrier_sha="c" * 64, donor_sha="d" * 64,
                               extra_metadata={"run": "test", "none": None})
    # only the output is hashed here
    assert calls == [out]
    meta, _ = cat.header_from_local(out)
    assert meta["run"] == "test"
    assert "none" not in meta
    assert result["fused"]["sha256"] == real(out)


def test_merge_files_fails_on_a_failed_verification(weights_factory, monkeypatch):
    carrier = weights_factory("carrier.safetensors")
    donor = weights_factory("donor.safetensors")
    monkeypatch.setattr(merge, "verify", lambda *a, **k: {
        "passed": False, "max_rel_error": 1.0, "tolerance": 0.005})
    with pytest.raises(merge.MergeError, match="verification failed"):
        merge.merge_files(carrier, donor, 0.5, carrier.parent / "out.safetensors")


def test_load_tensors_reports_missing_and_garbage_files(tmp_path):
    with pytest.raises(merge.MergeError, match="input not found"):
        merge.load_tensors(tmp_path / "nope.safetensors")
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(merge.MergeError, match="not a readable safetensors"):
        merge.load_tensors(bad)


def test_save_tensors_reports_an_unwritable_target(tmp_path):
    with pytest.raises(merge.MergeError, match="cannot write"):
        merge.save_tensors({}, tmp_path)   # a directory


def test_environment_says_how_to_install_when_torch_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(merge.MergeError, match=r"\.\[merge\]"):
        merge.environment()


def test_sha256_file_is_the_content_hash(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert merge.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_read_metadata_round_trips(weights_factory):
    path = weights_factory("meta.safetensors", meta={"ss_output_name": "demo"})
    assert merge.read_metadata(path)["ss_output_name"] == "demo"


def test_build_metadata_handles_a_donor_with_no_branches():
    meta = merge.build_metadata({}, alpha=0.0, carrier_name="c", carrier_sha="c",
                                donor_name="d", donor_sha="d",
                                stats={"donor_branches": [],
                                       "out_ranks": {"ar": 4}})
    assert meta["merge_kind"] == "rank_concat_none"
    assert meta["merge_nar_rank"] == "None"


def test_stem_ab_and_proj_cover_both_key_shapes():
    assert merge._ab("x.lora_A") == "A"
    assert merge._ab("x.lora_B.weight") == "B"
    assert merge._ab("x.other") is None
    assert merge._proj("x.lora_A.weight") == "x"
    assert merge._proj("x.alpha") == "x"
    assert merge._proj("x.other") == "x.other"
