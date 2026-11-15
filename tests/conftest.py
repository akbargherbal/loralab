"""Shared fixtures: synthetic safetensors files, built byte-for-byte."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from loralab import catalog as cat
from loralab.config import Config


def st_bytes(meta: dict | None = None, tensors: dict | None = None,
             pad: int = 64) -> bytes:
    """A minimal but byte-accurate safetensors file (header + dummy data)."""
    doc: dict = {}
    if meta is not None:
        doc["__metadata__"] = meta
    doc.update(tensors or {})
    header = json.dumps(doc).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * pad


def tensor(shape, dtype="BF16"):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def fused_tensors(rank=32):
    out = {}
    for prefix in ("text_encoders.layers.0.attn.to_q",
                   "diffusion_model.layers.0.attn.to_q"):
        out[f"{prefix}.lora_A.weight"] = tensor((rank, 8))
        out[f"{prefix}.lora_B.weight"] = tensor((8, rank))
    return out


def converted_tensors(rank=64):
    out = {}
    for kind in ("lora_A", "lora_B"):
        key = f"layers.0.attn.to_q.{kind}.weight"
        out[key] = tensor((rank, 8) if kind == "lora_A" else (8, rank))
    return out


@pytest.fixture
def cfg(tmp_path) -> Config:
    data = tmp_path / "data"
    data.mkdir()
    return Config(bucket="demo-bucket", root="ROOT", prefixes=("loras/",),
                  data_dir=data, config_path=tmp_path / "config.toml")


@pytest.fixture
def entry_factory(tmp_path):
    """Build catalog entries from synthetic headers."""
    def make(path, *, kind="fused", meta=None, tensors=None, size=1024):
        base = {"format": "pt", "ss_base_model_version": "yue2",
                "ss_output_name": "demo"}
        if kind == "merged":
            base.update({"merge_tool": "merge_quran_lora.py",
                         "merge_alpha": "0.2", "merge_ar_rank": "64",
                         "merge_nar_rank": "64"})
            tensors = tensors or fused_tensors(rank=64)
        elif kind == "converted":
            base = {"format": "pt", "converted_from": "ai-toolkit fused YuE2 LoRA",
                    "branch": "ar", "rank": "64", "source_file": "demo.safetensors"}
            tensors = tensors or converted_tensors(rank=64)
        else:
            tensors = tensors or fused_tensors(rank=32)
        base.update(meta or {})
        meta_out, tens = cat.parse_header(
            lambda s, e, b=st_bytes(base, tensors): b[s:e + 1])
        return cat.make_entry(path, meta_out, tens, size=size,
                              crc32c="AAAAAA==", updated="2026-11-15T00:00:00Z")
    return make


@pytest.fixture
def weights_factory(tmp_path):
    """Real (tiny) safetensors files for merge tests. Torch is imported lazily,
    so the catalog-only test suite still runs without it."""
    def make(name, *, rank=4, seed=1, branches=("ar", "nar"), in_dim=8,
             out_dim=6, dtype="float32", meta=None, alpha=None):
        import torch
        from safetensors.torch import save_file

        torch_dtype = getattr(torch, dtype)
        gen = torch.Generator().manual_seed(seed)
        tensors = {}
        for branch in branches:
            prefix = cat.AR_PREFIX if branch == "ar" else cat.NAR_PREFIX
            base = f"{prefix}layers.0.attn.to_q"
            tensors[f"{base}.lora_A.weight"] = torch.randn(
                rank, in_dim, generator=gen).to(torch_dtype)
            tensors[f"{base}.lora_B.weight"] = torch.randn(
                out_dim, rank, generator=gen).to(torch_dtype)
            if alpha is not None:
                tensors[f"{base}.alpha"] = torch.tensor([float(alpha)],
                                                        dtype=torch_dtype)
        path = tmp_path / name
        save_file(tensors, str(path),
                  metadata={"ss_output_name": Path(name).stem, **(meta or {})})
        return path
    return make


def converter_source(*, exit_code=0, write_nar=True, verify=True, sleep=0,
                     stem=None, noise=True) -> str:
    """A stand-in for the real converter: writes a pair, prints the OK line."""
    write_ar = (
        'save_file({k: v for k, v in t.items() if k.startswith("text_encoders")},'
        ' str(out / (stem + "_ar.safetensors")),'
        ' metadata={"converted_from": "x", "branch": "ar", "rank": "4",'
        ' "source_file": src.name})'
    )
    write_nar = (
        'save_file({k: v for k, v in t.items() if k.startswith("diffusion_model")},'
        ' str(out / (stem + "_nar.safetensors")),'
        ' metadata={"converted_from": "x", "branch": "nar", "rank": "4",'
        ' "source_file": src.name})'
    ) if write_nar else ""
    return f'''
import sys, time
from pathlib import Path
from safetensors.torch import load_file, save_file

time.sleep({sleep})
src = Path(sys.argv[1])
out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
stem = {stem!r} or sys.argv[sys.argv.index("--stem") + 1]
t = load_file(str(src))
{write_ar}
{write_nar}
{"print('converter chatter')" if noise else ""}
{"print('byte-level verification against the source: OK')" if verify else ""}
sys.exit({exit_code})
'''


@pytest.fixture
def fake_converter(tmp_path):
    def make(**kwargs):
        path = tmp_path / "convert_fake.py"
        path.write_text(converter_source(**kwargs))
        return path
    return make

