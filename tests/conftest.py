"""Shared fixtures: synthetic safetensors files, built byte-for-byte."""
from __future__ import annotations

import json
import struct

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
