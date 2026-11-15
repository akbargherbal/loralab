"""loralab — a local, single-user workbench for YuE2 LoRA adapters.

Two jobs, one process:
  * catalog  — read every adapter's safetensors **header** over an HTTP Range
               request (no tensor downloads) and show what it is.
  * mix      — carrier + donor at an alpha, merged by rank concatenation,
               converted to the audio.cpp pair, with a manifest.

This package deliberately imports nothing from the project repo it was written
for (`maqamrock-yue2-lora-finetuning`). The conventions it implements are
documented there; the code is its own.
"""

__version__ = "0.1.0"
