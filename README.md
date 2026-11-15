# loralab

A local, single-user workbench for **YuE2 LoRA adapters** backed by a Google Cloud
Storage bucket.

It does two things, and deliberately nothing else:

1. **Catalog** — reads every adapter's *safetensors header only* (the 8-byte length
   prefix plus the JSON that follows) over an HTTP `Range` request. A 133 MB file
   costs ~41 KB of traffic. 122 adapters take ~15 s and a few megabytes, instead of
   downloading ~7 GB.
2. **Mix** *(next)* — carrier + donor at an alpha, merged by rank concatenation,
   converted to the audio.cpp `_ar`/`_nar` pair, with a manifest.

No training. No audio rendering. No GPU.

> Status: **P1 ships the catalog.** The web app runs; you can browse, search, filter
> and inspect provenance and lineage. Mixing is P2 in [`PLAN.md`](PLAN.md).

## Install

```bash
git clone https://github.com/akbargherbal/loralab.git
cd loralab
python3 -m pip install -e ".[gcs]"     # flask + google-auth
```

P1 has **no torch and no safetensors dependency** — it parses headers itself. P2 will
add `torch` + `safetensors`; install torch from the CPU index when it does
(`--index-url https://download.pytorch.org/whl/cpu`) or you will download ~2 GB of CUDA
wheels to run `torch.cat` on a laptop.

## Quickstart

```bash
loralab doctor --bucket my-bucket          # auth, listing, catalog, disk
loralab scan   --bucket my-bucket          # build ~/.local/share/loralab/catalog.json
loralab serve  --bucket my-bucket          # http://127.0.0.1:5000
```

The **bucket is not baked in** — it is the one thing you must supply. Everything after
it (root prefix, the five adapter prefixes, the converter object, the upload prefix)
has a working default.

## Credentials

The app never asks for a token; it walks this ladder and uses the first rung that works:

| rung | how |
|---|---|
| 1 | `GOOGLE_APPLICATION_CREDENTIALS=/path/key.json` (service account) |
| 2 | `gcloud auth application-default login` (ADC) |
| 3 | `gcloud auth print-access-token` (the store gsutil uses) |

Rungs **fall through** on failure — `google.auth.default()` can resolve to compute-engine
metadata credentials on a non-GCP box and then fail, which would otherwise look like
"the app is broken". The winning rung is shown in the header and in `loralab doctor`.

Permissions needed: `storage.objects.list` + `storage.objects.get` on the prefix;
`storage.objects.create` only for uploads (P3). **Do not grant delete** — grant
`roles/storage.objectViewer` plus a custom role containing only `objects.create`, and the
app cannot remove an adapter even if a bug says it should.

## Commands

```bash
loralab serve   [--bucket B] [--root R] [--host H] [--port P] [--data-dir D]
loralab scan    [--bucket B] [--prefix REL ...]
loralab catalog [--json]        # print the cached catalog, no network
loralab doctor                  # what it thinks it has; paste this in a bug report
loralab init    [--bucket B]    # write ~/.config/loralab/config.toml
```

Config precedence: **CLI flag > `LORALAB_*` env > `~/.config/loralab/config.toml` > defaults**.
Unknown keys in the file are an error, not silently ignored.

## What the catalog recovers

`kind` is inferred from the file, never from the path:

| kind | marker | fields |
|---|---|---|
| `fused` | ai-toolkit raw: `text_encoders.*` + `diffusion_model.*` | base model, toolkit version, `ss_output_name`, **`ss_tag_frequency`** (the training tags), step/epoch, per-branch rank |
| `converted` | `__metadata__.converted_from` | `branch`, `rank`, `source_file` |
| `merged` | `__metadata__.merge_tool` | `merge.alpha`, tool, carrier/donor sha256, output ranks |

Plus `size_bytes`, `crc32c`, `md5` (absent on composite objects — GCS only stores md5 for
single-shot uploads), `updated`, and `lineage`.

**A lineage link is never inferred from a filename.** `converted → fused` is linked by
`source_file`; `merged → carrier/donor` only when that sha256 is actually known.
Otherwise the page says *link unknown* — that is deliberate, not a bug.

## Safety

* Binds `127.0.0.1` by default. `--host 0.0.0.0` prints a warning: the app can write to a
  bucket. Prefer a tunnel.
* No auth on the app itself; single user, by design.
* P1 is read-only. Nothing in the UI writes to GCS.
* Jobs live on disk (`<data-dir>/jobs/<id>/`). Kill the server any time; a job left
  `running` becomes `failed — engine restarted` on the next start rather than a zombie.

## Layout

```
loralab/
  config.py     # defaults, precedence, ~/.config + ~/.local/share
  gcs.py        # auth ladder, listing, Range reads
  catalog.py    # header parse → classify → lineage → catalog.json
  jobs.py       # on-disk job state, one FIFO worker, cooperative cancel
  web/          # Flask app, Jinja templates, one small CSS file
  cli.py        # serve | scan | catalog | doctor | init
```

Data lives in `~/.local/share/loralab/` (`catalog.json`, `jobs/`, `cache/`).

## Tests

```bash
python3 -m pytest -q                              # 116 tests, no network, no credentials

python3 -m coverage run -m pytest -q              # coverage gate
python3 -m coverage report                        # fails non-zero below fail_under = 95
```

**99% statement and branch coverage** (805 statements, 4 missed; 6 partial branches).
The floor is enforced twice: `[tool.coverage.report] fail_under = 95` in `pyproject.toml`,
and `.github/workflows/tests.yml` runs the gate on every push.

The four uncovered statements are: the two-line `python -m loralab` shim (exercised by a
subprocess test, not measurable in-process), and the `except OSError` in `jobs.log_text`
(the `is_file()` check above it makes the branch unreachable without patching `open`).

Synthetic safetensors are built byte-for-byte in the tests and go through the same
header parser the GCS path uses. Transport is exercised with a fake HTTP layer
(in-memory ranges, paginated listings, 401 refresh, non-401 re-raise). The web layer is
driven through Flask's test client, including the job state machine and the restart
reconcile.

## Provenance

The merge convention (`A_mix = cat([A_c, A_d])`, `B_mix = cat([B_c, α·B_d])`, alpha folded
into the up-projection), the alpha=0 identity invariant, and the converter's single-rank
constraint are documented in the upstream project
(`maqamrock-yue2-lora-finetuning`): `docs/PRON_LORA_MERGE.md`, `docs/LORA_LAB.md`. This
package re-implements them so it can stand alone; P2 pins the re-implementation to
published artifacts by sha256.

Code MIT. Adapter weights are not distributed here.
