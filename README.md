# loralab

A local, single-user workbench for **YuE2 LoRA adapters** backed by a Google Cloud
Storage bucket.

It does three things, and deliberately nothing else:

1. **Catalog** — reads every adapter's *safetensors header only* (the 8-byte length
   prefix plus the JSON that follows) over an HTTP `Range` request. A 133 MB file
   costs ~41 KB of traffic. 122 adapters take ~15 s and a few megabytes, instead of
   downloading ~7 GB.
2. **Mix** — carrier + donor at an alpha, merged by rank concatenation, converted to
   the audio.cpp `_ar`/`_nar` pair, with a manifest. The merge keeps the LoRA in
   low-rank form; the dense weight delta is never materialised.
3. **Upload** — a finished mix to GCS, after an exact preview of every object name and
   size. The default destination is `_prototypes/mixes/`; production `loras/` needs an
   explicit flag, existing keys need an explicit overwrite, and there is no delete path.

No training. No audio rendering. No GPU.

> Status: **P1–P4 ship.** Browse, search, filter and inspect provenance and lineage;
> build `carrier × donor` grids from the browser or `loralab mix`; preview and upload a
> mix with guards; edit settings and inventory the cache. The PLAN §8 conformance gate
> passes against the real published pair (both arms, all four converted sha256s).

## Install

```bash
git clone https://github.com/akbargherbal/loralab.git
cd loralab
python3 -m pip install -e ".[gcs,merge]" \
    --extra-index-url https://download.pytorch.org/whl/cpu   # flask + google-auth + torch
```

The catalog does **not** need torch: install `.[gcs]` alone and it runs. Mixing needs
`torch` + `safetensors`, which are the `.[merge]` extra. Install torch from the CPU
index as above or you will download ~2 GB of CUDA wheels to run `torch.cat` on a
laptop. If they are absent, `loralab doctor` and the Doctor page say so, and a mix job
fails with the exact install command instead of a traceback.

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
`storage.objects.create` only for uploads. **Do not grant delete** — grant
`roles/storage.objectViewer` plus a custom role containing only `objects.create`, and the
app cannot remove an adapter even if a bug says it should.

## Commands

```bash
loralab serve   [--bucket B] [--root R] [--host H] [--port P] [--data-dir D]
loralab scan    [--bucket B] [--prefix REL ...]
loralab catalog [--json]        # print the cached catalog, no network
loralab mix     --carrier C --donor D --alpha A [--alpha A ...] [--out DIR]
                [--stem NAME] [--convert | --no-convert] [--converter PATH]
                [--dry-run]
loralab upload  (--job ID | --dir DIR) [--prefix P] [--name N]
                [--include-fused | --no-include-fused] [--overwrite]
                [--allow-production-prefix] [--dry-run] [--yes]
loralab conformance [--manifest PATH] [--json]
loralab doctor                  # what it thinks it has; paste this in a bug report
loralab init    [--bucket B]    # write ~/.config/loralab/config.toml
```

`--carrier` and `--donor` take a catalog path/name or a local `.safetensors` file.
`--alpha` is repeatable or comma-separated. `--dry-run` reads headers only: ranks,
sizes, the equivalent command, and any structural problem before a byte is written.

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

## Mixing

```bash
loralab mix \
  --carrier v3_arabmaqamrock_lora/output/v3_arabmaqamrock_lora.safetensors \
  --donor   arabic_instrumental_lora/output/arabic_instrumental_lora.safetensors \
  --alpha 0.1 --alpha 0.2
```

The merge is **rank concatenation**, per projection, with `A` = down and `B` = up:

```
A_mix = cat([A_carrier,      A_donor], dim=0)     # rank_c + rank_d
B_mix = cat([B_carrier, α ·  B_donor], dim=1)     # α folded into the up-projection
```

* **α is a multiplier in `[0, 1]`**, not a percentage. At α = 0 the donor contributes
  nothing and the output is the carrier tensor for tensor — a free control arm.
  Anything outside the interval is rejected loudly.
* **Inputs must be `fused` or `merged`.** A `converted` audio.cpp file has a different
  layout (`layers.*`, no `.alpha`) and cannot be an input.
* An **AR-only donor** merges only `text_encoders.*`; `diffusion_model.*` is copied
  through and zero-padded to the merged rank, because the converter requires one rank.
  Padding is a numerical no-op.
* Verification is **measured, not asserted**: α = 0 is `torch.equal` over every tensor;
  α > 0 reports the dense delta error (`B_mix@A_mix == B_c@A_c + α·B_d@A_d`) on a
  sample of projections, and the number goes in the manifest. A failed check fails the
  arm.
* **More than two adapters** is a chain: mix A with B, then use the result (kind
  `merged`) as the carrier or donor of the next mix. Each merge adds ranks.

Outputs live under `<data-dir>/mixes/<stem>/a<α>/` for the CLI, or
`<data-dir>/jobs/<id>/a<α>/` for a web job: the fused file, the converted `_ar`/`_nar`
pair, and one sidecar per arm. `MANIFEST.json` records input sha256s, α, ranks, the
measured error, the converted pair's sha256, the exact commands, and the
torch/safetensors versions.

The **converter** is downloaded once from `converter_object` in the bucket and cached in
`<data-dir>/cache/tools/`. Its first sha256 is written next to it as a
trust-on-first-use pin; later runs refuse to execute a changed file. Set
`converter_sha256` in the config to pin a known artifact instead, pass `--converter
PATH` to use a local one, or `--no-convert` to build only the fused file.

**Identity and size.** The converted AR/NAR sha256 is reproducible; the fused file's
whole-file sha256 is not (safetensors writes `__metadata__` in nondeterministic order),
so the UI labels it "this run only". Ranks add: two rank-32 adapters make a rank-64
fused file (~235 MB) and a converted pair (~280 MB).

## Uploading

Upload is a separate, explicit click after a mix finishes. The preview names every
destination object first; `--dry-run` stops there and writes nothing.

```bash
loralab upload --dir <mix-dir> --bucket my-bucket --dry-run
loralab upload --job <mix-job-id> --dry-run
loralab upload --job <mix-job-id> --yes
```

* **Destination.** `<upload_prefix>/<carrier>__<donor>` by default. With an explicit
  `--stem v3instr`, the directory is `v3instr` (arm files are `v3instr_a0.1_ar…`).
  `--prefix` and `--name` override the two parts.
* **Contents.** The converted AR/NAR pair and each arm's JSON sidecar, then
  `MANIFEST.json` **last**, so a reader never sees a manifest for a half-written
  directory. `--include-fused` adds the fused file, whose whole-file sha is run-only.
* **Production guard.** Any destination with a `loras` path segment is refused unless
  `--allow-production-prefix` is explicit. The default stays in `_prototypes/mixes/`.
* **Existing keys.** A create-only precondition (`ifGenerationMatch=0`) makes an
  existing key an HTTP 412. `--overwrite` is required to replace one, and it pins the
  generation the preview saw, so an object that changed in between fails instead of
  being replaced silently.
* **No delete.** The app never sends a DELETE, and the recommended IAM grant has no
  delete permission. Removing an object is a deliberate console/`gsutil` action.
* **Read-back.** After each object the service's `size` and `md5Hash` are checked
  (GCS stores md5/crc32c, not sha256); the local sha256 remains the identity recorded
  in the manifest. The upload record is `upload.json` next to the source manifest (or
  in the upload job directory).

In the browser, a finished mix gets an `Upload to GCS →` link. The page is a GET
preview with the same object table, guards and terminal command; the actual POST also
requires a confirmation checkbox and re-runs every check server-side.

## Safety

* Binds `127.0.0.1` by default. `--host 0.0.0.0` prints a warning: the app can write to a
  bucket and a mix reads and writes local files. Prefer a tunnel.
* No auth on the app itself; single user, by design.
* Catalog and mix write only to the data dir (`catalog.json`, job dirs, `mixes/`,
  `cache/`). Upload is the only bucket write, and it requires a preview plus an explicit
  confirmation.
* Uploads default to `_prototypes/mixes/`. A destination under `loras/` is blocked unless
  `--allow-production-prefix` is explicit; existing keys are blocked unless `--overwrite`
  is explicit; there is no delete path and no delete permission is requested.
* Jobs live on disk (`<data-dir>/jobs/<id>/`). Kill the server any time; a job left
  `running` becomes `failed — engine restarted` on the next start rather than a zombie.

## Layout

```
loralab/
  config.py     # defaults, precedence, ~/.config + ~/.local/share
  gcs.py        # auth ladder, listing, Range reads, streamed downloads, create-only upload
  catalog.py    # header parse → classify → lineage → catalog.json
  merge.py      # rank-concat math, alpha convention, verification
  convert.py    # run the pinned converter (cache + TOFU sha256 pin)
  manifests.py  # MANIFEST.json + per-arm sidecars
  mixjob.py     # staging → merge each α → convert → manifest (CLI + web)
  upload.py     # destination guards, preview, create-only upload, upload.json
  conformance.py# rebuild vs the published converted pair (PLAN §8 gate)
  jobs.py       # on-disk job state, one FIFO worker, cooperative cancel
  web/          # Flask app, Jinja templates, one CSS file, bundled fonts (see DESIGN.md)
  cli.py        # serve | scan | catalog | mix | upload | conformance | doctor | init
```

Data lives in `~/.local/share/loralab/` (`catalog.json`, `jobs/`, `mixes/`, `cache/`).

## Tests

```bash
python3 -m pytest -q                              # 378 tests, no network, no credentials
python3 -m coverage run -m pytest -q              # coverage gate
python3 -m coverage report                        # fails non-zero below fail_under = 95
```

**98% statement and branch coverage** (2203 statements, 31 missed). The floor is
enforced twice: `[tool.coverage.report] fail_under = 95` in `pyproject.toml`, and
`.github/workflows/tests.yml` runs the gate on every push. CI installs the merge extra
from the CPU torch index; tests that need torch skip if it is absent, but the gate
expects it.

The uncovered lines are mostly defensive: the `python -m loralab` shim (exercised by a
subprocess test, not measurable in-process), `except OSError` paths whose `is_file()`
guard makes them hard to reach, and a few branch-only edges in the upload planner.

Synthetic safetensors are built byte-for-byte in the tests and go through the same
header parser the GCS path uses. The merge tests use real tiny tensors (rank 4) and
assert the arithmetic, the α = 0 identity, padding, and the measured δ; a fake converter
writes the `_ar`/`_nar` pair so the pipeline is exercised without the real tool.
Transport is exercised with a fake HTTP layer (in-memory ranges, paginated listings,
401 refresh, streaming downloads, create-only uploads, 412/400 bodies). The web layer is
driven through Flask's test client, including mixer 409 gates, upload preview/confirm
gates, settings validation, artifact allow-lists, job state machine and restart
reconcile. `tests/test_conformance.py` pins all four published converted sha256s, both
manifest shapes, and the float32-alpha negative case.

The real §8 gate is a one-time check, not a CI test (it needs the two 117 MB inputs):

```bash
loralab mix --carrier ...v3_arabmaqamrock_lora.safetensors \
            --donor ...arabic_instrumental_lora.safetensors \
            --alpha 0.1 --alpha 0.2 --stem v3instr --converter .../convert_aitoolkit_yue2_lora.py
loralab conformance --manifest <out>/MANIFEST.json
```

## Provenance

The merge convention (`A_mix = cat([A_c, A_d])`, `B_mix = cat([B_c, α·B_d])`, alpha folded
into the up-projection), the alpha=0 identity invariant, and the converter's single-rank
constraint are documented in the upstream project
(`maqamrock-yue2-lora-finetuning`): `docs/PRON_LORA_MERGE.md`, `docs/LORA_LAB.md`. This
package re-implements them so it can stand alone; `tests/test_merge.py` pins the
arithmetic, the α=0 identity and the padding rule. The published-artifact gate in
`PLAN.md` §8 was run on 2026-11-15 against the real GCS pair: rebuilding
`v3 × arabic_instrumental` at α 0.1 and 0.2 reproduced all four published converted
sha256s exactly (`loralab conformance` now encodes the expected values). The embedded
reference hashes and both manifest shapes are also pinned by `tests/test_conformance.py`.

Code MIT. Adapter weights are not distributed here.
