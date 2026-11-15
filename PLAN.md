# LoRA lab UI — standalone Flask app plan

_**Plan only. Nothing here is built.** Revised 2026-11-15 (third pass): the app is its own
repo, not a view inside this one. It never imports this repo, never reads `/content`, and
never assumes Colab. This file is the design record — when the app repo is created, copy it
there as `PLAN.md` and keep this copy as the pointer._

Authorities that **stay** in this repo even though the code won't: `docs/PRON_LORA_MERGE.md`
(the scaling convention), `docs/LORA_LAB.md` (catalog semantics), `docs/LORA_INVENTORY.md`
(what exists). The app re-derives facts from files, and cites those docs; it does not read them
at runtime.


> **Status (2026-11-15):** P1 is built and shipped — `loralab serve` shows the catalog,
> adapter detail, provenance, lineage, jobs and a doctor page. 71 tests, no network, no
> credentials. P2 (mix) is next and carries the conformance gate described in §8.
> This file is the design record; keep it in sync with the code or delete it.

---

## 0. The one change that drives the rest

"Standalone" is not packaging. It moves three things:

| dependency the app used to lean on | today | standalone answer |
|---|---|---|
| `lora_toolkit.py` (header scan, classify, lineage, table) | import from this repo | **re-implement** (~180 lines) behind the same JSON shape; the spec is `docs/LORA_LAB.md` |
| `merge_quran_lora.py` / `merge_pron_lora.py` (rank-concat math) | import | **re-implement** (~80 lines); the math is 3 lines per projection — the *convention* is the hard part, and it is pinned by tests |
| `convert_aitoolkit_yue2_lora.py` | staged by `bootstrap/setup.sh` | **download once, pin the sha256**, cache it, refuse to run on a mismatch |
| `/content/...` paths, `gsutil`, `gcloud` login state | assumed | **no absolute paths**; auth is an explicit, probed ladder (§1) |

The app is therefore: a **single-user local tool** you `pip install`, point at a bucket, and
open at `http://127.0.0.1:5000`. It builds adapters. It does not train, render, or rate.

The risk this creates is **math drift**, and it is solved by one gate, not by discipline: the
app must reproduce published artifacts byte-for-byte (§8). If it can't, it isn't ready.

## 1. Prerequisites — what "I have credentials" has to mean

**GCS permissions, on the target bucket prefix:**

| operation | permission | needed for |
|---|---|---|
| list + get (incl. HTTP `Range`) | `storage.objects.list`, `storage.objects.get` | catalog, source staging |
| create | `storage.objects.create` | explicit upload only |
| delete | — | **never requested** |

Grant `roles/storage.objectViewer` plus a custom role containing only `storage.objects.create`.
Then the app *cannot* delete an adapter even if a bug says it should. That is cheaper than a
"no delete button" policy and it survives future maintainers.

**Auth ladder** (probe in order, report which one won in the header):

1. `GOOGLE_APPLICATION_CREDENTIALS` → service-account JSON via `google-auth` — the mode a
   "I have credentials" user actually wants.
2. Application Default Credentials (`gcloud auth application-default login`).
3. `gcloud auth print-access-token` — last resort, same store gsutil uses.

Each step must **fall through on failure**, not abort. (`google.auth.default()` can resolve to
compute-engine metadata creds that then fail with a 404 on a non-GCP box — that is exactly why
the repo tool shells out to `gcloud`; `docs/LORA_LAB.md`, "Auth".) A wrong guess here looks like
"the app is broken".

**`loralab doctor`** — one command, run before anything: prints the auth mode in use, a list+get
probe on one object, the converter sha256 vs the pin, `torch`/`safetensors` versions, free disk,
and the resolved data dir. If the app misbehaves, this is the first thing to paste.

**Config** (no hardcoded project): `~/.config/loralab/config.toml`, overridable by `LORALAB_*` env
and by the Settings page.

```toml
bucket  = "akbar-december-2024-backup"
root    = "OSTRIS_Arabic_Suno_Finetuning"     # prefix inside the bucket; "" is valid
prefixes = ["loras/", "v3_arabmaqamrock_lora/", "arabic_instrumental_lora/", "quran_ahh_r32/"]
converter = { object = "audiocpp_inference/converter/convert_aitoolkit_yue2_lora.py",
              sha256 = "<pinned>" }
upload_prefix = "_prototypes/mixes/"           # default destination; never loras/
```

## 2. What the app owns

Four small modules plus a web layer. Keep the **JSON shapes identical** to the ones already in
the bucket so a manifest from either side is interchangeable.

| module | job | size | spec |
|---|---|---|---|
| `gcs.py` | auth ladder, paginated listing, `Range` reads, upload | ~150 | `docs/LORA_LAB.md` "Why a header scan" |
| `catalog.py` | header parse → `kind/branches/rank/tensors/dtype`, provenance, lineage, table | ~180 | `docs/LORA_LAB.md` "What it recovers" |
| `merge.py` | rank-concat at α, metadata, stats | ~80 | `docs/PRON_LORA_MERGE.md` (scaling) |
| `convert.py` | run the pinned converter, parse its `byte-level verification … OK` | ~60 | `docs/PRON_LORA_MERGE.md` "One non-obvious constraint" |
| `manifests.py` | write `MANIFEST.json` + per-arm sidecars | ~80 | `results/pron_production_merge/merge_manifest.json` |

The merge itself, per projection (`A` = down, `B` = up):

```
A_mix = cat([A_carrier, A_donor],       dim=0)   # rank_c + rank_d
B_mix = cat([B_carrier, α · B_donor],   dim=1)
```

α is folded into `B` because that is the side ai-toolkit scales (`up(down(x)) * scale`). At
**α = 0 the carrier is copied verbatim** — that arm is free and is the control. α scales a
multiplier, so `α = 30` is not "30 %": reject anything outside `[0, 1]` loudly.

## 3. Layout, packaging, deployment

```
loralab/
  __init__.py  config.py  gcs.py  catalog.py  merge.py  convert.py  manifests.py
  jobs.py            # on-disk job state, single FIFO worker, startup reconcile
  web/app.py  web/templates/  web/static/
  cli.py             # loralab serve | doctor | scan | mix | upload
pyproject.toml       # console script: loralab
tests/
```

* **Deps:** `flask`, `safetensors`, `torch` (CPU only), optional `google-auth`. Install torch from
  the CPU index (`--index-url https://download.pytorch.org/whl/cpu`) or you download ~2 GB of
  CUDA wheels to run `torch.cat` on a laptop. State this in the README, loudly.
* **Data dir:** `~/.local/share/loralab/` — `catalog.json`, `jobs/<id>/`, `cache/src/` (sources,
  keyed by `crc32c`+size), `cache/tools/` (the pinned converter).
* **Server:** `loralab serve` → `127.0.0.1:5000`, `use_reloader=False`, no auth, **single user
  by design**. `--host 0.0.0.0` exists but prints a warning: the app can write to a bucket.
* **Docker (optional):** `-p 127.0.0.1:5000:5000 -v ~/.config/gcloud:/root/.config/gcloud`. The
  image ships no credentials.
* **No repo imports.** A test asserts `import loralab` works with this repo absent from
  `sys.path` — a one-line guard against the dependency creeping back.

## 4. Screens → routes

Server-rendered Jinja + ~150 lines of vanilla JS for polling. No build step.

| route | screen | notes |
|---|---|---|
| `GET /` → `/catalog` | catalog | grouped table, filter by kind/prefix/rank + search; shows `generated_utc` age and the auth mode |
| `GET /catalog/<id>` | adapter detail | full record, lineage (`resolved: false` shown as "link unknown"), copy path, **use as carrier / donor** |
| `GET /mix/new?carrier=&donor=` | mixer | α chips + free entry; live preview of rank sum, size, disk headroom, compatibility |
| `POST /mix` | → `202 {job}` | 409 with the reason if preflight fails |
| `GET /jobs/<id>` | job page | per-arm state, log tail, artifacts + sha256, **the CLI command it ran**, Stop |
| `GET /api/jobs/<id>?log_offset=N` | poll | 1 s while running, stop on terminal |
| `POST /jobs/<id>/cancel` | cancel | flag file; render-free, so cancel is always safe |
| `GET /mix/<id>/manifest.json` | download | the manifest, exactly as uploaded |
| `POST /mix/<id>/upload` | upload | dry-run preview first: destination object names + sizes; refuses `loras/` unless `--allow-production-prefix` |
| `GET /settings` | settings | bucket/root/prefixes, upload prefix, converter pin, auth probe |

Naming: `<carrier>__<donor>__a<α>` by default, overridable. The converter writes
`<stem>_ar.safetensors` / `<stem>_nar.safetensors`; since the stem is not
`akbar_arabic_rock_lora`, the "use it" card must emit the **explicit** `ar`+`nar` form of a
songs-JSON alias, not `dir` (that shorthand only resolves the legacy basenames).

## 5. The mix job

Per α, checkpointed so a crash resumes:

1. **stage** — resolve each input (catalog entry → GCS object, or a local path); skip if already
   in `cache/src/` (keyed by `crc32c`); verify size always, sha256 when known.
2. **preflight** — both inputs `fused`/`merged` (a `converted` file is `layers.*` with no
   `.alpha` key and cannot be an input); branch layout (`ar`+`nar` → `merge_quran` style,
   `ar`-only donor → `merge_pron` style, NAR copied and zero-padded); one rank per branch per
   input; same tensor key set; α ∈ [0,1]; disk ≥ 1.5× the estimate.
3. **merge** → `jobs/<id>/fused/a<α>/<carrier-basename>.safetensors`. At α = 0 keep the carrier's
   basename — otherwise the identity check below loses meaning.
4. **verify** — α = 0: `torch.equal` over every tensor vs the carrier. α ≠ 0: `B_mix@A_mix ==
   B_c@A_c + α·(B_d@A_d)` over the projections, and **report the measured error** (3.0e-8 at
   α = 0.5 is exact; 3.5e-5 at α = 0.2 is bf16 rounding of the scaled block — both correct).
5. **convert** — run the pinned converter, capture stdout, parse
   `byte-level verification against the source: OK`; a non-zero exit fails the arm with its stderr.
6. **record** — `MANIFEST.json` (tool sha, input shas, α, ranks, converted AR/NAR sha256, sizes,
   exact commands, `torch`/`safetensors` versions) + one sidecar per arm.
7. **stop.** Upload is a separate click. Promotion into `loras/` is out of scope — curation, not build.

Sizes to show before the click: fused ~112 MB at rank 32, **~235 MB at 64**; converted pair
~140 MB (rank 32), **~280 MB (rank 64)**; a 5-arm grid ≈ 3 GB. Compute ~0.5 s merge + ~0.3 s
convert per arm — the first build of a pair also pays ~20 s to stage 224 MiB.

## 6. Invariants

1. **No repo imports, no absolute paths, no `/content`.** The app is installable and runnable
   anywhere Python + the deps exist.
2. **The JSON is the contract.** Catalog records and manifests match the existing shapes so a
   file from either side is readable by the other.
3. **Identity is the converted AR/NAR sha256.** The fused file's whole-file sha256 is *not*
   reproducible (safetensors 0.8.0 serialises `__metadata__` in nondeterministic order) — label
   it "this run only" in the UI.
4. **Measured, not asserted** — α = 0 is `torch.equal`; α ≠ 0 ships its δ-sum number; a failed
   check fails the arm.
5. **No in-memory-only state** — `job.json` on every transition; on startup a `running` job whose
   pid is dead becomes `failed` (`note: "engine restarted"`).
6. **Idempotency key** = hash(carrier, donor, sorted α, merger version, converter sha) — a
   double-click reuses a finished job instead of writing 3 GB again.
7. **The CLI equivalent is always on screen and in the log** — the app is a client of its own
   CLI, never a second implementation.
8. **Unknown ≠ guessed** — unresolved lineage stays unresolved; a missing/mismatched converter
   disables Build with the exact remedy; an auth probe failure says which rung failed.
9. **No delete, no overwrite-by-accident** — upload previews exact object names; existing keys
   require an explicit overwrite flag.

## 7. Phases

The previous attempt spent a whole session and shipped no UI. Order is chosen against that.

| phase | deliverable | done when |
|---|---|---|
| **P1** | `loralab serve` — app shell + catalog + adapter detail, reading a configured bucket | you open `127.0.0.1:5000` in a browser and see the real adapter table; no page can write |
| **P2** | mix: form → job → verify → convert → manifest → result card | the conformance gate (§8) passes: published arms reproduced |
| **P3** | upload with dry-run preview + prefix guard | upload is off by default; the destination is named before it happens |
| **P4** | polish: cache panel, settings page, `doctor`, keyboard/URL state | a repeat build on cached sources is < 3 s wall |

P1 is deliberately a UI phase, not a library phase. The catalog logic it needs is ~330 lines and
is written **inside** P1.

## 8. Tests

* Unit: header parse on synthetic safetensors (the same `_read_header` path both transports use),
  classification, provenance, lineage resolution, α validation, merge tensor shapes, manifest
  shape, job state machine + restart reconcile, idempotency key, 409 gates.
* Transport: a fake HTTP layer (in-memory ranges, paginated listings, 401 refresh, non-401
  re-raise) — no network, no credentials, same discipline as `tests/test_lora_toolkit.py`.
* **Conformance gate (the important one).** Rebuild `v3 × arabic_instrumental` and compare the
  **converted pair's sha256** against the published manifest
  (`<base>/arabic_instrumental_lora/mix_v3/MANIFEST.json`):

  | arm | converted AR | converted NAR |
  |---|---|---|
  | `a0.1` | `d48109b3…4a0` | `88f00bcb…b58` |
  | `a0.2` | `820f1ce9…dd6` | `f9738e21…25f` |

  Caveat, stated so the gate isn't used dishonestly: the converted bytes are reproducible only
  when the fused file keeps the arm's basename and the converter stem matches (a different
  basename changes only the embedded `source_file`). Otherwise compare the tensor set, ranks and
  the δ-sum — never the fused file's sha.
* A negative test: the app must fail the gate when α round-trips through `float32` sloppily or
  when the donor is scaled into `A` instead of `B` (the two most plausible re-implementation bugs).

## 9. Risks

| risk | effect | guard |
|---|---|---|
| re-implemented merge drifts from the repo's | adapters that are subtly wrong, silently | conformance gate on published shas + δ-sum check |
| converter version differs | converted ranks/layout differ | sha256 pin, refuse on mismatch |
| auth ladder guesses wrong | "app is broken" | probe every rung, show which won, `doctor` |
| upload writes the wrong prefix | pollutes the library | preview + `_prototypes/` default + no delete permission |
| torch pulls CUDA wheels | 2 GB download to do `torch.cat` | CPU index in the README and an extra in `pyproject` |
| folding α into `A` instead of `B` | wrong-sounding adapter, no error | negative test in §8 |
| merged file sha used as identity | false mismatch reports | display converted sha; label fused sha run-only |
| laptop opens the app on a LAN | anyone can write to the bucket | `127.0.0.1` default, warning on `0.0.0.0` |
| bf16 rounding read as a bug | chasing a phantom | the UI shows the measured error next to the check |

## 10. Open decisions

1. **Repo name** (`loralab`?) and whether it is public from day one — the merge math is a
   re-derivation of published conventions, so a licence header citing `docs/PRON_LORA_MERGE.md`
   is the honest move.
2. **Default bucket prefix** — prefill this project's bucket, or ship empty and make Settings
   mandatory on first run? (Prefill is friendlier; empty is more portable.)
3. **α grid vs one-at-a-time** in P2 — grids are cheap (0.5 s/arm) but ~280 MB/arm.
4. **Naming** — `<carrier>__<donor>__a<α>`, or keep the shipped `v3instr_a0.2` style? It is the
   string you will type into every later songs JSON.
