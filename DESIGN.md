# Design notes

The web UI is plain Flask + Jinja + **one CSS file** (`loralab/web/static/app.css`).
No build step, no CDN, no JavaScript framework: the app has to work on a laptop with
no network, so the two typefaces are bundled too.

This document records the *why* behind the look, so later changes extend it instead of
drifting from it.

## Idea

loralab is a workbench for audio-model adapters, used by one person sweeping hundreds of
safetensors files. Two consequences:

1. **Scanning beats decorating.** Tables are the main surface, so they get fixed column
   widths (columns line up from one prefix group to the next), tabular numerals, and
   generous row height.
2. **Colour carries meaning, and nothing else.** Neutrals are ink on cool paper. Colour
   is spent only on the three adapter kinds, job states and ok/fail. Buttons are ink,
   links are underlined ink; there is no brand accent competing with the data.

## Kinds have a colour *and* a shape

Colour alone fails for colour-blind readers and in greyscale, so each kind also has a
glyph (drawn with a CSS mask, so it inherits the text colour):

| kind        | glyph                 | colour  | reads as                    |
|-------------|-----------------------|---------|-----------------------------|
| `fused`     | solid disc            | blue    | trained, as-is              |
| `converted` | single ring           | teal    | same weights, re-wrapped    |
| `merged`    | two overlapping rings | magenta | carrier + donor combined    |

The logo is three bars in those same three colours, so brand and data share one
vocabulary. Job states follow the same rule (dot = done, square = failed, ring = queued,
dash = cancelled, pulsing dot = running).

## Two views built from existing data

* **Lineage as a flow** (detail page). Upstream files sit on the left, the current
  adapter on the right. Merged adapters show carrier and donor stacked with a `+`;
  converted adapters show their fused source. An unresolved link renders as a dashed
  "unknown" node and keeps the original wording: it is never guessed from a filename.
* **Merge α as a fader** (merged adapters). α is drawn on a 0 to 1 scale from
  "carrier only" to "full donor". It is a position on a scale, not a percentage split,
  because the donor's up-projection is scaled by α rather than blended convexly. Values
  above 1 pin the thumb to the right edge while the number still shows the real value.

## Tokens

Defined at the top of `app.css`; light and dark follow `prefers-color-scheme`.

| token | light | dark | use |
|---|---|---|---|
| `--paper` | `#eef0f6` | `#131626` | page |
| `--surface` | `#ffffff` | `#1b1f33` | panels, tables, inputs |
| `--ink` | `#1b1f33` | `#e6e9f6` | text, primary button |
| `--dim` | `#545b75` | `#9ba3c1` | secondary text |
| `--line` | `#d8dce8` | `#2a2f48` | borders |
| `--fused` / `--converted` / `--merged` | `#2260c4` / `#0a7787` / `#b02d7b` | `#7aadff` / `#55c8d8` / `#f27ab9` | kinds |
| `--ok` / `--warn` / `--bad` | `#23724a` / `#9a5a08` / `#bb3a30` | `#62d394` / `#ebb862` / `#ff8077` | status |

Terminal-like surfaces (the command and the job log) stay dark in both themes.

## Type

* **Instrument Sans** (variable, 400 to 700) for everything people read.
* **Spline Sans Mono** (variable, 300 to 700) only for strings a machine produced:
  paths, hashes, commands, logs. It is not used as decoration on labels.

Both are SIL OFL 1.1, Latin subset, self-hosted in `static/fonts/` with their licences.
Greek α and a few arrows fall back to the system font.

## Small UX additions

These add convenience without changing what any page does or which routes exist:

* Kind counts at the top of the catalog are links that apply the existing `?kind=`
  filter (and keep the current search and prefix). Click the active one to clear it.
* `/` focuses the catalog search box.
* Copy buttons on paths, hashes, the terminal command and the record JSON.
* The job log follows new output, unless you have scrolled up to read.
* Active tab in the header, a skip-to-content link, visible keyboard focus, and
  `prefers-reduced-motion` respected (the only non-interactive motion is the pulse on a
  running job).
* On narrow screens the header wraps to two rows and tables scroll inside their card
  instead of the page.

## What was deliberately not touched

Python code, routes, template variables, JSON endpoints, the job polling logic, CLI and
tests. Text that tests assert on (`No catalog yet`, `Merge provenance`, `Scan <scope>`,
`ok` / `fail`, ...) is unchanged. The one inline script that existed (`job.html`)
keeps its logic; it only gained the log-follow behaviour.

## Changing the look

* Recolour: edit the tokens in the two `:root` blocks.
* New adapter kind: add a `.k-<name>` rule (colour + `--glyph` SVG) next to the others.
* New job state: add an `.s-<name>` rule next to the others.
* Browser baseline: evergreen browsers (uses `color-mix()` and CSS `mask`).
  Older browsers lose tints and glyphs but stay readable.
