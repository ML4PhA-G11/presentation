# Learning the LBM Collision Operator — Slides

reveal.js presentation for the **Physics-Informed ML** project that learns the
Lattice Boltzmann (D2Q9) BGK collision operator with a neural network embedding
D4 lattice symmetry and exact mass/momentum conservation.

Source project: [`../learning_lbm_collision_operator/`](../../../learning_lbm_collision_operator) — see
`run-all-tensorflow.py`.

---

## ⚠️ Move out of this folder first

This project currently lives in a directory named
`git@github.com:ML4PhA-G11/presentation.git/` — that **colon** breaks `uv`
(`error: path segment contains separator ':'`), so the figure helper won't run.
Move the contents somewhere normal before working, e.g.:

```bash
# from the parent directory
mv "git@github.com:ML4PhA-G11/presentation.git" slides
cd slides
git remote -v   # the SSH remote stays intact — only the local path changes
```

After that, the rest of this README "just works".

---

## The site

Two plain static HTML files, matching the framework and styling of the ML4PhA
lecture decks (reveal.js 5.1.0, `white` theme, Space Grotesk / Inter fonts,
MathJax + chalkboard plugins):

- [`index.html`](./index.html) — the **landing page** (lecturer-style cards
  linking to the talk, the code, and the reference paper).
- [`talk.html`](./talk.html) — the **deck itself**, a single self-contained
  reveal.js file.

reveal.js, MathJax and the plugins are loaded from a CDN, so **there is no build
step** — edit the HTML and refresh.

## Tooling

- **A browser** — the pages are plain HTML; just open `index.html`.
- **uv** — manages the small Python helper (`prepare_figures.py`) that prepares
  the figure files in `assets/`.

## 1. Prepare the figure assets (uv)

```bash
uv sync                    # installs Pillow into .venv
uv run prepare_figures.py  # copies real artifacts, or writes placeholders
```

`prepare_figures.py` looks for the outputs of
`../../../learning_lbm_collision_operator/run-all-tensorflow.py` (in
`artifacts-run-all-tensorflow/`). If it finds them it copies them into
`assets/`; otherwise it generates labelled placeholder PNGs (and animated-GIF
stand-ins) so the deck still renders. Re-run it any time you regenerate the
artifacts.

## 2. Local preview

Serve the folder (needed for the CDN assets and MathJax to load over `http`),
then open the landing page:

```bash
python3 -m http.server 3030   # then open http://localhost:3030/
```

`index.html` links to `talk.html`. reveal.js shortcuts in the deck: `f`
fullscreen, `o` overview, `s` speaker notes, `b` / `c` chalkboard.

## 3. PDF download

The landing page card has a **PDF** link (`talk.pdf`), just like the ML4PhA
lecture pages. The deck is rendered to PDF with
[decktape](https://github.com/astefanutti/decktape):

```bash
./make-pdf.sh        # serves the deck + renders talk.pdf (needs internet + Chrome)
```

CI runs the same step on every push, so the published `talk.pdf` always matches
the live deck. `talk.pdf` is a build artifact — it's `.gitignore`d, not
committed.

## 4. Build a static site

No build needed — `index.html`, `talk.html` and `assets/` *are* the site
(plus `talk.pdf`, generated as above).

## 5. Hosting on GitHub Pages

A workflow in `.github/workflows/deploy.yml` stages `index.html`, `talk.html`
and `assets/`, renders `talk.pdf`, and publishes them on every push to `main`.
After the first successful run, enable Pages:

1. Repo Settings → Pages → Source = **GitHub Actions**.
2. Push to `main`; the deck appears at
   `https://ML4PhA-G11.github.io/presentation/`.

> The workflow publishes the static files directly (no `gh-pages` branch to
> manage). Commit your placeholder or real assets before pushing.

## Generating the real figures

```bash
cd ../../../learning_lbm_collision_operator
python run-all-tensorflow.py   # or `uv run` if you set that up there
cd -
uv run prepare_figures.py      # re-copies the new artifacts into assets/
```

| Artifact                                                                 | Slide asset                  |
|--------------------------------------------------------------------------|------------------------------|
| `artifacts-run-all-tensorflow/training_loss.png`                         | `assets/training_loss.png`   |
| `artifacts-run-all-tensorflow/velocity_decay.png`                        | `assets/velocity_decay.png`  |
| `artifacts-run-all-tensorflow/velocity_fields/velocity_field_t00000.png` | `assets/tgv_t0.png`          |
| `artifacts-run-all-tensorflow/velocity_fields/velocity_field_t00500.png` | `assets/tgv_t500.png`        |
| `artifacts-run-all-tensorflow/velocity_fields/velocity_field_t00900.png` | `assets/tgv_t900.png`        |

## Layout

```
.
├── index.html                   # landing page (lecturer-style cards + PDF link)
├── talk.html                    # the reveal.js deck — edit this
├── make-pdf.sh                  # render talk.html -> talk.pdf (decktape)
├── pyproject.toml               # Pillow (Python, via uv)
├── prepare_figures.py           # asset prep helper
├── assets/                      # images/GIFs referenced by the deck
└── .github/workflows/deploy.yml # GitHub Pages deploy (+ PDF render)
```
