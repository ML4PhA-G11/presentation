# Learning the LBM Collision Operator — Slides

Slidev presentation for the **Physics-Informed ML** project that learns the
Lattice Boltzmann (D2Q9) BGK collision operator with a neural network embedding
D4 lattice symmetry and exact mass/momentum conservation.

Source project: [`../learning_lbm_collision_operator/`](../../../learning_lbm_collision_operator) — see
`run-all-tensorflow.py`.

---

## ⚠️ Move out of this folder first

This project currently lives in a directory named
`git@github.com:ML4PhA-G11/presentation.git/` — that **colon** breaks `uv`
(`error: path segment contains separator ':'`) and will also confuse Vite /
Slidev asset resolution. Move the contents somewhere normal before working,
e.g.:

```bash
# from the parent directory
mv "git@github.com:ML4PhA-G11/presentation.git" slides
cd slides
git remote -v   # the SSH remote stays intact — only the local path changes
```

After that, the rest of this README "just works".

---

## Tooling

- **Node / npm** — runs Slidev (the deck itself).
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
`assets/`; otherwise it generates labelled placeholder PNGs so the deck still
renders. Re-run the script any time you regenerate the artifacts.

## 2. Local preview (npm)

```bash
npm install
npm run dev          # opens http://localhost:3030
```

Press `e` in the browser to edit a slide live, `o` for slide overview,
`d` for dark mode.

## 3. Build a static site

```bash
npm run build        # outputs ./dist with base path /presentation/
```

## 4. Hosting on GitHub Pages

A workflow in `.github/workflows/deploy.yml` builds on every push to `main`
and publishes to GitHub Pages. After the first successful run, enable Pages:

1. Repo Settings → Pages → Source = **GitHub Actions**.
2. Push to `main`; the deck appears at
   `https://ML4PhA-G11.github.io/presentation/`.

> The workflow commits *built* output via the Pages action — you don't manage
> a `gh-pages` branch yourself. It builds against whatever assets are in
> `assets/`, so commit your placeholder or real PNGs before pushing.

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
├── slides.md                    # the deck — edit this
├── package.json                 # Slidev (Node)
├── pyproject.toml               # Pillow (Python, via uv)
├── prepare_figures.py           # asset prep helper
├── assets/                      # PNGs referenced by the slides
└── .github/workflows/deploy.yml # GitHub Pages build & deploy
```
