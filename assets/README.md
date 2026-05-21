# assets/

Figures referenced by the deck. Run `uv run prepare_figures.py` to copy real
artifacts in, or generate labelled placeholders when they're missing.

**Animations (GIF):**

- `teaser.gif`            — title-page teaser (Kármán vortex street)
- `karman_classical.gif`  — Kármán street, classical BGK-LBM
- `karman_ml.gif`         — Kármán street, ML-LBM (learned collision)

**Figures (PNG):**

- `training_loss.png`     — training/validation RMSRE curves
- `naive_vs_gavg.png`     — naive MLP diverges vs GAVG tracks the reference
- `resnet_experiment.png` — ResNet vs plain MLP (loss / stability)
- `workspace_analysis.png`— appendix: workspace effort breakdown
- `velocity_decay.png`, `tgv_t0.png`, `tgv_t500.png`, `tgv_t900.png`
  — Taylor–Green benchmark figures (legacy)

Until real images are dropped in, the deck shows labelled placeholders. GIF
placeholders are kept unless a real animation (>200 KB) is already present.
