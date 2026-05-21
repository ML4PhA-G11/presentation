"""Populate ``assets/`` with the figures the deck references.

For each expected figure:
  - if the corresponding artifact exists in
    ``../learning_lbm_collision_operator/artifacts-run-all-tensorflow/``,
    copy it in;
  - otherwise generate a placeholder PNG so the deck still renders.

Run with ``uv run prepare_figures.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# Source artifacts produced by ../learning_lbm_collision_operator/run-all-tensorflow.py
SRC_ROOT = HERE.parent.parent.parent / "learning_lbm_collision_operator" / "artifacts-run-all-tensorflow"

# (slide_asset_name, source_path_relative_to_SRC_ROOT, size, label)
FIGURES = [
    ("training_loss.png",     "training_loss.png",                           (800, 500), "training_loss"),
    ("velocity_decay.png",    "velocity_decay.png",                          (800, 500), "velocity_decay"),
    ("tgv_t0.png",            "velocity_fields/velocity_field_t00000.png",   (500, 400), "Taylor-Green @ t=0"),
    ("tgv_t500.png",          "velocity_fields/velocity_field_t00500.png",   (500, 400), "Taylor-Green @ t=500"),
    ("tgv_t900.png",          "velocity_fields/velocity_field_t00900.png",   (500, 400), "Taylor-Green @ t=900"),
    # Figures for the current outline (Karman / naive-vs-GAVG / ResNet / appendix).
    # No upstream artifact yet — these stay as placeholders until you drop in real ones.
    ("naive_vs_gavg.png",     "naive_vs_gavg.png",                           (900, 320), "naive (diverges) vs GAVG"),
    ("resnet_experiment.png", "resnet_experiment.png",                       (520, 400), "ResNet vs plain MLP"),
    ("workspace_analysis.png","workspace_analysis.png",                      (520, 400), "workspace analysis"),
]

# (slide_asset_name, size, label) — animated GIF stand-ins (the deck shows GIFs here)
GIFS = [
    ("teaser.gif",          (560, 360), "Karman vortex street (teaser)"),
    ("karman_classical.gif",(420, 300), "classical BGK-LBM"),
    ("karman_ml.gif",       (420, 300), "ML-LBM (learned collision)"),
]


def _placeholder(out: Path, size: tuple[int, int], label: str) -> None:
    """Render a labelled grey grid as a placeholder PNG."""
    w, h = size
    img = Image.new("RGB", size, "#f9fafb")
    draw = ImageDraw.Draw(img)

    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill="#e5e7eb", width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill="#e5e7eb", width=1)

    draw.rectangle([(1, 1), (w - 2, h - 2)], outline="#9ca3af", width=2)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    sub = "placeholder — run run-all-tensorflow.py, then prepare_figures.py"
    tw = draw.textlength(label, font=font)
    sw = draw.textlength(sub, font=sub_font)
    draw.text(((w - tw) / 2, h / 2 - 28), label, fill="#374151", font=font)
    draw.text(((w - sw) / 2, h / 2 + 8),  sub,   fill="#6b7280", font=sub_font)

    img.save(out, "PNG")


def _placeholder_gif(out: Path, size: tuple[int, int], label: str) -> None:
    """Render a 2-frame animated grey-grid placeholder GIF (so <img> renders)."""
    frames = []
    for shift in (0, 20):
        w, h = size
        img = Image.new("RGB", size, "#f9fafb")
        draw = ImageDraw.Draw(img)
        for x in range(-40 + shift, w, 40):
            draw.line([(x, 0), (x, h)], fill="#e5e7eb", width=1)
        for y in range(0, h, 40):
            draw.line([(0, y), (w, y)], fill="#e5e7eb", width=1)
        draw.rectangle([(1, 1), (w - 2, h - 2)], outline="#9ca3af", width=2)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            sub_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        except OSError:
            font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
        sub = "placeholder GIF — drop in the real animation"
        tw = draw.textlength(label, font=font)
        sw = draw.textlength(sub, font=sub_font)
        draw.text(((w - tw) / 2, h / 2 - 24), label, fill="#374151", font=font)
        draw.text(((w - sw) / 2, h / 2 + 8), sub, fill="#6b7280", font=sub_font)
        frames.append(img)
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:], duration=600, loop=0)


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    copied, placeheld = 0, 0
    for dest_name, src_rel, size, label in FIGURES:
        dest = ASSETS / dest_name
        src = SRC_ROOT / src_rel
        if src.is_file():
            shutil.copyfile(src, dest)
            print(f"  copied  {src_rel:50s} -> assets/{dest_name}")
            copied += 1
        else:
            _placeholder(dest, size, label)
            print(f"  placeholder for assets/{dest_name}  (missing {src_rel})")
            placeheld += 1

    for dest_name, size, label in GIFS:
        dest = ASSETS / dest_name
        if dest.is_file() and dest.stat().st_size > 200_000:
            # Looks like a real animation already dropped in — leave it alone.
            print(f"  keeping existing assets/{dest_name}")
        else:
            _placeholder_gif(dest, size, label)
            print(f"  placeholder for assets/{dest_name}")
            placeheld += 1

    print(f"\nDone. {copied} real, {placeheld} placeholder.")
    if placeheld:
        print(
            f"Generate real figures by running:\n"
            f"  cd {SRC_ROOT.parent}\n"
            f"  python run-all-tensorflow.py\n"
            f"...then re-run this script."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
