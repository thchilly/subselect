"""SSIM + side-by-side diff for an M9 figure port.

Usage:
    python scripts/m9_ssim_diff.py <truth.png> <m9_output.png> <out_sidebyside.png>

Computes greyscale SSIM via ``skimage.metrics.structural_similarity`` and
writes a 1×2 panel: truth | M9 output, with the SSIM number in the suptitle.
Both images are loaded as-is (no resize) — if dimensions differ, this script
exits non-zero so dimension drift surfaces loudly rather than getting masked
by a forced resize.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity


def _load_grey(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB")) / 255.0
    return rgb2gray(img)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    truth, m9, out = (Path(p) for p in sys.argv[1:4])
    truth_grey, m9_grey = _load_grey(truth), _load_grey(m9)
    if truth_grey.shape != m9_grey.shape:
        print(f"DIMENSION MISMATCH: truth {truth_grey.shape} vs m9 {m9_grey.shape}")
        raise SystemExit(1)
    ssim = structural_similarity(truth_grey, m9_grey, data_range=1.0)
    truth_rgb, m9_rgb = _load_rgb(truth), _load_rgb(m9)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(truth_rgb); axes[0].set_title(f"truth (cell re-execute)\n{truth.name}", fontsize=11)
    axes[1].imshow(m9_rgb); axes[1].set_title(f"M9 port output\n{m9.name}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"SSIM = {ssim:.4f}  (target ≥ 0.98)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"SSIM = {ssim:.6f}")
    print(f"side-by-side → {out}")


if __name__ == "__main__":
    main()
