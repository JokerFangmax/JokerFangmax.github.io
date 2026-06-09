#!/usr/bin/env python3
"""Merge individual mesh figures into side-by-side panels."""

from PIL import Image
from pathlib import Path

PROJECT_DIR = Path("content/project/wost-final-project")


def hmerge(left_path: Path, right_path: Path, out_path: Path, pad: int = 20, bg: tuple = (255, 255, 255)):
    """Merge two images horizontally with padding."""
    left = Image.open(left_path)
    right = Image.open(right_path)

    # Resize to same height for clean alignment
    target_height = max(left.height, right.height)
    left = left.resize((int(left.width * target_height / left.height), target_height), Image.LANCZOS)
    right = right.resize((int(right.width * target_height / right.height), target_height), Image.LANCZOS)

    total_width = left.width + right.width + pad
    canvas = Image.new("RGB", (total_width, target_height), bg)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + pad, 0))
    canvas.save(out_path, "PNG")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Fig 1: Dirichlet panels
    hmerge(
        PROJECT_DIR / "fig1a_bunny_dirichlet_rmse_vs_walks.png",
        PROJECT_DIR / "fig1b_spot_dirichlet_rmse_vs_walks.png",
        PROJECT_DIR / "fig1_merged_dirichlet_panels.png",
    )

    # Fig 2: Neumann panels
    hmerge(
        PROJECT_DIR / "fig2a_bunny_neumann_rmse_vs_walks.png",
        PROJECT_DIR / "fig2b_spot_neumann_rmse_vs_walks.png",
        PROJECT_DIR / "fig2_merged_neumann_panels.png",
    )
