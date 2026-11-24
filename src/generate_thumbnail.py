"""
src/generate_thumbnail.py
Generates a clean, modern, minimalist Card & Thumbnail (560 x 280 pixels).
Spacious executive design: big typography on the left, iconic glowing tracking visual on the right.
Zero text overlap, zero clutter.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from kinematics import smooth_trajectory

PROCESSED_DIR = ROOT_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Minimalist Palette
BG_COLOR = "#070D18"        # Deep Midnight
ACCENT_CYAN = "#38BDF8"     # Electric Sky Blue
ACCENT_AMBER = "#F59E0B"    # Combine Amber
TEXT_WHITE = "#FFFFFF"      # Crisp White
TEXT_SLATE = "#94A3B8"      # Cool Slate
PILL_BG = "#0E1E38"         # Dark Blue Pill
PILL_BORDER = "#1E3A66"     # Pill Border


def generate_simple_card():
    print("Generating clean, minimalist 560 x 280 Card & Thumbnail...")

    ct = pl.read_parquet(PROCESSED_DIR / "combine_tracking.parquet").to_pandas()
    hoop = ct[(ct["drill_name"] == "RUN_THE_HOOP_DRILL") & (ct["entity_type"] == "PLAYER")].copy()
    sub = hoop[hoop["event_id"] == "20240229100100_1709262683937_000"].sort_values("time")
    kin = smooth_trajectory(sub["x"].values, sub["y"].values, window=7, polyorder=2)
    xh, yh = kin["x_hat"], kin["y_hat"]
    vx, vy, v = kin["vx"], kin["vy"], kin["v"]
    ax_v, ay_v, a_tau = kin["ax"], kin["ay"], kin["a_tau"]

    xc = xh - np.mean(xh)
    yc = yh - np.mean(yh)

    # Centripetal normal acceleration vectors
    v_safe = np.maximum(v, 0.2)
    tx, ty = vx / v_safe, vy / v_safe
    an_x = ax_v - a_tau * tx
    an_y = ay_v - a_tau * ty

    # Canvas: 5.6 x 2.8 inches at 100 DPI = exactly 560 x 280 pixels
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100, facecolor=BG_COLOR)

    # Base background coordinates: (0, 560) x (0, 280)
    ax_bg = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax_bg.set_xlim(0, 560)
    ax_bg.set_ylim(0, 280)
    ax_bg.axis("off")

    # Subtle decorative field hash marks in background (faint, elegant)
    for x in np.arange(40, 540, 60):
        ax_bg.plot([x, x], [0, 280], color="#0F1C31", lw=0.8, alpha=0.5, zorder=1)

    # -------------------------------------------------------------
    # LEFT COLUMN: Clean, Bold Typography & Messaging
    # -------------------------------------------------------------
    # 1. Category Pill Badge
    pill = FancyBboxPatch(
        (28, 234), 160, 22,
        boxstyle="round,pad=1.5,rounding_size=6",
        ec=ACCENT_CYAN, fc=PILL_BG, lw=1.1, zorder=5
    )
    ax_bg.add_patch(pill)
    ax_bg.text(
        108, 245, "NFL BIG DATA BOWL 2027",
        ha="center", va="center", color=ACCENT_CYAN,
        fontsize=7.2, fontweight="bold", fontfamily="sans-serif", zorder=6
    )

    # 2. Main Title (Stacked, High Impact, Large & Bold)
    ax_bg.text(
        28, 192, "BEYOND THE",
        ha="left", va="center", color=TEXT_WHITE,
        fontsize=21.0, fontweight="bold", fontfamily="sans-serif", zorder=6
    )
    ax_bg.text(
        28, 164, "STOPWATCH",
        ha="left", va="center", color=TEXT_WHITE,
        fontsize=21.0, fontweight="bold", fontfamily="sans-serif", zorder=6
    )

    # 3. Subtitle (Generous spacing, crisp color)
    ax_bg.text(
        28, 131, "First-Step Get-Off & Arc Cornering",
        ha="left", va="center", color=ACCENT_CYAN,
        fontsize=9.2, fontweight="bold", fontfamily="sans-serif", zorder=6
    )
    ax_bg.text(
        28, 115, "in NFL Pass Rushers via 10 Hz Tracking",
        ha="left", va="center", color=TEXT_SLATE,
        fontsize=8.5, fontfamily="sans-serif", zorder=6
    )

    # 4. Highlight Stat Pill
    stat_pill = FancyBboxPatch(
        (28, 42), 250, 44,
        boxstyle="round,pad=2,rounding_size=6",
        ec=PILL_BORDER, fc=PILL_BG, lw=1.1, zorder=5
    )
    ax_bg.add_patch(stat_pill)

    # Left colored accent pip
    ax_bg.scatter([42], [64], color=ACCENT_AMBER, s=36, zorder=7)
    ax_bg.text(
        54, 69, "+0.50 Quick Pressures / 100 Snaps",
        ha="left", va="center", color=TEXT_WHITE,
        fontsize=8.0, fontweight="bold", fontfamily="sans-serif", zorder=6
    )
    ax_bg.text(
        54, 53, "Open Track  •  N = 85 Regular-Season Cohort",
        ha="left", va="center", color=TEXT_SLATE,
        fontsize=6.8, fontfamily="sans-serif", zorder=6
    )

    # -------------------------------------------------------------
    # RIGHT COLUMN: Single Iconic Kinetic Tracking Visual
    # -------------------------------------------------------------
    # Dedicated visual container (subtle card backdrop with soft glow)
    card_right = FancyBboxPatch(
        (304, 24), 228, 232,
        boxstyle="round,pad=3,rounding_size=10",
        ec=PILL_BORDER, fc="#091426", lw=1.2, zorder=3
    )
    ax_bg.add_patch(card_right)

    # Visual Title inside container
    ax_bg.text(
        418, 236, "10 Hz Figure-8 Hoop Trace",
        ha="center", va="center", color=ACCENT_CYAN,
        fontsize=7.8, fontweight="bold", fontfamily="sans-serif", zorder=6
    )

    # Subplot for the Hoop Drill Trajectory
    ax_hoop = fig.add_axes([0.56, 0.17, 0.38, 0.60], facecolor="#091426")

    points = np.array([xc, yc]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0.0, 7.2)
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=3.4, zorder=4)
    lc.set_array(v[:-1])
    ax_hoop.add_collection(lc)

    # Inward normal acceleration vectors (quivers)
    idx = [i for i in range(4, len(xc) - 4, 3) if v[i] > 2.0]
    ax_hoop.quiver(
        xc[idx], yc[idx], an_x[idx], an_y[idx],
        color=ACCENT_CYAN, alpha=0.90, scale=38, width=0.012, headwidth=4.0, zorder=5
    )

    # Start (white circle) and Finish (amber square)
    ax_hoop.scatter([xc[0]], [yc[0]], color="#FFFFFF", edgecolors=ACCENT_CYAN, s=60, lw=1.8, zorder=7)
    ax_hoop.scatter([xc[-1]], [yc[-1]], color=ACCENT_AMBER, edgecolors="#FFFFFF", marker="s", s=50, lw=1.5, zorder=7)

    ax_hoop.set_xlim(-5.8, 5.8)
    ax_hoop.set_ylim(-4.6, 4.6)
    ax_hoop.set_aspect("equal")
    ax_hoop.axis("off")

    # Caption at bottom of right card
    ax_bg.text(
        418, 42, "Speed Colormap + Inward Force Vectors",
        ha="center", va="center", color=TEXT_SLATE,
        fontsize=6.8, fontfamily="sans-serif", zorder=6
    )

    # Save thumbnail.png at EXACTLY 560 x 280
    thumb_path = FIGURES_DIR / "thumbnail.png"
    fig.savefig(thumb_path, dpi=100, facecolor=BG_COLOR, edgecolor="none")
    plt.close()

    print(f"Clean Thumbnail successfully saved to {thumb_path} ({thumb_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_simple_card()
