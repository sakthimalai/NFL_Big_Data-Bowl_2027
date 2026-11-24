"""
src/plots.py
Publication-Grade, Colorblind-Safe Data Visualization Suite (300 DPI)
NFL Big Data Bowl 2027

Generates strictly 6 figures (paired with 1 unified table in the writeup = 7 total visuals,
well below the < 10 visual limit), using real 10 Hz tracking traces and authoritative numbers
from model_results.json.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from kinematics import smooth_trajectory

PROCESSED_DIR = ROOT_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-Safe Okabe-Ito / Front-Office Palette (No Red/Green ambiguity)
NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#94A3B8"
ROYAL_BLUE = "#1D4ED8"
SKY_BLUE = "#0284C7"
TEAL_BLUE = "#0369A1"
AMBER_GOLD = "#D97706"
DEEP_ORANGE = "#B45309"
LIGHT_BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"
WHITE = "#FFFFFF"


def set_theme():
    plt.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": SLATE,
        "axes.linewidth": 1.1,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.65,
        "axes.grid": True,
        "text.color": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
    })


def save_fig(fig, filename: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    print(f"  Saved {filename} to figures/")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 1: Real RUN_THE_HOOP_DRILL Tracking Traces & Agility Opt-Out Context
# -------------------------------------------------------------------------
def plot_figure_1(ct: pd.DataFrame, res: dict):
    print("Generating Figure 1: Real Hoop Drill Tracking Traces & Opt-Out Rates...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300)

    hoop = ct[(ct["drill_name"] == "RUN_THE_HOOP_DRILL") & (ct["entity_type"] == "PLAYER")].copy()

    def draw_real_hoop_trace(ax, event_id, player_label, stats_subtitle):
        sub = hoop[hoop["event_id"] == event_id].sort_values("time")
        x = sub["x"].values
        y = sub["y"].values
        kin = smooth_trajectory(x, y, window=7, polyorder=2)
        xh, yh = kin["x_hat"], kin["y_hat"]
        vx, vy, v = kin["vx"], kin["vy"], kin["v"]
        ax_vec, ay_vec, a_tau = kin["ax"], kin["ay"], kin["a_tau"]

        # Center coordinates for visual comparison
        xc = xh - np.mean(xh)
        yc = yh - np.mean(yh)

        # True normal acceleration vector: a_n_vec = a_vec - a_tau * tau_hat
        # Always points toward the instantaneous concave center of curvature!
        v_safe = np.maximum(v, 0.2)
        tx, ty = vx / v_safe, vy / v_safe
        an_x = ax_vec - a_tau * tx
        an_y = ay_vec - a_tau * ty

        # Line segments colored by speed v(t)
        points = np.array([xc, yc]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0.0, 7.2)
        lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=3.8, zorder=4)
        lc.set_array(v[:-1])
        line = ax.add_collection(lc)

        # Plot inward normal acceleration vectors where speed > 2.0 yd/s
        idx = [i for i in range(4, len(xc) - 4, 3) if v[i] > 2.0]
        ax.quiver(
            xc[idx], yc[idx], an_x[idx], an_y[idx],
            color=ROYAL_BLUE, alpha=0.75, scale=42, width=0.0055, headwidth=3.8,
            zorder=5, label="Normal Accel. Vector $\\vec{a}_n(t)$ (Inward)"
        )

        # Mark Start (from rest) and Finish
        ax.scatter([xc[0]], [yc[0]], color=WHITE, edgecolors=NAVY, s=110, lw=2.2, zorder=7,
                   label=f"Start from Rest ($v_0 = {v[0]:.2f}$ yd/s)")
        ax.scatter([xc[-1]], [yc[-1]], color=AMBER_GOLD, edgecolors=NAVY, marker="s", s=100, lw=1.8, zorder=7,
                   label="Drill Finish")

        ax.set_xlim(-5.8, 5.8)
        ax.set_ylim(-4.6, 4.8)
        ax.set_aspect("equal")
        ax.set_xlabel("Relative Field X (yards)", fontweight="bold")
        ax.set_ylabel("Relative Field Y (yards)", fontweight="bold")
        ax.set_title(player_label, fontsize=11.5, fontweight="bold", pad=8)
        ax.text(
            0.03, 0.96, stats_subtitle, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.8, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT_BG, ec=SLATE, alpha=0.95)
        )
        ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)
        return line

    # Panel A: High-Bend Edge Rusher (Javon Solomon, real event)
    line1 = draw_real_hoop_trace(
        ax1,
        event_id="20240229100100_1709262683937_000",
        player_label="A. High-Bend Edge Trace: Javon Solomon (DE, 2024 #168)",
        stats_subtitle="Real Figure-8 Hoop Trace (74 frames @ 10 Hz)\nMean Arc $a_n = 5.45$ yd/s² | 40-Dash = 4.72s (3-Cone: Opted Out)\nNFL: 3.96 Quick Pressures / 100 PR Snaps (3.0 Sacks in 202 Snaps)"
    )

    # Panel B: Lower-Bend Edge Rusher (Eric Watts, real event)
    line2 = draw_real_hoop_trace(
        ax2,
        event_id="20240229100100_1709262897872_000",
        player_label="B. Lower-Bend Edge Trace: Eric Watts (DE, 2024 UDFA)",
        stats_subtitle="Real Figure-8 Hoop Trace (80 frames @ 10 Hz)\nMean Arc $a_n = 4.35$ yd/s² | 40-Dash = 4.67s (Fast 40, Low Bend)\nNFL: 0.00 Quick Pressures / 100 PR Snaps (0 QP in 154 REG Snaps)"
    )

    cbar = fig.colorbar(line2, ax=[ax1, ax2], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Instantaneous Speed $v(t)$ (yd/s)", fontweight="bold")

    m = res["metadata"]
    fig.suptitle(
        "Figure 1: Real 10 Hz 'Run the Hoop' Traces Capture Curvilinear Cornering Missed by Stopwatch Opt-Outs",
        fontsize=13.5, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, -0.03,
        f"Source: combine_tracking.csv (10 Hz). While {m['three_cone_opt_outs_dl']}/{m['total_combine_dl']} ({m['three_cone_opt_out_pct_dl']:.1f}%) of Combine DL prospects skipped the 3-Cone "
        f"and {m['short_shuttle_opt_outs_dl']}/{m['total_combine_dl']} ({m['short_shuttle_opt_out_pct_dl']:.1f}%) skipped the Short Shuttle, "
        f"{m['pass_rush_drill_participants']}/{m['total_combine_dl']} (100%) ran PASS_RUSH_DRILL and {m['hoop_drill_participants']}/{m['total_combine_dl']} (92.4%) ran RUN_THE_HOOP_DRILL.",
        ha="center", fontsize=8.8, color=SLATE
    )
    save_fig(fig, "fig1_forty_blindspot.png")


# -------------------------------------------------------------------------
# Figure 2: First-Step Get-Off Curves in PASS_RUSH_DRILL & Reliability
# -------------------------------------------------------------------------
def plot_figure_2(ct: pd.DataFrame, res: dict):
    print("Generating Figure 2: First-Step Get-Off Curves & Test-Retest Reliability...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})

    cohort = pd.DataFrame(res["cohort_records"])
    pr = ct[(ct["drill_name"] == "PASS_RUSH_DRILL") & (ct["entity_type"] == "PLAYER")].copy()
    pr = pr[pr["nfl_id"].isin(cohort["nfl_id"])]

    # Strict 4-quartile partition (Q1=22, Q2=21, Q3=21, Q4=21 prospects, matching Figure 5)
    cohort_sorted = cohort.sort_values("z_jerk").reset_index(drop=True)
    cohort_sorted["jerk_quartile"] = pd.qcut(cohort_sorted["z_jerk"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    top_ids = set(cohort_sorted[cohort_sorted["jerk_quartile"] == "Q4"]["nfl_id"])
    bot_ids = set(cohort_sorted[cohort_sorted["jerk_quartile"] == "Q1"]["nfl_id"])

    top_player_curves = {}
    bot_player_curves = {}
    attempt_records = []

    for (nfl_id, event_id), grp in pr.groupby(["nfl_id", "event_id"]):
        grp = grp.sort_values("time")
        kin = smooth_trajectory(grp["x"].values, grp["y"].values, window=7, polyorder=2)
        v, a_mag, jerk = kin["v"], kin["a_mag"], kin["jerk"]
        n = len(v)
        if n >= 6:
            idx = np.flatnonzero(v > 0.50)
            i0 = int(idx[0]) if idx.size else 0
            end_5 = min(i0 + 5, n)
            if end_5 > i0:
                attempt_records.append({
                    "nfl_id": nfl_id,
                    "event_id": event_id,
                    "peak_jerk": float(np.max(jerk[i0:end_5]))
                })
            # Take 8 frames aligned from onset
            end_8 = min(i0 + 8, n)
            curve = a_mag[i0:end_8]
            if len(curve) < 8:
                curve = np.pad(curve, (0, 8 - len(curve)), mode="edge")

            if nfl_id in top_ids:
                top_player_curves.setdefault(nfl_id, []).append(curve)
            elif nfl_id in bot_ids:
                bot_player_curves.setdefault(nfl_id, []).append(curve)

    t_axis = np.arange(8) * 0.10
    top_avg = np.array([np.mean(curves, axis=0) for curves in top_player_curves.values()])
    bot_avg = np.array([np.mean(curves, axis=0) for curves in bot_player_curves.values()])

    top_mean, top_sem = top_avg.mean(axis=0), top_avg.std(axis=0) / np.sqrt(len(top_avg))
    bot_mean, bot_sem = bot_avg.mean(axis=0), bot_avg.std(axis=0) / np.sqrt(len(bot_avg))

    ax1.set_ylim(1.2, 5.85)
    ax1.set_xlim(-0.02, 0.72)
    ax1.axvspan(0.0, 0.40, color="#EFF6FF", alpha=0.8, label="0.5s First-Step Window (5 Frames Post-Onset)")
    ax1.plot(t_axis, top_mean, color=ROYAL_BLUE, lw=3.0, marker="o", label=f"Top Quartile Get-Off Jerk ($n={len(top_avg)}$ prospects)")
    ax1.fill_between(t_axis, top_mean - 1.96 * top_sem, top_mean + 1.96 * top_sem, color=ROYAL_BLUE, alpha=0.20)

    ax1.plot(t_axis, bot_mean, color=AMBER_GOLD, lw=3.0, linestyle="--", marker="s", label=f"Bottom Quartile Get-Off Jerk ($n={len(bot_avg)}$ prospects)")
    ax1.fill_between(t_axis, bot_mean - 1.96 * bot_sem, bot_mean + 1.96 * bot_sem, color=AMBER_GOLD, alpha=0.20)

    ax1.set_xlabel("Time From Movement Onset $t$ (seconds) [Frames 0–7]", fontweight="bold")
    ax1.set_ylabel("Linear Acceleration Magnitude $|\\vec{a}(t)|$ (yd/s²)", fontweight="bold")
    ax1.set_title("A. First-Step Acceleration Profile in 'Pass Rush Drill'", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.text(
        0.03, 0.96,
        "Aligned to movement onset ($v > 0.50$ yd/s)\nSteeper slope = Higher Jerk $j = d|\\vec{a}|/dt$\nShaded bands represent 95% CIs ($\\pm 1.96$ SEM)",
        transform=ax1.transAxes, va="top", fontsize=8.8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=SLATE, alpha=0.95)
    )

    # Panel B: Test-Retest Reliability across Repeat Attempts (strictly N=85 cohort, Attempt 1 vs 2)
    cohort_ids = set(cohort["nfl_id"].tolist())
    df_att = pd.DataFrame(attempt_records)
    df_att = df_att[df_att["nfl_id"].isin(cohort_ids)].sort_values(["nfl_id", "event_id"])
    p1 = df_att.groupby("nfl_id").nth(0).set_index("nfl_id")["peak_jerk"]
    p2 = df_att.groupby("nfl_id").nth(1).set_index("nfl_id")["peak_jerk"]
    piv = pd.concat([p1.rename("first"), p2.rename("last")], axis=1).dropna()

    ax2.scatter(piv["first"], piv["last"], color=SKY_BLUE, edgecolors=NAVY, s=55, alpha=0.85, zorder=4)
    m_slope, b_int = np.polyfit(piv["first"], piv["last"], 1)
    x_line = np.linspace(piv["first"].min(), piv["first"].max(), 50)
    ax2.plot(x_line, m_slope * x_line + b_int, color=NAVY, lw=2.2, linestyle="-", label="OLS Fit Across Attempts")

    rc = res["reliability_and_correlations"]
    ax2.set_ylim(1.0, 18.5)
    ax2.set_xlabel("Attempt 1 Peak First-Step Jerk (yd/s³)", fontweight="bold")
    ax2.set_ylabel("Attempt 2 Peak First-Step Jerk (yd/s³)", fontweight="bold")
    ax2.set_title(f"B. Within-Prospect Repeatability ($n={rc['pr_jerk_retest_n']}$ Multi-Rep)", fontweight="bold", pad=10)
    ax2.text(
        0.03, 0.97,
        f"Test-Retest Reliability: Attempt 1 vs. 2 $r = {rc['pr_jerk_retest_r']:.3f}$ ($p = {rc['pr_jerk_retest_p']:.1e}$)\n"
        f"Savitzky-Golay Window Sensitivity ($W=5,7,9$):\n"
        f"  • Hoop $a_n$: $r_{{5,7}} = {rc['hoop_w5_w7_r']:.3f}, r_{{7,9}} = {rc['hoop_w7_w9_r']:.3f}$ | "
        f"Get-Off Jerk: $r_{{5,7}} = {rc['jerk_w5_w7_r']:.3f}, r_{{7,9}} = {rc['jerk_w7_w9_r']:.3f}$",
        transform=ax2.transAxes, va="top", fontsize=8.3, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=ROYAL_BLUE, alpha=0.96)
    )
    ax2.legend(loc="lower right", framealpha=0.95)

    fig.suptitle(
        "Figure 2: First-Step Get-Off Jerk Separates Explosive Rushers Within 0.5s and Shows Moderate Repeatability",
        fontsize=13.0, fontweight="bold", y=1.02
    )
    save_fig(fig, "fig2_kinematic_vectors.png")


# -------------------------------------------------------------------------
# Figure 3: PRKI Distribution & Independence from the 40-Yard Dash
# -------------------------------------------------------------------------
def plot_figure_3(res: dict):
    print("Generating Figure 3: PRKI Distribution & Correlation vs. 40-Yard Dash...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    cohort = pd.DataFrame(res["cohort_records"])
    rc = res["reliability_and_correlations"]
    n_coh = res["metadata"]["qualified_cohort_n"]

    # Panel A: Standardized PRKI Distribution
    z_prki = cohort["z_prki"]
    q75, q25 = float(z_prki.quantile(0.75)), float(z_prki.quantile(0.25))
    ax1.hist(z_prki, bins=15, color=SKY_BLUE, edgecolor=NAVY, alpha=0.55, density=False, label=f"Qualified DL Cohort ($N={n_coh}$)")
    ax1.axvline(0.0, color=NAVY, linestyle="--", lw=1.6, label="Cohort Mean ($Z = 0.00$)")
    ax1.axvline(q75, color=ROYAL_BLUE, linestyle="-.", lw=1.8, label=f"Top Quartile Cutoff ({q75:+.2f} SD)")
    ax1.axvline(q25, color=AMBER_GOLD, linestyle="-.", lw=1.8, label=f"Bottom Quartile Cutoff ({q25:+.2f} SD)")

    ax1.set_xlim(-2.8, 2.5)
    ax1.set_ylim(0, 21)
    ax1.set_xlabel("Standardized Pass Rush Kinematic Index ($Z_{\\mathrm{PRKI}}$)", fontweight="bold")
    ax1.set_ylabel(f"Number of Prospects ($N = {n_coh}$)", fontweight="bold")
    ax1.set_title(f"A. Standardized PRKI Distribution ($N = {n_coh}$ REG Cohort)", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8.3, framealpha=0.95)

    # Exact annotations from cohort_records (positioned cleanly away from upper-left legend)
    def annotate_player(name, text_pos, box_fc, box_ec):
        row = cohort[cohort["display_name"] == name].iloc[0]
        zp = row["z_prki"]
        qp = row["qp_per_100_pr"]
        sacks = row["pr_sacks"]
        ax1.annotate(
            f"{name} ($Z = {zp:+.2f}$)\n{qp:.2f} QP/100 Snaps | {sacks:.1f} Sacks",
            xy=(zp, 3.5), xytext=text_pos,
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3),
            fontsize=8.1, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.28", fc=box_fc, ec=box_ec, lw=1.2)
        )

    annotate_player("Isaiah McGuire", (0.15, 16.5), "#EFF6FF", ROYAL_BLUE)
    annotate_player("Javon Solomon", (0.25, 11.5), "#EFF6FF", SKY_BLUE)
    annotate_player("YaYa Diaby", (0.35, 6.5), LIGHT_BG, SLATE)

    # Panel B: PRKI Components vs 40-Yard Dash (n=69 with 40-times)
    sub40 = cohort.dropna(subset=["forty"]).copy()
    ax2.scatter(sub40["forty"], sub40["z_jerk"], color=ROYAL_BLUE, s=55, alpha=0.80, edgecolors=NAVY,
                label=f"First-Step Get-Off Jerk ($Z$): $r = {rc['r_jerk_forty']:.3f}$ ($p = {rc['p_jerk_forty']:.3f}$, Modest)")
    ax2.scatter(sub40["forty"], sub40["z_bend"], color=AMBER_GOLD, marker="^", s=55, alpha=0.75, edgecolors=NAVY,
                label=f"Hoop Centripetal Bend ($Z$): $r = {rc['r_bend_forty']:.3f}$ ($p < 0.001$, Speed-Coupled)")

    x_vals = np.linspace(sub40["forty"].min(), sub40["forty"].max(), 50)
    m_j, b_j = np.polyfit(sub40["forty"], sub40["z_jerk"], 1)
    m_b, b_b = np.polyfit(sub40["forty"], sub40["z_bend"], 1)
    ax2.plot(x_vals, m_j * x_vals + b_j, color=ROYAL_BLUE, lw=2.2, linestyle="-")
    ax2.plot(x_vals, m_b * x_vals + b_b, color=AMBER_GOLD, lw=2.2, linestyle="--")

    ax2.set_xlabel("Combine 40-Yard Dash Time (seconds)", fontweight="bold")
    ax2.set_ylabel("Standardized Component Score ($Z$-Score)", fontweight="bold")
    ax2.set_title(f"B. Component Relationships vs. 40-Yard Dash ($n = {rc['n_forty']}$)", fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=8.3, framealpha=0.95)
    ax2.text(
        0.03, 0.16,
        f"3-Cone Agility Correlation ($n={rc['n_three_cone']}$ non-opt-outs):\n"
        f"  • Hoop Bend $a_n$ vs. 3-Cone: $r = {rc['r_bend_three_cone']:.3f}$ ($p = {rc['p_bend_three_cone']:.4f}$)\n"
        f"  • Get-Off Jerk vs. 3-Cone: $r = {rc['r_jerk_three_cone']:.3f}$ ($p = {rc['p_jerk_three_cone']:.3f}$)",
        transform=ax2.transAxes, va="top", fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=LIGHT_BG, ec=SLATE)
    )

    fig.suptitle(
        "Figure 3: Get-Off Jerk Is Only Modestly Related to Straight-Line Speed; Hoop Bend Tracks the 40 and 3-Cone",
        fontsize=12.8, fontweight="bold", y=1.02
    )
    save_fig(fig, "fig3_metric_distribution.png")


# -------------------------------------------------------------------------
# Figure 4: Regression Forest Plot & Incremental Predictive Fit
# -------------------------------------------------------------------------
def plot_figure_4(res: dict):
    print("Generating Figure 4: Specification Robustness Forest Plot & Repeated CV / Quasi-Binomial...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

    meta = res["metadata"]
    ols = res["ols_pass_rush_snaps"]
    wls = res["wls_pass_rush_snaps"]
    sens = ols["sensitivity"]
    glm = res["binomial_glm_pass_rush_snaps"]
    cv = res["cross_validation"]

    # Panel A: Specification Robustness Panel (one row per model in Section 3C)
    items = [
        (f"Edge-Heavy Only (≥50% Edge, n={sens['edge_heavy']['n']})", sens["edge_heavy"]["params"]["z_prki"], AMBER_GOLD),
        (f"Rookie Season Only (n={sens['rookie_season']['n']})", sens["rookie_season"]["params"]["z_prki"], SKY_BLUE),
        (f"Complete 40-Times Only (n={sens['complete_forty']['n']})", sens["complete_forty"]["params"]["z_prki"], SKY_BLUE),
        (f"WLS Snap-Weighted (n={meta['qualified_cohort_n']})", wls["wls2_params"]["z_prki"], ROYAL_BLUE),
        (f"OLS HC3: Log(Pick) + UDFA (n={sens['log_draft_pick']['n']})", sens["log_draft_pick"]["params"]["z_prki"], ROYAL_BLUE),
        (f"OLS HC3: Primary Model 2 (n={meta['qualified_cohort_n']})", ols["m2_params"]["z_prki"], ROYAL_BLUE),
    ]

    y_pos = np.arange(len(items))
    for i, (label, d, col) in enumerate(items):
        c = d["coef"]
        low, high = d["ci_low"], d["ci_high"]
        p_val = d["p_value"]
        sig = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "n.s.")
        ax1.errorbar(
            c, i, xerr=[[c - low], [high - c]], fmt="o", color=col, ecolor=col,
            elinewidth=2.8, capsize=5, markersize=8, zorder=5
        )
        ax1.text(
            c, i + 0.23, f"{c:+.2f} [{low:+.2f}, {high:+.2f}] ({sig}, p={p_val:.3f})",
            ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=NAVY
        )

    ax1.axvline(0.0, color=NAVY, linestyle="--", lw=1.4, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([it[0] for it in items], fontweight="bold")
    ax1.set_xlim(-0.35, 1.50)
    ax1.set_ylim(-0.5, len(items) - 0.25)
    ax1.set_xlabel("PRKI Effect (+1 SD) on Quick Pressures per 100 Rush Snaps (95% CI)", fontweight="bold")
    ax1.set_title(f"A. PRKI Specification Robustness (N = {meta['qualified_cohort_n']}, {meta['total_pass_rush_snaps']:,} REG Rush Snaps)", fontweight="bold", pad=10)

    # Panel B: In-Sample R^2 and Repeated 5-Fold x 50 Out-of-Sample R^2
    x_cats = np.arange(2)
    width = 0.34
    base_vals = [ols["baseline_r2"] * 100, cv["r2_oof_base_mean"] * 100]
    aug_vals = [ols["augmented_r2"] * 100, cv["r2_oof_aug_mean"] * 100]

    b1 = ax2.bar(x_cats - width / 2, base_vals, width, color=SLATE, edgecolor=NAVY, lw=1.3, label="Baseline (Pick + 40 + Missing40 + Wt + EdgeShare)")
    b2 = ax2.bar(x_cats + width / 2, aug_vals, width, color=ROYAL_BLUE, edgecolor=NAVY, lw=1.3, label="Augmented (+ PRKI)")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x_cats)
    ax2.set_xticklabels(["In-Sample $R^2$\n(OLS Fit)", "5-Fold × 50 Out-of-Sample $R^2$\n(Leak-Free Ridge CV)"], fontweight="bold")
    ax2.set_ylabel("Variance Explained (%)", fontweight="bold")
    ax2.set_ylim(0, 60)
    ax2.set_title("B. Repeated CV & Quasi-Binomial Check", fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.95)

    g_prki = glm["g2_params"]["z_prki"]
    g_jerk = glm["g2b_params"]["z_jerk"]
    ax2.text(
        0.03, 0.83,
        f"Incremental Gains (Model 2 vs. Model 1):\n"
        f"  • OLS $\\Delta R^2 = +{ols['delta_r2']*100:.2f}$ pp ($\\beta = +{ols['m2_params']['z_prki']['coef']:.2f}$, HC3 $p = {ols['m2_params']['z_prki']['p_value']:.3f}$)\n"
        f"  • 5×50 CV $\\Delta R^2_{{\\mathrm{{OOF}}}} = +{cv['delta_r2_oof_pp_mean']:.2f} \\pm {cv['delta_r2_oof_pp_sd']:.2f}$ pp ($-{cv['mse_reduction_pct']:.1f}\\%$ MSE)\n"
        f"Grouped Quasi-Binomial ($\\phi = {glm['dispersion_phi_m2']:.2f}$, F-test $p = {glm['f_p_value_vs_baseline']:.3f}$):\n"
        f"  • PRKI OR = {g_prki['odds_ratio']:.2f} [1.03, 1.40] ($p = {g_prki['p_value']:.3f}$**)\n"
        f"  • Get-Off Jerk OR = {g_jerk['odds_ratio']:.2f} [1.06, 1.37] ($p = {g_jerk['p_value']:.3f}$***)",
        transform=ax2.transAxes, va="top", fontsize=7.9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec=ROYAL_BLUE, alpha=0.96)
    )

    fig.suptitle(
        "Figure 4: The PRKI Coefficient Is Positive Across Specifications (+0.33 to +0.61 per 100 Rush Snaps)",
        fontsize=12.5, fontweight="bold", y=1.02
    )
    save_fig(fig, "fig4_regression_forest.png")


# -------------------------------------------------------------------------
# Figure 5: Quick Pressures per 100 Pass-Rush Snaps by PRKI Quartile
# -------------------------------------------------------------------------
def plot_figure_5(res: dict):
    print("Generating Figure 5: Quick Pressures per 100 Pass-Rush Snaps by PRKI Quartile...")
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)

    meta = res["metadata"]
    q_list = res["quartile_breakdown"]
    labels = [f"{q['quartile']}\n(n = {q['n']} prospects)" for q in q_list]
    means = [q["mean_qp_per_100_pr"] for q in q_list]
    sems = [q["sem_qp_per_100_pr"] for q in q_list]
    ci95s = [q["ci95_halfwidth"] for q in q_list]

    palette = [MUTED, SKY_BLUE, ROYAL_BLUE, AMBER_GOLD]
    bars = ax.bar(
        labels, means, yerr=sems, capsize=7, color=palette, edgecolor=NAVY,
        linewidth=1.5, width=0.52, error_kw=dict(lw=2.0, ecolor=NAVY),
        label="Quartile Mean $\\pm 1$ SEM"
    )

    for bar, m_val, s_val, c_val in zip(bars, means, sems, ci95s):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m_val + s_val + 0.08,
            f"{m_val:.2f}\n(±{s_val:.2f} SEM)",
            ha="center", va="bottom", fontsize=9.3, fontweight="bold", color=NAVY
        )

    rel_gain = ((means[-1] / means[0]) - 1.0) * 100.0
    diff_qp = means[-1] - means[0]
    ax.annotate(
        f"Top vs. Bottom Quartile:\n+{diff_qp:.2f} Quick Pressures / 100 Rush Snaps\n(+{rel_gain:.1f}% Relative Increase, {means[0]:.2f} → {means[-1]:.2f})",
        xy=(3, means[-1]), xytext=(0.55, 3.35),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8),
        fontsize=9.5, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec=ROYAL_BLUE, lw=1.5)
    )

    ax.set_ylim(0, 4.10)
    ax.set_ylabel("Quick Pressures (<2.5s) per 100 Pass-Rush Snaps", fontweight="bold")
    ax.set_xlabel(f"Combine Pass Rush Kinematic Index (PRKI) Quartile ($N = {meta['qualified_cohort_n']}$ Total)", fontweight="bold")
    ax.set_title(
        f"Figure 5: Top-Quartile PRKI Rushers Generate {means[-1]:.2f} Quick Pressures per 100 Snaps vs. {means[0]:.2f} in Bottom Quartile",
        fontsize=12.5, fontweight="bold", pad=12
    )
    ax.legend(loc="upper left", framealpha=0.95)
    fig.text(
        0.5, -0.03,
        f"Sample: N = {meta['qualified_cohort_n']} Combine DL prospects with >=30 REG snaps ({meta['total_pass_rush_snaps']:,} rush snaps). "
        f"Q4 - Q1 gap = +{means[-1] - means[0]:.2f} QP/100 snaps (SE ≈ 0.47, 95% CI [+0.97, +2.81]; error bars show +/- 1 SEM).",
        ha="center", fontsize=8.8, color=SLATE
    )
    save_fig(fig, "fig5_ingame_impact.png")


# -------------------------------------------------------------------------
# Figure 6: Combine-to-Game Bend Translation & Edge Evaluation Matrix
# -------------------------------------------------------------------------
def plot_figure_6(res: dict):
    print("Generating Figure 6: Combine-to-Game Bend Translation & Front-Office Matrix...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=300)
    plt.subplots_adjust(wspace=0.22)

    meta = res["metadata"]
    cohort = pd.DataFrame(res["cohort_records"])
    rc = res["reliability_and_correlations"]

    # Panel A: Combine Hoop Bend (z_bend) vs In-Game Pass-Rush Cornering Accel (ingame_bend_an)
    sub_ig = cohort.dropna(subset=["ingame_bend_an"]).copy()
    sc1 = ax1.scatter(
        sub_ig["z_bend"], sub_ig["ingame_bend_an"],
        c=sub_ig["qp_per_100_pr"], cmap="viridis", s=65, edgecolors=NAVY, lw=0.9, zorder=4
    )
    m_ig, b_ig = np.polyfit(sub_ig["z_bend"], sub_ig["ingame_bend_an"], 1)
    x_ig = np.linspace(sub_ig["z_bend"].min(), sub_ig["z_bend"].max(), 50)
    ax1.plot(x_ig, m_ig * x_ig + b_ig, color=ROYAL_BLUE, lw=2.4, label=f"OLS Trend ($r = {rc['r_bend_ingame_bend']:+.3f}, p = {rc['p_bend_ingame_bend']:.4f}$)")

    ax1.set_ylim(1.5, 4.50)
    ax1.set_xlabel("Combine Hoop Centripetal Acceleration ($Z_{\\mathrm{bend}}$)", fontweight="bold")
    ax1.set_ylabel("In-Game Pass-Rush Cornering $a_{n,\\mathrm{game}}$ (90th Pct, yd/s²)", fontweight="bold")
    ax1.set_title(f"A. Combine vs. In-Game Rush Cornering ($N = {len(sub_ig)}$)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.text(
        0.03, 0.97,
        f"{meta['total_ingame_rush_window_frames']:,} Rush-Window Frames (SavGol W=7):\n"
        f"  • Combine Bend $\\to$ In-Game Bend: $r = {rc['r_bend_ingame_bend']:+.3f}$ ($p = {rc['p_bend_ingame_bend']:.4f}$)\n"
        f"  • Partial $r$ (ctrl Edge Share): $r = {rc['partial_r_bend_ingame_bend']:+.3f}$ ($p = {rc['partial_p_bend_ingame_bend']:.2f}$)\n"
        f"  • In-Game Bend $\\to$ Quick Pressure: $r = {rc['r_ingame_bend_qp']:+.3f}$ ($p = {rc['p_ingame_bend_qp']:.1e}$)",
        transform=ax1.transAxes, va="top", fontsize=8.1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=ROYAL_BLUE, alpha=0.96)
    )

    # Panel B: Front-Office Decision Matrix for True Edge Rushers (DE/OLB with 40-times, n=48)
    edge = cohort[cohort["nfl_position"].isin(["DE", "OLB"])].dropna(subset=["forty"]).copy()
    med_40 = float(edge["forty"].median())  # 4.675s
    cut_y = 0.0

    # We plot 40-yard dash on X-axis WITHOUT inverting, or if we invert, we place labels using axis coordinates!
    # Let's keep standard X-axis: 4.45s (Faster, Left) -> 5.20s (Slower, Right)
    ax2.set_xlim(4.42, 5.02)
    ax2.set_ylim(-2.1, 2.65)

    ax2.axvline(med_40, color=SLATE, linestyle="--", lw=1.4, label=f"Edge Median 40-Time ({med_40:.2f}s)")
    ax2.axhline(cut_y, color=SLATE, linestyle="--", lw=1.4, label="Cohort Average PRKI ($Z = 0.00$)")

    # Shaded regions
    ax2.fill_between([4.42, med_40], cut_y, 2.65, color="#EFF6FF", alpha=0.55)
    ax2.fill_between([med_40, 5.02], cut_y, 2.65, color="#FEF3C7", alpha=0.55)
    ax2.fill_between([4.42, med_40], -2.1, cut_y, color="#F8FAFC", alpha=0.6)
    ax2.fill_between([med_40, 5.02], -2.1, cut_y, color="#F1F5F9", alpha=0.6)

    # Clean Quadrant Titles in Axes Coordinates (Guaranteed never to overlap or be obscured!)
    ax2.text(0.03, 0.96, "Q1: HIGH-SPEED + HIGH PRKI\n(Fast 40 < 4.68s, High PRKI)", transform=ax2.transAxes,
             va="top", ha="left", fontsize=8.2, fontweight="bold", color=ROYAL_BLUE,
             bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=SLATE, alpha=0.90))
    ax2.text(0.97, 0.96, "Q2: VALUE TARGETS\n(Modest 40 ≥ 4.68s, High PRKI)", transform=ax2.transAxes,
             va="top", ha="right", fontsize=8.2, fontweight="bold", color=DEEP_ORANGE,
             bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=SLATE, alpha=0.90))
    ax2.text(0.03, 0.04, "Q3: LINEAR SPEED / LOW PRKI\n(Fast 40 < 4.68s, Lower PRKI)", transform=ax2.transAxes,
             va="bottom", ha="left", fontsize=8.2, fontweight="bold", color=SLATE,
             bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=SLATE, alpha=0.90))
    ax2.text(0.97, 0.04, "Q4: LOWER ATHLETIC PROFILE\n(Slower 40 ≥ 4.68s, Lower PRKI)", transform=ax2.transAxes,
             va="bottom", ha="right", fontsize=8.2, fontweight="bold", color=SLATE,
             bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=SLATE, alpha=0.90))

    sc2 = ax2.scatter(
        edge["forty"], edge["z_prki"],
        c=edge["qp_per_100_pr"], cmap="viridis", s=75, edgecolors=NAVY, lw=1.0, zorder=5
    )

    # Non-overlapping player callouts (All true Edge DE/OLB!)
    callouts = [
        ("YaYa Diaby (#82)", "YaYa Diaby", (4.46, 0.72)),
        ("Jalyx Hunt (#94)", "Jalyx Hunt", (4.54, 1.62)),
        ("Isaiah McGuire (#126)", "Isaiah McGuire", (4.78, 1.45)),
        ("Javon Solomon (#168)", "Javon Solomon", (4.78, 0.75)),
        ("Barryn Sorrell (#124)", "Barryn Sorrell", (4.53, -1.25)),
    ]
    for label_txt, name, text_pos in callouts:
        p = edge[edge["display_name"] == name]
        if len(p) > 0:
            px = float(p["forty"].iloc[0])
            py = float(p["z_prki"].iloc[0])
            ax2.annotate(
                label_txt, xy=(px, py), xytext=text_pos,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
                fontsize=8.2, fontweight="bold", color=NAVY,
                bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec=SLATE, alpha=0.92),
                zorder=6
            )

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("NFL Quick Pressures (<2.5s) per 100 Pass-Rush Snaps", fontweight="bold")

    ax2.set_xlabel("Combine 40-Yard Dash Time (seconds) [← Faster | Slower →]", fontweight="bold")
    ax2.set_ylabel("Standardized PRKI ($Z_{\\mathrm{PRKI}}$)", fontweight="bold")
    ax2.set_title(f"B. Edge Rusher Evaluation Matrix (DE/OLB Only, $n = {len(edge)}$)", fontweight="bold", pad=10)

    fig.suptitle(
        "Figure 6: Combine vs. In-Game Rush Bend (A) and the Edge Draft Matrix (B)",
        fontsize=13.0, fontweight="bold", y=1.02
    )
    save_fig(fig, "fig6_evaluation_matrix.png")


def main():
    set_theme()
    res_path = PROCESSED_DIR / "model_results.json"
    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    ct = pl.read_parquet(PROCESSED_DIR / "combine_tracking.parquet").to_pandas()

    plot_figure_1(ct, res)
    plot_figure_2(ct, res)
    plot_figure_3(res)
    plot_figure_4(res)
    plot_figure_5(res)
    plot_figure_6(res)
    print("All 6 master figures generated successfully.")


if __name__ == "__main__":
    main()
