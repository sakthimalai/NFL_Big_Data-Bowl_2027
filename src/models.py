"""
src/models.py
Hierarchical Modeling, In-Game Tracking Translation & Econometric Inference Engine
NFL Big Data Bowl 2027

Implements:
1. Sample Waterfall accountability (118 Combine DL -> 109 Hoop+PR -> 106 Qualified >=30 REG snaps).
2. Savitzky-Golay filter window sensitivity (W=5, 7, 9) and Attempt 1 vs Attempt 2 repeatability (N=106).
3. In-game 10 Hz tracking validation restricted to [ball_snap -> pass_forward/sack] using identical
   Savitzky-Golay smoothing and masks (v > 1.50 yd/s, kappa > 0.15 yd^-1).
4. Prospect-level OLS (HC3 robust SEs), Snap-Weighted WLS, and Grouped Quasi-Binomial GLM (phi-adjusted).
5. MixedLM get-off translation test (z_jerk -> NFL player_get_off) on EDGE pass-rush snaps.
6. Repeated 5-Fold x 50 Out-of-Sample Cross-Validation with in-fold StandardScaler + Ridge.
7. Serializes single source of truth to processed/model_results.json.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.stats import chi2, f as f_dist, norm
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from kinematics import smooth_trajectory

PROCESSED_DIR = ROOT_DIR / "processed"
DL_POS = ["EDGE", "INTERIOR_LINE"]


def compute_sensitivity_and_reliability(qual_ids: list):
    print("Computing Savitzky-Golay window sensitivity (W=5,7,9) and Attempt 1 vs 2 reliability (N=106)...")
    ct = pl.read_parquet(PROCESSED_DIR / "combine_tracking.parquet")
    pct_dl = (
        ct.filter(pl.col("nfl_id").is_in(qual_ids) & (pl.col("entity_type") == "PLAYER"))
        .sort(["nfl_id", "drill_name", "attempt", "time"])
        .to_pandas()
    )

    h_all, p_all = None, None
    rel_stats = {}

    for w in [5, 7, 9]:
        hoop_rows = []
        for (nfl_id, attempt, event_id), grp in pct_dl[pct_dl["drill_name"] == "RUN_THE_HOOP_DRILL"].groupby(
            ["nfl_id", "attempt", "event_id"], sort=False
        ):
            kin = smooth_trajectory(grp["x"].values, grp["y"].values, window=w, polyorder=2)
            v, an, kap = kin["v"], kin["a_n"], kin["kappa"]
            mask = (v > 1.50) & (kap > 0.15)
            if np.any(mask):
                hoop_rows.append({"nfl_id": nfl_id, "attempt": attempt, f"an_w{w}": float(np.mean(an[mask]))})
        df_hw = pd.DataFrame(hoop_rows)

        pr_rows = []
        for (nfl_id, attempt, event_id), grp in pct_dl[pct_dl["drill_name"] == "PASS_RUSH_DRILL"].groupby(
            ["nfl_id", "attempt", "event_id"], sort=False
        ):
            kin = smooth_trajectory(grp["x"].values, grp["y"].values, window=w, polyorder=2)
            v, a_mag, jerk = kin["v"], kin["a_mag"], kin["jerk"]
            n = len(v)
            if n >= 6:
                idx = np.flatnonzero(v > 0.5)
                i0 = int(idx[0]) if idx.size else 0
                end_i = min(i0 + 5, n)
                if end_i > i0:
                    pr_rows.append({
                        "nfl_id": nfl_id,
                        "attempt": attempt,
                        f"jerk_w{w}": float(np.max(jerk[i0:end_i])),
                        f"acc_w{w}": float(np.max(a_mag[i0:end_i])),
                    })
        df_pw = pd.DataFrame(pr_rows)

        if w == 5:
            h_all = df_hw.groupby("nfl_id")[[f"an_w{w}"]].mean()
            p_all = df_pw.groupby("nfl_id")[[f"jerk_w{w}", f"acc_w{w}"]].mean()
        else:
            h_all = h_all.join(df_hw.groupby("nfl_id")[[f"an_w{w}"]].mean())
            p_all = p_all.join(df_pw.groupby("nfl_id")[[f"jerk_w{w}", f"acc_w{w}"]].mean())

        if w == 7:
            df_pw_sorted = df_pw.sort_values(["nfl_id", "attempt"])
            p_att1 = df_pw_sorted.groupby("nfl_id").nth(0).set_index("nfl_id")["jerk_w7"]
            p_att2 = df_pw_sorted.groupby("nfl_id").nth(1).set_index("nfl_id")["jerk_w7"]
            both = pd.concat([p_att1.rename("att1"), p_att2.rename("att2")], axis=1).dropna()
            r_p, p_p = stats.pearsonr(both["att1"], both["att2"])
            rel_stats["pr_jerk_retest_r"] = float(r_p)
            rel_stats["pr_jerk_retest_p"] = float(p_p)
            rel_stats["pr_jerk_retest_n"] = int(len(both))

    h_corr = h_all.corr()
    j_corr = p_all[["jerk_w5", "jerk_w7", "jerk_w9"]].corr()
    rel_stats["hoop_w5_w7_r"] = float(h_corr.loc["an_w5", "an_w7"])
    rel_stats["hoop_w7_w9_r"] = float(h_corr.loc["an_w7", "an_w9"])
    rel_stats["jerk_w5_w7_r"] = float(j_corr.loc["jerk_w5", "jerk_w7"])
    rel_stats["jerk_w7_w9_r"] = float(j_corr.loc["jerk_w7", "jerk_w9"])
    return rel_stats


def compute_ingame_tracking_metrics(dl_ids: list, rush_keys: pl.DataFrame):
    print("Extracting in-game pass-rush tracking kinematics [ball_snap -> pass_forward/sack] across 2023-2025...")
    gt_list = []
    for yr in [2023, 2024, 2025]:
        gt_yr = pl.read_parquet(PROCESSED_DIR / f"game_tracking_{yr}.parquet")
        gt_sub = gt_yr.join(rush_keys, on=["game_id", "play_id", "nfl_id"], how="inner")
        gt_list.append(gt_sub)

    gt_all = pl.concat(gt_list).sort(["game_id", "play_id", "nfl_id", "time"])
    total_raw_frames = len(gt_all)

    # Identify ball_snap and end-of-rush events per play-player
    end_events = ["pass_forward", "qb_sack", "qb_strip_sack", "run", "fumble"]
    gt_pd = gt_all.select(["game_id", "play_id", "nfl_id", "time", "x", "y", "event"]).to_pandas()

    # Vectorized snap/end time bounds per (game_id, play_id, nfl_id)
    snap_df = (
        gt_pd[gt_pd["event"] == "ball_snap"]
        .groupby(["game_id", "play_id", "nfl_id"], as_index=False)["time"]
        .min()
        .rename(columns={"time": "t_snap"})
    )
    end_df = (
        gt_pd[gt_pd["event"].isin(end_events)]
        .groupby(["game_id", "play_id", "nfl_id"], as_index=False)["time"]
        .min()
        .rename(columns={"time": "t_end"})
    )
    gt_pd = gt_pd.merge(snap_df, on=["game_id", "play_id", "nfl_id"], how="left")
    gt_pd = gt_pd.merge(end_df, on=["game_id", "play_id", "nfl_id"], how="left")
    gt_pd["t_snap"] = gt_pd["t_snap"].fillna(gt_pd["time"])
    t_cap = gt_pd["t_snap"] + pd.Timedelta(seconds=4.0)
    gt_pd["t_end_cap"] = gt_pd["t_end"].fillna(t_cap).where(gt_pd["t_end"].fillna(t_cap) < t_cap, t_cap)

    rush_window = gt_pd[(gt_pd["time"] >= gt_pd["t_snap"]) & (gt_pd["time"] <= gt_pd["t_end_cap"])].copy()
    total_rush_window_frames = len(rush_window)

    # Apply the EXACT same smooth_trajectory(W=7, p=2) and masks (v > 1.50, kappa > 0.15)
    play_records = []
    for (gid, pid, nid), grp in rush_window.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        if len(grp) < 7:
            continue
        kin = smooth_trajectory(grp["x"].values, grp["y"].values, window=7, polyorder=2)
        v, an, kap = kin["v"], kin["a_n"], kin["kappa"]
        mask = (v > 1.50) & (kap > 0.15)
        if np.any(mask):
            play_records.append({
                "nfl_id": nid,
                "play_p90_an": float(np.percentile(an[mask], 90)),
                "play_mean_an": float(np.mean(an[mask])),
                "play_max_a": float(np.max(kin["a_mag"])),
            })

    play_df = pd.DataFrame(play_records)
    player_game_kin = (
        play_df.groupby("nfl_id", as_index=False)
        .agg(
            tracked_pr_plays=("play_p90_an", "count"),
            ingame_bend_an=("play_p90_an", "mean"),
            ingame_mean_an=("play_mean_an", "mean"),
            ingame_peak_accel=("play_max_a", "mean"),
        )
    )
    return player_game_kin, total_raw_frames, total_rush_window_frames


def build_prospect_and_play_datasets():
    print("Building unified regular-season econometric modeling datasets...")
    games = pl.read_parquet(PROCESSED_DIR / "games.parquet")
    pp = pl.read_parquet(PROCESSED_DIR / "player_play.parquet").join(
        games.select(["game_id", "season", "season_type"]), on="game_id", how="left"
    ).filter(pl.col("season_type") == "REG")

    cf = pl.read_parquet(PROCESSED_DIR / "combine_features.parquet").to_pandas()
    players = pl.read_parquet(PROCESSED_DIR / "players.parquet").to_pandas()[
        ["nfl_id", "display_name", "draft_overall_pick", "nfl_position", "college_name"]
    ]

    dl_prospects = cf[cf["combine_position"] == "DL"].copy()
    dl_prospects = dl_prospects.merge(players[["nfl_id", "college_name"]], on="nfl_id", how="left")
    dl_ids = dl_prospects["nfl_id"].tolist()

    # Regular-season defensive snaps (EDGE + INTERIOR_LINE)
    def_snaps_reg = pp.filter(
        pl.col("nfl_id").is_in(dl_ids) &
        pl.col("lined_up_position").is_in(DL_POS)
    ).group_by("nfl_id").agg([
        pl.len().alias("dl_snaps_reg")
    ]).to_pandas()

    # Play-based regular-season pass-rush snaps (no selection on player_get_off!)
    rush_filter = (
        pl.col("nfl_id").is_in(dl_ids) &
        pl.col("lined_up_position").is_in(DL_POS) &
        pl.col("pass_result").is_not_null() &
        (pl.col("coverage_assignment").is_null() | (pl.col("coverage_assignment") == "NO_ASSIGNMENT"))
    )
    rush_pp = pp.filter(rush_filter).with_columns([
        (pl.col("time_to_pressure") <= 2.5).fill_null(False).cast(pl.Int8).alias("qp_own"),
        pl.col("quick_pressure").fill_null(0).cast(pl.Int8).alias("qp_flag"),
    ])

    pr_all = rush_pp.group_by("nfl_id").agg([
        pl.len().alias("pr_snaps"),
        (pl.col("lined_up_position") == "EDGE").cast(pl.Float64).mean().alias("edge_align_share"),
        (pl.col("lined_up_position") == "EDGE").cast(pl.Int32).sum().alias("edge_pr_snaps"),
        pl.col("qp_flag").filter(pl.col("lined_up_position") == "EDGE").sum().alias("edge_qp"),
        (pl.col("lined_up_position") == "INTERIOR_LINE").cast(pl.Int32).sum().alias("int_pr_snaps"),
        pl.col("qp_flag").filter(pl.col("lined_up_position") == "INTERIOR_LINE").sum().alias("int_qp"),
        pl.col("qp_flag").cast(pl.Float64).sum().alias("pr_quick_pressures"),
        pl.col("qp_own").cast(pl.Float64).sum().alias("pr_qp_own"),
        pl.col("time_to_pressure").is_not_null().cast(pl.Float64).sum().alias("pr_total_pressures"),
        pl.col("sack").fill_null(0).cast(pl.Float64).sum().alias("pr_sacks"),
        pl.col("player_get_off").is_null().cast(pl.Float64).mean().alias("get_off_null_share"),
        pl.col("player_get_off").mean().alias("ingame_get_off"),
    ]).to_pandas()

    # Rookie-season regular-season pass-rush snaps
    rush_draft = rush_pp.join(pl.from_pandas(dl_prospects[["nfl_id", "draft_year"]]), on="nfl_id", how="inner")
    pr_rookie = rush_draft.filter(pl.col("season") == pl.col("draft_year")).group_by("nfl_id").agg([
        pl.len().alias("rookie_pr_snaps"),
        pl.col("qp_flag").cast(pl.Float64).sum().alias("rookie_quick_pressures"),
    ]).to_pandas()

    dl_prospects = (
        dl_prospects.merge(def_snaps_reg, on="nfl_id", how="left")
        .merge(pr_all, on="nfl_id", how="left")
        .merge(pr_rookie, on="nfl_id", how="left")
    )

    # In-game tracking kinematics strictly on rush_pp plays
    rush_keys = rush_pp.select(["game_id", "play_id", "nfl_id"]).unique()
    game_kin_df, total_raw_frames, total_window_frames = compute_ingame_tracking_metrics(dl_ids, rush_keys)
    dl_prospects = dl_prospects.merge(game_kin_df, on="nfl_id", how="left")
    dl_prospects.attrs["total_raw_frames"] = total_raw_frames
    dl_prospects.attrs["total_window_frames"] = total_window_frames

    # Clean covariates & missing indicators
    dl_prospects["is_udfa"] = dl_prospects["draft_overall_pick"].isna().astype(int)
    dl_prospects["draft_pick"] = dl_prospects["draft_overall_pick"].fillna(260.0)
    dl_prospects["log_draft_pick"] = np.log(dl_prospects["draft_pick"])
    dl_prospects["forty_missing"] = dl_prospects["forty"].isna().astype(int)
    dl_prospects["forty_clean"] = dl_prospects["forty"].fillna(dl_prospects["forty"].mean())
    dl_prospects["weight_clean"] = dl_prospects["combine_weight"].fillna(dl_prospects["combine_weight"].mean())

    # Outcomes in football units (per 100 regular-season pass-rush snaps)
    dl_prospects["qp_rate_pr"] = dl_prospects["pr_quick_pressures"] / dl_prospects["pr_snaps"]
    dl_prospects["qp_per_100_pr"] = dl_prospects["qp_rate_pr"] * 100.0
    dl_prospects["total_press_per_100_pr"] = (dl_prospects["pr_total_pressures"] / dl_prospects["pr_snaps"]) * 100.0
    dl_prospects["qp_per_100_def"] = (dl_prospects["pr_quick_pressures"] / dl_prospects["dl_snaps_reg"]) * 100.0
    dl_prospects["rookie_qp_per_100_pr"] = (dl_prospects["rookie_quick_pressures"] / dl_prospects["rookie_pr_snaps"]) * 100.0

    # Qualified modeling cohort (N = 106 with both drills & >= 30 REG defensive snaps)
    qual = dl_prospects[
        (dl_prospects["dl_snaps_reg"] >= 30) &
        dl_prospects["hoop_centripetal_acc"].notna() &
        dl_prospects["pr_first_step_jerk"].notna()
    ].copy()
    assert len(qual) > 0, "Qualified REG cohort is empty!"
    for req_col in ["draft_pick", "forty_clean", "forty_missing", "weight_clean", "edge_align_share", "qp_per_100_pr"]:
        assert qual[req_col].notna().all(), f"NaN found in {req_col}"

    # Audit agreement between quick_pressure and (time_to_pressure <= 2.5) on qualified rush snaps
    q_ids = qual["nfl_id"].tolist()
    q_rush = rush_pp.filter(pl.col("nfl_id").is_in(q_ids))
    qp_agree = float((q_rush["qp_flag"] == q_rush["qp_own"]).mean())
    dl_prospects.attrs["qp_own_agreement"] = qp_agree

    # Standardize strictly within N=106 cohort
    for col, zcol in [
        ("hoop_centripetal_acc", "z_bend"),
        ("pr_first_step_jerk", "z_jerk"),
        ("draft_pick", "z_draft_pick"),
        ("log_draft_pick", "z_log_draft_pick"),
        ("forty_clean", "z_forty"),
        ("weight_clean", "z_weight"),
    ]:
        qual[zcol] = (qual[col] - qual[col].mean()) / qual[col].std()

    qual["prki_sum"] = qual["z_bend"] + qual["z_jerk"]
    qual["z_prki"] = (qual["prki_sum"] - qual["prki_sum"].mean()) / qual["prki_sum"].std()

    # Edge-aligned play-level dataset (REG, EDGE, pass_result!=null, player_get_off!=null) for MixedLM
    edge_plays = rush_pp.filter(
        pl.col("nfl_id").is_in(q_ids) &
        (pl.col("lined_up_position") == "EDGE") &
        pl.col("player_get_off").is_not_null()
    ).to_pandas()

    edge_plays = edge_plays.merge(
        qual[["nfl_id", "z_prki", "z_bend", "z_jerk", "z_draft_pick", "z_forty", "forty_missing", "z_weight"]],
        on="nfl_id", how="inner"
    )
    edge_plays["is_third_down"] = (edge_plays["down"] == 3).fillna(False).astype(float)
    edge_plays["yards_to_go_clean"] = edge_plays["yards_to_go"].fillna(10).clip(1, 25).astype(float)
    edge_plays["z_ytg"] = (edge_plays["yards_to_go_clean"] - edge_plays["yards_to_go_clean"].mean()) / edge_plays["yards_to_go_clean"].std()
    edge_plays["get_off_sec"] = edge_plays["player_get_off"].astype(float)

    print(f"Total Combine DL: {len(dl_prospects)} | Both drills: {dl_prospects['hoop_centripetal_acc'].notna().sum()} | Qualified REG Cohort: N = {len(qual)}")
    return dl_prospects, qual, edge_plays


def partial_pearsonr(x: np.ndarray, y: np.ndarray, cov: np.ndarray):
    """Partial Pearson correlation between x and y controlling for cov (1D or 2D)."""
    C = np.column_stack([np.ones(len(x)), cov])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def run_all_models():
    dl_all, qual, edge_plays = build_prospect_and_play_datasets()
    rel_stats = compute_sensitivity_and_reliability(qual["nfl_id"].tolist())

    # Audit get_off_null_share vs z_prki
    r_null_prki, p_null_prki = stats.pearsonr(qual["get_off_null_share"], qual["z_prki"])

    # 1. Primary OLS Models with HC3 robust SEs (controlling for edge_align_share & forty_missing)
    base_f = "qp_per_100_pr ~ z_draft_pick + z_forty + forty_missing + z_weight + edge_align_share"
    m1_pr = smf.ols(base_f, data=qual).fit(cov_type="HC3")
    m2_pr = smf.ols(f"{base_f} + z_prki", data=qual).fit(cov_type="HC3")
    m2b_pr = smf.ols(f"{base_f} + z_bend + z_jerk", data=qual).fit(cov_type="HC3")

    # Sensitivity subsets: Edge-heavy (edge_align_share >= 0.5), Complete 40-times, Rookie-season, Nonlinear Draft Pick
    edge_heavy = qual[qual["edge_align_share"] >= 0.5].copy()
    m_edge_heavy = smf.ols(f"{base_f} + z_prki", data=edge_heavy).fit(cov_type="HC3")

    comp40 = qual[qual["forty_missing"] == 0].copy()
    m_comp40 = smf.ols("qp_per_100_pr ~ z_draft_pick + z_forty + z_weight + edge_align_share + z_prki", data=comp40).fit(cov_type="HC3")

    rookie_sub = qual.dropna(subset=["rookie_qp_per_100_pr"]).copy()
    m_rookie = smf.ols(f"rookie_qp_per_100_pr ~ z_draft_pick + z_forty + forty_missing + z_weight + edge_align_share + z_prki", data=rookie_sub).fit(cov_type="HC3")

    m_logpick = smf.ols("qp_per_100_pr ~ z_log_draft_pick + is_udfa + z_forty + forty_missing + z_weight + edge_align_share + z_prki", data=qual).fit(cov_type="HC3")

    # 2. Grouped Binomial GLM & Quasi-Binomial (phi-adjusted)
    endog = np.column_stack([qual["pr_quick_pressures"].values, (qual["pr_snaps"] - qual["pr_quick_pressures"]).values])
    base_cols = ["z_draft_pick", "z_forty", "forty_missing", "z_weight", "edge_align_share"]
    X1 = sm.add_constant(qual[base_cols])
    X2 = sm.add_constant(qual[base_cols + ["z_prki"]])
    X2b = sm.add_constant(qual[base_cols + ["z_bend", "z_jerk"]])

    g1 = sm.GLM(endog, X1, family=sm.families.Binomial()).fit()
    g2 = sm.GLM(endog, X2, family=sm.families.Binomial()).fit()
    g2b = sm.GLM(endog, X2b, family=sm.families.Binomial()).fit()

    phi2 = float(g2.pearson_chi2 / g2.df_resid)
    phi2b = float(g2b.pearson_chi2 / g2b.df_resid)
    f_stat_glm = float(((g1.deviance - g2.deviance) / 1.0) / phi2)
    p_f_glm = float(f_dist.sf(f_stat_glm, dfn=1, dfd=int(g2.df_resid)))

    # 3. Snap-Weighted WLS
    wls1 = smf.wls(base_f, data=qual, weights=qual["pr_snaps"]).fit()
    wls2 = smf.wls(f"{base_f} + z_prki", data=qual, weights=qual["pr_snaps"]).fit()
    wls2b = smf.wls(f"{base_f} + z_bend + z_jerk", data=qual, weights=qual["pr_snaps"]).fit()

    # 4. Play-Level Hierarchical MixedLM for NFL get-off translation (z_jerk -> get_off_sec)
    m_mix_base = smf.mixedlm("get_off_sec ~ is_third_down + z_ytg + z_draft_pick + z_forty + forty_missing", edge_plays, groups=edge_plays["nfl_id"]).fit(reml=False)
    m_mix_jerk = smf.mixedlm("get_off_sec ~ is_third_down + z_ytg + z_draft_pick + z_forty + forty_missing + z_jerk", edge_plays, groups=edge_plays["nfl_id"]).fit(reml=False)
    lrt_mix = -2 * (m_mix_base.llf - m_mix_jerk.llf)
    p_lrt_mix = float(chi2.sf(lrt_mix, df=1))
    jerk_go_beta = float(m_mix_jerk.params["z_jerk"])
    jerk_go_se = float(m_mix_jerk.bse["z_jerk"])
    jerk_go_p = float(m_mix_jerk.pvalues["z_jerk"])

    # 5. Repeated 5-Fold x 50 Out-of-Sample Cross-Validation (leak-free Pipeline)
    rkf = RepeatedKFold(n_splits=5, n_repeats=50, random_state=42)
    raw_base_cols = ["draft_pick", "forty_clean", "forty_missing", "weight_clean", "edge_align_share"]
    raw_aug_cols = raw_base_cols + ["prki_sum"]
    X_raw_base = qual[raw_base_cols].values
    X_raw_aug = qual[raw_aug_cols].values
    y_pr = qual["qp_per_100_pr"].values

    r2_base_repeats, r2_aug_repeats, delta_r2_repeats, mse_base_repeats, mse_aug_repeats = [], [], [], [], []
    for rep in range(50):
        oof_b = np.zeros(len(qual))
        oof_a = np.zeros(len(qual))
        # Use 5 folds per repeat
        from sklearn.model_selection import KFold
        kf_rep = KFold(n_splits=5, shuffle=True, random_state=42 + rep)
        for tr, te in kf_rep.split(qual):
            pipe_b = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            pipe_a = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            oof_b[te] = pipe_b.fit(X_raw_base[tr], y_pr[tr]).predict(X_raw_base[te])
            oof_a[te] = pipe_a.fit(X_raw_aug[tr], y_pr[tr]).predict(X_raw_aug[te])
        r2_b = r2_score(y_pr, oof_b)
        r2_a = r2_score(y_pr, oof_a)
        r2_base_repeats.append(r2_b)
        r2_aug_repeats.append(r2_a)
        delta_r2_repeats.append((r2_a - r2_b) * 100.0)
        mse_base_repeats.append(mean_squared_error(y_pr, oof_b))
        mse_aug_repeats.append(mean_squared_error(y_pr, oof_a))

    # 6. Quartile Tier Breakdown (N=106)
    qual["prki_quartile"] = pd.qcut(qual["z_prki"], q=4, labels=["Q1 (Bottom 25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (Top 25%)"])
    q_table = []
    for q_label, grp in qual.groupby("prki_quartile", observed=False):
        mean_val = float(grp["qp_per_100_pr"].mean())
        sd_val = float(grp["qp_per_100_pr"].std())
        n_val = int(len(grp))
        sem_val = sd_val / np.sqrt(n_val)
        q_table.append({
            "quartile": str(q_label),
            "n": n_val,
            "mean_qp_per_100_pr": mean_val,
            "pooled_qp_per_100_pr": float(grp["pr_quick_pressures"].sum() / grp["pr_snaps"].sum() * 100.0),
            "sd_qp_per_100_pr": sd_val,
            "sem_qp_per_100_pr": float(sem_val),
            "ci95_halfwidth": float(1.96 * sem_val),
        })

    # 7. Correlations & Partial Correlations (controlling for edge_align_share)
    sub_40 = qual.dropna(subset=["forty"])
    r_bend_40, p_bend_40 = stats.pearsonr(sub_40["z_bend"], sub_40["forty"])
    r_jerk_40, p_jerk_40 = stats.pearsonr(sub_40["z_jerk"], sub_40["forty"])

    sub_3c = qual.dropna(subset=["three_cone"])
    r_bend_3c, p_bend_3c = stats.pearsonr(sub_3c["z_bend"], sub_3c["three_cone"])
    r_jerk_3c, p_jerk_3c = stats.pearsonr(sub_3c["z_jerk"], sub_3c["three_cone"])

    sub_ig = qual.dropna(subset=["ingame_bend_an"])
    r_prki_ig, p_prki_ig = stats.pearsonr(sub_ig["z_prki"], sub_ig["ingame_bend_an"])
    r_bend_ig, p_bend_ig = stats.pearsonr(sub_ig["z_bend"], sub_ig["ingame_bend_an"])
    r_ig_qp, p_ig_qp = stats.pearsonr(sub_ig["ingame_bend_an"], sub_ig["qp_per_100_pr"])

    pr_bend_ig, pp_bend_ig = partial_pearsonr(sub_ig["z_bend"].values, sub_ig["ingame_bend_an"].values, sub_ig["edge_align_share"].values)
    pr_ig_qp, pp_ig_qp = partial_pearsonr(sub_ig["ingame_bend_an"].values, sub_ig["qp_per_100_pr"].values, sub_ig["edge_align_share"].values)

    def extract_ols_dict(model):
        ci = model.conf_int()
        out = {}
        for k in model.params.index:
            out[k] = {
                "coef": float(model.params[k]),
                "std_err": float(model.bse[k]),
                "t_stat": float(model.tvalues[k]),
                "p_value": float(model.pvalues[k]),
                "ci_low": float(ci.loc[k, 0]),
                "ci_high": float(ci.loc[k, 1]),
            }
        return out

    def extract_qb_dict(model, phi):
        out = {}
        for k in model.params.index:
            b = float(model.params[k])
            se_nom = float(model.bse[k])
            se_qb = float(se_nom * np.sqrt(phi))
            z_qb = b / se_qb
            p_qb = float(2 * (1.0 - norm.cdf(abs(z_qb))))
            out[k] = {
                "log_odds": b,
                "odds_ratio": float(np.exp(b)),
                "std_err_nominal": se_nom,
                "std_err_qb": se_qb,
                "z_stat_qb": float(z_qb),
                "p_value_nominal": float(model.pvalues[k]),
                "p_value": p_qb,
                "or_ci_low": float(np.exp(b - 1.96 * se_qb)),
                "or_ci_high": float(np.exp(b + 1.96 * se_qb)),
            }
        return out

    cohort_cols = [
        "nfl_id", "display_name", "draft_year", "nfl_position", "college_name",
        "draft_round", "draft_overall_pick", "is_udfa", "draft_pick",
        "forty", "forty_missing", "ten_yd_split", "three_cone", "short_shuttle", "combine_weight",
        "hoop_centripetal_acc", "pr_first_step_jerk", "pr_first_step_accel",
        "z_bend", "z_jerk", "prki_sum", "z_prki", "z_draft_pick", "z_forty", "z_weight",
        "dl_snaps_reg", "pr_snaps", "edge_align_share", "pr_quick_pressures", "pr_total_pressures",
        "pr_sacks", "ingame_get_off", "ingame_bend_an", "qp_per_100_pr", "total_press_per_100_pr",
        "qp_per_100_def", "prki_quartile"
    ]
    cohort_records = json.loads(qual[cohort_cols].to_json(orient="records"))

    pooled_pr_rate = float(qual["pr_quick_pressures"].sum() / qual["pr_snaps"].sum() * 100.0)
    pooled_edge_rate = float(qual["edge_qp"].sum() / qual["edge_pr_snaps"].sum() * 100.0)
    pooled_int_rate = float(qual["int_qp"].sum() / qual["int_pr_snaps"].sum() * 100.0)
    pooled_def_rate = float(qual["pr_quick_pressures"].sum() / qual["dl_snaps_reg"].sum() * 100.0)

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "season_type": "REG",
            "total_combine_dl": int(len(dl_all)),
            "edge_de_olb_total": int(dl_all["nfl_position"].isin(["DE", "OLB"]).sum()),
            "interior_dt_nt_total": int(dl_all["nfl_position"].isin(["DT", "NT"]).sum()),
            "pass_rush_drill_participants": int(dl_all["pr_first_step_jerk"].notna().sum()),
            "hoop_drill_participants": int(dl_all["hoop_centripetal_acc"].notna().sum()),
            "three_cone_opt_outs_dl": int(dl_all["three_cone"].isna().sum()),
            "three_cone_opt_out_pct_dl": float(dl_all["three_cone"].isna().mean() * 100),
            "short_shuttle_opt_outs_dl": int(dl_all["short_shuttle"].isna().sum()),
            "short_shuttle_opt_out_pct_dl": float(dl_all["short_shuttle"].isna().mean() * 100),
            "three_cone_opt_out_pct_edge": float(dl_all[dl_all["nfl_position"].isin(["DE", "OLB"])]["three_cone"].isna().mean() * 100),
            "qualified_cohort_n": int(len(qual)),
            "qualified_edge_de_olb_n": int(qual["nfl_position"].isin(["DE", "OLB"]).sum()),
            "qualified_edge_heavy_n": int(len(edge_heavy)),
            "qualified_forty_complete_n": int(len(comp40)),
            "qualified_forty_missing_n": int(qual["forty_missing"].sum()),
            "qualified_udfa_n": int(qual["is_udfa"].sum()),
            "total_defensive_snaps_reg": int(qual["dl_snaps_reg"].sum()),
            "total_pass_rush_snaps": int(qual["pr_snaps"].sum()),
            "total_quick_pressures": int(qual["pr_quick_pressures"].sum()),
            "pooled_qp_per_100_pr": pooled_pr_rate,
            "pooled_edge_qp_per_100": pooled_edge_rate,
            "pooled_int_qp_per_100": pooled_int_rate,
            "pooled_qp_per_100_def": pooled_def_rate,
            "player_mean_qp_per_100_pr": float(qual["qp_per_100_pr"].mean()),
            "qp_own_agreement_pct": float(dl_all.attrs["qp_own_agreement"] * 100.0),
            "get_off_null_share_pct": float((qual["get_off_null_share"] * qual["pr_snaps"]).sum() / qual["pr_snaps"].sum() * 100.0),
            "r_get_off_null_vs_prki": float(r_null_prki),
            "p_get_off_null_vs_prki": float(p_null_prki),
            "total_edge_pass_rush_snaps": int(len(edge_plays)),
            "total_ingame_raw_frames": int(dl_all.attrs["total_raw_frames"]),
            "total_ingame_rush_window_frames": int(dl_all.attrs["total_window_frames"]),
        },
        "reliability_and_correlations": {
            **rel_stats,
            "n_forty": int(len(sub_40)),
            "r_bend_forty": float(r_bend_40), "p_bend_forty": float(p_bend_40),
            "r_jerk_forty": float(r_jerk_40), "p_jerk_forty": float(p_jerk_40),
            "n_three_cone": int(len(sub_3c)),
            "r_bend_three_cone": float(r_bend_3c), "p_bend_three_cone": float(p_bend_3c),
            "r_jerk_three_cone": float(r_jerk_3c), "p_jerk_three_cone": float(p_jerk_3c),
            "n_ingame": int(len(sub_ig)),
            "r_prki_ingame_bend": float(r_prki_ig), "p_prki_ingame_bend": float(p_prki_ig),
            "r_bend_ingame_bend": float(r_bend_ig), "p_bend_ingame_bend": float(p_bend_ig),
            "partial_r_bend_ingame_bend": float(pr_bend_ig), "partial_p_bend_ingame_bend": float(pp_bend_ig),
            "r_ingame_bend_qp": float(r_ig_qp), "p_ingame_bend_qp": float(p_ig_qp),
            "partial_r_ingame_bend_qp": float(pr_ig_qp), "partial_p_ingame_bend_qp": float(pp_ig_qp),
        },
        "ols_pass_rush_snaps": {
            "cov_type": "HC3",
            "baseline_r2": float(m1_pr.rsquared),
            "baseline_adj_r2": float(m1_pr.rsquared_adj),
            "augmented_r2": float(m2_pr.rsquared),
            "augmented_adj_r2": float(m2_pr.rsquared_adj),
            "decoupled_r2": float(m2b_pr.rsquared),
            "decoupled_adj_r2": float(m2b_pr.rsquared_adj),
            "delta_r2": float(m2_pr.rsquared - m1_pr.rsquared),
            "m1_params": extract_ols_dict(m1_pr),
            "m2_params": extract_ols_dict(m2_pr),
            "m2b_params": extract_ols_dict(m2b_pr),
            "sensitivity": {
                "edge_heavy": {"n": int(len(edge_heavy)), "r2": float(m_edge_heavy.rsquared), "params": extract_ols_dict(m_edge_heavy)},
                "complete_forty": {"n": int(len(comp40)), "r2": float(m_comp40.rsquared), "params": extract_ols_dict(m_comp40)},
                "rookie_season": {"n": int(len(rookie_sub)), "snaps": int(rookie_sub["rookie_pr_snaps"].sum()), "r2": float(m_rookie.rsquared), "params": extract_ols_dict(m_rookie)},
                "log_draft_pick": {"n": int(len(qual)), "r2": float(m_logpick.rsquared), "params": extract_ols_dict(m_logpick)},
            },
        },
        "binomial_glm_pass_rush_snaps": {
            "df_resid_m2": int(g2.df_resid),
            "dispersion_phi_m2": phi2,
            "dispersion_phi_m2b": phi2b,
            "deviance_m1": float(g1.deviance),
            "deviance_m2": float(g2.deviance),
            "f_stat_vs_baseline": f_stat_glm,
            "f_p_value_vs_baseline": p_f_glm,
            "g2_params": extract_qb_dict(g2, phi2),
            "g2b_params": extract_qb_dict(g2b, phi2b),
        },
        "wls_pass_rush_snaps": {
            "baseline_r2": float(wls1.rsquared),
            "augmented_r2": float(wls2.rsquared),
            "decoupled_r2": float(wls2b.rsquared),
            "wls2_params": extract_ols_dict(wls2),
            "wls2b_params": extract_ols_dict(wls2b),
        },
        "ingame_getoff_mixedlm": {
            "n_edge_snaps": int(len(edge_plays)),
            "n_players": int(edge_plays["nfl_id"].nunique()),
            "z_jerk_beta": jerk_go_beta,
            "z_jerk_se": jerk_go_se,
            "z_jerk_ci_low": float(jerk_go_beta - 1.96 * jerk_go_se),
            "z_jerk_ci_high": float(jerk_go_beta + 1.96 * jerk_go_se),
            "z_jerk_p": jerk_go_p,
            "lrt_chi2": float(lrt_mix),
            "lrt_p": p_lrt_mix,
        },
        "cross_validation": {
            "method": "RepeatedKFold(5 folds x 50 repeats), in-fold StandardScaler + Ridge(alpha=1.0)",
            "r2_oof_base_mean": float(np.mean(r2_base_repeats)),
            "r2_oof_aug_mean": float(np.mean(r2_aug_repeats)),
            "delta_r2_oof_pp_mean": float(np.mean(delta_r2_repeats)),
            "delta_r2_oof_pp_sd": float(np.std(delta_r2_repeats)),
            "mse_base_mean": float(np.mean(mse_base_repeats)),
            "mse_aug_mean": float(np.mean(mse_aug_repeats)),
            "mse_reduction_pct": float((1.0 - np.mean(mse_aug_repeats) / np.mean(mse_base_repeats)) * 100.0),
        },
        "quartile_breakdown": q_table,
        "cohort_records": cohort_records,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved authoritative model_results.json to processed/ successfully.")
    return results


if __name__ == "__main__":
    run_all_models()
