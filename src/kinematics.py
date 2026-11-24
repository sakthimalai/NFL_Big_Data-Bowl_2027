"""
src/kinematics.py
Biomechanical Signal Processing & Feature Engineering Engine
NFL Big Data Bowl 2027

Implements:
1. Savitzky-Golay trajectory smoothing on 10 Hz optical tracking coordinates (x, y).
2. Numerical derivation of velocity, acceleration, curvature (kappa), centripetal acceleration (a_n), and jerk (j).
3. Extraction of position-specific kinematic metrics across 4 candidate tracks:
   - Track A: WR Break Point Energy Absorption & Re-acceleration Index (BERI)
   - Track B: DL Pass-Rush Bend & Hoop Centripetal Acceleration
   - Track C: DB Backpedal-to-Transition Dwell Latency
   - Track D: OL Lateral Mirror Acceleration Stability
4. Multi-track correlation scan against NFL regular season performance.
"""

import os
import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.signal import savgol_filter

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "processed"

DT = 0.10  # 10 Hz tracking interval (seconds)

def smooth_trajectory(x: np.ndarray, y: np.ndarray, window: int = 7, polyorder: int = 2):
    """
    Applies Savitzky-Golay filter to smooth coordinates and estimate derivatives.
    Window length 7 frames = 0.7s at 10 Hz.
    """
    n = len(x)
    if n < window:
        # If trajectory is shorter than window, fallback to smaller odd window or finite diff
        w = n if n % 2 != 0 else n - 1
        if w >= 3:
            x_hat = savgol_filter(x, window_length=w, polyorder=min(polyorder, w - 1))
            y_hat = savgol_filter(y, window_length=w, polyorder=min(polyorder, w - 1))
        else:
            x_hat, y_hat = x.copy(), y.copy()
    else:
        x_hat = savgol_filter(x, window_length=window, polyorder=polyorder)
        y_hat = savgol_filter(y, window_length=window, polyorder=polyorder)

    # First and second numerical derivatives via central differences
    vx = np.gradient(x_hat, DT)
    vy = np.gradient(y_hat, DT)
    v = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)
    a_mag = np.sqrt(ax**2 + ay**2)

    # Tangential acceleration: projection of acceleration along velocity vector
    # a_tau = (vx * ax + vy * ay) / max(v, 0.01)
    speed_safe = np.maximum(v, 0.05)
    a_tau = (vx * ax + vy * ay) / speed_safe

    # Curvature: kappa = |vx * ay - vy * ax| / (vx^2 + vy^2)^(3/2)
    cross_prod = np.abs(vx * ay - vy * ax)
    denom = np.maximum((vx**2 + vy**2)**1.5, 1e-4)
    kappa = cross_prod / denom
    kappa = np.clip(kappa, 0.0, 5.0)  # Bound extreme numerical curvature at near-zero speed

    # Centripetal / Normal acceleration: a_n = v^2 * kappa
    a_n = (v**2) * kappa

    # Jerk: time derivative of total acceleration magnitude
    jerk = np.gradient(a_mag, DT)

    return {
        "x_hat": x_hat,
        "y_hat": y_hat,
        "vx": vx,
        "vy": vy,
        "v": v,
        "ax": ax,
        "ay": ay,
        "a_mag": a_mag,
        "a_tau": a_tau,
        "kappa": kappa,
        "a_n": a_n,
        "jerk": jerk
    }

def extract_wr_route_metrics(drill_name: str, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Extracts route stem deceleration, cut mechanics, and exit burst for WRs.
    Target drills: COMEBACK, CURL, DAGGER, SLANT, POST_CORNER, GAUNTLET.
    """
    kin = smooth_trajectory(x, y)
    v = kin["v"]
    a_tau = kin["a_tau"]
    kappa = kin["kappa"]
    
    # Peak entry speed in the initial sprint phase (stem)
    n = len(v)
    if n < 8:
        return {}
    
    # Deceleration phase: minimum (most negative) tangential acceleration
    # Exclude first 0.5s (start of drill) and last 0.3s (end of drill)
    valid_idx = slice(min(5, n // 4), max(n - 3, n // 2))
    a_tau_mid = a_tau[valid_idx]
    v_mid = v[valid_idx]
    kappa_mid = kappa[valid_idx]
    
    if len(a_tau_mid) == 0:
        return {}
        
    peak_decel = float(np.min(a_tau_mid)) # Most negative deceleration rate (yd/s^2)
    max_entry_speed = float(np.max(v[:n // 2])) if n // 2 > 0 else float(np.max(v))
    
    # Apex of the cut: frame of maximum curvature or minimum speed during deceleration
    # We identify the cut apex where curvature is highest under deceleration
    decel_mask = a_tau_mid < -0.5
    if np.any(decel_mask):
        cut_rel_idx = np.argmax(kappa_mid)
        cut_idx = min(5, n // 4) + cut_rel_idx
    else:
        cut_idx = int(np.argmin(v_mid)) + min(5, n // 4)
        
    v_cut_apex = float(v[cut_idx])
    
    # Exit burst: maximum speed in the 1.0s (10 frames) following the cut apex
    exit_window = slice(cut_idx + 1, min(cut_idx + 11, n))
    if exit_window.stop > exit_window.start:
        v_exit_peak = float(np.max(v[exit_window]))
        exit_accel = float(np.max(a_tau[exit_window]))
    else:
        v_exit_peak = v_cut_apex
        exit_accel = 0.0
        
    # Re-acceleration ratio: exit speed relative to cut apex
    reaccel_ratio = v_exit_peak / max(v_cut_apex, 0.5)
    
    # Break Energy Retention Index (BERI):
    # Quantifies efficiency: high decel into cut + fast exit burst relative to entry speed
    # BERI = ( |peak_decel| * (v_exit_peak / max_entry_speed) )
    speed_retention = v_exit_peak / max(max_entry_speed, 1.0)
    beri_score = abs(peak_decel) * speed_retention
    
    return {
        "wr_max_entry_speed": max_entry_speed,
        "wr_peak_deceleration": abs(peak_decel),
        "wr_cut_apex_speed": v_cut_apex,
        "wr_exit_peak_speed": v_exit_peak,
        "wr_exit_accel": exit_accel,
        "wr_speed_retention": speed_retention,
        "wr_beri_score": beri_score,
        "wr_max_curvature": float(np.max(kappa_mid)),
    }

def extract_dl_hoop_metrics(drill_name: str, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Extracts pass-rush bend, centripetal acceleration, and get-off burst for DL/EDGE.
    Drills: RUN_THE_HOOP_DRILL, PASS_RUSH_DRILL, FOUR_BAG_AGILITY_DRILL.
    """
    kin = smooth_trajectory(x, y)
    v = kin["v"]
    a_n = kin["a_n"]
    kappa = kin["kappa"]
    jerk = kin["jerk"]
    n = len(v)
    if n < 8:
        return {}
        
    # First-step get-off burst: peak jerk and peak acceleration in initial 0.8s (8 frames)
    initial_idx = slice(0, min(8, n))
    first_step_jerk = float(np.max(jerk[initial_idx]))
    first_step_accel = float(np.max(kin["a_mag"][initial_idx]))
    
    # Cornering / Bend metrics around the hoop (high curvature zone)
    # Filter to frames where player is active and turning (kappa > 0.10)
    turning_mask = (kappa > 0.10) & (v > 1.5)
    if np.any(turning_mask):
        peak_centripetal_acc = float(np.max(a_n[turning_mask]))
        mean_centripetal_acc = float(np.mean(a_n[turning_mask]))
        bend_speed = float(np.mean(v[turning_mask]))
        max_curvature = float(np.max(kappa[turning_mask]))
    else:
        peak_centripetal_acc = float(np.max(a_n))
        mean_centripetal_acc = float(np.mean(a_n))
        bend_speed = float(np.mean(v))
        max_curvature = float(np.max(kappa))
        
    # Pass-Rush Bend Efficiency (PRBE) = Sustained Centripetal Acc * Bend Speed
    prbe_score = mean_centripetal_acc * bend_speed
    
    return {
        "dl_first_step_jerk": first_step_jerk,
        "dl_first_step_accel": first_step_accel,
        "dl_peak_centripetal_acc": peak_centripetal_acc,
        "dl_mean_centripetal_acc": mean_centripetal_acc,
        "dl_bend_speed": bend_speed,
        "dl_max_curvature": max_curvature,
        "dl_prbe_score": prbe_score,
    }

def extract_db_transition_metrics(drill_name: str, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Extracts backpedal-to-drive transition dwell time and redirect efficiency for DBs.
    Drills: BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION, BACK_PEDAL_AND_90_DEGREE_BREAK, W_DRILL.
    """
    kin = smooth_trajectory(x, y)
    v = kin["v"]
    a_tau = kin["a_tau"]
    n = len(v)
    if n < 8:
        return {}
        
    # DB backpedal drill starts with backward pedal, hits break point, drives forward.
    # The break transition occurs where speed reaches local minimum while changing direction.
    mid_start = min(5, n // 4)
    mid_end = max(n - 5, 3 * n // 4)
    if mid_end <= mid_start:
        return {}
        
    v_mid = v[mid_start:mid_end]
    min_speed_idx = mid_start + int(np.argmin(v_mid))
    
    # Transition dwell time: number of frames speed stays below 2.0 yd/s during cut
    low_speed_frames = np.sum(v[max(0, min_speed_idx - 4): min(n, min_speed_idx + 5)] < 2.5)
    dwell_time_sec = float(low_speed_frames * DT)
    
    # Deceleration rate into transition and acceleration rate out
    entry_decel = float(np.min(a_tau[max(0, min_speed_idx - 5): min_speed_idx + 1]))
    exit_accel = float(np.max(a_tau[min_speed_idx: min(n, min_speed_idx + 6)])) if min_speed_idx < n - 1 else 0.0
    
    # Click-and-Close Index (CCI) = Exit Accel / (Dwell Time + 0.1)
    cci_score = exit_accel / (dwell_time_sec + 0.1)
    
    return {
        "db_transition_dwell_sec": dwell_time_sec,
        "db_entry_decel": abs(entry_decel),
        "db_exit_accel": exit_accel,
        "db_cci_score": cci_score,
    }

def extract_ol_mirror_metrics(drill_name: str, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Extracts lateral agility, mirror stability, and burst for OL.
    """
    kin = smooth_trajectory(x, y)
    v = kin["v"]
    a_mag = kin["a_mag"]
    n = len(v)
    if n < 8:
        return {}
        
    peak_lateral_acc = float(np.max(a_mag))
    mean_speed = float(np.mean(v))
    lateral_burst_jerk = float(np.max(kin["jerk"]))
    
    return {
        "ol_peak_lateral_acc": peak_lateral_acc,
        "ol_mean_speed": mean_speed,
        "ol_lateral_burst_jerk": lateral_burst_jerk,
    }

