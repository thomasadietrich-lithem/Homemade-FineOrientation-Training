import os
import re
import glob
import json
import html
import webbrowser
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# Huxlin-style standalone dashboard for Fine Orientation / Gabor MetaCog CSV files
# ============================================================
#
# Scope:
# - descriptive analysis only; no automatic clinical decision rule
# - one psychometric Weibull fit per session
# - longitudinal dashboard per trained retinotopic location
# - within-session block performance
# - LEFT vs RIGHT orientation comparison
# - staircase trace / distribution / RT-fatigue summaries
#
# Reference lineage:
# - TestingCodes-main/FineOrientation/FineOrientationDiscrimination_MetaCog.m
# - DataFitting-style psychometric fitting conventions
#
# Designed for outputs from:
# - FineOrientation_GaborPatch_MetaCog_PsychoPy_v4_4_no_metacog_default.py
#
# Important:
# - The practical default exercise may disable metacognition. This analyser therefore
#   treats confidence and score as optional metadata, not core performance metrics.
# - Bonus/catch trials (staircase == 0) are shown but excluded by default from
#   threshold fitting, because they do not update the 3 adaptive staircases.
#

CHANCE = 0.5
LAPSE_RATE = 0.05
THRESHOLD_TARGET = 0.725

MIN_TRIALS_FOR_SESSION_FIT = 80
MIN_UNIQUE_LEVELS_FOR_SESSION_FIT = 4
N_BLOCKS_WITHIN_SESSION = 6
RECENT_SESSION_WINDOW = 5
POOLED_THRESHOLD_WINDOW = 3

# The task scale used by the Rochester FineOrientation script.
BASE_ANGLE_DEG = [53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5, 0.1]

# Include bonus trials in descriptive accuracy but exclude from fitted thresholds.
FIT_EXCLUDES_BONUS_TRIALS = True


def make_writable_output_path(path):
    base, ext = os.path.splitext(path)
    candidate = path
    idx = 1
    while True:
        try:
            parent = os.path.dirname(candidate)
            os.makedirs(parent, exist_ok=True)
            with open(candidate, "ab"):
                pass
            return candidate
        except Exception:
            candidate = f"{base}_{idx}{ext}"
            idx += 1


def parse_timestamp_from_name(path):
    name = os.path.basename(path)
    m = re.search(r'_(\d{8}_\d{6})_(?:results|summary)\.', name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None


def companion_summary_path(csv_path):
    # Typical v4_4 name:
    # subject_OriMetaCog_H_V_YYYYMMDD_HHMMSS_results.csv
    return csv_path.replace("_results.csv", "_summary.json")


def load_companion_summary(csv_path):
    p = companion_summary_path(csv_path)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_csvs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    patterns = [
        os.path.join(cwd, "DataFineOrientation", "**", "*_results.csv"),
        os.path.join(script_dir, "DataFineOrientation", "**", "*_results.csv"),
        os.path.join(cwd, "*_results.csv"),
        os.path.join(script_dir, "*_results.csv"),
    ]
    for patt in patterns:
        found = sorted(glob.glob(patt, recursive=True))
        if found:
            return found
    return []


def output_directory_for(csv_paths):
    first_dir = os.path.dirname(os.path.abspath(csv_paths[0]))
    # Prefer placing analysis next to DataFineOrientation root if possible.
    parts = os.path.abspath(first_dir).split(os.sep)
    if "DataFineOrientation" in parts:
        idx = parts.index("DataFineOrientation")
        root = os.sep.join(parts[:idx+1])
        candidate = os.path.join(root, "analysis_results")
    else:
        candidate = os.path.join(first_dir, "analysis_results")
    try:
        os.makedirs(candidate, exist_ok=True)
        testfile = os.path.join(candidate, ".write_test")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return candidate
    except Exception:
        fallback = os.path.join(first_dir, "analysis_results")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def infer_location_id(csv_path):
    meta = load_companion_summary(csv_path)
    if meta:
        loc = meta.get("stimulus_location_deg") or {}
        try:
            h = float(loc.get("H"))
            v = float(loc.get("V"))
            return f"H{h:+.2f}_V{v:+.2f}", meta
        except Exception:
            pass
    # Fallback parse from name: subject_OriMetaCog_H_V_timestamp
    name = os.path.basename(csv_path)
    m = re.search(r'_OriMetaCog_(-?\d+)_(-?\d+)_\d{8}_\d{6}_results\.csv$', name)
    if m:
        return f"H{float(m.group(1)):+.2f}_V{float(m.group(2)):+.2f}", meta
    return "unknown_location", meta


# -------------------------------------------------------------------------
# Weibull fitting
# -------------------------------------------------------------------------

def weibull_python_true_725(x_log, alpha_log, beta):
    """Python explicit 72.5% threshold convention.

    This is the mathematically direct convention:
        threshold_python_true_725 = angle where p(correct)=0.725.
    """
    xx = np.maximum(np.asarray(x_log, dtype=float), 1e-9)
    aa = max(float(alpha_log), 1e-9)
    return CHANCE + (1.0 - CHANCE - LAPSE_RATE) * (1.0 - np.exp(- (xx / aa) ** beta))


def threshold_from_python_params(alpha_log, beta):
    scaled = (THRESHOLD_TARGET - CHANCE) / (1.0 - CHANCE - LAPSE_RATE)
    scaled = np.clip(scaled, 1e-9, 1.0 - 1e-9)
    x_thr_log = alpha_log * ((-np.log(1.0 - scaled)) ** (1.0 / beta))
    thr_deg = (10.0 ** x_thr_log) - 1.0
    return max(0.0, float(thr_deg)), float(x_thr_log)


def matlab_k(beta):
    """Replicate the threshold scaling convention in Rochester/DataFitting-style weibull.m.

    In the MATLAB code, k is based on target/chance and does not include lapse in
    the same way as the direct 72.5% extraction. Therefore the returned threshold
    parameter is slightly lower than the direct 72.5% point when lapse > 0.
    """
    return (-np.log((1.0 - THRESHOLD_TARGET) / (1.0 - CHANCE))) ** (1.0 / beta)


def weibull_rochester_exact(x_log, thresh_log, beta):
    """Rochester/DataFitting-compatible Weibull convention."""
    x = np.maximum(np.asarray(x_log, dtype=float), 1e-9)
    thr = max(float(thresh_log), 1e-9)
    k = matlab_k(beta)
    return 1.0 - LAPSE_RATE - (1.0 - CHANCE - LAPSE_RATE) * np.exp(-((k * x / thr) ** beta))


def neg_log_likelihood_python(params, x_log, y_bin):
    alpha_log, beta = params
    p = np.clip(weibull_python_true_725(x_log, alpha_log, beta), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_bin, dtype=float)
    return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def neg_log_likelihood_rochester(params, x_log, y_bin):
    thresh_log, beta = params
    p = np.clip(weibull_rochester_exact(x_log, thresh_log, beta), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_bin, dtype=float)
    return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def bernoulli_log_likelihood(y_bin, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_bin, dtype=float)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_quality_metrics(x_log, y_bin, pred_p):
    ll_model = bernoulli_log_likelihood(y_bin, pred_p)
    empirical_p = float(np.mean(y_bin)) if len(y_bin) else CHANCE
    empirical_p = float(np.clip(empirical_p, 1e-6, 1.0 - 1e-6))
    ll_null = bernoulli_log_likelihood(y_bin, np.full_like(y_bin, empirical_p, dtype=float))
    pseudo_r2 = None
    if ll_null != 0 and np.isfinite(ll_null):
        pseudo_r2 = 1.0 - (ll_model / ll_null)
    n = int(len(y_bin)); k = 2
    return {
        "fit_log_likelihood": float(ll_model),
        "null_log_likelihood": float(ll_null),
        "fit_pseudo_r2": float(pseudo_r2) if pseudo_r2 is not None and np.isfinite(pseudo_r2) else None,
        "fit_aic": float(2 * k - 2 * ll_model),
        "fit_bic": float(k * np.log(max(n, 1)) - 2 * ll_model),
    }


def _fit_common(objective, x_log, y_bin):
    xmin = max(1e-6, float(np.min(x_log)))
    xmax = max(xmin + 1e-6, float(np.max(x_log)))
    starts = [
        (np.median(x_log), 2.0),
        (xmax * 0.75, 2.0),
        (xmax * 0.90, 3.0),
        (max(xmin, xmax * 0.50), 1.5),
        (np.median(x_log), 5.0),
    ]
    best = None
    for a0, b0 in starts:
        try:
            res = minimize(
                objective,
                x0=np.array([a0, b0], dtype=float),
                args=(x_log, y_bin),
                method="L-BFGS-B",
                bounds=[(xmin, xmax), (0.25, 12.0)],
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return best


def fit_weibulls(dd_fit):
    if len(dd_fit) < MIN_TRIALS_FOR_SESSION_FIT:
        return {"fit_valid": False, "fit_failure_reason": f"too_few_trials (<{MIN_TRIALS_FOR_SESSION_FIT})"}
    if dd_fit["difficulty_abs_deg"].nunique() < MIN_UNIQUE_LEVELS_FOR_SESSION_FIT:
        return {"fit_valid": False, "fit_failure_reason": f"too_few_unique_levels (<{MIN_UNIQUE_LEVELS_FOR_SESSION_FIT})"}

    x_log = np.log10(dd_fit["difficulty_abs_deg"].to_numpy(dtype=float) + 1.0)
    y_bin = dd_fit["correct"].to_numpy(dtype=float)
    observed_max = float(np.max(dd_fit["difficulty_abs_deg"]))

    best_py = _fit_common(neg_log_likelihood_python, x_log, y_bin)
    best_roch = _fit_common(neg_log_likelihood_rochester, x_log, y_bin)
    if best_py is None or not np.isfinite(best_py.fun) or best_roch is None or not np.isfinite(best_roch.fun):
        return {"fit_valid": False, "fit_failure_reason": "optimizer_failed"}

    alpha_log, beta_py = float(best_py.x[0]), float(best_py.x[1])
    thr_py_deg, thr_py_log = threshold_from_python_params(alpha_log, beta_py)

    thr_roch_log, beta_roch = float(best_roch.x[0]), float(best_roch.x[1])
    thr_roch_deg = max(0.0, float((10.0 ** thr_roch_log) - 1.0))

    if not np.isfinite(thr_py_deg) or not np.isfinite(thr_roch_deg):
        return {"fit_valid": False, "fit_failure_reason": "non_finite_threshold"}

    # QC: descriptive. Keep both failures explicit.
    max_reasonable = max(observed_max * 1.5, observed_max + 5)
    if thr_py_deg < 0 or thr_py_deg > max_reasonable or thr_roch_deg < 0 or thr_roch_deg > max_reasonable:
        return {
            "fit_valid": False,
            "fit_failure_reason": "threshold_outside_reasonable_session_range",
            "threshold_python_true_725_deg_raw": thr_py_deg,
            "threshold_rochester_exact_deg_raw": thr_roch_deg,
        }

    pred_roch = weibull_rochester_exact(x_log, thr_roch_log, beta_roch)
    pred_py = weibull_python_true_725(x_log, alpha_log, beta_py)

    delta_deg = thr_py_deg - thr_roch_deg
    delta_percent = (delta_deg / thr_roch_deg * 100.0) if thr_roch_deg > 0 else None

    out = {
        "fit_valid": True,
        "fit_failure_reason": None,

        # Main Rochester-comparable values.
        "threshold_rochester_exact_deg": float(thr_roch_deg),
        "threshold_rochester_exact_log10_plus1": float(thr_roch_log),
        "beta_rochester_exact": float(beta_roch),

        # Direct mathematical 72.5% value.
        "threshold_python_true_725_deg": float(thr_py_deg),
        "threshold_python_true_725_log10_plus1": float(thr_py_log),
        "alpha_log_python": float(alpha_log),
        "beta_python": float(beta_py),

        # For backward dashboard compatibility.
        "threshold_deg": float(thr_roch_deg),
        "threshold_log10_plus1": float(thr_roch_log),
        "alpha_log": float(thr_roch_log),
        "beta": float(beta_roch),

        "threshold_delta_python_minus_rochester_deg": float(delta_deg),
        "threshold_delta_python_minus_rochester_percent": float(delta_percent) if delta_percent is not None and np.isfinite(delta_percent) else None,
    }
    out.update({f"rochester_{k}": v for k, v in fit_quality_metrics(x_log, y_bin, pred_roch).items()})
    out.update({f"python_{k}": v for k, v in fit_quality_metrics(x_log, y_bin, pred_py).items()})
    # Also provide generic metrics for table compatibility.
    out.update(fit_quality_metrics(x_log, y_bin, pred_roch))
    return out


# -------------------------------------------------------------------------
# Data loading / cleaning
# -------------------------------------------------------------------------

def clean_trial_frame(csv_path):
    df = pd.read_csv(csv_path)
    required = {"trial", "difficulty_abs_deg", "correct", "staircase"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing} in {os.path.basename(csv_path)}")

    keep = [
        "trial", "difficulty_abs_deg", "response_time_s", "signed_angle_deg", "correct",
        "confidence", "staircase", "response_key", "running_score",
        # optional enriched v4_4 columns
        "is_bonus_trial", "target_side", "response_side", "points_delta",
        "stair_step_pre", "stair_step_post", "stair_value_pre_deg", "stair_value_post_deg",
        "response_phase", "stimulus_duration_actual_s", "mask_delay_actual_s", "iti_actual_s",
    ]
    keep = [c for c in keep if c in df.columns]
    dd = df[keep].copy()
    dd = dd.replace([np.inf, -np.inf], np.nan)

    numeric_cols = [
        "trial", "difficulty_abs_deg", "response_time_s", "signed_angle_deg", "correct",
        "confidence", "staircase", "running_score", "is_bonus_trial", "points_delta",
        "stair_step_pre", "stair_step_post", "stair_value_pre_deg", "stair_value_post_deg",
        "stimulus_duration_actual_s", "mask_delay_actual_s", "iti_actual_s",
    ]
    for c in numeric_cols:
        if c in dd.columns:
            dd[c] = pd.to_numeric(dd[c], errors="coerce")

    dd["correct"] = dd["correct"].replace({
        True: 1, False: 0,
        "True": 1, "False": 0, "true": 1, "false": 0,
        "correct": 1, "incorrect": 0, "Correct": 1, "Incorrect": 0,
    })
    dd["correct"] = pd.to_numeric(dd["correct"], errors="coerce")

    dd = dd.dropna(subset=["trial", "difficulty_abs_deg", "correct", "staircase"]).copy()
    dd = dd[(dd["difficulty_abs_deg"] >= 0) & (dd["correct"].isin([0, 1]))].copy()
    dd["correct"] = dd["correct"].astype(float)
    dd["trial"] = dd["trial"].astype(int)
    dd["staircase"] = dd["staircase"].astype(int)
    if "is_bonus_trial" not in dd.columns:
        dd["is_bonus_trial"] = (dd["staircase"] == 0).astype(int)
    else:
        dd["is_bonus_trial"] = dd["is_bonus_trial"].fillna((dd["staircase"] == 0).astype(int)).astype(int)
    dd = dd.sort_values("trial")
    if len(dd) == 0:
        raise ValueError(f"No valid trial rows after cleaning {os.path.basename(csv_path)}")
    return dd


def stimulus_class_from_trials(dd):
    if "signed_angle_deg" in dd.columns and dd["signed_angle_deg"].notna().any():
        return pd.Series(np.where(dd["signed_angle_deg"].astype(float) < 0, "LEFT", "RIGHT"), index=dd.index)
    if "target_side" in dd.columns:
        return dd["target_side"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(dd), index=dd.index)


def fit_session(csv_path):
    dd = clean_trial_frame(csv_path)
    timestamp = parse_timestamp_from_name(csv_path)
    location_id, meta = infer_location_id(csv_path)

    dd_fit = dd[dd["is_bonus_trial"] == 0].copy() if FIT_EXCLUDES_BONUS_TRIALS else dd.copy()

    result = {
        "source_csv": os.path.basename(csv_path),
        "source_csv_abs": os.path.abspath(csv_path),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else None,
        "location_id": location_id,
        "n_trials": int(len(dd)),
        "n_fit_trials": int(len(dd_fit)),
        "n_bonus_trials": int((dd["is_bonus_trial"] == 1).sum()),
        "n_unique_levels": int(dd_fit["difficulty_abs_deg"].nunique()) if len(dd_fit) else 0,
        "accuracy_percent": float(dd["correct"].mean() * 100.0),
        "accuracy_fit_trials_percent": float(dd_fit["correct"].mean() * 100.0) if len(dd_fit) else None,
        "fit_valid": False,
        "fit_failure_reason": None,
        "threshold_deg": None,
        "threshold_rochester_exact_deg": None,
        "threshold_python_true_725_deg": None,
        "threshold_delta_python_minus_rochester_percent": None,
        "beta_rochester_exact": None,
        "beta_python": None,
        "percent_correct_easiest": None,
        "percent_correct_table": [],
        "block_accuracy": [],
        "stimulus_class_accuracy": [],
        "rt_summary": {},
        "staircase_summary": {},
        "protocol_warnings": [],
    }

    if meta is not None:
        result["companion_summary"] = os.path.basename(companion_summary_path(csv_path))
        result["subject"] = meta.get("subject")
        result["stimulus_location_deg"] = meta.get("stimulus_location_deg")
        result["fixation_location_deg"] = meta.get("fixation_location_deg")
        result["metacognition_enabled"] = meta.get("metacognition_enabled")
        result["iso_reference_mode"] = meta.get("iso_reference_mode")
        result["final_running_score"] = meta.get("final_running_score")
        result["session_total_s"] = meta.get("session_total_s")
        result["protocol_constants"] = meta.get("protocol_constants")
        result["monitor_geometry"] = meta.get("monitor_geometry")
        result["n_trials_requested"] = meta.get("n_trials_requested")
        result["n_trials_completed_summary"] = meta.get("n_trials_completed")
        result["aborted"] = meta.get("aborted")
        result["error"] = meta.get("error")
        if meta.get("aborted"):
            result["protocol_warnings"].append("session_aborted")
        if meta.get("error"):
            result["protocol_warnings"].append("session_error")
        if meta.get("n_trials_completed") is not None and meta.get("n_trials_requested") is not None:
            if int(meta.get("n_trials_completed")) < int(meta.get("n_trials_requested")):
                result["protocol_warnings"].append("partial_session")
        try:
            spm = (meta.get("monitor_geometry") or {}).get("screen_profile_metadata") or {}
            if spm.get("resolution_changed"):
                result["protocol_warnings"].append("screen_resolution_changed")
        except Exception:
            pass

    # Percent correct by stimulus level for fit trials.
    grouped = dd_fit.groupby("difficulty_abs_deg")["correct"].agg(["mean", "count"]).reset_index().sort_values("difficulty_abs_deg")
    for _, row in grouped.iterrows():
        result["percent_correct_table"].append({
            "angle_deg": float(row["difficulty_abs_deg"]),
            "angle_plus1_deg": float(row["difficulty_abs_deg"] + 1.0),
            "n": int(row["count"]),
            "percent_correct": float(row["mean"] * 100.0),
            "proportion_correct": float(row["mean"]),
        })
    if len(grouped) > 0:
        # Easiest = largest angle.
        result["percent_correct_easiest"] = float(grouped.iloc[-1]["mean"] * 100.0)

    # Block accuracy for all trials and fit trials.
    if len(dd) > 0:
        block_ids = np.floor(np.linspace(0, N_BLOCKS_WITHIN_SESSION, len(dd), endpoint=False)).astype(int) + 1
        block_ids = np.clip(block_ids, 1, N_BLOCKS_WITHIN_SESSION)
        dd_blocks = dd.copy()
        dd_blocks["block"] = block_ids
        btab = dd_blocks.groupby("block")["correct"].agg(["mean", "count"]).reset_index()
        for _, row in btab.iterrows():
            result["block_accuracy"].append({
                "block": int(row["block"]),
                "n": int(row["count"]),
                "accuracy_percent": float(row["mean"] * 100.0),
            })

    labels = stimulus_class_from_trials(dd)
    dd_cls = dd.copy()
    dd_cls["stimulus_class"] = labels
    ctab = dd_cls.groupby("stimulus_class")["correct"].agg(["mean", "count"]).reset_index()
    for _, row in ctab.iterrows():
        result["stimulus_class_accuracy"].append({
            "stimulus_class": str(row["stimulus_class"]),
            "n": int(row["count"]),
            "accuracy_percent": float(row["mean"] * 100.0),
        })

    # RT summary.
    if "response_time_s" in dd.columns and dd["response_time_s"].notna().any():
        rt = dd["response_time_s"].dropna().astype(float)
        result["rt_summary"] = {
            "rt_mean_s": float(rt.mean()),
            "rt_median_s": float(rt.median()),
            "rt_std_s": float(rt.std(ddof=0)),
            "rt_min_s": float(rt.min()),
            "rt_max_s": float(rt.max()),
        }
        if len(dd) > 0:
            first = dd.iloc[:max(1, len(dd)//6)]
            last = dd.iloc[-max(1, len(dd)//6):]
            if "response_time_s" in first.columns and "response_time_s" in last.columns:
                result["rt_summary"]["rt_first_block_median_s"] = float(first["response_time_s"].median())
                result["rt_summary"]["rt_last_block_median_s"] = float(last["response_time_s"].median())
                result["rt_summary"]["rt_last_minus_first_median_s"] = float(last["response_time_s"].median() - first["response_time_s"].median())

    # Staircase summary.
    stair_summary = {}
    for stair, sub in dd.groupby("staircase"):
        key = f"staircase_{int(stair)}"
        stair_summary[key] = {
            "n": int(len(sub)),
            "accuracy_percent": float(sub["correct"].mean() * 100.0),
            "first_level_deg": float(sub["difficulty_abs_deg"].iloc[0]),
            "last_level_deg": float(sub["difficulty_abs_deg"].iloc[-1]),
            "mean_level_deg": float(sub["difficulty_abs_deg"].mean()),
            "n_unique_levels": int(sub["difficulty_abs_deg"].nunique()),
        }
        if "stair_step_post" in sub.columns and sub["stair_step_post"].notna().any():
            stair_summary[key]["last_stair_step_post"] = float(sub["stair_step_post"].dropna().iloc[-1])
        if "stair_value_post_deg" in sub.columns and sub["stair_value_post_deg"].notna().any():
            stair_summary[key]["last_stair_value_post_deg"] = float(sub["stair_value_post_deg"].dropna().iloc[-1])
    result["staircase_summary"] = stair_summary

    # Fit.
    fit = fit_weibulls(dd_fit)
    result.update(fit)
    return result


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def add_secondary_degree_axis(ax):
    def forward_log_to_deg(x):
        return (10.0 ** np.asarray(x)) - 1.0
    def inverse_deg_to_log(x):
        x = np.asarray(x)
        return np.log10(np.maximum(x, 0.0) + 1.0)
    secax = ax.secondary_xaxis('top', functions=(forward_log_to_deg, inverse_deg_to_log))
    secax.set_xlabel("orientation difference (deg)")
    deg_ticks = [0.1, 0.5, 0.8, 1.2, 2, 3.2, 5.1, 8.1, 12.97, 20.75, 33.2, 53.1]
    xlo, xhi = ax.get_xlim()
    allowed = [d for d in deg_ticks if xlo <= np.log10(d + 1.0) <= xhi]
    if allowed:
        secax.set_xticks([np.log10(d + 1.0) for d in allowed])
        secax.set_xticklabels([f"{d:g}" for d in allowed])
    return secax


def save_session_plot(session_result, out_dir):
    csv_path = session_result["source_csv_abs"]
    out_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_results", "")
    out_png = make_writable_output_path(os.path.join(out_dir, f"{out_base}_psychometric_weibull.png"))
    out_json = make_writable_output_path(os.path.join(out_dir, f"{out_base}_session_summary.json"))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(session_result, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    tbl = session_result.get("percent_correct_table", [])
    if tbl:
        xs = np.array([r["angle_plus1_deg"] for r in tbl], dtype=float)
        ys = np.array([r["percent_correct"] for r in tbl], dtype=float)
        ns = np.array([r["n"] for r in tbl], dtype=int)
        xlog = np.log10(xs)
        ax.scatter(xlog, ys, s=np.clip(ns * 8, 30, 220), alpha=0.9, label="observed")
        for x, y, n in zip(xlog, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        if session_result.get("fit_valid"):
            grid_deg = np.linspace(max(0, min(xs) - 1.0), max(xs) - 1.0, 400)
            grid_log = np.log10(grid_deg + 1.0)

            curve_roch = weibull_rochester_exact(
                grid_log,
                session_result["threshold_rochester_exact_log10_plus1"],
                session_result["beta_rochester_exact"],
            ) * 100.0
            ax.plot(grid_log, curve_roch, linewidth=2, label="Rochester exact Weibull")

            curve_py = weibull_python_true_725(
                grid_log,
                session_result["alpha_log_python"],
                session_result["beta_python"],
            ) * 100.0
            ax.plot(grid_log, curve_py, linestyle=":", linewidth=1.8, label="Python true 72.5% Weibull")

            thr_roch = session_result["threshold_rochester_exact_deg"]
            thr_roch_log = np.log10(thr_roch + 1.0)
            ax.axvline(thr_roch_log, linestyle="--", linewidth=1)
            ax.annotate(f"Roch={thr_roch:.2f}°", (thr_roch_log, THRESHOLD_TARGET * 100.0),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)

            thr_py = session_result["threshold_python_true_725_deg"]
            thr_py_log = np.log10(thr_py + 1.0)
            ax.axvline(thr_py_log, linestyle=":", linewidth=1)
            ax.annotate(f"72.5={thr_py:.2f}°", (thr_py_log, THRESHOLD_TARGET * 100.0),
                        textcoords="offset points", xytext=(6, -14), fontsize=9)

        title = (
            f"{session_result['source_csv']} | {session_result['location_id']}\n"
            f"trials={session_result['n_trials']} | fit trials={session_result['n_fit_trials']} | "
            f"levels={session_result['n_unique_levels']}"
        )
    else:
        title = f"{session_result['source_csv']} | no usable data"
    ax.set_xlabel("log10(angle + 1)")
    ax.set_ylabel("percent correct (%)")
    ax.set_ylim(0, 102)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(fontsize=8, loc="best")
    add_secondary_degree_axis(ax)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png, out_json


def save_staircase_plot(session_result, out_dir):
    csv_path = session_result["source_csv_abs"]
    out_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_results", "")
    out_png = make_writable_output_path(os.path.join(out_dir, f"{out_base}_staircase_trace.png"))
    try:
        dd = clean_trial_frame(csv_path)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for stair, sub in dd.groupby("staircase"):
        label = "bonus/catch" if int(stair) == 0 else f"staircase {int(stair)}"
        ax.plot(sub["trial"], sub["difficulty_abs_deg"], marker="o", markersize=2.6, linewidth=0.8, label=label)
    ax.set_xlabel("trial")
    ax.set_ylabel("orientation difference (deg)")
    ax.set_title(f"Staircase trace | {session_result['source_csv']} | {session_result['location_id']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_distribution_plot(session_result, out_dir):
    csv_path = session_result["source_csv_abs"]
    out_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_results", "")
    out_png = make_writable_output_path(os.path.join(out_dir, f"{out_base}_stimulus_distribution.png"))
    try:
        dd = clean_trial_frame(csv_path)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    levels = np.sort(dd["difficulty_abs_deg"].dropna().unique())
    fit_counts = dd[dd["is_bonus_trial"] == 0].groupby("difficulty_abs_deg")["correct"].count().reindex(levels).fillna(0).to_numpy()
    bonus_counts = dd[dd["is_bonus_trial"] == 1].groupby("difficulty_abs_deg")["correct"].count().reindex(levels).fillna(0).to_numpy()
    x = np.arange(len(levels))
    ax.bar(x, fit_counts, width=0.8, label="adaptive staircases")
    if bonus_counts.sum() > 0:
        ax.bar(x, bonus_counts, bottom=fit_counts, width=0.8, label="bonus/catch")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in levels], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("orientation difference level (deg)")
    ax.set_ylabel("trial count")
    ax.set_title(f"Stimulus-level distribution | {session_result['source_csv']} | {session_result['location_id']}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def save_rt_block_plot(session_result, out_dir):
    csv_path = session_result["source_csv_abs"]
    out_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_results", "")
    out_png = make_writable_output_path(os.path.join(out_dir, f"{out_base}_rt_and_block_accuracy.png"))
    try:
        dd = clean_trial_frame(csv_path)
    except Exception:
        return None
    if "response_time_s" not in dd.columns or not dd["response_time_s"].notna().any():
        return None

    block_ids = np.floor(np.linspace(0, N_BLOCKS_WITHIN_SESSION, len(dd), endpoint=False)).astype(int) + 1
    block_ids = np.clip(block_ids, 1, N_BLOCKS_WITHIN_SESSION)
    dd = dd.copy()
    dd["block"] = block_ids
    btab = dd.groupby("block").agg(
        acc=("correct", "mean"),
        rt_med=("response_time_s", "median"),
        n=("correct", "count"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(btab["block"], btab["acc"] * 100.0, marker="o", linewidth=1.5, label="accuracy (%)")
    ax1.set_xlabel("within-session block")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_yticks(np.arange(0, 101, 10))

    ax2 = ax1.twinx()
    ax2.plot(btab["block"], btab["rt_med"], marker="s", linewidth=1.4, linestyle="--", label="median RT (s)")
    ax2.set_ylabel("median RT (s)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    ax1.set_title(f"Within-session accuracy and RT | {session_result['source_csv']}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _fit_threshold_from_dataframe(dd):
    dd_fit = dd[dd["is_bonus_trial"] == 0].copy() if FIT_EXCLUDES_BONUS_TRIALS else dd.copy()
    fit = fit_weibulls(dd_fit)
    return float(fit["threshold_rochester_exact_deg"]) if fit.get("fit_valid") else np.nan


def pooled_thresholds_by_window(sessions, window=POOLED_THRESHOLD_WINDOW):
    ordered = sorted([s for s in sessions if s.get("timestamp") is not None], key=lambda s: s["timestamp"])
    pooled = []
    for i in range(len(ordered)):
        subset = ordered[max(0, i - window + 1): i + 1]
        dfs = []
        for s in subset:
            try:
                dfs.append(clean_trial_frame(s["source_csv_abs"]))
            except Exception:
                pass
        pooled.append(np.nan if not dfs else _fit_threshold_from_dataframe(pd.concat(dfs, ignore_index=True)))
    return np.array(pooled, dtype=float)


def rolling_mean(values, window=3):
    return pd.Series(values, dtype=float).rolling(window=window, min_periods=1).mean().to_numpy()


def save_longitudinal_overlay_plot(location_id, sessions, out_dir):
    valid = [s for s in sessions if s.get("timestamp") is not None]
    valid = sorted(valid, key=lambda s: s["timestamp"])
    out_png = make_writable_output_path(os.path.join(out_dir, f"longitudinal_{location_id}_orientation_accuracy_threshold_overlay.png"))
    fig, ax_thr = plt.subplots(figsize=(13, 6))
    if valid:
        x = np.arange(1, len(valid) + 1)
        dates = [datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%m-%d\n%H:%M") for s in valid]
        thresholds = np.array([s.get("threshold_rochester_exact_deg", np.nan) if s.get("fit_valid") else np.nan for s in valid], dtype=float)
        thresholds_py = np.array([s.get("threshold_python_true_725_deg", np.nan) if s.get("fit_valid") else np.nan for s in valid], dtype=float)
        acc = np.array([s.get("accuracy_fit_trials_percent", np.nan) for s in valid], dtype=float)

        ax_thr.plot(x, thresholds, marker="o", linewidth=1.5, label="Rochester exact threshold (deg)")
        ax_thr.plot(x, thresholds_py, marker="o", linewidth=1.0, linestyle=":", label="True 72.5% threshold (deg)")
        ax_thr.plot(x, rolling_mean(thresholds, 3), linestyle="--", linewidth=1.4, label="3-session mean Rochester threshold")
        pooled = pooled_thresholds_by_window(valid)
        ax_thr.plot(x, pooled, linestyle="-.", linewidth=1.7, label="3-session pooled Rochester threshold")
        ax_thr.set_ylabel("threshold (deg; lower is better)")
        ax_thr.set_xlabel("session")
        ax_thr.set_xticks(x)
        ax_thr.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)

        ax_acc = ax_thr.twinx()
        ax_acc.plot(x, acc, marker="s", linewidth=1.2, label="fit-trial accuracy (%)")
        ax_acc.plot(x, rolling_mean(acc, 3), linestyle=":", linewidth=1.5, label="3-session mean accuracy")
        ax_acc.set_ylabel("accuracy (%; higher is better)")
        ax_acc.set_ylim(0, 100)
        ax_acc.set_yticks(np.arange(0, 101, 10))

        lines, labels = ax_thr.get_legend_handles_labels()
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        ax_thr.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax_thr.set_title(f"Fine Orientation longitudinal dashboard - {location_id} (n={len(valid)} sessions)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_block_accuracy_recent_plot(location_id, sessions, out_dir):
    valid = [s for s in sessions if s.get("timestamp") is not None]
    valid = sorted(valid, key=lambda s: s["timestamp"])[-RECENT_SESSION_WINDOW:]
    out_png = make_writable_output_path(os.path.join(out_dir, f"recent_{location_id}_orientation_block_accuracy.png"))
    fig, ax = plt.subplots(figsize=(9, 5))
    block_matrix = []
    for s in valid:
        blocks = s.get("block_accuracy", [])
        vals = np.full(N_BLOCKS_WITHIN_SESSION, np.nan)
        for b in blocks:
            idx = int(b["block"]) - 1
            if 0 <= idx < N_BLOCKS_WITHIN_SESSION:
                vals[idx] = b["accuracy_percent"]
        block_matrix.append(vals)
        label = datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%m-%d %H:%M")
        ax.plot(np.arange(1, N_BLOCKS_WITHIN_SESSION + 1), vals, marker="o", linewidth=1.0, alpha=0.65, label=label)
    if block_matrix:
        arr = np.vstack(block_matrix)
        ax.plot(np.arange(1, N_BLOCKS_WITHIN_SESSION + 1), np.nanmean(arr, axis=0), marker="s", linewidth=2.4, label=f"mean last {len(block_matrix)}")
    ax.set_xlabel("within-session block")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_title(f"Within-session raw accuracy, recent sessions - {location_id}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_stimulus_class_plot(location_id, sessions, out_dir):
    valid = [s for s in sessions if s.get("timestamp") is not None]
    valid = sorted(valid, key=lambda s: s["timestamp"])
    out_png = make_writable_output_path(os.path.join(out_dir, f"longitudinal_{location_id}_left_right_accuracy.png"))
    classes = ["LEFT", "RIGHT"]
    fig, ax = plt.subplots(figsize=(13, 5))
    if valid:
        x = np.arange(1, len(valid) + 1)
        dates = [datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%m-%d\n%H:%M") for s in valid]
        for cls in classes:
            vals = []
            for s in valid:
                found = np.nan
                for row in s.get("stimulus_class_accuracy", []):
                    if row.get("stimulus_class") == cls:
                        found = row.get("accuracy_percent", np.nan)
                        break
                vals.append(found)
            ax.plot(x, vals, marker="o", linewidth=1.2, label=cls)
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("session")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_title(f"LEFT vs RIGHT orientation accuracy - {location_id}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_summary_csv(session_results, out_dir):
    rows = []
    for s in session_results:
        rows.append({
            "timestamp": s.get("timestamp"),
            "subject": s.get("subject"),
            "source_csv": s.get("source_csv"),
            "location_id": s.get("location_id"),
            "n_trials": s.get("n_trials"),
            "n_fit_trials": s.get("n_fit_trials"),
            "n_bonus_trials": s.get("n_bonus_trials"),
            "accuracy_percent_all": s.get("accuracy_percent"),
            "accuracy_percent_fit_trials": s.get("accuracy_fit_trials_percent"),
            "fit_valid": s.get("fit_valid"),
            "fit_failure_reason": s.get("fit_failure_reason"),
            "threshold_rochester_exact_deg": s.get("threshold_rochester_exact_deg"),
            "threshold_python_true_725_deg": s.get("threshold_python_true_725_deg"),
            "threshold_delta_percent": s.get("threshold_delta_python_minus_rochester_percent"),
            "beta_rochester_exact": s.get("beta_rochester_exact"),
            "beta_python": s.get("beta_python"),
            "percent_correct_easiest": s.get("percent_correct_easiest"),
            "rt_median_s": (s.get("rt_summary") or {}).get("rt_median_s"),
            "rt_last_minus_first_median_s": (s.get("rt_summary") or {}).get("rt_last_minus_first_median_s"),
            "metacognition_enabled": s.get("metacognition_enabled"),
            "protocol_warnings": ";".join(s.get("protocol_warnings", [])),
        })
    out_csv = make_writable_output_path(os.path.join(out_dir, "fine_orientation_session_summary_table.csv"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def _relpath_for_html(path, base_dir):
    if path is None:
        return None
    try:
        return os.path.relpath(path, base_dir).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")


def _safe_anchor(text):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(text or 'section'))


def generate_html_dashboard(longitudinal, out_dir):
    out_dir = os.path.abspath(out_dir)
    dashboard_path = make_writable_output_path(os.path.join(out_dir, "fine_orientation_dashboard.html"))

    def esc(x):
        return html.escape("" if x is None else str(x))

    # Data payload for the interactive longitudinal chart.
    interactive_payload = {"locations": {}}
    for location_id, loc in longitudinal.get("locations", {}).items():
        sessions = sorted(loc.get("sessions", []), key=lambda s: (s.get("timestamp") is None, s.get("timestamp") or ""))
        interactive_payload["locations"][location_id] = []
        pooled = loc.get("pooled_3session_thresholds_rochester_deg", []) or []
        for idx, s in enumerate(sessions, 1):
            interactive_payload["locations"][location_id].append({
                "index": idx,
                "timestamp": s.get("timestamp"),
                "accuracy_fit_trials_percent": s.get("accuracy_fit_trials_percent"),
                "accuracy_percent": s.get("accuracy_percent"),
                "threshold_rochester_exact_deg": s.get("threshold_rochester_exact_deg"),
                "threshold_python_true_725_deg": s.get("threshold_python_true_725_deg"),
                "threshold_delta_percent": s.get("threshold_delta_python_minus_rochester_percent"),
                "pooled_3session_threshold_rochester_deg": pooled[idx-1] if idx-1 < len(pooled) else None,
                "rt_median_s": (s.get("rt_summary") or {}).get("rt_median_s"),
                "n_fit_trials": s.get("n_fit_trials"),
                "fit_valid": bool(s.get("fit_valid")),
            })

    css = """
    body { font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }
    header { position: sticky; top: 0; background: #111; color: white; padding: 14px 22px; z-index: 10; }
    header h1 { margin: 0 0 8px 0; font-size: 20px; }
    nav { line-height: 1.7; max-height: 85px; overflow-y: auto; }
    nav a { color: #d8e6ff; margin-right: 14px; text-decoration: none; font-size: 13px; }
    main { padding: 24px; max-width: 1500px; margin: auto; }
    section { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.12); }
    h2 { margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { background: #eee; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; background: white; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 18px; }
    .session-card { border: 1px solid #ddd; border-radius: 6px; padding: 12px; background: #fafafa; break-inside: avoid; }
    .meta, .small { color: #666; font-size: 13px; }
    .note { background:#fff9db; border:1px solid #e5d27a; padding:10px; border-radius:6px; font-size:13px; }
    .warn { color:#8a2a00; font-weight:bold; }
    .toplink { float:right; font-size:12px; }
    canvas.longitudinal-canvas { width: 100%; max-width: 1150px; border: 1px solid #ddd; background: #fff; }
    .controls { display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin: 10px 0 14px 0; font-size:13px; }
    .controls label { white-space:nowrap; }
    button { border: 1px solid #999; background: #f4f4f4; padding: 7px 10px; border-radius: 5px; cursor: pointer; margin: 3px; }
    .print-button { background:#0b57d0; color:white; border-color:#0b57d0; font-weight:bold; }
    @media print {
      header { position: static; background: white; color: #111; border-bottom:1px solid #ccc; }
      nav, button, .toplink { display:none !important; }
      body { background:white; }
      main { padding:0; max-width:none; }
      section { box-shadow:none; border:none; padding:0; margin-bottom:10mm; }
      .session-card { page-break-inside: avoid; }
    }
    """

    nav = ['<a href="#summary">Summary</a>']
    sections = []
    summary_rel = _relpath_for_html(longitudinal.get("summary_csv"), out_dir)
    sections.append(
        '<section id="summary">'
        '<h2>Fine Orientation Analysis Summary</h2>'
        '<button class="print-button" onclick="window.print()">Export / print as PDF</button>'
        f'<p class="meta">Sessions analysed: {esc(longitudinal.get("n_sessions_total"))}. '
        f'Chance={CHANCE}, lapse={LAPSE_RATE}, target={THRESHOLD_TARGET}. '
        f'Bonus trials excluded from threshold fits: {FIT_EXCLUDES_BONUS_TRIALS}.</p>'
        '<div class="note">Main threshold is the Rochester/DataFitting-compatible convention. '
        'The dashboard also reports the direct 72.5% threshold and their percent difference.</div>'
        + (f'<p><a href="{esc(summary_rel)}">Open fine_orientation_session_summary_table.csv</a></p>' if summary_rel else '')
        + '</section>'
    )

    for location_id, loc in longitudinal.get("locations", {}).items():
        anchor = _safe_anchor(location_id)
        nav.append(f'<a href="#loc_{esc(anchor)}">{esc(location_id)}</a>')
        overlay = _relpath_for_html(loc.get("accuracy_threshold_overlay_png"), out_dir)
        block = _relpath_for_html(loc.get("recent_block_accuracy_png"), out_dir)
        lr = _relpath_for_html(loc.get("left_right_accuracy_png"), out_dir)

        sorted_sessions = sorted(loc.get("sessions", []), key=lambda s: (s.get("timestamp") is None, s.get("timestamp") or ""))
        rows, cards = [], []
        for idx, s in enumerate(sorted_sessions, 1):
            sess_anchor = f"{anchor}_session_{idx}"
            warn = "; ".join(s.get("protocol_warnings", []))
            rows.append(
                '<tr>'
                f'<td><a href="#{esc(sess_anchor)}">{idx}</a></td>'
                f'<td>{esc(s.get("timestamp"))}</td>'
                f'<td>{esc(s.get("source_csv"))}</td>'
                f'<td>{esc(s.get("n_trials"))}</td>'
                f'<td>{esc(s.get("n_fit_trials"))}</td>'
                f'<td>{"" if s.get("accuracy_fit_trials_percent") is None else "{:.2f}".format(s.get("accuracy_fit_trials_percent"))}</td>'
                f'<td>{"" if s.get("threshold_rochester_exact_deg") is None else "{:.2f}".format(s.get("threshold_rochester_exact_deg"))}</td>'
                f'<td>{"" if s.get("threshold_python_true_725_deg") is None else "{:.2f}".format(s.get("threshold_python_true_725_deg"))}</td>'
                f'<td>{"" if s.get("threshold_delta_python_minus_rochester_percent") is None else "{:.2f}%".format(s.get("threshold_delta_python_minus_rochester_percent"))}</td>'
                f'<td>{esc(s.get("fit_valid"))}</td>'
                f'<td class="warn">{esc(warn)}</td>'
                '</tr>'
            )
            psych = _relpath_for_html(s.get("psychometric_plot_png"), out_dir)
            stair = _relpath_for_html(s.get("staircase_plot_png"), out_dir)
            dist = _relpath_for_html(s.get("distribution_plot_png"), out_dir)
            rt = _relpath_for_html(s.get("rt_block_plot_png"), out_dir)
            cards.append(
                f'<div class="session-card" id="{esc(sess_anchor)}">'
                '<a class="toplink" href="#summary">top</a>'
                f'<h3>Session {idx}: {esc(s.get("timestamp"))}</h3>'
                f'<div class="small">{esc(s.get("source_csv"))}<br>'
                f'fit trials={esc(s.get("n_fit_trials"))}; accuracy={"" if s.get("accuracy_fit_trials_percent") is None else "{:.2f}%".format(s.get("accuracy_fit_trials_percent"))}; '
                f'Roch threshold={"" if s.get("threshold_rochester_exact_deg") is None else "{:.2f}°".format(s.get("threshold_rochester_exact_deg"))}; '
                f'true 72.5={"" if s.get("threshold_python_true_725_deg") is None else "{:.2f}°".format(s.get("threshold_python_true_725_deg"))}</div>'
                '<p><strong>Psychometric Weibull</strong></p>'
                f'fit trials={esc(s.get("n_fit_trials"))}; accuracy={"" if s.get("accuracy_fit_trials_percent") is None else "{:.2f}%".format(s.get("accuracy_fit_trials_percent"))}; '
                f'Roch threshold={"" if s.get("threshold_rochester_exact_deg") is None else "{:.2f}°".format(s.get("threshold_rochester_exact_deg"))}; '
                f'true 72.5={"" if s.get("threshold_python_true_725_deg") is None else "{:.2f}°".format(s.get("threshold_python_true_725_deg"))}</div>'
                + '<p><strong>Stimulus-level distribution</strong></p>'
                + (f'<img src="{esc(dist)}">' if dist else '<p>No distribution plot.</p>')
                + '<p><strong>RT and block accuracy</strong></p>'
                + (f'<img src="{esc(rt)}">' if rt else '<p>No RT plot.</p>')
                + '</div>'
            )

        sections.append(
            f'<section id="loc_{esc(anchor)}">'
            '<a class="toplink" href="#summary">top</a>'
            f'<h2>Location {esc(location_id)}</h2>'
            f'<p class="meta">Sessions: {esc(loc.get("n_sessions"))}; valid fits: {esc(loc.get("n_valid_fits"))}; '
            f'threshold SD: {esc((loc.get("variability") or {}).get("threshold_rochester_std_deg"))}</p>'
            '<h3>Interactive longitudinal threshold + accuracy</h3>'
            + '<div class="controls">'
            + f'<label><input type="checkbox" id="show_roch_{esc(anchor)}" checked onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> Rochester-like threshold</label>'
            + f'<label><input type="checkbox" id="show_true_{esc(anchor)}" checked onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> True 72.5% threshold</label>'
            + f'<label><input type="checkbox" id="show_pooled_{esc(anchor)}" onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> 3-session pooled threshold</label>'
            + f'<label><input type="checkbox" id="show_acc_{esc(anchor)}" checked onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> Accuracy</label>'
            + f'<label><input type="checkbox" id="show_rt_{esc(anchor)}" onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> Median RT</label>'
            + f'<label><input type="checkbox" id="show_delta_{esc(anchor)}" onchange="renderLongitudinal(\'{esc(location_id)}\', \'{esc(anchor)}\')"> Threshold delta %</label>'
            + '</div>'
            + f'<canvas class="longitudinal-canvas" id="long_canvas_{esc(anchor)}" width="1150" height="520"></canvas>'
            + f'<div class="small" id="long_note_{esc(anchor)}"></div>'
            + '<h3>Static longitudinal reference</h3>'
            + (f'<img src="{esc(overlay)}">' if overlay else '<p>No longitudinal plot.</p>')
            + '<h3>Recent within-session block accuracy</h3>'
            + (f'<img src="{esc(block)}">' if block else '<p>No block plot.</p>')
            + '<h3>LEFT vs RIGHT accuracy</h3>'
            + (f'<img src="{esc(lr)}">' if lr else '<p>No left/right plot.</p>')
            + '<h3>Session table</h3>'
            + '<table><thead><tr><th>#</th><th>Timestamp</th><th>CSV</th><th>Trials</th><th>Fit trials</th><th>Acc %</th><th>Roch thr °</th><th>72.5 thr °</th><th>Delta</th><th>Fit</th><th>Warnings</th></tr></thead><tbody>'
            + ''.join(rows)
            + '</tbody></table></section>'
            + f'<section><h2>Session-level plots — {esc(location_id)}</h2><div class="grid">'
            + ''.join(cards)
            + '</div></section>'
        )

    payload_json = json.dumps(interactive_payload)
    js = r"""
    const DASHBOARD_DATA = __PAYLOAD__;

    function safeAnchor(loc) {
      return String(loc || 'section').replace(/[^A-Za-z0-9_-]+/g, '_');
    }
    function checked(id) {
      const el = document.getElementById(id);
      return el ? el.checked : false;
    }
    function finiteValues(arr) {
      return arr.filter(v => Number.isFinite(v));
    }
    function drawLine(ctx, x1, y1, x2, y2) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    function seriesFrom(sessions, field) {
      return sessions.map(s => {
        const v = s[field];
        return (v === null || v === undefined || Number.isNaN(Number(v))) ? NaN : Number(v);
      });
    }
    function renderLongitudinal(locationId, anchor) {
      const canvas = document.getElementById('long_canvas_' + anchor);
      const note = document.getElementById('long_note_' + anchor);
      if (!canvas) return;

      const sessions = (DASHBOARD_DATA.locations[locationId] || []).filter(s => s.timestamp !== null);
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const m = {l:70, r:72, t:38, b:75};
      ctx.clearRect(0, 0, W, H);
      ctx.font = '13px Arial';

      if (!sessions.length) {
        ctx.fillText('No longitudinal data.', m.l, m.t);
        return;
      }

      const xN = sessions.length;
      function X(i) {
        return m.l + (xN <= 1 ? 0.5 : (i / (xN - 1))) * (W - m.l - m.r);
      }

      const showRoch = checked('show_roch_' + anchor);
      const showTrue = checked('show_true_' + anchor);
      const showPooled = checked('show_pooled_' + anchor);
      const showAcc = checked('show_acc_' + anchor);
      const showRT = checked('show_rt_' + anchor);
      const showDelta = checked('show_delta_' + anchor);

      const thrR = seriesFrom(sessions, 'threshold_rochester_exact_deg');
      const thrT = seriesFrom(sessions, 'threshold_python_true_725_deg');
      const thrP = seriesFrom(sessions, 'pooled_3session_threshold_rochester_deg');
      const acc = seriesFrom(sessions, 'accuracy_fit_trials_percent');
      const rt = seriesFrom(sessions, 'rt_median_s');
      const delta = seriesFrom(sessions, 'threshold_delta_percent');

      let thresholdValues = [];
      if (showRoch) thresholdValues = thresholdValues.concat(finiteValues(thrR));
      if (showTrue) thresholdValues = thresholdValues.concat(finiteValues(thrT));
      if (showPooled) thresholdValues = thresholdValues.concat(finiteValues(thrP));
      if (!thresholdValues.length) thresholdValues = [10];

      const yThrMax = Math.max(1, Math.ceil(Math.max(...thresholdValues) * 1.15));
      const yThrMin = 0;
      function Ythr(v) {
        return H - m.b - ((v - yThrMin) / Math.max(1e-9, yThrMax - yThrMin)) * (H - m.t - m.b);
      }
      function Ypct(v) {
        return H - m.b - (v / 100) * (H - m.t - m.b);
      }
      const rtVals = finiteValues(rt);
      const rtMax = Math.max(1, rtVals.length ? Math.max(...rtVals) * 1.15 : 1);
      function Yrt(v) {
        return H - m.b - (v / rtMax) * (H - m.t - m.b);
      }

      // Axes and grid
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      drawLine(ctx, m.l, H - m.b, W - m.r, H - m.b);
      drawLine(ctx, m.l, m.t, m.l, H - m.b);
      drawLine(ctx, W - m.r, m.t, W - m.r, H - m.b);

      ctx.fillStyle = '#333';
      for (let y = 0; y <= yThrMax; y += Math.max(1, Math.ceil(yThrMax / 5))) {
        const yy = Ythr(y);
        ctx.strokeStyle = '#eee';
        drawLine(ctx, m.l, yy, W - m.r, yy);
        ctx.fillStyle = '#333';
        ctx.fillText(String(y), 24, yy + 4);
      }
      for (let y = 0; y <= 100; y += 20) {
        const yy = Ypct(y);
        ctx.fillStyle = '#666';
        ctx.fillText(String(y), W - m.r + 10, yy + 4);
      }

      const step = Math.max(1, Math.ceil(xN / 12));
      for (let i = 0; i < xN; i += step) {
        const xx = X(i);
        ctx.strokeStyle = '#ddd';
        drawLine(ctx, xx, H - m.b, xx, H - m.b + 5);
        ctx.save();
        ctx.translate(xx - 8, H - m.b + 16);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#333';
        ctx.fillText('S' + (i + 1), 0, 0);
        ctx.restore();
      }

      ctx.fillStyle = '#333';
      ctx.fillText('threshold (deg; lower is better)', 8, 22);
      ctx.fillText('accuracy / delta %', W - m.r - 22, 22);

      function plot(vals, color, yFn, dash, label, marker) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        let started = false;
        vals.forEach((v, i) => {
          if (!Number.isFinite(v)) { started = false; return; }
          const xx = X(i), yy = yFn(v);
          if (!started) { ctx.moveTo(xx, yy); started = true; }
          else ctx.lineTo(xx, yy);
        });
        ctx.stroke();
        ctx.setLineDash([]);

        if (marker) {
          vals.forEach((v, i) => {
            if (!Number.isFinite(v)) return;
            ctx.beginPath();
            ctx.arc(X(i), yFn(v), 3.3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      const legend = [];
      if (showRoch) { plot(thrR, '#1f77b4', Ythr, [], 'Rochester-like threshold', true); legend.push(['#1f77b4', 'Rochester-like threshold']); }
      if (showTrue) { plot(thrT, '#9467bd', Ythr, [6,4], 'True 72.5% threshold', true); legend.push(['#9467bd', 'True 72.5% threshold']); }
      if (showPooled) { plot(thrP, '#2ca02c', Ythr, [2,4], '3-session pooled threshold', false); legend.push(['#2ca02c', '3-session pooled threshold']); }
      if (showAcc) { plot(acc, '#d62728', Ypct, [], 'Accuracy', true); legend.push(['#d62728', 'Accuracy (%)']); }
      if (showDelta) { plot(delta, '#ff7f0e', Ypct, [5,3], 'Threshold delta %', true); legend.push(['#ff7f0e', 'Threshold delta (%)']); }
      if (showRT) { plot(rt, '#7f7f7f', Yrt, [8,4], 'Median RT', true); legend.push(['#7f7f7f', 'Median RT (own scale)']); }

      legend.forEach((item, idx) => {
        ctx.fillStyle = item[0];
        ctx.fillRect(m.l + 10, m.t + 8 + idx * 18, 12, 12);
        ctx.fillStyle = '#333';
        ctx.fillText(item[1], m.l + 28, m.t + 19 + idx * 18);
      });

      if (note) {
        const validFits = sessions.filter(s => s.fit_valid).length;
        const trialCounts = finiteValues(seriesFrom(sessions, 'n_fit_trials'));
        const minTrials = trialCounts.length ? Math.min(...trialCounts) : 'n/a';
        note.textContent = `Sessions: ${sessions.length}; valid fits: ${validFits}; minimum fit-trials/session: ${minTrials}. With very short test sessions, session-level fits are intentionally absent or unstable.`;
      }
    }

    window.addEventListener('load', () => {
      Object.keys(DASHBOARD_DATA.locations || {}).forEach(loc => {
        renderLongitudinal(loc, safeAnchor(loc));
      });
    });
    """.replace("__PAYLOAD__", payload_json)

    body = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>Fine Orientation Analysis Dashboard</title>'
        f'<style>{css}</style></head><body>'
        '<header><h1>Fine Orientation / Gabor Analysis Dashboard</h1><nav>'
        + "\n".join(nav)
        + '</nav></header><main>'
        + ''.join(sections)
        + f'<script>{js}</script>'
        + '</main></body></html>'
    )
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(body)
    return dashboard_path


def open_dashboard(path):
    try:
        webbrowser.open("file://" + os.path.abspath(path))
    except Exception:
        pass


def main():
    csv_paths = discover_csvs()
    if not csv_paths:
        print("No Fine Orientation *_results.csv files found.")
        print("Expected locations include ./DataFineOrientation/**/_results.csv")
        return

    out_dir = output_directory_for(csv_paths)
    os.makedirs(out_dir, exist_ok=True)

    session_results = []
    for csv_path in sorted(csv_paths):
        try:
            res = fit_session(csv_path)
        except Exception as e:
            ts = parse_timestamp_from_name(csv_path)
            loc, _ = infer_location_id(csv_path)
            res = {
                "source_csv": os.path.basename(csv_path),
                "source_csv_abs": os.path.abspath(csv_path),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None,
                "location_id": loc,
                "fit_valid": False,
                "fit_failure_reason": f"analysis_exception: {repr(e)}",
                "n_trials": None,
                "n_fit_trials": None,
                "accuracy_percent": None,
                "accuracy_fit_trials_percent": None,
                "percent_correct_table": [],
                "block_accuracy": [],
                "stimulus_class_accuracy": [],
                "protocol_warnings": ["analysis_exception"],
            }

        psych_png, session_json = save_session_plot(res, out_dir)
        stair_png = save_staircase_plot(res, out_dir)
        dist_png = save_distribution_plot(res, out_dir)
        rt_png = save_rt_block_plot(res, out_dir)

        res["psychometric_plot_png"] = os.path.abspath(psych_png) if psych_png else None
        res["staircase_plot_png"] = os.path.abspath(stair_png) if stair_png else None
        res["distribution_plot_png"] = os.path.abspath(dist_png) if dist_png else None
        res["rt_block_plot_png"] = os.path.abspath(rt_png) if rt_png else None
        res["session_json_output"] = os.path.abspath(session_json)
        session_results.append(res)

    grouped = defaultdict(list)
    for res in session_results:
        grouped[res["location_id"]].append(res)

    summary_csv = save_summary_csv(session_results, out_dir)

    longitudinal = {
        "source_lineage": "TestingCodes/FineOrientation + DataFitting-compatible Rochester exact threshold; descriptive dashboard additions",
        "constants": {
            "thresholdTarget": THRESHOLD_TARGET,
            "lapse_rate": LAPSE_RATE,
            "chance": CHANCE,
            "within_session_blocks": N_BLOCKS_WITHIN_SESSION,
            "recent_session_window": RECENT_SESSION_WINDOW,
            "pooled_threshold_window": POOLED_THRESHOLD_WINDOW,
            "fit_excludes_bonus_trials": FIT_EXCLUDES_BONUS_TRIALS,
        },
        "n_sessions_total": len(session_results),
        "summary_csv": os.path.abspath(summary_csv),
        "locations": {},
    }

    for location_id, sessions in grouped.items():
        sessions_sorted = sorted(sessions, key=lambda s: (s.get("timestamp") is None, s.get("timestamp") or ""))
        overlay_png = save_longitudinal_overlay_plot(location_id, sessions_sorted, out_dir)
        block_png = save_block_accuracy_recent_plot(location_id, sessions_sorted, out_dir)
        lr_png = save_stimulus_class_plot(location_id, sessions_sorted, out_dir)

        valid = [s for s in sessions_sorted if s.get("timestamp") is not None]
        thr_roch = np.array([s.get("threshold_rochester_exact_deg", np.nan) if s.get("fit_valid") else np.nan for s in valid], dtype=float)
        thr_py = np.array([s.get("threshold_python_true_725_deg", np.nan) if s.get("fit_valid") else np.nan for s in valid], dtype=float)
        acc = np.array([s.get("accuracy_fit_trials_percent", np.nan) for s in valid], dtype=float)
        pooled = pooled_thresholds_by_window(valid, window=POOLED_THRESHOLD_WINDOW) if valid else np.array([], dtype=float)

        longitudinal["locations"][location_id] = {
            "n_sessions": len(sessions_sorted),
            "n_valid_fits": sum(1 for s in sessions_sorted if s.get("fit_valid")),
            "variability": {
                "threshold_rochester_std_deg": float(np.nanstd(thr_roch)) if np.isfinite(thr_roch).any() else None,
                "threshold_python_true_725_std_deg": float(np.nanstd(thr_py)) if np.isfinite(thr_py).any() else None,
                "accuracy_std_percent": float(np.nanstd(acc)) if np.isfinite(acc).any() else None,
                "pooled_3session_threshold_std_deg": float(np.nanstd(pooled)) if np.isfinite(pooled).any() else None,
            },
            "pooled_3session_thresholds_rochester_deg": [None if not np.isfinite(v) else float(v) for v in pooled],
            "accuracy_threshold_overlay_png": os.path.abspath(overlay_png),
            "recent_block_accuracy_png": os.path.abspath(block_png),
            "left_right_accuracy_png": os.path.abspath(lr_png),
            "sessions": sessions_sorted,
        }

    long_json = make_writable_output_path(os.path.join(out_dir, "fine_orientation_analysis_dashboard_summary.json"))
    with open(long_json, "w", encoding="utf-8") as f:
        json.dump(longitudinal, f, indent=2)

    dashboard_html = generate_html_dashboard(longitudinal, out_dir)
    open_dashboard(dashboard_html)

    print(json.dumps({
        "n_sessions_total": len(session_results),
        "output_directory": os.path.abspath(out_dir),
        "dashboard_html": os.path.abspath(dashboard_html),
        "dashboard_json": os.path.abspath(long_json),
        "summary_csv": os.path.abspath(summary_csv),
        "locations": {k: {"n_sessions": v["n_sessions"], "n_valid_fits": v["n_valid_fits"]}
                      for k, v in longitudinal["locations"].items()}
    }, indent=2))


if __name__ == "__main__":
    main()