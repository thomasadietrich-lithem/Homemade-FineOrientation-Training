"""
PsychoPy port of Huxlin Lab Fine Orientation Discrimination with Metacognition.

Reference MATLAB source:
    TestingCodes-main/FineOrientation/FineOrientationDiscrimination_MetaCog.m

Protocol fidelity target:
    This script mirrors the MATLAB task structure, constants, trial logic, response
    mapping, staircase rules, stimulus envelope, mask delay, response timing, ITI,
    and output columns as closely as PsychoPy permits.

Important compliance notes:
    - The MATLAB reference task has NO FBA pre-cue. No pre-cue option is exposed.
    - EyeLink is not implemented here. The MATLAB file sets ET=0 by default, so the
      default home/no-eye-tracking path is the mirrored path.
    - Gamma-table loading from the MATLAB lab computer is not reproduced. Visual
      values are generated in rgb255 space to match MATLAB 0..255 grayscale values.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import tempfile
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from psychopy import core, event, gui, monitors, sound, visual

try:
    from scipy.io import savemat  # Optional, but preferred for MATLAB-compatible output.
except Exception:  # pragma: no cover - PsychoPy installs vary.
    savemat = None

# -----------------------------------------------------------------------------
# Paths / monitor UX inherited from the existing PsychoPy Global Tilt port
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_DIRNAME = "DataFineOrientation"
MONITOR_CACHE_FILE = os.path.join(BASE_DIR, "monitor_profiles.json")

# -----------------------------------------------------------------------------
# MATLAB protocol constants from FineOrientationDiscrimination_MetaCog.m
# -----------------------------------------------------------------------------

# Subject/menu defaults requested by user, with MATLAB task defaults preserved where
# they are part of the protocol.
DEFAULT_SUBJECT_ID = "0000"
DEFAULT_VIEWING_DISTANCE_CM = 42.0
DEFAULT_SCREEN_WIDTH_CM = 54.3  # MATLAB file value; user can calibrate in dialog.

# MATLAB Identify starting location section.
DEFAULT_H_ECC_STIM_DEG = 0.0
DEFAULT_V_ECC_STIM_DEG = 0.0
DEFAULT_H_ECC_FIX_DEG = 0.0
DEFAULT_V_ECC_FIX_DEG = 0.0

# MATLAB Other task parameters.
DELAY_MASK_S = 0.25
N_TRIALS_PER_STAIRCASE = 90
BONUS_TRIALS = 30
ITI_S = 1.0
TOTAL_TRIALS_DEFAULT = BONUS_TRIALS + 3 * N_TRIALS_PER_STAIRCASE

# MATLAB baseAngle vector. This differs from the RDK/global-tilt scale.
BASE_ANGLE_DEG: List[float] = [53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5, 0.1]
INITIAL_STAIR_STEPS = {1: 1, 2: 4, 3: 9}  # MATLAB 1-based stairStep1/2/3.
CORRECTS_TO_STEP_DOWN = {1: 4, 2: 3, 3: 2}  # MATLAB comments: 4:1, 3:1, 2:1.

# Stimulus parameters.
ONSET_S = 0.25
OFFSET_S = 0.25
DURATION_S = 0.0
SF_CYCLES_PER_DEG = 1.0
STIM_RADIUS_DEG = 2.5
CONTRAST_PERCENT = 100.0
SPATIAL_ENVELOPE = 1  # 0=disk, 1=Gabor/Gaussian envelope, 2=raised cosine.
WHICH_ENVELOPE = 1  # Preserved in metadata; MATLAB's dujeEnvelope is not used downstream.
MOTION_DURATION_MS = 200.0  # Used only for MATLAB's unused dujeEnvelope calculation.

# Mask parameters from MATLAB.
MASK_ORIENTATION_DEG = 30.0  # irrelevant because MASK_COHERENCE = 0.
MASK_COHERENCE = 0.0
MASK_WIDTH_PX = 151  # MUST BE ODD in MATLAB.
MASK_SF_CPP = 0.001
MASK_STD_SF_CPP = 0.01

# Visual appearance.
FONT = "Arial"
FONT_SIZE_PX = 40
FIX_SIZE = 8
BACKGROUND_255 = 128.0
BLACK_255 = 0.0
WHITE_255 = 255.0
BACKGROUND_RGB = BACKGROUND_255 / 127.5 - 1.0
BLACK_RGB = -1.0
WHITE_RGB = 1.0


# -----------------------------------------------------------------------------
# Monitor calibration helpers
# -----------------------------------------------------------------------------


def get_screen_pixels() -> Tuple[int, int]:
    """Best-effort screen pixel query before opening the PsychoPy window."""
    try:
        import pyglet

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        return int(screen.width), int(screen.height)
    except Exception:
        return 1920, 1080


@dataclass
class MonitorGeometry:
    width_cm: float
    distance_cm: float
    res_x: int
    res_y: int
    arcmin_per_pix: float
    refresh_hz: float


def estimate_arcmin_per_pixel(screen_width_cm: float, viewing_distance_cm: float, res_x: int) -> float:
    # MATLAB: theta = atand((screen_width/2)/viewing_dist);
    #         scale_factor = theta*60/(resolution(1)/2);
    theta_deg = math.degrees(math.atan((screen_width_cm / 2.0) / viewing_distance_cm))
    return theta_deg * 60.0 / (res_x / 2.0)


def deg_to_matlab_pixels(deg: float, geom: MonitorGeometry) -> float:
    # MATLAB: deg * 60 / scale_factor
    return deg * 60.0 / geom.arcmin_per_pix


def _safe_json_load(path: str) -> Dict[str, dict]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_json_save(path: str, payload: Dict[str, dict]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _profile_key() -> str:
    w_px, h_px = get_screen_pixels()
    return f"{platform.system()}:{w_px}x{h_px}"


def load_or_calibrate_monitor() -> Optional[Tuple[monitors.Monitor, MonitorGeometry]]:
    """Load/create monitor profile, but always show the active profile before starting.

    Rationale
    ---------
    The task is expressed in degrees of visual angle. A silent mismatch in screen
    width, viewing distance or resolution changes:
    - stimulus eccentricity,
    - Gabor size,
    - spatial frequency,
    - fixation/stimulus geometry.

    Therefore we do NOT force a full recalibration at every launch, but we do
    force a lightweight profile confirmation screen with a Recalibrate option.
    """
    ident = _profile_key()
    profiles = _safe_json_load(MONITOR_CACHE_FILE)
    profile = profiles.get(ident)
    w_px, h_px = get_screen_pixels()

    def _ask_for_profile(existing: Optional[dict] = None) -> Optional[dict]:
        dlg_dict = {
            "Screen width (cm)": float(existing.get("width_cm", DEFAULT_SCREEN_WIDTH_CM)) if existing else DEFAULT_SCREEN_WIDTH_CM,
            "Viewing distance (cm)": float(existing.get("distance_cm", DEFAULT_VIEWING_DISTANCE_CM)) if existing else DEFAULT_VIEWING_DISTANCE_CM,
        }
        dlg = gui.DlgFromDict(
            dlg_dict,
            title="Screen calibration",
            order=["Screen width (cm)", "Viewing distance (cm)"],
        )
        if not dlg.OK:
            return None
        try:
            width_cm = float(dlg_dict["Screen width (cm)"])
            distance_cm = float(dlg_dict["Viewing distance (cm)"])
        except Exception:
            width_cm = DEFAULT_SCREEN_WIDTH_CM
            distance_cm = DEFAULT_VIEWING_DISTANCE_CM
        return {
            "width_cm": width_cm,
            "distance_cm": distance_cm,
            "size_pix": [w_px, h_px],
            "calibrated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "profile_key": ident,
        }

    # First-time calibration.
    if profile is None:
        profile = _ask_for_profile(None)
        if profile is None:
            return None
        profiles[ident] = profile
        _safe_json_save(MONITOR_CACHE_FILE, profiles)
    else:
        # Backfill older profiles created by prior versions.
        profile.setdefault("calibrated_at", "unknown")
        profile.setdefault("profile_key", ident)

    stored_res = profile.get("size_pix", [None, None])
    resolution_changed = stored_res != [w_px, h_px]

    # Lightweight confirmation shown at every launch.
    profile_text = (
        "ACTIVE SCREEN PROFILE\n\n"
        f"Profile key: {ident}\n"
        f"Screen width: {float(profile['width_cm']):.2f} cm\n"
        f"Viewing distance: {float(profile['distance_cm']):.2f} cm\n"
        f"Stored resolution: {stored_res[0]} × {stored_res[1]} px\n"
        f"Current resolution: {w_px} × {h_px} px\n"
        f"Last calibration: {profile.get('calibrated_at', 'unknown')}\n\n"
        "Use the same physical screen, same resolution, same display scaling, "
        "same chin-rest position, and same viewing distance for longitudinal comparisons.\n\n"
    )
    if resolution_changed:
        profile_text += (
            "WARNING: the current resolution differs from the stored profile. "
            "Recalibration is strongly recommended before continuing.\n\n"
        )

    confirm = gui.Dlg(title="Confirm screen profile")
    confirm.addText(profile_text)
    confirm.addField("Action", choices=["Continue with this profile", "Recalibrate", "Cancel"])
    confirm.show()
    if not confirm.OK:
        return None

    action = confirm.data[0] if confirm.data else "Cancel"
    if action == "Cancel":
        return None
    if action == "Recalibrate":
        new_profile = _ask_for_profile(profile)
        if new_profile is None:
            return None
        profile = new_profile
        profiles[ident] = profile
        _safe_json_save(MONITOR_CACHE_FILE, profiles)
    elif resolution_changed:
        # If the user continues despite changed resolution, record the current
        # resolution in metadata later but do not silently overwrite calibration.
        pass

    mon = monitors.Monitor(ident)
    mon.setWidth(float(profile["width_cm"]))
    mon.setDistance(float(profile["distance_cm"]))

    # If resolution changed and user continued, use the actual resolution for
    # rendering but retain the physical calibration values. This prevents window
    # size mismatch while preserving an explicit warning in the metadata.
    active_size_pix = [w_px, h_px]
    mon.setSizePix(active_size_pix)

    geom = MonitorGeometry(
        width_cm=float(profile["width_cm"]),
        distance_cm=float(profile["distance_cm"]),
        res_x=int(active_size_pix[0]),
        res_y=int(active_size_pix[1]),
        arcmin_per_pix=estimate_arcmin_per_pixel(
            float(profile["width_cm"]), float(profile["distance_cm"]), int(active_size_pix[0])
        ),
        refresh_hz=60.0,  # overwritten after window opens.
    )

    # Attach non-dataclass metadata to the monitor object for later JSON summary.
    # This keeps MonitorGeometry simple and backwards-compatible.
    try:
        mon._psychopi_profile_metadata = {
            "profile_key": ident,
            "stored_resolution_px": stored_res,
            "current_resolution_px": active_size_pix,
            "resolution_changed": bool(resolution_changed),
            "calibrated_at": profile.get("calibrated_at", "unknown"),
            "profile_file": MONITOR_CACHE_FILE,
        }
    except Exception:
        pass

    return mon, geom

# -----------------------------------------------------------------------------
# MATLAB-equivalent stimulus helpers
# -----------------------------------------------------------------------------


def make_spatial_envelope(radius_px: int, gaussian_stdev_px: int, spatial_envelope: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror MATLAB block that constructs Env.x1, Env.y1, Env.circle1."""
    axis = np.arange(-radius_px, radius_px + 1)
    x, y = np.meshgrid(axis, axis)
    circle_binary = ((radius_px ** 2 - (x ** 2 + y ** 2)) >= 0).astype(float)

    if spatial_envelope == 1:
        # MATLAB:
        # circle = exp(-((x/(sqrt(2)*Gaussian_stdev/2))^2)-(...)) .* circle
        denom = math.sqrt(2.0) * gaussian_stdev_px / 2.0
        if denom <= 0:
            circle = circle_binary
        else:
            circle = np.exp(-((x / denom) ** 2) - ((y / denom) ** 2)) * circle_binary
    elif spatial_envelope == 2:
        r = (np.sqrt(x ** 2 + y ** 2) + np.finfo(float).eps) * circle_binary
        max_r = np.max(r) if np.max(r) > 0 else 1.0
        r = r / max_r
        circle = ((np.cos(r * math.pi) + 1.0) / 2.0) * circle_binary
    else:
        circle = circle_binary
    return x.astype(float), y.astype(float), circle.astype(float)


def make_temporal_envelope(refresh_hz: float) -> np.ndarray:
    """Mirror MATLAB onset_env/offset_env inclusive indexing."""
    n_onset = int(round(ONSET_S * refresh_hz))
    n_offset = int(round(OFFSET_S * refresh_hz))

    # MATLAB: sind((0:n_onset)*90/n_onset)
    onset_env = np.sin(np.deg2rad(np.arange(0, n_onset + 1) * 90.0 / n_onset)) if n_onset > 0 else np.array([])
    # MATLAB: sind(fliplr(0:n_offset)*90/n_offset)
    offset_env = np.sin(np.deg2rad(np.arange(n_offset, -1, -1) * 90.0 / n_offset)) if n_offset > 0 else np.array([])
    plateau = np.ones(int(round(DURATION_S * refresh_hz)))
    return np.concatenate([onset_env, plateau, offset_env]).astype(float)


def make_gabor_frame(
    x: np.ndarray,
    y: np.ndarray,
    envelope: np.ndarray,
    angle_deviation_deg: float,
    ramp_value: float,
    geom: MonitorGeometry,
) -> np.ndarray:
    """Mirror MATLAB movie{i} = round((sin(a*x+b*y).*circle*ramp_amp)+background)."""
    f = (SF_CYCLES_PER_DEG * geom.arcmin_per_pix / 60.0) * 2.0 * math.pi
    angle_rad = math.radians(angle_deviation_deg)
    a = math.cos(angle_rad) * f
    b = math.sin(angle_rad) * f
    amplitude = BACKGROUND_255 * CONTRAST_PERCENT / 100.0
    ramp_amp = amplitude * ramp_value
    image = np.round((np.sin(a * x + b * y) * envelope * ramp_amp) + BACKGROUND_255)
    return np.clip(image, 0, 255).astype(np.uint8)

def image255_to_psychopy_rgb(image_255: np.ndarray) -> np.ndarray:
    """Convert MATLAB 0..255 grayscale texture to PsychoPy RGB float texture.

    MATLAB/Photchtoolbox draws uint8 luminance textures directly: 0=black,
    128=background gray, 255=white. For PsychoPy the safest backend-independent
    representation is a 3-channel float texture in the -1..1 rgb domain. This
    avoids black rectangular artifacts caused by backend-dependent interpretation
    of grayscale/PIL images.
    """
    arr = np.asarray(image_255, dtype=np.float32)
    arr = np.clip(arr, 0.0, 255.0)
    arr = arr / 127.5 - 1.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr.astype(np.float32)




def image255_to_pil_rgb(image_255: np.ndarray) -> Image.Image:
    """Convert MATLAB 0..255 luminance image to a PIL RGB texture.

    This is the most robust PsychoPy representation for matching MATLAB
    Screen('MakeTexture', uint8_luminance_image) on Windows/PsychoPy 2025.
    It avoids backend-dependent interpretation of float numpy arrays that can
    produce black square artifacts around the aperture.
    """
    arr = np.asarray(image_255, dtype=np.float32)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return Image.fromarray(arr, mode="RGB")

def crop_or_pad_to_size(image_255: np.ndarray, target_size: int, background: float = BACKGROUND_255) -> np.ndarray:
    """Center-crop or center-pad an image to the MATLAB destination patch size.

    MATLAB draws the mask texture through movie_rect/screen_patch, so the visible
    mask is constrained to the same destination rectangle as the Gabor patch. This
    helper makes that constraint explicit in PsychoPy.
    """
    src = np.asarray(image_255, dtype=np.float32)
    h, w = src.shape[:2]
    out = np.ones((target_size, target_size), dtype=np.float32) * background
    crop_h = min(h, target_size)
    crop_w = min(w, target_size)
    src_y0 = max(0, (h - crop_h) // 2)
    src_x0 = max(0, (w - crop_w) // 2)
    dst_y0 = max(0, (target_size - crop_h) // 2)
    dst_x0 = max(0, (target_size - crop_w) // 2)
    out[dst_y0:dst_y0+crop_h, dst_x0:dst_x0+crop_w] = src[src_y0:src_y0+crop_h, src_x0:src_x0+crop_w]
    return np.clip(out, 0, 255).astype(np.uint8)


def freq_coords(im_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of freq_coords.m."""
    range_start = -math.ceil((im_size - 1) / 2.0)
    range_end = range_start + im_size - 1
    vals = np.arange(range_start, range_end + 1)
    fx, fy = np.meshgrid(vals, vals)
    rho = np.sqrt(fx ** 2 + fy ** 2)
    theta = np.arctan2(fx, -fy)
    return fx, fy, rho, theta


def rician_pdf(x: np.ndarray, nu: float, sigma: float) -> np.ndarray:
    """Rician PDF equivalent to MATLAB pdf('rician', x, nu, sigma)."""
    sigma = max(float(sigma), np.finfo(float).eps)
    z = x * nu / (sigma ** 2)
    return (x / (sigma ** 2)) * np.exp(-((x ** 2 + nu ** 2) / (2.0 * sigma ** 2))) * np.i0(z)


def gen_images_dt(width: int, sp_freq_cpp: float, sp_freq_std_cpp: float, ori_deg: float, ori_kappa: float, aperture: np.ndarray) -> np.ndarray:
    """Port of genImagesDT.m for the single-frame post-stimulus mask."""
    noise = np.random.randn(width, width)
    noise_f = np.fft.fftshift(np.fft.fft2(noise))
    _, _, rho, theta = freq_coords(width)
    sp_freq_filter = rician_pdf(rho / width, sp_freq_cpp, sp_freq_std_cpp)

    if abs(ori_kappa) < 1e-12:
        ori_filter = np.ones_like(theta)
    else:
        ori_filter = np.exp(ori_kappa * np.cos((2.0 * theta) - (2.0 * math.radians(ori_deg)))) / np.i0(ori_kappa)

    filter_f = sp_freq_filter * ori_filter
    total = np.sum(filter_f)
    if total == 0 or not np.isfinite(total):
        filter_f = np.ones_like(filter_f) / filter_f.size
    else:
        filter_f = filter_f / total

    im_f = noise_f * filter_f
    im = aperture * np.real(np.fft.ifft2(np.fft.ifftshift(im_f)))
    max_abs = np.max(np.abs(im))
    if max_abs > 0:
        im = im / max_abs
    return im


def make_mask_aperture() -> np.ndarray:
    """Mirror MATLAB circleCOS used as genImagesDT aperture."""
    envelope_size = math.floor(MASK_WIDTH_PX / 2)
    axis = np.arange(-envelope_size, envelope_size + 1)
    x, y = np.meshgrid(axis, axis)
    circle = ((envelope_size ** 2 - (x ** 2 + y ** 2)) > 0).astype(float)
    r = (np.sqrt(x ** 2 + y ** 2) + np.finfo(float).eps) * circle
    max_r = np.max(r) if np.max(r) > 0 else 1.0
    r = r / max_r
    cos2d = (np.cos(r * math.pi) + 1.0) / 2.0
    return cos2d * circle


# -----------------------------------------------------------------------------
# Trial / scoring helpers
# -----------------------------------------------------------------------------


def build_trial_structure(total_trials: int = TOTAL_TRIALS_DEFAULT) -> List[int]:
    """Mirror MATLAB default exactly when total_trials=300.

    MATLAB default: [zeros(30); ones(90); twos(90); threes(90)], shuffled.
    If the user changes total_trials, we preserve the 10% bonus / 30% per staircase
    ratio as a UX extension and mark the exact number in the output.
    """
    if total_trials == TOTAL_TRIALS_DEFAULT:
        trial_structure = [0] * BONUS_TRIALS + [1] * N_TRIALS_PER_STAIRCASE + [2] * N_TRIALS_PER_STAIRCASE + [3] * N_TRIALS_PER_STAIRCASE
    else:
        bonus = int(round(total_trials * 0.10))
        remaining = max(0, total_trials - bonus)
        base = remaining // 3
        rem = remaining % 3
        counts = [base + (1 if i < rem else 0) for i in range(3)]
        trial_structure = [0] * bonus + [1] * counts[0] + [2] * counts[1] + [3] * counts[2]
    np.random.shuffle(trial_structure)
    return trial_structure


def ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        fallback = os.path.join(tempfile.gettempdir(), DATA_ROOT_DIRNAME)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def draw_score_and_fixation(
    win: visual.Window,
    running_score: int,
    trial_count: int,
    total_trials: int,
    fixation_pos: Tuple[float, float],
    fix_diameter_px: float,
    stimulus_radius_px: float,
    background_rect: Optional[visual.Rect] = None,
    show_score: bool = True,
) -> None:
    """Draw trial counter, optional score, and fixation.

    MATLAB-ISO MetaCog mode displays running score because score is part of the
    confidence-weighted response design. In the default non-metacognitive training
    mode, score is hidden because it no longer reflects a meaningful variable and
    could distract from perceptual discrimination.
    """
    if background_rect is not None:
        background_rect.draw()

    fx, fy = fixation_pos
    counter_side = -0.8
    trial_str = f"{trial_count}/{total_trials}"

    # MATLAB-like vertical layout below fixation. If score is disabled, place the
    # trial counter in the score slot to avoid a large empty gap.
    score_y = fy + (stimulus_radius_px / 2.0 * counter_side)
    trial_y_original = fy + (stimulus_radius_px * 1.5 * counter_side)
    trial_y = score_y + 0.5 * (trial_y_original - score_y)
    if not show_score:
        trial_y = score_y

    if show_score:
        score = visual.TextStim(
            win,
            text=str(running_score),
            pos=(fx, score_y),
            color=BLACK_RGB,
            colorSpace="rgb",
            height=FONT_SIZE_PX,
            font=FONT,
            units="pix",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        score.draw()

    counter = visual.TextStim(
        win,
        text=trial_str,
        pos=(fx, trial_y),
        color=BLACK_RGB,
        colorSpace="rgb",
        height=FONT_SIZE_PX / 2.0,
        font=FONT,
        units="pix",
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
    )
    fix = visual.Circle(
        win,
        radius=max(1.0, fix_diameter_px / 2.0),
        pos=fixation_pos,
        fillColor=BLACK_RGB,
        lineColor=BLACK_RGB,
        colorSpace="rgb",
        units="pix",
    )

    counter.draw()
    fix.draw()


@dataclass
class TrialRow:
    trial: int
    difficulty_abs_deg: float
    response_time_s: float
    signed_angle_deg: float
    correct: int
    confidence: int
    staircase: int
    response_key: str
    running_score: int
    is_bonus_trial: int
    stair_step_pre: int
    stair_step_post: int
    stair_count_pre: int
    stair_count_post: int
    target_side: str
    response_side: str
    points_delta: int
    stimulus_start_s: float
    stimulus_end_s: float
    stimulus_duration_actual_s: float
    response_phase: str


# -----------------------------------------------------------------------------
# Main session
# -----------------------------------------------------------------------------


def run_session(
    win: visual.Window,
    geom: MonitorGeometry,
    subject_id: str,
    h_ecc_stim_deg: float,
    v_ecc_stim_deg: float,
    h_ecc_fix_deg: float,
    v_ecc_fix_deg: float,
    total_trials: int,
    metacognition_enabled: bool,
    save_root: str,
    screen_profile_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run the Fine Orientation/Gabor session."""
    output_dir = ensure_dir(os.path.join(save_root, DATA_ROOT_DIRNAME, subject_id))
    trial_structure = build_trial_structure(total_trials)
    total_trials_actual = len(trial_structure)

    # MATLAB coordinate handling:
    #   h_ecc_orig = H_ecc_stim; v_ecc_orig = V_ecc_stim;
    #   V_ecc_stim = -V_ecc_stim; then Psychtoolbox screen y positive down.
    # PsychoPy pix y positive up, so the perceptual equivalent position is +v_ecc_stim_deg.
    stim_pos = (deg_to_matlab_pixels(h_ecc_stim_deg, geom), deg_to_matlab_pixels(v_ecc_stim_deg, geom))
    fix_pos = (deg_to_matlab_pixels(h_ecc_fix_deg, geom), deg_to_matlab_pixels(v_ecc_fix_deg, geom))

    stimulus_radius_px = int(round(60.0 * STIM_RADIUS_DEG / geom.arcmin_per_pix))
    gaussian_stdev_px = int(round(stimulus_radius_px / 1.5))
    x, y, spatial_env = make_spatial_envelope(stimulus_radius_px, gaussian_stdev_px, SPATIAL_ENVELOPE)
    bps = spatial_env.shape[0]
    temporal_env = make_temporal_envelope(geom.refresh_hz)
    mv_length = len(temporal_env)
    mask_aperture = make_mask_aperture()
    fix_diameter_px = FIX_SIZE * geom.arcmin_per_pix

    # MATLAB stair steps and counts are 1-based. We preserve that convention internally.
    stair_step = dict(INITIAL_STAIR_STEPS)
    stair_count = {1: 0, 2: 0, 3: 0}
    running_score = 0
    show_score = bool(metacognition_enabled)
    results: List[TrialRow] = []

    try:
        snd_start = sound.Sound(value=1000, secs=0.05)
        snd_correct = sound.Sound(value=1200, secs=0.12)
        snd_incorrect = sound.Sound(value=800, secs=0.12)
    except Exception:
        snd_start = snd_correct = snd_incorrect = None

    # MATLAB explicitly fills the full screen background before drawing each
    # texture. We reproduce that with an explicit Rect rather than relying on
    # Window.flip clearing behavior.
    background_rect = visual.Rect(
        win,
        width=geom.res_x,
        height=geom.res_y,
        pos=(0, 0),
        units="pix",
        fillColor=BACKGROUND_RGB,
        lineColor=BACKGROUND_RGB,
        colorSpace="rgb",
    )

    # Pre-created image stimuli for reuse; image content is swapped each frame.
    # PIL RGB images preserve MATLAB uint8 0..255 luminance behavior more robustly
    # than backend-dependent numpy/float textures on PsychoPy 2025 Windows.
    gabor_stim = visual.ImageStim(
        win,
        image=image255_to_pil_rgb(np.zeros((bps, bps), dtype=np.uint8) + int(BACKGROUND_255)),
        pos=stim_pos,
        size=(bps, bps),
        units="pix",
        interpolate=False,
    )
    mask_stim = visual.ImageStim(
        win,
        image=image255_to_pil_rgb(np.zeros((bps, bps), dtype=np.uint8) + int(BACKGROUND_255)),
        pos=stim_pos,
        size=(bps, bps),
        units="pix",
        interpolate=False,
    )

    response_clock = core.Clock()
    aborted = False
    error_info: Optional[str] = None
    session_start = core.getTime()

    try:
        for trial_count, staircase_id in enumerate(trial_structure, start=1):
            # MATLAB presentScore before selecting staircase/difficulty.
            draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, background_rect, show_score=show_score)
            win.flip()

            # MATLAB Select Staircase and current difficulty.
            if staircase_id == 0:
                if stair_step[1] < 5:
                    angle_deviation = BASE_ANGLE_DEG[0]
                else:
                    angle_deviation = BASE_ANGLE_DEG[stair_step[1] - 4]  # MATLAB stairStep1-3 -> Python -4
            else:
                if stair_count[staircase_id] == CORRECTS_TO_STEP_DOWN[staircase_id]:
                    stair_step[staircase_id] += 1
                    stair_count[staircase_id] = 0
                    if stair_step[staircase_id] > len(BASE_ANGLE_DEG):
                        stair_step[staircase_id] = len(BASE_ANGLE_DEG)
                angle_deviation = BASE_ANGLE_DEG[stair_step[staircase_id] - 1]

            stair_step_pre = int(stair_step.get(staircase_id, 0)) if staircase_id in (1, 2, 3) else 0
            stair_count_pre = int(stair_count.get(staircase_id, 0)) if staircase_id in (1, 2, 3) else 0

            # MATLAB randomizes tilt sign with CoinFlip(1,.5).
            # Negative angle = left response correct; positive angle = right response correct.
            if np.random.rand() < 0.5:
                signed_angle = -float(angle_deviation)
                target_side = "left"
                correct_hc, correct_lc = "1", "2"
                incorrect_hc, incorrect_lc = "0", "9"
            else:
                signed_angle = float(angle_deviation)
                target_side = "right"
                correct_hc, correct_lc = "0", "9"
                incorrect_hc, incorrect_lc = "1", "2"

            # MATLAB Beeper(1000) immediately before stimulus presentation.
            if snd_start is not None:
                snd_start.play()

            # For Stimulus: build and draw each movie frame with score/counter/fixation.
            # MATLAB does not collect behavioral responses during this period.
            stimulus_start_s = core.getTime()
            for ramp_value in temporal_env:
                frame = make_gabor_frame(x, y, spatial_env, signed_angle, ramp_value, geom)
                gabor_stim.image = image255_to_pil_rgb(frame)
                background_rect.draw()
                gabor_stim.draw()
                # Draw only text/fixation overlay after the stimulus, matching MATLAB
                # order: FillRect -> DrawTexture -> DrawText -> FillOval -> Flip.
                draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, show_score=show_score)
                win.flip()

            stimulus_end_s = core.getTime()
            stimulus_duration_actual_s = stimulus_end_s - stimulus_start_s

            # MATLAB presentScore immediately after stimulus.
            draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, background_rect, show_score=show_score)
            win.flip()

            # Draw Mask for delayMask seconds.
            mask_float = gen_images_dt(MASK_WIDTH_PX, MASK_SF_CPP, MASK_STD_SF_CPP, MASK_ORIENTATION_DEG, MASK_COHERENCE, mask_aperture)
            mask_img = np.uint8(np.clip(mask_float * CONTRAST_PERCENT + BACKGROUND_255, 0, 255))
            mask_img = crop_or_pad_to_size(mask_img, bps, BACKGROUND_255)
            mask_stim.image = image255_to_pil_rgb(mask_img)
            background_rect.draw()
            mask_stim.draw()
            draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, show_score=show_score)
            win.flip()
            core.wait(DELAY_MASK_S)

            # MATLAB presentScore again before collecting response, then tic.
            draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, background_rect, show_score=show_score)
            win.flip()
            response_clock.reset()

            rs = 0
            confidence = 0
            response_key = ""
            response_side = ""
            points_delta = 0
            response_phase = "post_mask_only"
            if metacognition_enabled:
                key_list = ["1", "2", "9", "0", "escape"]
            else:
                # Default pragmatic training mode: decision only, no confidence,
                # no confidence-weighted score.
                key_list = ["left", "right", "escape"]

            while True:
                keys = event.waitKeys(keyList=key_list, timeStamped=response_clock)
                if not keys:
                    continue
                key, rt = keys[0]
                response_key = key
                if key == "escape":
                    aborted = True
                    raise KeyboardInterrupt("Session aborted by Escape")

                if metacognition_enabled:
                    if key == correct_hc:
                        rs = 1
                        confidence = 1
                        points_delta = 50
                        response_side = target_side
                        running_score += points_delta
                    elif key == correct_lc:
                        rs = 1
                        confidence = 0
                        points_delta = 20
                        response_side = target_side
                        running_score += points_delta
                    elif key == incorrect_lc:
                        rs = 0
                        confidence = 0
                        points_delta = 0
                        response_side = "right" if target_side == "left" else "left"
                    elif key == incorrect_hc:
                        rs = 0
                        confidence = 1
                        points_delta = -50
                        response_side = "right" if target_side == "left" else "left"
                        running_score += points_delta
                else:
                    left_correct = signed_angle < 0
                    response_side = key
                    if (key == "left" and left_correct) or (key == "right" and not left_correct):
                        rs = 1
                    else:
                        rs = 0
                    confidence = -1
                    points_delta = 0
                response_time = float(rt)
                break

            # MATLAB staircase update. Catch trials (0) do not update staircase counts/steps.
            if rs == 1:
                if staircase_id in (1, 2, 3):
                    stair_count[staircase_id] += 1
                if snd_correct is not None:
                    snd_correct.play()
            else:
                if staircase_id in (1, 2, 3):
                    if stair_step[staircase_id] > 1:
                        stair_step[staircase_id] -= 1
                    stair_count[staircase_id] = 0
                if snd_incorrect is not None:
                    snd_incorrect.play()

            stair_step_post = int(stair_step.get(staircase_id, 0)) if staircase_id in (1, 2, 3) else 0
            stair_count_post = int(stair_count.get(staircase_id, 0)) if staircase_id in (1, 2, 3) else 0

            # MATLAB result columns:
            # 1 trial, 2 abs difficulty, 3 RT, 4 signed angle, 5 correct, 6 confidence, 7 staircase.
            # Extra columns are added to the CSV/JSON-oriented output for analysis
            # transparency; the .mat results matrix keeps the 7 MATLAB columns.
            results.append(
                TrialRow(
                    trial=trial_count,
                    difficulty_abs_deg=abs(float(signed_angle)),
                    response_time_s=response_time,
                    signed_angle_deg=float(signed_angle),
                    correct=int(rs),
                    confidence=int(confidence),
                    staircase=int(staircase_id),
                    response_key=response_key,
                    running_score=int(running_score),
                    is_bonus_trial=int(staircase_id == 0),
                    stair_step_pre=int(stair_step_pre),
                    stair_step_post=int(stair_step_post),
                    stair_count_pre=int(stair_count_pre),
                    stair_count_post=int(stair_count_post),
                    target_side=target_side,
                    response_side=response_side,
                    points_delta=int(points_delta),
                    stimulus_start_s=float(stimulus_start_s - session_start),
                    stimulus_end_s=float(stimulus_end_s - session_start),
                    stimulus_duration_actual_s=float(stimulus_duration_actual_s),
                    response_phase=response_phase,
                )
            )

            draw_score_and_fixation(win, running_score, trial_count, total_trials_actual, fix_pos, fix_diameter_px, stimulus_radius_px, background_rect, show_score=show_score)
            win.flip()
            core.wait(ITI_S)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error_info = repr(exc)
        traceback.print_exc()

    session_total_s = core.getTime() - session_start

    # -------------------------------------------------------------------------
    # Output block: MATLAB-compatible matrix + CSV + JSON detailed report.
    # -------------------------------------------------------------------------
    date_str = _dt.datetime.now().strftime("%Y%m%d")
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{subject_id}_OriMetaCog_{int(round(h_ecc_stim_deg))}_{int(round(v_ecc_stim_deg))}_{timestamp}"

    results_matrix = np.zeros((len(results), 7), dtype=float)
    for idx, row in enumerate(results):
        results_matrix[idx, :] = [
            row.trial,
            row.difficulty_abs_deg,
            row.response_time_s,
            row.signed_angle_deg,
            row.correct,
            row.confidence,
            row.staircase,
        ]

    csv_path = os.path.join(output_dir, base_name + "_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "difficulty_abs_deg",
            "response_time_s",
            "signed_angle_deg",
            "correct",
            "confidence",
            "staircase",
            "response_key",
            "running_score",
            "is_bonus_trial",
            "stair_step_pre",
            "stair_step_post",
            "stair_count_pre",
            "stair_count_post",
            "target_side",
            "response_side",
            "points_delta",
            "stimulus_start_s",
            "stimulus_end_s",
            "stimulus_duration_actual_s",
            "response_phase",
        ])
        for row in results:
            writer.writerow([
                row.trial,
                row.difficulty_abs_deg,
                row.response_time_s,
                row.signed_angle_deg,
                row.correct,
                row.confidence,
                row.staircase,
                row.response_key,
                row.running_score,
                row.is_bonus_trial,
                row.stair_step_pre,
                row.stair_step_post,
                row.stair_count_pre,
                row.stair_count_post,
                row.target_side,
                row.response_side,
                row.points_delta,
                row.stimulus_start_s,
                row.stimulus_end_s,
                row.stimulus_duration_actual_s,
                row.response_phase,
            ])

    mat_path = os.path.join(output_dir, base_name + ".mat")
    if savemat is not None:
        savemat(
            mat_path,
            {
                "results": results_matrix,
                "initials": subject_id,
                "h_ecc_orig": h_ecc_stim_deg,
                "v_ecc_orig": v_ecc_stim_deg,
                "baseAngle": np.array(BASE_ANGLE_DEG),
                "trialStructure": np.array(trial_structure[: len(results)]),
                "runningScore": running_score,
                "time": date_str,
            },
        )
    else:
        mat_path = ""

    accuracy = float(np.mean([r.correct for r in results]) * 100.0) if results else float("nan")
    confidence_rate = float(np.mean([r.confidence for r in results if r.confidence >= 0]) * 100.0) if any(r.confidence >= 0 for r in results) else float("nan")
    completed = len(results) == total_trials_actual and not aborted and error_info is None

    final_staircase_steps = {str(k): int(v) for k, v in stair_step.items()}
    final_staircase_values_deg = {
        str(k): float(BASE_ANGLE_DEG[max(0, min(len(BASE_ANGLE_DEG)-1, v-1))])
        for k, v in stair_step.items()
    }
    final_values = list(final_staircase_values_deg.values())
    final_staircase_mean_deg = float(np.mean(final_values)) if final_values else float("nan")
    final_staircase_sd_deg = float(np.std(final_values)) if final_values else float("nan")
    trial_counts_by_staircase = {
        str(k): int(sum(1 for r in results if r.staircase == k))
        for k in [0, 1, 2, 3]
    }
    correct_counts_by_staircase = {
        str(k): int(sum(1 for r in results if r.staircase == k and r.correct == 1))
        for k in [0, 1, 2, 3]
    }
    accuracy_by_staircase = {
        str(k): (
            100.0 * correct_counts_by_staircase[str(k)] / trial_counts_by_staircase[str(k)]
            if trial_counts_by_staircase[str(k)] > 0 else None
        )
        for k in [0, 1, 2, 3]
    }
    response_side_counts = {
        "left": int(sum(1 for r in results if r.response_side == "left")),
        "right": int(sum(1 for r in results if r.response_side == "right")),
    }
    target_side_counts = {
        "left": int(sum(1 for r in results if r.target_side == "left")),
        "right": int(sum(1 for r in results if r.target_side == "right")),
    }
    rt_values = np.array([r.response_time_s for r in results if np.isfinite(r.response_time_s)], dtype=float)
    rt_summary = {
        "median_s": float(np.median(rt_values)) if rt_values.size else None,
        "mean_s": float(np.mean(rt_values)) if rt_values.size else None,
        "sd_s": float(np.std(rt_values)) if rt_values.size else None,
    }
    stimulus_duration_values = np.array([r.stimulus_duration_actual_s for r in results if np.isfinite(r.stimulus_duration_actual_s)], dtype=float)
    timing_summary = {
        "stimulus_duration_actual_mean_s": float(np.mean(stimulus_duration_values)) if stimulus_duration_values.size else None,
        "stimulus_duration_actual_sd_s": float(np.std(stimulus_duration_values)) if stimulus_duration_values.size else None,
        "stimulus_duration_actual_min_s": float(np.min(stimulus_duration_values)) if stimulus_duration_values.size else None,
        "stimulus_duration_actual_max_s": float(np.max(stimulus_duration_values)) if stimulus_duration_values.size else None,
        "response_collection_phase": "post_mask_only",
        "matlab_allows_response_during_stimulus": False,
    }

    summary = {
        "subject": subject_id,
        "task": "FineOrientation_GaborPatch_MetaCog",
        "source_lineage": "TestingCodes/FineOrientation/FineOrientationDiscrimination_MetaCog.m",
        "iso_reference_mode": bool(metacognition_enabled),
        "metacognition_enabled": bool(metacognition_enabled),
        "n_trials_requested": int(total_trials),
        "n_trials_completed": len(results),
        "n_trials_planned_after_ratio_adjustment": int(total_trials_actual),
        "n_trials_matlab_default": TOTAL_TRIALS_DEFAULT,
        "session_complete": bool(completed),
        "accuracy_percent": accuracy,
        "confidence_high_percent": confidence_rate,
        "final_running_score": int(running_score),
        "score_visible_during_task": bool(show_score),
        "session_total_s": session_total_s,
        "rt_summary": rt_summary,
        "timing_summary": timing_summary,
        "trial_counts_by_staircase": trial_counts_by_staircase,
        "correct_counts_by_staircase": correct_counts_by_staircase,
        "accuracy_by_staircase": accuracy_by_staircase,
        "target_side_counts": target_side_counts,
        "response_side_counts": response_side_counts,
        "final_staircase_steps_1_based": final_staircase_steps,
        "final_staircase_values_deg": final_staircase_values_deg,
        "final_staircase_mean_deg": final_staircase_mean_deg,
        "final_staircase_sd_deg": final_staircase_sd_deg,
        "stimulus_location_deg": {"H": h_ecc_stim_deg, "V": v_ecc_stim_deg},
        "fixation_location_deg": {"H": h_ecc_fix_deg, "V": v_ecc_fix_deg},
        "protocol_constants": {
            "delay_mask_s": DELAY_MASK_S,
            "n_trials_per_staircase": N_TRIALS_PER_STAIRCASE,
            "bonus_trials": BONUS_TRIALS,
            "iti_s": ITI_S,
            "base_angle_deg": BASE_ANGLE_DEG,
            "initial_stair_steps_1_based": INITIAL_STAIR_STEPS,
            "corrects_to_step_down": CORRECTS_TO_STEP_DOWN,
            "onset_s": ONSET_S,
            "offset_s": OFFSET_S,
            "duration_s": DURATION_S,
            "sf_cycles_per_deg": SF_CYCLES_PER_DEG,
            "stim_radius_deg": STIM_RADIUS_DEG,
            "contrast_percent": CONTRAST_PERCENT,
            "spatial_envelope": SPATIAL_ENVELOPE,
            "mask_width_px": MASK_WIDTH_PX,
            "mask_delay_s": DELAY_MASK_S,
            "response_collection_phase": "post_mask_only",
            "metacognition_default_enabled": False,
            "default_response_keys": "left/right arrows when metacognition disabled",
            "matlab_response_keys": "1/2/9/0 when metacognition enabled",
        },
        "monitor_geometry": {
            "screen_width_cm": geom.width_cm,
            "viewing_distance_cm": geom.distance_cm,
            "resolution_px": [geom.res_x, geom.res_y],
            "arcmin_per_pix": geom.arcmin_per_pix,
            "refresh_hz_actual": geom.refresh_hz,
            "stimulus_radius_px": stimulus_radius_px,
            "patch_size_px": bps,
            "temporal_envelope_frames": mv_length,
            "screen_profile_metadata": screen_profile_metadata or {},
        },
        "outputs": {
            "csv": os.path.basename(csv_path),
            "mat": os.path.basename(mat_path) if mat_path else None,
        },
        "aborted": aborted,
        "error": error_info,
        "timestamp": timestamp,
    }
    json_path = os.path.join(output_dir, base_name + "_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# Entry point / UX
# -----------------------------------------------------------------------------


def main() -> Optional[Dict[str, object]]:
    info = {
        "Subject ID": DEFAULT_SUBJECT_ID,
        "Stimulus horizontal eccentricity X (deg; Left(-) to Right(+))": str(DEFAULT_H_ECC_STIM_DEG),
        "Stimulus vertical eccentricity Y (deg; Down(-) to Up(+))": str(DEFAULT_V_ECC_STIM_DEG),
        "Number of trials": TOTAL_TRIALS_DEFAULT,
        "Activate metacognition?": False,
    }
    dlg = gui.DlgFromDict(
        info,
        title="Fine Orientation / Gabor Patch MetaCog",
        order=[
            "Subject ID",
            "Stimulus horizontal eccentricity X (deg; Left(-) to Right(+))",
            "Stimulus vertical eccentricity Y (deg; Down(-) to Up(+))",
            "Number of trials",
            "Activate metacognition?",
        ],
    )
    if not dlg.OK:
        return None

    try:
        subject_id = str(info["Subject ID"])
        h_ecc_stim = float(info["Stimulus horizontal eccentricity X (deg; Left(-) to Right(+))"])
        v_ecc_stim = float(info["Stimulus vertical eccentricity Y (deg; Down(-) to Up(+))"])
        total_trials = int(info["Number of trials"])
        metacognition_enabled = bool(info["Activate metacognition?"])
    except Exception:
        err = gui.Dlg(title="Input error")
        err.addText("One or more inputs were invalid.")
        err.show()
        return None

    mon_geom = load_or_calibrate_monitor()
    if mon_geom is None:
        return None
    mon, geom = mon_geom

    win = visual.Window(
        size=(geom.res_x, geom.res_y),
        fullscr=True,
        monitor=mon,
        units="pix",
        color=BACKGROUND_RGB,
        colorSpace="rgb",
        allowGUI=False,
    )

    refresh = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=20)
    if refresh is None or refresh <= 0:
        refresh = 60.0
    geom.refresh_hz = float(refresh)

    if metacognition_enabled:
        response_text = (
            "Response keys (MATLAB-ISO MetaCog mode):\n"
            "1 = confident LEFT\n"
            "2 = uncertain LEFT\n"
            "9 = uncertain RIGHT\n"
            "0 = confident RIGHT\n\n"
            "Score:\n"
            "Confident + Correct = +50\n"
            "Not Confident + Correct = +20\n"
            "Not Confident + Incorrect = 0\n"
            "Confident + Incorrect = -50"
        )
    else:
        response_text = (
            "Response keys (default practical training mode):\n"
            "LEFT arrow = left tilt\n"
            "RIGHT arrow = right tilt\n\n"
            "Metacognition and confidence-weighted score are disabled. "
            "The exercise is scored by accuracy and threshold in the analysis dashboard."
        )

    readme = (
        "READ ME / SETUP\n\n"
        "This task ports the MATLAB Fine Orientation Discrimination MetaCog exercise.\n"
        "A static Gabor patch appears briefly, followed by a mask. Judge whether the\n"
        "patch orientation tilts LEFT or RIGHT. It does not move.\n\n"
        "Default mode uses LEFT/RIGHT arrows only. The original MATLAB metacognition\n"
        "keys (1/2/9/0) can still be activated from the launch dialog for ISO research mode.\n\n"
        "SCREEN REQUIREMENTS FOR LONGITUDINAL COMPARABILITY\n"
        "- Use the same physical screen every session.\n"
        "- Keep the same screen resolution and Windows/macOS display scaling.\n"
        "- Keep brightness and contrast fixed; do not use automatic brightness.\n"
        "- Disable Night Light / Night Shift, blue-light filters, True Tone, HDR, adaptive contrast, gaming modes, and dynamic color modes.\n"
        "- Use SDR mode rather than HDR.\n"
        "- Let the screen warm up for a few minutes before training if possible.\n"
        "- Avoid direct sunlight, reflections, and large room-light changes.\n"
        "- Keep the chin/head position at the calibrated viewing distance.\n"
        "- Do not compare thresholds across different screens unless this is explicitly marked as a new setup.\n\n"
        f"Subject: {subject_id}\n"
        f"Stimulus location: X={h_ecc_stim:.2f}°, Y={v_ecc_stim:.2f}°\n"
        f"Trials: {total_trials} (MATLAB default = {TOTAL_TRIALS_DEFAULT})\n"
        f"Viewing distance: {geom.distance_cm:.2f} cm\n"
        f"Screen width: {geom.width_cm:.2f} cm\n"
        f"Detected refresh: {geom.refresh_hz:.2f} Hz\n"
        f"Difficulty scale: {BASE_ANGLE_DEG}\n"
        f"Stimulus radius: {STIM_RADIUS_DEG}°\n"
        f"Onset ramp: {ONSET_S}s | Offset ramp: {OFFSET_S}s | Mask: {DELAY_MASK_S}s | ITI: {ITI_S}s\n\n"
        + response_text
        + "\n\nPress SPACE to start or ESC to quit."
    )
    visual.TextStim(
        win,
        text=readme,
        color=BLACK_RGB,
        colorSpace="rgb",
        height=24,
        wrapWidth=min(1100, geom.res_x * 0.85),
        font=FONT,
        units="pix",
    ).draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        win.close()
        return None

    summary = run_session(
        win=win,
        geom=geom,
        subject_id=subject_id,
        h_ecc_stim_deg=h_ecc_stim,
        v_ecc_stim_deg=v_ecc_stim,
        h_ecc_fix_deg=DEFAULT_H_ECC_FIX_DEG,
        v_ecc_fix_deg=DEFAULT_V_ECC_FIX_DEG,
        total_trials=total_trials,
        metacognition_enabled=metacognition_enabled,
        save_root=BASE_DIR,
        screen_profile_metadata=getattr(mon, "_psychopi_profile_metadata", {}),
    )

    if summary.get("error"):
        msg = (
            "Session ended due to an unexpected error.\n\n"
            f"Trials saved: {summary.get('n_trials_completed', 0)}\n"
            "Review the CSV/JSON/MAT output in the exercise output folder.\n\n"
            "Press any key to exit."
        )
    elif summary.get("aborted"):
        msg = (
            "Session interrupted.\n\n"
            f"Trials saved: {summary.get('n_trials_completed', 0)}\n"
            f"Accuracy so far: {summary.get('accuracy_percent', float('nan')):.1f}%\n"
            "Press any key to exit."
        )
    else:
        msg = (
            "Training complete.\n\n"
            f"Accuracy: {summary.get('accuracy_percent', float('nan')):.1f}%\n"
            f"Final score: {summary.get('final_running_score', 0)}\n"
            "Outputs saved in DataFineOrientation.\n\n"
            "Press any key to exit."
        )
    visual.TextStim(
        win,
        text=msg,
        color=BLACK_RGB,
        colorSpace="rgb",
        height=30,
        wrapWidth=min(1100, geom.res_x * 0.85),
        font=FONT,
        units="pix",
    ).draw()
    win.flip()
    event.waitKeys()
    win.close()
    return summary


if __name__ == "__main__":
    main()
