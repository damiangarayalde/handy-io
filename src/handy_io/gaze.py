"""
Gaze-to-screen mapping.

Two modes, selected by the MODE constant in main.py:

  "calibration_only"
      Measures iris position relative to the eye-corner span (2-D image coords),
      then maps that ratio through a fitted calibration matrix.
      Simple, no 3-D geometry — but still requires a calibration pass so the
      mapping is per-person and per-camera-placement rather than hard-coded magic
      numbers.  DEFAULT.

  "full"
      Extracts a head-pose-compensated 3-D gaze ray using the facial
      transformation matrix from MediaPipe, estimates camera-to-face distance
      from the inter-pupillary distance in pixels, then projects the ray onto a
      virtual screen plane.  Also requires calibration (to correct residual
      per-person angular offsets), but additionally handles head distance changes.

Each mode exposes two functions:

  extract_features(gl)  →  np.ndarray  feature vector fed to calibration / mapping
  map_to_screen(feat, calib, screen_wh, smoothing, _prev)  →  (x, y)
"""

from __future__ import annotations
import numpy as np
from .landmarks import GazeLandmarks

# ── assumed average inter-pupillary distance (metres) ────────────────────────
_IPD_METRES = 0.063
# ── camera focal length resolved at runtime from actual FOV ──────────────────
# Filled in by _get_focal_px(); falls back to 800 if the camera reports no FOV.
_FOCAL_PX: float | None = None


def _get_focal_px(cap=None) -> float:
    """
    Estimate focal length in pixels from the camera's actual frame width and
    an assumed 60° horizontal FOV (typical for built-in webcams).
    Formula: f = (image_width / 2) / tan(HFOV / 2)
    """
    import math
    import cv2 as _cv2
    width = cap.get(_cv2.CAP_PROP_FRAME_WIDTH) if cap is not None else 1280.0
    if width <= 0:
        width = 1280.0
    hfov_deg = 60.0  # conservative default; override HFOV_DEG if you know yours
    return (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))


def init_focal_px(cap=None) -> float:
    """Call once after opening the camera to set the module-level focal length."""
    global _FOCAL_PX
    _FOCAL_PX = _get_focal_px(cap)
    print(f"[gaze] focal length set to {_FOCAL_PX:.1f} px")
    return _FOCAL_PX

# ── left/right eye corner indices inside GazeLandmarks arrays ────────────────
_OUTER, _INNER = 0, 1


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iris_2d_offset(iris: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Return (ox, oy) in roughly [-1, 1]: iris centre relative to eye-corner span,
    measured in 2-D normalised image coordinates.
    """
    center = iris[0, :2]
    outer, inner = corners[_OUTER, :2], corners[_INNER, :2]
    eye_mid = (outer + inner) / 2.0
    eye_width = np.linalg.norm(inner - outer) + 1e-6
    return (center - eye_mid) / eye_width


def _smooth(new_xy: np.ndarray, alpha: float, store: list[np.ndarray]) -> np.ndarray:
    """Exponential moving average.  store is a one-element list used as mutable state."""
    if not store:
        store.append(new_xy.copy())
    else:
        store[0] = alpha * store[0] + (1.0 - alpha) * new_xy
    return store[0]


# ─────────────────────────────────────────────────────────────────────────────
# Mode: "calibration_only"
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_simple(gl: GazeLandmarks) -> np.ndarray:
    """
    Feature vector for calibration_only mode.

    Returns a 4-element vector:
      [left_ox, left_oy, right_ox, right_oy]
    where ox/oy are iris-in-eye offsets in 2-D image space averaged.
    """
    left_off = _iris_2d_offset(gl.left_iris, gl.left_eye_corners)
    right_off = _iris_2d_offset(gl.right_iris, gl.right_eye_corners)
    return np.array([left_off[0], left_off[1], right_off[0], right_off[1]])


# ─────────────────────────────────────────────────────────────────────────────
# Mode: "full"
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_distance_m(gl: GazeLandmarks) -> float:
    """
    Estimate camera-to-face distance in metres from the pixel span between iris
    centres using the thin-lens formula:  Z = f * IPD_real / IPD_pixels.

    Falls back to 0.65 m if landmarks are degenerate.
    """
    w, h = gl.image_wh
    left_px = gl.left_iris[0, :2] * np.array([w, h])
    right_px = gl.right_iris[0, :2] * np.array([w, h])
    ipd_px = np.linalg.norm(right_px - left_px)
    if ipd_px < 5:
        return 0.65
    focal = _FOCAL_PX if _FOCAL_PX is not None else 800.0
    return focal * _IPD_METRES / ipd_px


def _gaze_ray_head_compensated(gl: GazeLandmarks) -> np.ndarray:
    """
    Compute a unit gaze-direction vector in camera space by:
      1. Measuring the average iris offset in the eye-corner frame (2-D).
      2. Lifting that to a 3-D direction in head-local space.
      3. Rotating it by the head-pose matrix into camera space.

    If face_transform is unavailable, falls back to a simple forward vector
    with the 2-D offset applied directly (same quality as calibration_only).

    Returns shape (3,) unit vector pointing from the camera toward where the
    user is looking.
    """
    left_off = _iris_2d_offset(gl.left_iris, gl.left_eye_corners)
    right_off = _iris_2d_offset(gl.right_iris, gl.right_eye_corners)
    avg_off = (left_off + right_off) / 2.0

    # Gaze direction in head-local space: x/y from iris offset, z forward.
    # Scale of ~0.2 maps a max iris offset of 1.0 to ~11° — roughly the
    # physical limit of comfortable eye rotation.
    local_dir = np.array([avg_off[0] * 0.2, avg_off[1] * 0.2, 1.0])
    local_dir /= np.linalg.norm(local_dir)

    if gl.face_transform is None:
        return local_dir

    # Extract 3×3 rotation from the 4×4 transform (upper-left block).
    R = gl.face_transform[:3, :3]
    world_dir = R @ local_dir
    norm = np.linalg.norm(world_dir)
    return world_dir / (norm + 1e-9)


def extract_features_full(gl: GazeLandmarks) -> np.ndarray:
    """
    Feature vector for full mode.

    Distance is used to project the gaze ray onto a virtual screen plane at a
    fixed reference depth, so the calibration matrix only needs to correct
    angular offsets — not compensate for nonlinear depth-scale variation.

    Screen-plane intersection at reference depth D:
      hit_x = D * ray_x / ray_z
      hit_y = D * ray_y / ray_z

    This is the correct perspective projection; it automatically expands/
    contracts with distance so the calibration holds regardless of how far
    the user sits from the screen.

    Returns a 3-element vector [hit_x, hit_y, 1.0].
    """
    ray  = _gaze_ray_head_compensated(gl)
    # Reference plane depth: use the estimated distance so the projection
    # stays in a stable coordinate range across sessions.
    dist = _estimate_distance_m(gl)
    rz   = ray[2] if abs(ray[2]) > 1e-3 else 1e-3
    hit_x = dist * ray[0] / rz
    hit_y = dist * ray[1] / rz
    return np.array([hit_x, hit_y, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Unified screen mapping (both modes)
# ─────────────────────────────────────────────────────────────────────────────

def map_to_screen(
    features: np.ndarray,
    calib_matrix: np.ndarray,
    screen_wh: tuple[int, int],
    smoothing: float = 0.75,
    _prev: list[np.ndarray] | None = None,
) -> tuple[int, int]:
    """
    Apply the calibration matrix and return a screen pixel coordinate.

    calib_matrix shape: (2, len(features))  — maps feature vector to (gaze_x_norm, gaze_y_norm).
    """
    sw, sh = screen_wh
    raw = calib_matrix @ features          # shape (2,)
    gaze = np.clip(raw, 0.0, 1.0)

    if smoothing > 0.0 and _prev is not None:
        gaze = _smooth(gaze, smoothing, _prev)

    return int(gaze[0] * sw), int(gaze[1] * sh)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy uncalibrated fallback (used before calibration is complete)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_gaze_uncalibrated(
    gl: GazeLandmarks,
    screen_wh: tuple[int, int],
    h_scale: float = 3.5,
    v_scale: float = 4.5,
    smoothing: float = 0.75,
    _prev: list[np.ndarray] | None = None,
) -> tuple[int, int]:
    """
    Original heuristic mapping — used only during the calibration phase when
    the calibration matrix is not yet available (so the display has something
    to show).  Not used in normal operation after calibration completes.
    """
    sw, sh = screen_wh
    feat = extract_features_simple(gl)
    avg_off = np.array([(feat[0] + feat[2]) / 2.0, (feat[1] + feat[3]) / 2.0])
    gaze_x = float(np.clip(0.5 - avg_off[0] * h_scale, 0.0, 1.0))
    gaze_y = float(np.clip(0.5 + avg_off[1] * v_scale, 0.0, 1.0))
    gaze = np.array([gaze_x, gaze_y])
    if smoothing > 0.0 and _prev is not None:
        gaze = _smooth(gaze, smoothing, _prev)
    return int(gaze[0] * sw), int(gaze[1] * sh)
