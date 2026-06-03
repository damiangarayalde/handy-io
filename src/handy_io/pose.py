"""
Head-to-screen pose estimation via ArUco markers.

How it works
------------
Four ArUco markers (IDs 0-3) are rendered into the corners of the main display
window by draw_corner_markers().  A glasses-mounted camera sees those markers on
the physical screen.  detect_screen_pose() finds them and calls solvePnP to
recover a 4×4 rigid transform T_head_screen: the pose of the screen relative to
the glasses-cam (i.e. head) frame.

With T_head_screen available, gaze_to_screen_geometric() projects a 3-D gaze ray
(from gaze.py's _gaze_ray_head_compensated) directly onto the screen plane using
geometry, bypassing the regression calibration matrix entirely.

Coordinate convention
---------------------
Screen-plane origin = top-left corner of the screen.
+X = right, +Y = down, +Z = out of the screen toward the viewer.
Units = metres (MARKER_SIZE_M and SCREEN_SIZE_M define the scale).
"""

from __future__ import annotations
import cv2
import numpy as np


# ── tunables ──────────────────────────────────────────────────────────────────

# Physical side-length of each printed/rendered ArUco marker in metres.
# At a typical screen pixel density a marker that is ~80 px wide on a 27"
# 2560-wide display is about 0.025 m.  Adjust to match your actual screen.
MARKER_SIZE_M = 0.025

# Physical screen dimensions in metres (width, height).
# Defaults to a 27" 16:9 panel.  Override in main.py for your screen.
SCREEN_SIZE_M = (0.597, 0.336)

# Margin from the screen corner to the marker centre, in normalised [0,1] coords.
# 0.03 keeps the marker just inside the visible area.
CORNER_MARGIN = 0.03

# ArUco dictionary — 4×4 markers are fast to detect and robust at a distance.
_ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# Minimum number of markers that must be visible to consider the pose valid.
MIN_MARKERS = 2

# ── marker corner positions in screen-plane space (metres) ───────────────────
# Corners are ordered: top-left (0), top-right (1), bottom-right (2), bottom-left (3).
# Each entry is the 2-D position of that marker's centre on the screen plane.
def _marker_centres_m(screen_wm: float, screen_hm: float) -> np.ndarray:
    """Return (4, 2) array of marker centre positions in metres."""
    mx = MARKER_SIZE_M / 2
    my = MARKER_SIZE_M / 2
    return np.array([
        [mx,              my             ],   # ID 0: top-left
        [screen_wm - mx,  my             ],   # ID 1: top-right
        [screen_wm - mx,  screen_hm - my],   # ID 2: bottom-right
        [mx,              screen_hm - my],   # ID 3: bottom-left
    ], dtype=np.float64)


def _marker_object_points(centre_m: np.ndarray) -> np.ndarray:
    """
    Return the four corner 3-D object points (in screen-plane space) for one
    marker given its centre in metres.  Z = 0 for all (flat screen plane).
    """
    h = MARKER_SIZE_M / 2
    cx, cy = centre_m
    return np.array([
        [cx - h, cy - h, 0.0],  # top-left corner of marker
        [cx + h, cy - h, 0.0],  # top-right
        [cx + h, cy + h, 0.0],  # bottom-right
        [cx - h, cy + h, 0.0],  # bottom-left
    ], dtype=np.float64)


# ── drawing helpers ───────────────────────────────────────────────────────────

def _build_aruco():
    """Return (dictionary, detector_params) for the chosen ArUco config."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    return aruco_dict, params


_ARUCO_DICT, _DETECTOR_PARAMS = _build_aruco()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _DETECTOR_PARAMS)


def draw_corner_markers(
    canvas: np.ndarray,
    marker_px: int = 80,
    border_bits: int = 1,
) -> np.ndarray:
    """
    Draw four ArUco markers (IDs 0-3) near the corners of `canvas` in-place.

    Parameters
    ----------
    canvas      BGR image (the fullscreen display frame).
    marker_px   Side-length of each marker in pixels.
    border_bits Number of quiet-zone bits around the marker pattern.

    Returns the same canvas (modified in-place) for convenience.
    """
    h, w = canvas.shape[:2]
    margin_x = int(w * CORNER_MARGIN)
    margin_y = int(h * CORNER_MARGIN)

    positions = [
        (margin_x,     margin_y    ),   # ID 0: top-left
        (w - margin_x - marker_px, margin_y    ),   # ID 1: top-right
        (w - margin_x - marker_px, h - margin_y - marker_px),   # ID 2: bottom-right
        (margin_x,     h - margin_y - marker_px),   # ID 3: bottom-left
    ]

    for marker_id, (px, py) in enumerate(positions):
        marker_img = cv2.aruco.generateImageMarker(
            _ARUCO_DICT, marker_id, marker_px + 2 * border_bits * (marker_px // 6)
        )
        # Resize to exact marker_px so positioning math is exact
        marker_img = cv2.resize(marker_img, (marker_px, marker_px), interpolation=cv2.INTER_NEAREST)
        # Convert grayscale marker to BGR
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        x0, y0 = px, py
        x1, y1 = x0 + marker_px, y0 + marker_px
        # Clamp to canvas bounds
        x1 = min(x1, w)
        y1 = min(y1, h)
        canvas[y0:y1, x0:x1] = marker_bgr[: y1 - y0, : x1 - x0]

    return canvas


# ── pose estimation ───────────────────────────────────────────────────────────

class ScreenPoseEstimator:
    """
    Maintains camera intrinsics for the glasses-cam and estimates the
    head-to-screen transform each frame.

    Usage
    -----
    estimator = ScreenPoseEstimator(glasses_cap)
    # each frame from the glasses camera:
    T = estimator.update(glasses_frame_bgr)   # None if pose unavailable
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        screen_size_m: tuple[float, float] = SCREEN_SIZE_M,
    ) -> None:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Estimate intrinsics from FOV — same approach as gaze.init_focal_px.
        import math
        hfov_deg = 60.0  # conservative default for typical webcams
        fx = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        fy = fx  # square pixels assumed
        self._K = np.array([
            [fx,  0.0, w / 2.0],
            [0.0, fy,  h / 2.0],
            [0.0, 0.0, 1.0    ],
        ], dtype=np.float64)
        self._dist = np.zeros((4,), dtype=np.float64)  # assume undistorted

        sw_m, sh_m = screen_size_m
        self._centres_m = _marker_centres_m(sw_m, sh_m)  # (4, 2)

        # Last successful transform; None until first valid detection.
        self.transform: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Detect ArUco markers in `frame_bgr` and update the head-to-screen
        transform.  Returns the current 4×4 transform (or None).
        """
        corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame_bgr)
        if ids is None or len(ids) < MIN_MARKERS:
            return self.transform  # keep last good pose

        # Collect matched object/image point pairs
        obj_pts_list: list[np.ndarray] = []
        img_pts_list: list[np.ndarray] = []

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id > 3:
                continue
            centre_m = self._centres_m[marker_id]
            obj_pts_list.append(_marker_object_points(centre_m))
            # corners[i] shape: (1, 4, 2) — four corners of the detected marker
            img_pts_list.append(corners[i].reshape(4, 2))

        if len(obj_pts_list) < MIN_MARKERS:
            return self.transform

        obj_pts = np.vstack(obj_pts_list).astype(np.float64)
        img_pts = np.vstack(img_pts_list).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self._K, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self.transform

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = tvec.flatten()
        self.transform = T
        return T

    def draw_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Overlay detected marker outlines onto a copy of frame_bgr."""
        out = frame_bgr.copy()
        corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(out)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        return out


# ── geometric gaze projection ─────────────────────────────────────────────────

def gaze_to_screen_geometric(
    gaze_ray_cam: np.ndarray,
    T_head_screen: np.ndarray,
    screen_size_m: tuple[float, float] = SCREEN_SIZE_M,
) -> np.ndarray | None:
    """
    Project a 3-D gaze ray (in glasses/head-cam space) onto the screen plane
    using the head-to-screen transform from ArUco detection.

    Parameters
    ----------
    gaze_ray_cam    Unit direction vector (3,) in camera/head space.
    T_head_screen   4×4 transform: screen frame relative to head/cam frame.
                    (output of ScreenPoseEstimator.update)
    screen_size_m   Physical (width, height) of screen in metres.

    Returns
    -------
    np.ndarray of shape (2,) with normalised [0,1] screen coordinates (x, y),
    or None if the ray doesn't intersect the screen plane.
    """
    sw_m, sh_m = screen_size_m

    # Invert T to get the head/cam pose in screen coordinates.
    # T_head_screen maps screen-space → cam-space:  p_cam = T @ p_screen
    # We need the ray origin and direction in screen space.
    R = T_head_screen[:3, :3]
    t = T_head_screen[:3, 3]

    # Camera origin in screen space
    cam_origin_screen = -R.T @ t  # (3,)

    # Gaze direction in screen space
    ray_dir_screen = R.T @ gaze_ray_cam  # (3,)

    # Screen plane: Z = 0 in screen space.
    # Parametric ray: P(λ) = cam_origin_screen + λ * ray_dir_screen
    # Intersect with Z = 0: cam_origin_screen[2] + λ * ray_dir_screen[2] = 0
    rz = ray_dir_screen[2]
    if abs(rz) < 1e-6:
        return None  # ray parallel to screen

    lam = -cam_origin_screen[2] / rz
    if lam < 0:
        return None  # intersection behind the camera

    hit = cam_origin_screen + lam * ray_dir_screen  # (3,), Z≈0

    # Normalise to [0, 1] in screen space
    nx = hit[0] / sw_m
    ny = hit[1] / sh_m
    return np.array([nx, ny], dtype=np.float64)
