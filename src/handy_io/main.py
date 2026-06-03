"""
handy-io milestone 1 — live gaze heatmap with calibration.

Single fullscreen window:
  • During calibration: shows the 9-dot calibration screen.
  • After calibration: shows the camera feed with heatmap blended on top.
  • ArUco markers are always rendered in the four corners of the display so a
    glasses-mounted camera can see them and recover the head-to-screen pose.

Keys:
  q / ESC  — quit
  c        — clear heatmap
  r        — re-run calibration
  d        — toggle glasses-cam debug overlay (shows detected ArUco corners)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE controls which gaze pipeline is used:

  "calibration_only"  (DEFAULT)
      2-D iris-in-eye offset features fed through a polynomial calibration
      matrix.  Simple, single-camera.

  "full"
      Head-pose-compensated 3-D gaze ray + distance estimation.
      Also requires calibration but additionally handles the user
      leaning forward or back during a session.

GLASSES_CAM controls the glasses-mounted second camera:

  -1  (DEFAULT) — disabled; falls back to single-camera calibration.
  0, 1, 2, …   — camera index of the glasses-mounted camera.
                  When active the ArUco markers in the screen corners are used
                  to geometrically project the gaze ray onto the screen,
                  replacing the regression calibration matrix.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import threading
import cv2
import numpy as np

from .capture import open_camera, frame_generator
from .landmarks import FaceMesh
from .heatmap import GazeHeatmap
from . import gaze, calibration, pose

# ── MODE ────────────────────────────────────────────────────────────────────
MODE: str = "calibration_only"   # "calibration_only" | "full"

# ── tunables ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_GLASSES = -1        # set to 1 (or 2, …) when glasses-cam is plugged in
CAM_W, CAM_H = 1280, 720
GLASSES_W, GLASSES_H = 640, 480

# Physical screen size in metres — update for your display.
SCREEN_SIZE_M = (0.334, 0.215)

SIGMA = 55.0
DECAY = 0.985
SMOOTHING = 0.75
ALPHA = 0.55        # heatmap blend opacity over camera feed
ARUCO_MARKER_PX = 80   # side-length of corner ArUco markers in pixels
WIN_NAME = "handy-io"
# ────────────────────────────────────────────────────────────────────────────

assert MODE in ("calibration_only", "full"), f"Unknown MODE: {MODE!r}"

_EXTRACT_FN = gaze.extract_features_simple if MODE == "calibration_only" \
    else gaze.extract_features_full
_NEED_FACE_TRANSFORM = MODE == "full"


def _make_fullscreen_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def _screen_size() -> tuple[int, int]:
    """Query the primary display resolution via tkinter (reliable on macOS)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def _run_calibration(
    cap: cv2.VideoCapture,
    mesh: FaceMesh,
    screen_wh: tuple[int, int],
) -> tuple[np.ndarray | None, int]:
    """Run the calibration sequence and return (matrix, poly_degree)."""
    cal = calibration.Calibrator(canvas_wh=screen_wh, extract_fn=_EXTRACT_FN)
    print("[calibration] fixate each dot — stable gaze auto-captures.  "
          "SPACE=manual  ESC=skip")

    frame_count = 0
    for _, frame_rgb in frame_generator(cap):
        frame_count += 1
        gl = mesh.process(frame_rgb)
        key = cv2.waitKey(16) & 0xFF
        canvas = cal.render_frame(gl, key)
        # Always draw ArUco markers even during calibration so the glasses-cam
        # can see them from the very first frame.
        pose.draw_corner_markers(canvas, marker_px=ARUCO_MARKER_PX)
        cv2.imshow(WIN_NAME, canvas)
        if cal.done:
            print(f"[calibration] complete — {frame_count} frames, "
                  f"{len(cal._samples)} dots captured")
            break
        if key in (ord("q"),):
            break

    if frame_count == 0:
        print("[calibration] ERROR: no camera frames received")

    poly_degree = getattr(cal, "poly_degree", calibration.POLY_DEGREE)
    return cal.matrix, poly_degree


def _draw_gaze_marker(frame: np.ndarray, x: int, y: int) -> None:
    r, thickness = 18, 2
    arm = 28
    col_outer = (255, 255, 255)
    col_inner = (0, 180, 255)
    cv2.circle(frame, (x, y), r, col_outer, thickness, cv2.LINE_AA)
    cv2.line(frame, (x - arm, y), (x - r - 2, y),
             col_outer, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + r + 2, y), (x + arm, y),
             col_outer, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y - arm), (x, y - r - 2),
             col_outer, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + r + 2), (x, y + arm),
             col_outer, thickness, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4, col_inner, -1, cv2.LINE_AA)


# ── glasses-cam thread ────────────────────────────────────────────────────────

class _GlassesCamThread(threading.Thread):
    """
    Reads frames from the glasses-mounted camera on a background thread and
    keeps the latest head-to-screen transform available via .transform.

    Runs as a daemon so it dies automatically when the main thread exits.
    """

    def __init__(self, cap: cv2.VideoCapture, estimator: pose.ScreenPoseEstimator) -> None:
        super().__init__(daemon=True)
        self._cap = cap
        self._estimator = estimator
        self._lock = threading.Lock()
        self._transform: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None
        self.running = True

    @property
    def transform(self) -> np.ndarray | None:
        with self._lock:
            return self._transform

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame

    def run(self) -> None:
        for frame_bgr, _ in frame_generator(self._cap):
            if not self.running:
                break
            t = self._estimator.update(frame_bgr)
            with self._lock:
                self._transform = t
                self._latest_frame = frame_bgr

    def stop(self) -> None:
        self.running = False


def run() -> None:
    cap = open_camera(CAMERA_INDEX, CAM_W, CAM_H)
    gaze.init_focal_px(cap)
    mesh = FaceMesh(output_face_transform=_NEED_FACE_TRANSFORM)

    # ── single shared fullscreen window ──────────────────────────────────────
    sw, sh = _screen_size()
    _make_fullscreen_window(WIN_NAME)
    print(f"handy-io  |  mode={MODE}  screen={sw}×{sh}")

    heatmap = GazeHeatmap(sw, sh, sigma=SIGMA, decay=DECAY)
    prev_gaze: list[np.ndarray] = []

    # ── optional glasses-cam setup ────────────────────────────────────────────
    glasses_thread: _GlassesCamThread | None = None
    if CAMERA_GLASSES >= 0:
        try:
            glasses_cap = open_camera(CAMERA_GLASSES, GLASSES_W, GLASSES_H)
            estimator = pose.ScreenPoseEstimator(
                glasses_cap, screen_size_m=SCREEN_SIZE_M)
            glasses_thread = _GlassesCamThread(glasses_cap, estimator)
            glasses_thread.start()
            print(f"[pose] glasses-cam started on index {CAMERA_GLASSES}")
        except RuntimeError as e:
            print(f"[pose] WARNING: could not open glasses-cam: {e}")
            glasses_thread = None

    # ── calibration phase ─────────────────────────────────────────────────────
    print("Running calibration …")
    calib_matrix, calib_poly_degree = _run_calibration(cap, mesh, (sw, sh))
    if calib_matrix is None:
        print("[calibration] not enough points — falling back to uncalibrated mode.")

    print("handy-io live  |  q=quit  c=clear  r=recalibrate  d=glasses-cam debug")

    show_glasses_debug = False

    # ── live phase ────────────────────────────────────────────────────────────
    try:
        for frame_bgr, frame_rgb in frame_generator(cap):
            heatmap.decay_frame()
            gl = mesh.process(frame_rgb)

            gx, gy = sw // 2, sh // 2  # fallback centre

            if gl is not None:
                # ── try geometric projection via glasses-cam pose first ───────
                screen_xy: np.ndarray | None = None

                if glasses_thread is not None:
                    T = glasses_thread.transform
                    if T is not None and MODE == "full":
                        ray = gaze._gaze_ray_head_compensated(gl)
                        screen_xy = pose.gaze_to_screen_geometric(
                            ray, T, screen_size_m=SCREEN_SIZE_M
                        )

                # ── fall back to calibration-matrix regression ────────────────
                if screen_xy is None and calib_matrix is not None:
                    feat = _EXTRACT_FN(gl)
                    screen_xy = calibration.apply(
                        feat, calib_matrix, calib_poly_degree)

                if screen_xy is not None:
                    if SMOOTHING > 0.0:
                        screen_xy = gaze._smooth(
                            screen_xy, SMOOTHING, prev_gaze)
                    gx = int(np.clip(screen_xy[0], 0.0, 1.0) * sw)
                    gy = int(np.clip(screen_xy[1], 0.0, 1.0) * sh)
                elif calib_matrix is None:
                    gx, gy = gaze.estimate_gaze_uncalibrated(
                        gl, (sw, sh), smoothing=SMOOTHING, _prev=prev_gaze
                    )

                heatmap.add_point(gx, gy)

            # ── compose display ───────────────────────────────────────────────
            display = cv2.resize(frame_bgr, (sw, sh))
            hm = heatmap.render()
            mask = hm.any(axis=2)
            display[mask] = cv2.addWeighted(
                display, 1 - ALPHA, hm, ALPHA, 0
            )[mask]

            if gl is not None:
                _draw_gaze_marker(display, gx, gy)

            # Draw ArUco markers in screen corners every frame
            pose.draw_corner_markers(display, marker_px=ARUCO_MARKER_PX)

            # Optional: small glasses-cam debug inset (top-right corner)
            if show_glasses_debug and glasses_thread is not None:
                gf = glasses_thread.latest_frame
                if gf is not None:
                    debug_frame = estimator.draw_debug(gf)
                    inset_w = sw // 4
                    inset_h = int(inset_w * gf.shape[0] / gf.shape[1])
                    inset = cv2.resize(debug_frame, (inset_w, inset_h))
                    x0 = sw - inset_w - 10
                    display[10: 10 + inset_h, x0: x0 + inset_w] = inset

            # Pose status indicator
            if glasses_thread is not None:
                T = glasses_thread.transform
                status = "POSE OK" if T is not None else "NO POSE"
                color = (0, 220, 0) if T is not None else (0, 80, 220)
                cv2.putText(display, status, (sw - 150, sh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                heatmap._buf[:] = 0
                prev_gaze.clear()
            elif key == ord("d"):
                show_glasses_debug = not show_glasses_debug
            elif key == ord("r"):
                prev_gaze.clear()
                heatmap._buf[:] = 0
                calib_matrix, calib_poly_degree = _run_calibration(
                    cap, mesh, (sw, sh))
                _make_fullscreen_window(WIN_NAME)

    finally:
        if glasses_thread is not None:
            glasses_thread.stop()
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
