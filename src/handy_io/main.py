"""
handy-io milestone 1 — live gaze heatmap with calibration.

Single fullscreen window:
  • During calibration: shows the 9-dot calibration screen.
  • After calibration: shows the camera feed with heatmap blended on top.

Keys:
  q / ESC  — quit
  c        — clear heatmap
  r        — re-run calibration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE controls which gaze pipeline is used:

  "calibration_only"  (DEFAULT)
      2-D iris-in-eye offset features fed through a calibration matrix.

  "full"
      Head-pose-compensated 3-D gaze ray + distance estimation.
      Also requires calibration but additionally handles the user
      leaning forward or back during a session.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import cv2
import numpy as np

from .capture import open_camera, frame_generator
from .landmarks import FaceMesh
from .heatmap import GazeHeatmap
from . import gaze, calibration

# ── MODE ────────────────────────────────────────────────────────────────────
MODE: str = "calibration_only"   # "calibration_only" | "full"

# ── tunables ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAM_W, CAM_H = 1280, 720
SIGMA    = 55.0
DECAY    = 0.985
SMOOTHING = 0.75
ALPHA    = 0.55        # heatmap blend opacity over camera feed
WIN_NAME = "handy-io"
# ────────────────────────────────────────────────────────────────────────────

assert MODE in ("calibration_only", "full"), f"Unknown MODE: {MODE!r}"

_EXTRACT_FN        = gaze.extract_features_simple if MODE == "calibration_only" \
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
) -> np.ndarray | None:
    cal = calibration.Calibrator(canvas_wh=screen_wh, extract_fn=_EXTRACT_FN)
    print("[calibration] fixate each dot — stable gaze auto-captures.  "
          "SPACE=manual  ESC=skip")

    frame_count = 0
    for _, frame_rgb in frame_generator(cap):
        frame_count += 1
        gl  = mesh.process(frame_rgb)
        key = cv2.waitKey(16) & 0xFF
        canvas = cal.render_frame(gl, key)
        cv2.imshow(WIN_NAME, canvas)
        if cal.done:
            print(f"[calibration] complete — {frame_count} frames, "
                  f"{len(cal._samples)} dots captured")
            break
        if key in (ord("q"),):
            break

    if frame_count == 0:
        print("[calibration] ERROR: no camera frames received")

    return cal.matrix


def run() -> None:
    cap  = open_camera(CAMERA_INDEX, CAM_W, CAM_H)
    mesh = FaceMesh(output_face_transform=_NEED_FACE_TRANSFORM)

    # ── single shared fullscreen window ──────────────────────────────────────
    sw, sh = _screen_size()
    _make_fullscreen_window(WIN_NAME)
    print(f"handy-io  |  mode={MODE}  screen={sw}×{sh}")

    heatmap = GazeHeatmap(sw, sh, sigma=SIGMA, decay=DECAY)
    prev_gaze: list[np.ndarray] = []

    # ── calibration phase ─────────────────────────────────────────────────────
    print("Running calibration …")
    calib_matrix = _run_calibration(cap, mesh, (sw, sh))
    if calib_matrix is None:
        print("[calibration] not enough points — falling back to uncalibrated mode.")

    print("handy-io live  |  q=quit  c=clear  r=recalibrate")

    # ── live phase ────────────────────────────────────────────────────────────
    try:
        for frame_bgr, frame_rgb in frame_generator(cap):
            heatmap.decay_frame()
            gl = mesh.process(frame_rgb)

            if gl is not None:
                if calib_matrix is not None:
                    feat      = _EXTRACT_FN(gl)
                    screen_xy = calibration.apply(feat, calib_matrix)
                    if SMOOTHING > 0.0:
                        screen_xy = gaze._smooth(screen_xy, SMOOTHING, prev_gaze)
                    gx = int(np.clip(screen_xy[0], 0.0, 1.0) * sw)
                    gy = int(np.clip(screen_xy[1], 0.0, 1.0) * sh)
                else:
                    gx, gy = gaze.estimate_gaze_uncalibrated(
                        gl, (sw, sh), smoothing=SMOOTHING, _prev=prev_gaze
                    )

                heatmap.add_point(gx, gy)

            # Camera frame resized to screen resolution, heatmap blended on top
            display = cv2.resize(frame_bgr, (sw, sh))
            hm      = heatmap.render()
            mask    = hm.any(axis=2)
            display[mask] = cv2.addWeighted(
                display, 1 - ALPHA, hm, ALPHA, 0
            )[mask]

            cv2.imshow(WIN_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                heatmap._buf[:] = 0
                prev_gaze.clear()
            elif key == ord("r"):
                prev_gaze.clear()
                heatmap._buf[:] = 0
                calib_matrix = _run_calibration(cap, mesh, (sw, sh))
                _make_fullscreen_window(WIN_NAME)

    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
