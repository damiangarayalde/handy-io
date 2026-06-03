"""
9-point gaze calibration.

Flow
----
1. Caller owns the display window and calls render_frame() each iteration to
   get a canvas to show. No window management happens inside this module.
2. User fixates each dot; stable gaze (DWELL_FRAMES frames below STABILITY_THRESH)
   triggers auto-capture, or press SPACE for manual capture.
3. After all 9 dots, fit a least-squares linear map:
       screen_xy_norm = calib_matrix @ feature_vector
4. Return the fitted matrix for use in calibration.apply().
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Callable


# ── tunables ─────────────────────────────────────────────────────────────────
DWELL_FRAMES     = 30    # frames of stable gaze before auto-capture triggers
STABILITY_THRESH = 0.06  # max std-dev of iris offset across the dwell window
SAMPLES_PER_DOT  = 15   # frames averaged together to form one calibration sample
DOT_RADIUS       = 22
DOT_COLOR_WAIT   = (0, 200, 255)   # amber: waiting for stable fixation
DOT_COLOR_READY  = (0, 255,   0)   # green: stable, capturing
DOT_COLOR_DONE   = (80,  80,  80)  # grey:  captured
BG_COLOR         = (20,  20,  20)

# 9-point grid in normalised [0, 1] screen coordinates
_GRID: list[tuple[float, float]] = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


class Calibrator:
    """
    Manages the calibration sequence.

    The caller owns the display window. Each frame, call:
      canvas = cal.render_frame(gl, key)
      cv2.imshow(win, canvas)

    Parameters
    ----------
    canvas_wh   Pixel size of the canvas to render onto (= screen resolution).
    extract_fn  Feature extractor: GazeLandmarks → np.ndarray.
    """

    def __init__(self, canvas_wh: tuple[int, int], extract_fn: Callable) -> None:
        self.cw, self.ch = canvas_wh
        self._extract = extract_fn
        self._dot_idx    = 0
        self._dwell_buf: list[np.ndarray] = []
        self._samples:   list[np.ndarray] = []
        self._targets:   list[np.ndarray] = []
        self._capture_buf: list[np.ndarray] = []
        self._capturing  = False
        self._done       = False
        self._frame_count = 0
        self.matrix: np.ndarray | None = None

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done

    def render_frame(self, gl, key: int) -> np.ndarray:
        """
        Process one frame of landmarks + key event; return the canvas to display.
        gl  may be None if face not detected.
        key is the result of cv2.waitKey() & 0xFF from the caller.
        """
        self._frame_count += 1

        if gl is not None:
            feat = self._extract(gl)
            self._dwell_buf.append(feat)
            if len(self._dwell_buf) > DWELL_FRAMES:
                self._dwell_buf.pop(0)

            stable = self._is_stable()

            if stable and not self._capturing:
                self._capturing = True
                self._capture_buf.clear()

            if self._capturing:
                self._capture_buf.append(feat)
                if len(self._capture_buf) >= SAMPLES_PER_DOT:
                    self._record_sample()
        else:
            stable = False

        # Honour keypresses only after a few frames so spurious ESC on macOS
        # at window-open time doesn't skip calibration immediately.
        if self._frame_count > 10:
            if key == ord(" ") and gl is not None:
                feat = self._extract(gl)
                self._capture_buf = [feat] * SAMPLES_PER_DOT
                self._capturing = True
                self._record_sample()
            elif key == 27:
                print(f"[calibration] ESC — using {len(self._samples)} samples")
                if len(self._samples) >= 4:
                    self._fit()
                self._done = True

        return self._draw(stable)

    # ── internals ────────────────────────────────────────────────────────────

    def _draw(self, stable: bool) -> np.ndarray:
        canvas = np.full((self.ch, self.cw, 3), BG_COLOR, dtype=np.uint8)

        dot_color = DOT_COLOR_READY if stable else DOT_COLOR_WAIT

        for i, (nx, ny) in enumerate(_GRID):
            px, py = int(nx * self.cw), int(ny * self.ch)
            if i < self._dot_idx:
                cv2.circle(canvas, (px, py), DOT_RADIUS, DOT_COLOR_DONE, -1)
            elif i == self._dot_idx:
                cv2.circle(canvas, (px, py), DOT_RADIUS, dot_color, -1)
                if self._capturing:
                    frac  = len(self._capture_buf) / SAMPLES_PER_DOT
                    angle = int(360 * frac)
                    cv2.ellipse(canvas, (px, py),
                                (DOT_RADIUS + 10, DOT_RADIUS + 10),
                                -90, 0, angle, (255, 255, 255), 3)
            else:
                cv2.circle(canvas, (px, py), DOT_RADIUS // 2, DOT_COLOR_WAIT, 1)

        buf_pct = int(100 * len(self._dwell_buf) / DWELL_FRAMES)
        cv2.putText(canvas,
                    f"Dot {self._dot_idx + 1}/9  —  fixate and hold  "
                    f"({buf_pct}% stable)",
                    (20, self.ch - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "SPACE = capture now   ESC = skip calibration",
                    (20, self.ch - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)
        return canvas

    def _is_stable(self) -> bool:
        if len(self._dwell_buf) < DWELL_FRAMES:
            return False
        arr = np.stack(self._dwell_buf)
        return float(arr.std(axis=0).max()) < STABILITY_THRESH

    def _record_sample(self) -> None:
        avg_feat = np.mean(self._capture_buf, axis=0)
        nx, ny   = _GRID[self._dot_idx]
        self._samples.append(avg_feat)
        self._targets.append(np.array([nx, ny]))

        self._dot_idx     += 1
        self._dwell_buf.clear()
        self._capture_buf.clear()
        self._capturing = False

        print(f"[calibration] dot {self._dot_idx}/9 captured")

        if self._dot_idx >= len(_GRID):
            self._fit()
            self._done = True

    def _fit(self) -> None:
        A      = np.stack(self._samples)
        b      = np.stack(self._targets)
        A_bias = np.hstack([A, np.ones((len(A), 1))])
        sol_x, _, _, _ = np.linalg.lstsq(A_bias, b[:, 0], rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(A_bias, b[:, 1], rcond=None)
        self.matrix = np.vstack([sol_x, sol_y])
        print(f"[calibration] fit complete — {len(self._samples)} points, "
              f"matrix {self.matrix.shape}")


def apply(features: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map feature vector → normalised screen (x, y) via calibration matrix."""
    feat_bias = np.append(features, 1.0)
    return np.clip(matrix @ feat_bias, 0.0, 1.0)
