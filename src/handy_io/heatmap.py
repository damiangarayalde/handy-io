"""
Gaze heatmap accumulator.

Each gaze sample splats a Gaussian blob onto an accumulation buffer.
The buffer is then normalised and colourised with a COLORMAP_JET overlay
that can be blended onto any BGR frame.
"""

from __future__ import annotations
import cv2
import numpy as np


class GazeHeatmap:
    def __init__(
        self,
        width: int,
        height: int,
        sigma: float = 40.0,
        decay: float = 0.97,
    ) -> None:
        self.w = width
        self.h = height
        self.sigma = sigma
        self.decay = decay          # multiply buffer by this each frame (fade old gaze)
        self._buf = np.zeros((height, width), dtype=np.float32)

        # Pre-build Gaussian kernel (odd size, at least 6*sigma)
        k = int(sigma * 6) | 1     # ensure odd
        self._kernel = cv2.getGaussianKernel(k, sigma)
        self._kernel_2d = self._kernel @ self._kernel.T

    def add_point(self, x: int, y: int) -> None:
        """Splat a Gaussian centred at (x, y) onto the accumulation buffer."""
        k = self._kernel_2d
        kh, kw = k.shape
        r, c = kh // 2, kw // 2

        # Destination ROI (clamped to buffer bounds)
        x0, y0 = x - c, y - r
        x1, y1 = x0 + kw, y0 + kh

        # Source ROI (handles kernel partially outside frame)
        kx0 = max(0, -x0)
        ky0 = max(0, -y0)
        kx1 = kw - max(0, x1 - self.w)
        ky1 = kh - max(0, y1 - self.h)

        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(self.w, x1), min(self.h, y1)

        if dx1 > dx0 and dy1 > dy0:
            self._buf[dy0:dy1, dx0:dx1] += k[ky0:ky1, kx0:kx1]

    def decay_frame(self) -> None:
        """Call once per frame to let old gaze points fade out."""
        self._buf *= self.decay

    def render(self, alpha: float = 0.45) -> np.ndarray:
        """
        Return a BGR heatmap image (same size as buffer) ready to overlay.
        alpha controls opacity when blending — returned image has full values,
        caller multiplies by alpha before adding to the camera frame.
        """
        if self._buf.max() < 1e-6:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        norm = self._buf / self._buf.max()
        grey = (norm * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
        # Make zero-gaze regions transparent by zeroing them out
        mask = (grey == 0)
        coloured[mask] = 0
        return coloured

    def overlay(self, frame_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Blend the heatmap onto frame_bgr and return the result."""
        hm = self.render()
        hm_resized = cv2.resize(hm, (frame_bgr.shape[1], frame_bgr.shape[0]))
        mask = hm_resized.any(axis=2)
        out = frame_bgr.copy()
        out[mask] = cv2.addWeighted(frame_bgr, 1 - alpha, hm_resized, alpha, 0)[mask]
        return out
