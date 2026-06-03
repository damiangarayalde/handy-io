"""MediaPipe face-mesh inference using the Tasks API (mediapipe >= 0.10)."""

from __future__ import annotations
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── model ────────────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model → {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


# ── landmark indices ──────────────────────────────────────────────────────────
# MediaPipe 478-point mesh: indices 468-477 are the iris refinement points.
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362


@dataclass
class GazeLandmarks:
    left_iris: np.ndarray         # (5, 3) normalised xyz
    right_iris: np.ndarray        # (5, 3) normalised xyz
    left_eye_corners: np.ndarray  # (2, 3) outer/inner
    right_eye_corners: np.ndarray # (2, 3) outer/inner
    image_wh: tuple[int, int]
    # 4×4 rigid transform: head pose in camera space (rotation + translation).
    # Available when FaceMesh is created with output_face_transform=True.
    face_transform: np.ndarray | None = None  # (4, 4) float64


def _make_base_options(model_path: Path, use_gpu: bool) -> BaseOptions:
    delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_path=str(model_path), delegate=delegate)


class FaceMesh:
    def __init__(
        self,
        max_faces: int = 1,
        output_face_transform: bool = True,
        use_gpu: bool = True,
    ) -> None:
        model_path = _ensure_model()
        self._output_face_transform = output_face_transform
        base = _make_base_options(model_path, use_gpu)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=output_face_transform,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_rgb: np.ndarray) -> GazeLandmarks | None:
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33  # ~30 fps synthetic timestamp
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]  # first face

        def pt(idx: int) -> list[float]:
            return [lm[idx].x, lm[idx].y, lm[idx].z]

        left_iris = np.array([pt(i) for i in LEFT_IRIS])
        right_iris = np.array([pt(i) for i in RIGHT_IRIS])
        left_corners = np.array([pt(LEFT_EYE_OUTER), pt(LEFT_EYE_INNER)])
        right_corners = np.array([pt(RIGHT_EYE_OUTER), pt(RIGHT_EYE_INNER)])

        face_transform: np.ndarray | None = None
        if self._output_face_transform and result.facial_transformation_matrixes:
            # MediaPipe returns a row-major flat list of 16 floats
            face_transform = np.array(
                result.facial_transformation_matrixes[0].data, dtype=np.float64
            ).reshape(4, 4)

        return GazeLandmarks(
            left_iris=left_iris,
            right_iris=right_iris,
            left_eye_corners=left_corners,
            right_eye_corners=right_corners,
            image_wh=(w, h),
            face_transform=face_transform,
        )

    def close(self) -> None:
        self._landmarker.close()
