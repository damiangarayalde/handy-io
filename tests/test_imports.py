import cv2
import mediapipe
import numpy


def test_core_dependencies_import():
    assert cv2.__name__ == "cv2"
    assert mediapipe.__name__ == "mediapipe"
    assert numpy.__name__ == "numpy"
