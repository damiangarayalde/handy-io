"""Webcam capture — yields BGR frames as a generator."""

import time
import cv2

_MAX_CONSEC_FAILURES = 100  # give up only after this many consecutive read failures
_RETRY_SLEEP_S = 0.033      # sleep ~1 frame interval between retries (30 fps = 33 ms)


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def frame_generator(cap: cv2.VideoCapture):
    """Yield (frame_bgr, frame_rgb); tolerate transient read failures."""
    failures = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures >= _MAX_CONSEC_FAILURES:
                print(f"[capture] camera stopped after {_MAX_CONSEC_FAILURES} consecutive failed reads")
                break           # camera truly gone
            time.sleep(_RETRY_SLEEP_S)
            continue            # wait for sensor to produce next frame
        failures = 0
        yield frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
