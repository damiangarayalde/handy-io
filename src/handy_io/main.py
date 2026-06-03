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
  h        — toggle hand skeleton overlay

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
import queue
import threading
import time
import cv2
import numpy as np

from .capture import open_camera, frame_generator
from .landmarks import FaceMesh, GazeLandmarks
from .heatmap import GazeHeatmap
from .hands import (
    HandTracker, HandsResult,
    draw_hands, draw_finger_occlusion_points, draw_pinch_square, pinch_transform,
    FINGERTIPS,
)
from . import gaze, calibration, pose

# ── MODE ────────────────────────────────────────────────────────────────────
MODE: str = "calibration_only"   # "calibration_only" | "full"

# ── feature flags ────────────────────────────────────────────────────────────
SHOW_HANDS: bool = True   # set False to disable hand tracking entirely
USE_GPU:    bool = False  # Metal GPU delegate crashes on macOS with BGR frames

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


def _try_gpu(make_model, label: str, **kw):
    """Instantiate *make_model* with GPU delegate; fall back to CPU on error."""
    try:
        m = make_model(use_gpu=True, **kw)
        print(f"[{label}] GPU delegate active")
        return m
    except Exception as e:
        print(f"[{label}] GPU delegate failed ({e}), falling back to CPU")
        return make_model(use_gpu=False, **kw)


class InferenceThread(threading.Thread):
    """
    Runs one MediaPipe model on a background thread.

    The main loop drops the latest frame into `submit()` (non-blocking —
    old frames are evicted so the thread always works on the freshest one)
    and reads the last completed result via the `result` property.

    Because MediaPipe VIDEO-mode landmarkers are not thread-safe, each
    thread owns its own model instance.
    """

    def __init__(self, model, name: str) -> None:
        super().__init__(daemon=True, name=name)
        self._model = model
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result = None
        self.running = True

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Send a frame for inference; drop the queued frame if one is waiting."""
        try:
            self._q.put_nowait(frame_rgb)
        except queue.Full:
            try:
                self._q.get_nowait()   # evict stale frame
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame_rgb)
            except queue.Full:
                pass

    @property
    def result(self):
        with self._lock:
            return self._result

    def run(self) -> None:
        while self.running:
            try:
                frame_rgb = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame_rgb is None:
                break
            r = self._model.process(frame_rgb)
            with self._lock:
                self._result = r

    def stop(self) -> None:
        self.running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._model.close()


from dataclasses import dataclass

@dataclass
class Monitor:
    index: int
    w: int
    h: int
    x: int   # top-left origin in the global desktop coordinate system
    y: int
    primary: bool

    @property
    def label(self) -> str:
        tag = " (built-in)" if self.primary else ""
        return f"Display {self.index + 1}  {self.w}×{self.h}{tag}"


def _get_monitors() -> list[Monitor]:
    """Enumerate connected displays via CoreGraphics ctypes (macOS). Falls back to single entry."""
    try:
        import ctypes, ctypes.util

        class _CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        class _CGSize(ctypes.Structure):
            _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

        class _CGRect(ctypes.Structure):
            _fields_ = [("origin", _CGPoint), ("size", _CGSize)]

        cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
        cg.CGGetActiveDisplayList.restype = ctypes.c_int32
        cg.CGDisplayBounds.restype = _CGRect
        cg.CGDisplayIsMain.restype = ctypes.c_bool

        max_d = 16
        ids = (ctypes.c_uint32 * max_d)()
        count = ctypes.c_uint32(0)
        cg.CGGetActiveDisplayList(max_d, ids, ctypes.byref(count))

        monitors = []
        for i in range(count.value):
            r = cg.CGDisplayBounds(ids[i])
            monitors.append(Monitor(
                index=i,
                w=int(r.size.width),
                h=int(r.size.height),
                x=int(r.origin.x),
                y=int(r.origin.y),
                primary=bool(cg.CGDisplayIsMain(ids[i])),
            ))
        if monitors:
            return monitors
    except Exception:
        pass
    return [Monitor(index=0, w=1920, h=1080, x=0, y=0, primary=True)]


def _make_fullscreen_window(name: str, monitor: Monitor | None = None) -> None:
    # GUI_NORMAL removes the macOS toolbar; WINDOW_NORMAL allows resizing.
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)

    # Show a placeholder so the window exists before we move/resize it.
    placeholder = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imshow(name, placeholder)
    cv2.waitKey(100)

    if monitor is not None and not monitor.primary:
        # cv2.WINDOW_FULLSCREEN always snaps to the primary display on macOS.
        # For external monitors: move then resize to cover the screen exactly.
        cv2.moveWindow(name, monitor.x, monitor.y)
        cv2.waitKey(100)
        cv2.resizeWindow(name, monitor.w, monitor.h)
        cv2.waitKey(100)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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


def _launcher() -> tuple[str, Monitor]:
    """
    Show a tkinter startup window and return (mode, monitor):
      mode: "gaze_and_hands" | "hands_only" | "quit"
    """
    import tkinter as tk
    from tkinter import font as tkfont

    monitors = _get_monitors()
    result: list[tuple[str, Monitor]] = []

    root = tk.Tk()
    root.title("handy-io")
    root.configure(bg="#111")
    root.resizable(False, False)

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    btn_font   = tkfont.Font(family="Helvetica", size=14)
    sub_font   = tkfont.Font(family="Helvetica", size=10)
    label_font = tkfont.Font(family="Helvetica", size=11, weight="bold")

    tk.Label(root, text="handy-io", font=title_font,
             bg="#111", fg="#fff").pack(pady=(28, 4))
    tk.Label(root, text="Select mode and target display", font=sub_font,
             bg="#111", fg="#888").pack(pady=(0, 20))

    # ── screen selector ───────────────────────────────────────────────────────
    tk.Label(root, text="PROJECT ON", font=label_font,
             bg="#111", fg="#aaa").pack(anchor="w", padx=40)

    selected_monitor = tk.IntVar(value=0)

    screen_frame = tk.Frame(root, bg="#111")
    screen_frame.pack(padx=40, pady=(4, 20), fill="x")

    CANVAS_H = 64
    TOTAL_W  = 380

    # Draw a mini desktop map showing monitor layout
    # Compute bounding box of all monitors
    min_x = min(m.x for m in monitors)
    min_y = min(m.y for m in monitors)  # noqa: F841 (used below)
    total_pw = max(m.x + m.w for m in monitors) - min_x
    total_ph = max(abs(m.y) + m.h for m in monitors)
    scale = min(TOTAL_W / total_pw, CANVAS_H / total_ph) * 0.85

    canvas = tk.Canvas(screen_frame, width=TOTAL_W, height=CANVAS_H,
                       bg="#1a1a1a", highlightthickness=0)
    canvas.pack(side="left", padx=(0, 16))

    rects: list[int] = []
    for m in monitors:
        cx = int((m.x - min_x) * scale) + 10
        cy = int(abs(m.y) * scale) + 4 if m.y < 0 else int(m.y * scale) + 4
        cw = max(int(m.w * scale), 20)
        ch = max(int(m.h * scale), 14)
        r = canvas.create_rectangle(cx, cy, cx + cw, cy + ch,
                                    fill="#333", outline="#555", width=1)
        canvas.create_text(cx + cw // 2, cy + ch // 2,
                           text=str(m.index + 1), fill="#aaa",
                           font=("Helvetica", max(8, int(ch * 0.4))))
        rects.append(r)

    def _highlight(idx: int) -> None:
        for i, r in enumerate(rects):
            canvas.itemconfig(r, fill="#0070f3" if i == idx else "#333",
                              outline="#60a0ff" if i == idx else "#555")

    _highlight(0)

    # Radio buttons for each monitor
    radio_frame = tk.Frame(screen_frame, bg="#111")
    radio_frame.pack(side="left", anchor="w")

    def _on_select(idx: int) -> None:
        selected_monitor.set(idx)
        _highlight(idx)

    for m in monitors:
        rb = tk.Radiobutton(
            radio_frame, text=m.label,
            variable=selected_monitor, value=m.index,
            command=lambda i=m.index: _on_select(i),
            bg="#111", fg="#ccc", selectcolor="#111",
            activebackground="#111", activeforeground="#fff",
            font=sub_font,
        )
        rb.pack(anchor="w", pady=2)

    # ── mode buttons ──────────────────────────────────────────────────────────
    tk.Label(root, text="MODE", font=label_font,
             bg="#111", fg="#aaa").pack(anchor="w", padx=40)

    btn_cfg = dict(font=btn_font, width=28, pady=12, relief="flat", cursor="hand2")

    def _pick(mode: str) -> None:
        mon = monitors[selected_monitor.get()]
        result.append((mode, mon))
        root.destroy()

    tk.Button(root, text="Gaze + Hand tracking",
              bg="#0070f3", fg="#fff", activebackground="#005cc5",
              command=lambda: _pick("gaze_and_hands"), **btn_cfg).pack(padx=40, pady=(6, 2))
    tk.Label(root, text="Eye gaze heatmap with calibration and hand skeleton",
             font=sub_font, bg="#111", fg="#666").pack()

    tk.Button(root, text="Hand tracking only",
              bg="#222", fg="#fff", activebackground="#333",
              command=lambda: _pick("hands_only"), **btn_cfg).pack(padx=40, pady=(14, 2))
    tk.Label(root, text="Hand skeleton only — no face model, no calibration",
             font=sub_font, bg="#111", fg="#666").pack()

    tk.Button(root, text="Quit", font=sub_font, bg="#111", fg="#555",
              activebackground="#222", relief="flat", cursor="hand2",
              command=root.destroy).pack(pady=(22, 20))

    root.eval("tk::PlaceWindow . center")
    root.mainloop()

    primary = next((m for m in monitors if m.primary), monitors[0])
    return result[0] if result else ("quit", primary)


def _run_hands_only(monitor: Monitor) -> None:
    """Live loop with hand tracking only — no face model, no calibration."""
    cap = open_camera(CAMERA_INDEX, CAM_W, CAM_H)
    sw, sh = monitor.w, monitor.h
    _make_fullscreen_window(WIN_NAME, monitor)

    hand_model = HandTracker(max_hands=2, use_gpu=False)
    hand_thread = InferenceThread(hand_model, "hand-inference")
    hand_thread.start()

    print("handy-io  [hands only]  |  q=quit  h=toggle skeleton")
    show_hands = True
    _fps_t0 = time.monotonic()
    _fps_count = 0
    _fps_display = 0.0

    try:
        for frame_bgr, frame_rgb in frame_generator(cap):
            hand_thread.submit(frame_rgb)
            hands_result: HandsResult | None = hand_thread.result

            display = cv2.flip(cv2.resize(frame_bgr, (sw, sh)), 1)

            if show_hands and hands_result:
                draw_hands(display, hands_result, mirror_x=True)
                pt = pinch_transform(hands_result, sw, sh)
                if pt is not None:
                    draw_pinch_square(display, pt)

            _fps_count += 1
            if _fps_count >= 30:
                _fps_display = _fps_count / (time.monotonic() - _fps_t0)
                _fps_t0 = time.monotonic()
                _fps_count = 0
            cv2.putText(display, f"{_fps_display:.1f} fps", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(display, "HANDS ONLY", (12, sh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                show_hands = not show_hands
    finally:
        hand_thread.stop()
        cap.release()
        cv2.destroyAllWindows()


def _run_gaze_and_hands(monitor: Monitor) -> None:
    cap = open_camera(CAMERA_INDEX, CAM_W, CAM_H)
    gaze.init_focal_px(cap)

    face_model = _try_gpu(
        lambda use_gpu: FaceMesh(output_face_transform=_NEED_FACE_TRANSFORM, use_gpu=use_gpu),
        "face",
    ) if USE_GPU else FaceMesh(output_face_transform=_NEED_FACE_TRANSFORM, use_gpu=False)

    hand_model: HandTracker | None = None
    if SHOW_HANDS:
        hand_model = (
            _try_gpu(lambda use_gpu: HandTracker(max_hands=2, use_gpu=use_gpu), "hands")
            if USE_GPU else HandTracker(max_hands=2, use_gpu=False)
        )

    # ── single shared fullscreen window ──────────────────────────────────────
    sw, sh = monitor.w, monitor.h
    _make_fullscreen_window(WIN_NAME, monitor)
    print(f"handy-io  |  mode={MODE}  screen={sw}×{sh}  display={monitor.label}")

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

    # ── calibration phase (synchronous — models called directly on main thread) ─
    print("Running calibration …")
    calib_matrix, calib_poly_degree = _run_calibration(cap, face_model, (sw, sh))
    if calib_matrix is None:
        print("[calibration] not enough points — falling back to uncalibrated mode.")

    # ── start inference threads now that calibration is done ─────────────────
    face_thread = InferenceThread(face_model, "face-inference")
    face_thread.start()
    hand_thread: InferenceThread | None = None
    if hand_model is not None:
        hand_thread = InferenceThread(hand_model, "hand-inference")
        hand_thread.start()

    print("handy-io live  |  q=quit  c=clear  r=recalibrate  d=glasses-cam debug  h=hands")

    show_glasses_debug = False
    show_hands = SHOW_HANDS

    # FPS tracking
    _fps_t0 = time.monotonic()
    _fps_count = 0
    _fps_display = 0.0

    # ── live phase ────────────────────────────────────────────────────────────
    try:
        for frame_bgr, frame_rgb in frame_generator(cap):
            # Submit to both inference threads (non-blocking)
            face_thread.submit(frame_rgb)
            if hand_thread is not None:
                hand_thread.submit(frame_rgb)

            # Read latest completed results (may be from a previous frame — that's fine)
            gl: GazeLandmarks | None = face_thread.result
            hands_result: HandsResult | None = hand_thread.result if hand_thread else None

            heatmap.decay_frame()

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

            # ── finger occlusion points ───────────────────────────────────────
            finger_occlusion: list[tuple[list[dict], list[int]]] = []
            if (gl is not None and calib_matrix is not None
                    and hands_result and show_hands):
                for hand in hands_result.hands:
                    if hand.handedness != "Right":
                        continue
                    tip_xy = hand.landmarks[FINGERTIPS, :2]  # (5, 2) normalised
                    pts = gaze.finger_screen_points(
                        gl, tip_xy, calib_matrix, calib_poly_degree)
                    finger_occlusion.append((pts, FINGERTIPS))

            # ── compose display ───────────────────────────────────────────────
            # 1. Mirror the camera feed so the user sees a natural reflection.
            display = cv2.flip(cv2.resize(frame_bgr, (sw, sh)), 1)

            # 2. Hand skeleton — drawn in camera-normalised coords onto the
            #    already-mirrored frame, so x must be flipped.
            if show_hands and hands_result:
                draw_hands(display, hands_result, mirror_x=True)
                pt = pinch_transform(hands_result, sw, sh)
                if pt is not None:
                    draw_pinch_square(display, pt)

            # 3. Gaze heatmap — true screen-space, no flip.
            hm = heatmap.render()
            mask = hm.any(axis=2)
            display[mask] = cv2.addWeighted(
                display, 1 - ALPHA, hm, ALPHA, 0
            )[mask]

            # 4. Gaze marker and occlusion points — true screen-space, no flip.
            if gl is not None:
                _draw_gaze_marker(display, gx, gy)

            for pts, tip_indices in finger_occlusion:
                draw_finger_occlusion_points(display, pts, tip_indices)

            # 5. ArUco markers in screen corners.
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

            # FPS counter (update every 30 frames)
            _fps_count += 1
            if _fps_count >= 30:
                _fps_display = _fps_count / (time.monotonic() - _fps_t0)
                _fps_t0 = time.monotonic()
                _fps_count = 0
            cv2.putText(display, f"{_fps_display:.1f} fps", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                heatmap._buf[:] = 0
                prev_gaze.clear()
            elif key == ord("d"):
                show_glasses_debug = not show_glasses_debug
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("r"):
                prev_gaze.clear()
                heatmap._buf[:] = 0
                # Stop face thread, recalibrate synchronously, restart thread
                face_thread.stop()
                face_thread.join(timeout=2.0)
                calib_matrix, calib_poly_degree = _run_calibration(
                    cap, face_model, (sw, sh))
                face_thread = InferenceThread(face_model, "face-inference")
                face_thread.start()
                _make_fullscreen_window(WIN_NAME, monitor)

    finally:
        face_thread.stop()
        if hand_thread is not None:
            hand_thread.stop()
        if glasses_thread is not None:
            glasses_thread.stop()
        cap.release()
        cv2.destroyAllWindows()


def run() -> None:
    mode, monitor = _launcher()
    if mode == "quit":
        return
    elif mode == "hands_only":
        _run_hands_only(monitor)
    else:
        _run_gaze_and_hands(monitor)


if __name__ == "__main__":
    run()
