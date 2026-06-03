"""MediaPipe hand landmark inference using the Tasks API (mediapipe >= 0.10).

Each hand yields 21 landmarks in normalised [0,1] x/y plus a metric-scale z
(depth relative to the wrist).  Landmark indices follow the canonical MediaPipe
hand topology:

     4        8       12       16       20   ← fingertips
     |        |        |        |        |
     3        7       11       15       19
     |        |        |        |        |
     2        6       10       14       18
     |        |        |        |        |
     1        5        9       13       17
      \\       |        |        |       /
            0  (wrist)

Finger groups (tip, dip, pip, mcp):
  THUMB  : 4, 3, 2, 1
  INDEX  : 8, 7, 6, 5
  MIDDLE : 12, 11, 10, 9
  RING   : 16, 15, 14, 13
  PINKY  : 20, 19, 18, 17
"""

from __future__ import annotations
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── model ────────────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model → {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


# ── landmark index constants ──────────────────────────────────────────────────
WRIST = 0

THUMB_MCP,  THUMB_IP,   THUMB_TIP  = 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Connections to draw the hand skeleton (pairs of landmark indices)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # palm
    (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    # thumb
    (1, 2), (2, 3), (3, 4),
    # index
    (5, 6), (6, 7), (7, 8),
    # middle
    (9, 10), (10, 11), (11, 12),
    # ring
    (13, 14), (14, 15), (15, 16),
    # pinky
    (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandResult:
    """Landmarks for one detected hand."""
    landmarks: np.ndarray   # (21, 3) normalised x,y + metric-ish z
    handedness: str         # "Left" or "Right"
    score: float

    @property
    def fingertip_xy(self) -> np.ndarray:
        """(5, 2) normalised xy of the five fingertips."""
        return self.landmarks[FINGERTIPS, :2]

    def finger_extended(self) -> list[bool]:
        """
        Heuristic: a finger is 'extended' when its tip is further from the
        wrist than its MCP joint in normalised y (screen coords — y grows down,
        so a raised finger has a *smaller* y than the wrist).
        """
        wrist_y = self.landmarks[WRIST, 1]
        mcp_indices = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        extended = []
        for tip_idx, mcp_idx in zip(FINGERTIPS, mcp_indices):
            tip_y = self.landmarks[tip_idx, 1]
            mcp_y = self.landmarks[mcp_idx, 1]
            # Tip is above MCP (smaller y) → extended.
            # For thumb: use x-distance from wrist instead.
            if tip_idx == THUMB_TIP:
                extended.append(
                    abs(self.landmarks[tip_idx, 0] - self.landmarks[WRIST, 0])
                    > abs(self.landmarks[THUMB_MCP, 0] - self.landmarks[WRIST, 0])
                )
            else:
                extended.append(tip_y < mcp_y - 0.02)
        return extended


@dataclass
class HandsResult:
    hands: list[HandResult] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.hands)


def _make_base_options(model_path: Path, use_gpu: bool) -> BaseOptions:
    delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_path=str(model_path), delegate=delegate)


class HandTracker:
    """Wraps MediaPipe HandLandmarker for VIDEO-mode inference."""

    def __init__(
        self,
        max_hands: int = 2,
        min_confidence: float = 0.5,
        use_gpu: bool = True,
    ) -> None:
        model_path = _ensure_model()
        base = _make_base_options(model_path, use_gpu)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_rgb: np.ndarray) -> HandsResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands: list[HandResult] = []
        for i, lm_list in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
            side = result.handedness[i][0].display_name if result.handedness else "Unknown"
            score = result.handedness[i][0].score if result.handedness else 0.0
            hands.append(HandResult(landmarks=pts, handedness=side, score=score))

        return HandsResult(hands=hands)

    def close(self) -> None:
        self._landmarker.close()


# ── overlay drawing ───────────────────────────────────────────────────────────

# Per-finger colours in BGR for OpenCV
_FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    THUMB_TIP:  ( 50, 180, 255),   # amber
    INDEX_TIP:  ( 80, 220,  80),   # green
    MIDDLE_TIP: (255, 160,  80),   # blue
    RING_TIP:   (220,  80, 220),   # magenta
    PINKY_TIP:  (220, 220,  50),   # cyan
}

# Which tip index belongs to each connection endpoint
_CONN_FINGER: dict[int, int] = {
    1: THUMB_TIP, 2: THUMB_TIP, 3: THUMB_TIP, 4: THUMB_TIP,
    5: INDEX_TIP, 6: INDEX_TIP, 7: INDEX_TIP, 8: INDEX_TIP,
    9: MIDDLE_TIP, 10: MIDDLE_TIP, 11: MIDDLE_TIP, 12: MIDDLE_TIP,
    13: RING_TIP, 14: RING_TIP, 15: RING_TIP, 16: RING_TIP,
    17: PINKY_TIP, 18: PINKY_TIP, 19: PINKY_TIP, 20: PINKY_TIP,
}


def draw_hands(
    frame: np.ndarray,
    result: HandsResult,
    *,
    show_labels: bool = True,
    show_all_dots: bool = True,
    mirror_x: bool = False,
) -> None:
    """
    Draw hand skeleton and fingertip highlights directly onto *frame* (BGR,
    in-place).  Set mirror_x=True when the frame has already been horizontally
    flipped so the landmarks (in original camera space) map correctly.
    """
    if not result:
        return

    h, w = frame.shape[:2]

    def to_px(lm: np.ndarray) -> tuple[int, int]:
        x = (1.0 - lm[0]) if mirror_x else lm[0]
        return int(x * w), int(lm[1] * h)

    for hand in result.hands:
        lm = hand.landmarks
        extended = hand.finger_extended()

        # ── skeleton connections ──────────────────────────────────────────────
        for a, b in HAND_CONNECTIONS:
            tip_key = _CONN_FINGER.get(b, _CONN_FINGER.get(a, INDEX_TIP))
            color = _FINGER_COLORS[tip_key]
            cv2.line(frame, to_px(lm[a]), to_px(lm[b]), color, 2, cv2.LINE_AA)

        # ── all joint dots ────────────────────────────────────────────────────
        if show_all_dots:
            for idx in range(21):
                tip_key = _CONN_FINGER.get(idx, INDEX_TIP)
                color = _FINGER_COLORS[tip_key]
                cv2.circle(frame, to_px(lm[idx]), 4, color, -1, cv2.LINE_AA)

        # ── fingertip highlights ──────────────────────────────────────────────
        for tip_idx, name, ext in zip(FINGERTIPS, FINGER_NAMES, extended):
            px = to_px(lm[tip_idx])
            color = _FINGER_COLORS[tip_idx]
            ring_color = color if ext else tuple(max(0, c - 80) for c in color)
            cv2.circle(frame, px, 12, ring_color, 2, cv2.LINE_AA)
            cv2.circle(frame, px, 5, color, -1, cv2.LINE_AA)

        # ── labels ───────────────────────────────────────────────────────────
        if show_labels:
            wrist_px = to_px(lm[WRIST])
            cv2.putText(frame, hand.handedness, (wrist_px[0] - 20, wrist_px[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


@dataclass
class PinchTransform:
    """Raw pinch state from the right hand's thumb–index pair (one frame)."""
    cx: int          # screen x of midpoint
    cy: int          # screen y of midpoint
    size: int        # side length in pixels
    angle_deg: float # XY-plane angle of thumb→index axis, degrees


@dataclass
class SquareObject:
    """A persistent interactive square on the viewport."""
    cx: float
    cy: float
    size: float
    angle_deg: float
    color: tuple[int, int, int] = (0, 220, 255)

    def corners(self) -> np.ndarray:
        """Return (4,2) int32 screen corners."""
        half = self.size / 2
        local = np.array([[-half, -half], [half, -half],
                          [half,  half], [-half,  half]], dtype=np.float32)
        rad = np.radians(self.angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return ((rot @ local.T).T + np.array([self.cx, self.cy])).astype(np.int32)

    def contains_point(self, px: float, py: float) -> bool:
        """True when (px, py) is inside the square (axis-aligned bounding check in local space)."""
        dx = px - self.cx
        dy = py - self.cy
        rad = np.radians(-self.angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy
        half = self.size / 2
        return abs(lx) <= half and abs(ly) <= half

    def distance_to(self, px: float, py: float) -> float:
        return float(np.hypot(px - self.cx, py - self.cy))


def make_default_squares(screen_w: int, screen_h: int) -> list[SquareObject]:
    """Spawn a small set of squares scattered across the viewport."""
    base = screen_h * 0.18
    positions = [
        (0.25, 0.30), (0.55, 0.25), (0.75, 0.60),
        (0.35, 0.65), (0.15, 0.55),
    ]
    colors = [
        (0, 220, 255), (80, 255, 120), (255, 160, 60),
        (220, 80, 220), (255, 220, 50),
    ]
    return [
        SquareObject(
            cx=x * screen_w, cy=y * screen_h,
            size=base, angle_deg=0.0, color=c,
        )
        for (x, y), c in zip(positions, colors)
    ]


def pinch_transform(
    hands_result: HandsResult,
    screen_w: int,
    screen_h: int,
    *,
    min_frac: float = 0.1,
    max_frac: float = 0.8,
    max_dist: float = 0.35,
    mirror_x: bool = True,
) -> PinchTransform | None:
    """
    Derive translation, rotation, and size from the right hand's thumb-tip (4)
    and index-tip (8).  Returns None when no right hand is detected.
    """
    for hand in hands_result.hands:
        if hand.handedness != "Right":
            continue
        thumb = hand.landmarks[THUMB_TIP]
        index = hand.landmarks[INDEX_TIP]

        dist = float(np.linalg.norm(thumb - index))
        t = np.clip(dist / max_dist, 0.0, 1.0)
        size = min_frac * screen_h + t * (max_frac - min_frac) * screen_h

        mid = (thumb + index) / 2.0
        mx = 1.0 - mid[0] if mirror_x else mid[0]
        cx = float(np.clip(mx,    0.0, 1.0) * screen_w)
        cy = float(np.clip(mid[1], 0.0, 1.0) * screen_h)

        dx = index[0] - thumb[0]
        if mirror_x:
            dx = -dx
        dy = index[1] - thumb[1]
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        return PinchTransform(cx=int(cx), cy=int(cy), size=int(size), angle_deg=angle_deg)
    return None


# ── grab state ────────────────────────────────────────────────────────────────

@dataclass
class GrabState:
    """
    Snapshot taken at spacebar press — holds the offset between the pinch
    midpoint and the grabbed square's centre, so the square doesn't jump.
    Also records the size and angle deltas so the object deforms continuously.
    """
    square_idx: int
    # pinch values at grab time
    grab_cx: float
    grab_cy: float
    grab_size: float
    grab_angle: float
    # square values at grab time
    sq_cx: float
    sq_cy: float
    sq_size: float
    sq_angle: float


def pick_square(
    pinch: PinchTransform,
    squares: list[SquareObject],
    *,
    max_dist_frac: float = 0.25,
    screen_h: int = 1080,
) -> int | None:
    """
    Return the index of the square whose centre is closest to *pinch* midpoint,
    giving priority to squares that actually contain the midpoint.
    Falls back to nearest centre within max_dist_frac * screen_h.
    """
    px, py = float(pinch.cx), float(pinch.cy)
    threshold = max_dist_frac * screen_h

    # Prefer a square that contains the pinch midpoint
    for i, sq in enumerate(squares):
        if sq.contains_point(px, py):
            return i

    # Fall back to nearest centre within threshold
    best_i, best_d = None, threshold
    for i, sq in enumerate(squares):
        d = sq.distance_to(px, py)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def apply_grab(
    grab: GrabState,
    pinch: PinchTransform,
    squares: list[SquareObject],
) -> None:
    """Update the grabbed square's transform to follow the current pinch."""
    sq = squares[grab.square_idx]
    dcx = pinch.cx - grab.grab_cx
    dcy = pinch.cy - grab.grab_cy
    sq.cx = grab.sq_cx + dcx
    sq.cy = grab.sq_cy + dcy
    sq.size = grab.sq_size * (pinch.size / grab.grab_size) if grab.grab_size > 0 else grab.sq_size
    sq.angle_deg = grab.sq_angle + (pinch.angle_deg - grab.grab_angle)


# ── drawing ───────────────────────────────────────────────────────────────────

def _draw_square_shape(
    frame: np.ndarray,
    sq: SquareObject,
    *,
    selected: bool = False,
) -> None:
    pts = sq.corners().reshape((-1, 1, 2))
    thickness = 4 if selected else 2
    cv2.polylines(frame, [pts], isClosed=True, color=sq.color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    if selected:
        # filled semi-transparent highlight
        overlay = frame.copy()
        cv2.fillPoly(overlay, [sq.corners()], sq.color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        # bright corner dots
        for corner in sq.corners():
            cv2.circle(frame, tuple(corner), 6, sq.color, -1, cv2.LINE_AA)


def draw_squares(
    frame: np.ndarray,
    squares: list[SquareObject],
    selected_idx: int | None = None,
) -> None:
    for i, sq in enumerate(squares):
        _draw_square_shape(frame, sq, selected=i == selected_idx)


def draw_pinch_cursor(frame: np.ndarray, pinch: PinchTransform) -> None:
    """Small crosshair at the pinch midpoint — visible when nothing is grabbed."""
    cx, cy = pinch.cx, pinch.cy
    arm = 18
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, (0, 220, 255), -1, cv2.LINE_AA)


def draw_finger_occlusion_points(
    frame: np.ndarray,
    occlusion_points: list[dict],    # output of gaze.finger_screen_points
    fingertip_indices: list[int],    # e.g. FINGERTIPS subset for the right hand
) -> None:
    """
    Draw on-screen interference points for each fingertip.

    For each fingertip: a small filled dot per eye (left/right) and a larger
    ring for the average, all in the finger's canonical colour.
    """
    h, w = frame.shape[:2]

    for tip_idx, pts in zip(fingertip_indices, occlusion_points):
        color = _FINGER_COLORS.get(tip_idx, (200, 200, 200))

        def _to_px(norm_xy) -> tuple[int, int] | None:
            if norm_xy is None:
                return None
            x = int(np.clip(norm_xy[0], 0.0, 1.0) * w)
            y = int(np.clip(norm_xy[1], 0.0, 1.0) * h)
            return x, y

        left_px  = _to_px(pts.get("left"))
        right_px = _to_px(pts.get("right"))
        avg_px   = _to_px(pts.get("avg"))

        # Left-eye dot: small filled circle
        if left_px is not None:
            cv2.circle(frame, left_px, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, left_px, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Right-eye dot: small filled circle with different outline
        if right_px is not None:
            cv2.circle(frame, right_px, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, right_px, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # Average: larger ring
        if avg_px is not None:
            cv2.circle(frame, avg_px, 12, color, 2, cv2.LINE_AA)
            cv2.circle(frame, avg_px,  4, color, -1, cv2.LINE_AA)
