"""
Collapsible cheat-sheet panel rendered directly onto OpenCV frames.

Usage:
    sheet = CheatSheet()
    # in the render loop:
    sheet.draw(frame)
    # in the key handler:
    consumed = sheet.handle_key(key)

Keys:
    ?          toggle panel open / collapsed
    [ / ]      step backward / forward through the guided tour
    Esc        close panel (if open)

The panel is a semi-transparent dark sidebar on the left edge.  When
collapsed it shrinks to a narrow tab showing "? help".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np


# ── colour palette ────────────────────────────────────────────────────────────
_BG        = (18,  18,  22)      # panel background
_BG_ALPHA  = 0.82               # panel opacity
_BORDER    = (55,  55,  65)
_HEADER_BG = (30,  30,  38)
_ACCENT    = (80, 200, 255)      # blue — matches MOVE colour
_TOUR_HL   = (80, 255, 140)      # green — tour highlight
_KEY_COL   = (255, 190,  60)     # amber — key badges
_TEXT      = (210, 210, 210)
_DIM       = (110, 110, 120)
_WHITE     = (255, 255, 255)

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX   # heavier weight for headers


# ── data model ───────────────────────────────────────────────────────────────

class Entry(NamedTuple):
    keys: str       # e.g. "G"  or  "Fist"
    desc: str       # short description shown in the panel
    detail: str     # one-sentence explanation shown during tour


@dataclass
class Section:
    title: str
    entries: list[Entry]


# ── cheat-sheet content ───────────────────────────────────────────────────────

_SECTIONS: list[Section] = [
    Section("TRANSFORM", [
        Entry("Hold G",   "Move object",        "Hold G while your pinch is near an object — it follows your hand.  Release G to commit."),
        Entry("Hold R",   "Rotate object",       "Hold R and twist your pinch hand.  Release R to lock in the new angle."),
        Entry("Hold S",   "Scale object",        "Hold S and spread or close your thumb-index gap.  Release S to set the size."),
        Entry("X / Y",    "Lock axis",           "While holding G, tap X to constrain motion horizontally, Y to constrain vertically."),
        Entry("Esc",      "Cancel",              "Press Esc (or Open Palm gesture) while holding a key to revert to the original transform."),
    ]),
    Section("SELECTION", [
        Entry("Tab",      "Cycle objects",       "Tab steps through each object in order, highlighting it for the next operation."),
        Entry("A",        "Select / deselect all","A selects every object at once.  Press again to deselect all."),
        Entry("Del / ⌫", "Delete selected",      "Delete removes the selected object(s).  There is no undo yet — be careful!"),
    ]),
    Section("SCENE", [
        Entry("N",        "New object",          "Spawns a new square at the current gaze point (or screen centre without gaze)."),
        Entry("Z",        "Cycle heatmap",        "Cycles gaze heatmap opacity: off → 30 % → 55 % → 80 % → off."),
        Entry("C",        "Clear heatmap",        "Wipes the accumulated gaze heatmap immediately."),
        Entry("H",        "Toggle skeleton",      "Shows or hides the hand skeleton overlay on the camera feed."),
    ]),
    Section("GESTURES", [
        Entry("Fist ✊",   "→ Move",              "Close all fingers into a fist to enter Move mode on the nearest object."),
        Entry("Pinch 🤌",  "→ Confirm",           "Bring thumb and index tip together to confirm (commit) the active transform."),
        Entry("Palm 🖐",   "→ Cancel",            "Open your palm with all five fingers extended to cancel the active transform."),
        Entry("Point ☝",  "Aim cursor",           "Extend only your index finger to aim a hover cursor without triggering actions."),
        Entry("V ✌",      "Cycle objects",        "Extend index and middle fingers (V-sign) to Tab-cycle through objects."),
    ]),
    Section("APP", [
        Entry("?",        "Toggle this panel",   "Press ? to show or hide this cheat-sheet at any time."),
        Entry("[ / ]",    "Tour steps",          "Step backward / forward through a guided tour explaining each feature."),
        Entry("R (no hand)", "Recalibrate gaze", "Press R when no hand is visible to restart the gaze calibration sequence."),
        Entry("D",        "Glasses-cam debug",   "Toggles an inset showing what the glasses-mounted camera sees (if connected)."),
        Entry("Q / Esc",  "Quit",                "Q or Esc (when no modal is active) closes the application."),
    ]),
]

# Flat list of all entries for tour stepping
_TOUR_ENTRIES: list[tuple[Section, Entry]] = [
    (sec, e) for sec in _SECTIONS for e in sec.entries
]


# ── panel layout constants ────────────────────────────────────────────────────

_PANEL_W         = 310     # expanded width in pixels
_COLLAPSED_W     = 52      # collapsed tab width
_MARGIN          = 10
_LINE_H          = 22      # vertical step between entries
_SECTION_HEADER_H= 28
_FONT_SCALE_HDR  = 0.48
_FONT_SCALE_KEY  = 0.42
_FONT_SCALE_BODY = 0.40
_FONT_SCALE_TOUR = 0.39
_TOUR_PANEL_H    = 72      # height of the tour detail box at the bottom
_BADGE_PAD_X     = 6
_BADGE_PAD_Y     = 3


# ── CheatSheet class ──────────────────────────────────────────────────────────

class CheatSheet:
    """
    Stateful collapsible cheat-sheet panel.

    Call draw(frame) every render loop.  Call handle_key(key) in the key
    handler; returns True if the key was consumed so the caller can skip
    its own handling.
    """

    def __init__(self) -> None:
        self.open: bool = False          # panel expanded?
        self.tour_idx: int | None = None # None = not in tour mode

    # ── public API ────────────────────────────────────────────────────────────

    def handle_key(self, key: int) -> bool:
        """Return True if the key was consumed by the cheat-sheet."""
        if key == ord("?"):
            self.open = not self.open
            if not self.open:
                self.tour_idx = None
            return True
        if self.open and key == 27:      # Esc closes panel
            self.open = False
            self.tour_idx = None
            return True
        if self.open and key == ord("]"):
            self._tour_step(+1)
            return True
        if self.open and key == ord("["):
            self._tour_step(-1)
            return True
        return False

    def draw(self, frame: np.ndarray) -> None:
        if self.open:
            self._draw_expanded(frame)
        else:
            self._draw_collapsed(frame)

    # ── private helpers ───────────────────────────────────────────────────────

    def _tour_step(self, delta: int) -> None:
        n = len(_TOUR_ENTRIES)
        if self.tour_idx is None:
            self.tour_idx = 0 if delta > 0 else n - 1
        else:
            self.tour_idx = (self.tour_idx + delta) % n

    def _draw_collapsed(self, frame: np.ndarray) -> None:
        sh, sw = frame.shape[:2]
        w = _COLLAPSED_W
        # Background tab
        _fill_rect(frame, 0, 0, w, 90, _BG, _BG_ALPHA)
        cv2.rectangle(frame, (w - 1, 0), (w - 1, 89), _BORDER, 1)
        # "?" icon
        cv2.putText(frame, "?", (14, 32), _FONT_BOLD, 0.9, _ACCENT, 2, cv2.LINE_AA)
        cv2.putText(frame, "help", (8, 54), _FONT, 0.38, _DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, "[ ]", (10, 74), _FONT, 0.33, _DIM, 1, cv2.LINE_AA)

    def _draw_expanded(self, frame: np.ndarray) -> None:
        sh, sw = frame.shape[:2]
        pw = _PANEL_W

        # How many entries fit? Compute total height needed.
        # We'll scroll-clip if needed.
        total_rows = sum(len(s.entries) + 1 for s in _SECTIONS)  # +1 for header
        content_h  = total_rows * _LINE_H + len(_SECTIONS) * 4
        tour_box_h = _TOUR_PANEL_H + _MARGIN if self.tour_idx is not None else 0
        panel_h    = min(sh, content_h + _MARGIN * 3 + 36 + tour_box_h)

        # Background
        _fill_rect(frame, 0, 0, pw, panel_h, _BG, _BG_ALPHA)
        cv2.rectangle(frame, (pw - 1, 0), (pw - 1, panel_h - 1), _BORDER, 1)
        cv2.rectangle(frame, (0, panel_h - 1), (pw - 1, panel_h - 1), _BORDER, 1)

        # ── header bar ────────────────────────────────────────────────────────
        _fill_rect(frame, 0, 0, pw, 30, _HEADER_BG, 1.0)
        cv2.putText(frame, "HANDY-IO  CHEAT SHEET", (_MARGIN, 21),
                    _FONT_BOLD, 0.44, _ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "?=close  [/]=tour", (pw - 118, 21),
                    _FONT, 0.33, _DIM, 1, cv2.LINE_AA)
        cv2.line(frame, (0, 30), (pw - 1, 30), _BORDER, 1)

        # ── tour highlight entry (which entry the tour is on) ─────────────────
        tour_sec: Section | None = None
        tour_entry: Entry | None = None
        if self.tour_idx is not None:
            tour_sec, tour_entry = _TOUR_ENTRIES[self.tour_idx]

        # ── section + entry rows ──────────────────────────────────────────────
        y = 36
        max_y = panel_h - tour_box_h - _MARGIN

        for sec in _SECTIONS:
            if y + _SECTION_HEADER_H > max_y:
                break

            # Section title
            is_active_sec = (tour_sec is sec)
            sec_col = _TOUR_HL if is_active_sec else _DIM
            cv2.putText(frame, sec.title, (_MARGIN, y + 14),
                        _FONT_BOLD, _FONT_SCALE_HDR, sec_col, 1, cv2.LINE_AA)
            cv2.line(frame, (_MARGIN, y + 18), (pw - _MARGIN, y + 18),
                     (40, 40, 50), 1)
            y += _SECTION_HEADER_H

            for entry in sec.entries:
                if y + _LINE_H > max_y:
                    break

                is_tour = (tour_entry is entry)

                # Tour highlight row background
                if is_tour:
                    _fill_rect(frame, 0, y - 14, pw, y + 6, _TOUR_HL, 0.08)
                    cv2.rectangle(frame, (0, y - 14), (pw - 1, y + 6),
                                  _TOUR_HL, 1)

                # Key badge
                key_text = entry.keys
                (kw, kh), _ = cv2.getTextSize(key_text, _FONT, _FONT_SCALE_KEY, 1)
                bx1, by1 = _MARGIN, y - kh - _BADGE_PAD_Y
                bx2, by2 = _MARGIN + kw + _BADGE_PAD_X * 2, y + _BADGE_PAD_Y - 2
                badge_col = _TOUR_HL if is_tour else _KEY_COL
                _fill_rect(frame, bx1, by1, bx2, by2, badge_col, 0.15)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_col, 1)
                cv2.putText(frame, key_text, (_MARGIN + _BADGE_PAD_X, y - 1),
                            _FONT, _FONT_SCALE_KEY, badge_col, 1, cv2.LINE_AA)

                # Description
                desc_x = bx2 + 8
                desc_col = _WHITE if is_tour else _TEXT
                cv2.putText(frame, entry.desc, (desc_x, y - 1),
                            _FONT, _FONT_SCALE_BODY, desc_col, 1, cv2.LINE_AA)

                y += _LINE_H

            y += 2  # gap between sections

        # ── tour detail box at the bottom ─────────────────────────────────────
        if self.tour_idx is not None and tour_entry is not None:
            n = len(_TOUR_ENTRIES)
            box_y0 = panel_h - _TOUR_PANEL_H - _MARGIN
            box_y1 = panel_h - _MARGIN

            _fill_rect(frame, _MARGIN - 2, box_y0, pw - _MARGIN + 2, box_y1,
                       _TOUR_HL, 0.07)
            cv2.rectangle(frame, (_MARGIN - 2, box_y0),
                          (pw - _MARGIN + 2, box_y1), _TOUR_HL, 1)

            # step counter
            counter = f"Step {self.tour_idx + 1} / {n}"
            cv2.putText(frame, counter, (_MARGIN + 4, box_y0 + 14),
                        _FONT_BOLD, 0.38, _TOUR_HL, 1, cv2.LINE_AA)
            cv2.putText(frame, "[ prev   next ]",
                        (pw - 108, box_y0 + 14),
                        _FONT, 0.33, _DIM, 1, cv2.LINE_AA)

            # detail text — word-wrap to fit panel width
            lines = _word_wrap(tour_entry.detail, pw - _MARGIN * 2 - 8,
                               _FONT, _FONT_SCALE_TOUR, 1)
            ty = box_y0 + 30
            for line in lines[:3]:   # max 3 lines
                cv2.putText(frame, line, (_MARGIN + 4, ty),
                            _FONT, _FONT_SCALE_TOUR, _TEXT, 1, cv2.LINE_AA)
                ty += 16


# ── drawing utilities ─────────────────────────────────────────────────────────

def _fill_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend a filled rectangle onto *frame* (BGR, in-place)."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _word_wrap(text: str, max_px: int, font, scale: float, thickness: int) -> list[str]:
    """Split *text* into lines that fit within *max_px* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        (w, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if w <= max_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
