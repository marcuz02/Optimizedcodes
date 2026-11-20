# ==========================================
# utils/debug_draw.py
# JOB: draw the debug overlay (face box + emotion bars) onto a frame.
# Used by both main.py's live preview and tests/demo_probs.py.
# No camera or model code in here - just drawing.
# ==========================================
import protocol as P

_cv2 = None
_FONT = None
_BOX_COLOR = (0, 255, 0)
_BAR_COLOR = (0, 200, 255)
_TEXT_COLOR = (255, 255, 255)
_LABEL_COLOR = (0, 0, 255)
_FPS_COLOR = (0, 255, 255)

_ROWS = None
_ROWS_SOURCE = None


def _get_cv2():
    global _cv2, _FONT
    if _cv2 is None:
        import cv2
        _cv2 = cv2
        _FONT = cv2.FONT_HERSHEY_SIMPLEX
    return _cv2


def _get_rows():
    global _ROWS, _ROWS_SOURCE
    emotions = P.EMOTIONS
    if _ROWS is None or _ROWS_SOURCE is not emotions:
        _ROWS = tuple(
            (
                30 + i * 26,
                18 + i * 26,
                36 + i * 26,
                34 + i * 26,
                f"{name:8}",
            )
            for i, name in enumerate(emotions)
        )
        _ROWS_SOURCE = emotions
    return _ROWS


def draw_overlay(frame, box, probs, label=None, fps=None):
    """Draws in place on `frame` (a BGR numpy array) and returns it."""
    cv2 = _get_cv2()
    font = _FONT
    rectangle = cv2.rectangle
    putText = cv2.putText

    if box is not None:
        x, y, w, h = box
        rectangle(frame, (x, y), (x + w, y + h), _BOX_COLOR, 2)

    if probs is not None:
        rows = _get_rows()
        for i, (_, bar_top, bar_bot, text_y, prefix) in enumerate(rows):
            p = float(probs[i])
            bar_x2 = 110 + int(p * 200)
            rectangle(frame, (110, bar_top), (bar_x2, bar_bot), _BAR_COLOR, -1)
            putText(frame, f"{prefix} {p:.2f}", (8, text_y), font, 0.5, _TEXT_COLOR, 1)
    elif label:
        putText(frame, label, (8, 30), font, 0.7, _LABEL_COLOR, 2)

    if fps is not None:
        h_frame = frame.shape[0]
        putText(frame, f"{fps:.0f} FPS", (8, h_frame - 12), font, 0.6, _FPS_COLOR, 2)

    return frame