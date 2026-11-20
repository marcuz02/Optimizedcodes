# ==========================================
# hardware/camera.py
# JOB: give camera pictures (frames). Nothing else.
# Fake mode (config.CAMERA_FAKE = True) gives grey test frames.
# ==========================================
import logging
import os

import config

log = logging.getLogger("camera")


class CameraError(Exception):
    """Raised when the camera cannot be opened."""


class Camera:
    def __init__(self, fake=None):
        self.fake = bool(config.CAMERA_FAKE if fake is None else fake)
        self._cap = None
        self._count = 0
        self._is_open = False
        self._cv2 = None
        self._np = None
        self._fake_frame = None

    def _ensure_imaging_libs(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        if not self.fake and self._np is None:
            import numpy as np
            self._np = np

    def open(self):
        if self._is_open:
            return

        if self.fake:
            self._ensure_imaging_libs()
            width = config.CAMERA_WIDTH
            height = config.CAMERA_HEIGHT
            base = self._np.full((height, width, 3), 80, dtype=self._np.uint8) if self._np else None
            if base is None:
                import numpy as np
                self._np = np
                base = np.full((height, width, 3), 80, dtype=np.uint8)
            self._fake_frame = base
            log.info("Camera fake mode: grey test frames")
        else:
            self._ensure_imaging_libs()
            cv2 = self._cv2
            src = config.CAMERA_SOURCE
            if os.name == "nt" and isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)

            if not cap.isOpened():
                raise CameraError(
                    f"Cannot open camera source {src!r}. Is it used by another program?"
                )

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
            log.info("Camera open: source %r", src)

        self._is_open = True

    def read(self):
        """Returns one BGR frame (numpy array), or None if no picture came."""
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")

        if self.fake:
            self._count += 1
            # Copy pre-filled buffer instead of allocating & filling from scratch
            frame = self._fake_frame.copy()
            self._cv2.putText(
                frame,
                f"fake frame {self._count}",
                (20, 40),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            return frame

        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._fake_frame = None
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()