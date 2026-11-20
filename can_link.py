# ==========================================
# hardware/can_link.py
# JOB: send messages to the STM32 over CAN. Nothing else.
# Fake mode (config.CAN_FAKE = True) works with no hardware:
# frames are stored in self.sent so tests can check them.
# ==========================================
import logging
import threading
from collections import deque

import config
import protocol as P

log = logging.getLogger("can_link")

_QUIET_FAKE_IDS = frozenset((P.ID_MOUTH, P.ID_HEARTBEAT))


def _clamp_byte(value):
    ival = int(value)
    if ival < 0:
        return 0
    if ival > 255:
        return 255
    return ival


def _clamp_percent(value):
    ival = int(value)
    if ival < 0:
        return 0
    if ival > 100:
        return 100
    return ival


class CanLinkError(Exception):
    """Raised when the CAN adapter cannot be opened."""


class CanLink:
    def __init__(self, fake=None):
        self.fake = bool(config.CAN_FAKE if fake is None else fake)
        self._lock = threading.Lock()
        self._mode = P.MODE_IDLE
        self.sent = deque(maxlen=500)      # fake mode: deque of (id, [data])
        self._can = None                   # python-can module (real mode)
        self._bus = None
        self._halt = threading.Event()
        self._thread = None
        self._counter = 0
        self._is_open = False

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value):
        with self._lock:
            self._mode = value

    # ---------- open / close ----------
    def open(self):
        if self._is_open:
            return

        if self.fake:
            log.info("CAN fake mode: frames are stored, not sent")
        else:
            try:
                import can
                self._can = can
                if config.CAN_INTERFACE == "socketcan":
                    # socketcan gets its bitrate from `ip link`
                    self._bus = can.interface.Bus(
                        channel=config.CAN_CHANNEL,
                        interface=config.CAN_INTERFACE,
                    )
                else:
                    self._bus = can.interface.Bus(
                        channel=config.CAN_CHANNEL,
                        interface=config.CAN_INTERFACE,
                        bitrate=config.CAN_BITRATE,
                    )
            except Exception as e:
                raise CanLinkError(
                    f"Cannot open CAN ({config.CAN_INTERFACE}/{config.CAN_CHANNEL}): {e}"
                ) from e
            log.info("CAN open: %s / %s", config.CAN_INTERFACE, config.CAN_CHANNEL)

        self._halt.clear()
        self._is_open = True
        thread = threading.Thread(target=self._heartbeat_loop, name="can-heartbeat", daemon=True)
        self._thread = thread
        thread.start()

    def close(self):
        if not self._is_open:
            return

        self._halt.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

        with self._lock:
            self._mode = P.MODE_IDLE

        # Leave the robot in a safe state before marking closed/shutting down
        self._send(P.ID_STATE, [P.MODE_IDLE])
        self._send(P.ID_MOUTH, [0])
        self._send(P.ID_HEAD, [P.HEAD_CENTER, P.HEAD_CENTER])

        self._is_open = False
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
        log.info("CAN closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------- public API ----------
    def send_state(self, mode):
        if mode not in P.MODES:
            log.warning("Unknown mode %r ignored", mode)
            return
        with self._lock:
            if mode == self._mode:         # only send when mode changes
                return
            self._mode = mode
        self._send(P.ID_STATE, [mode])

    def send_emotion(self, emotion, intensity=80, source=P.SRC_USER):
        emotion_id = P.EMOTION_ID.get(emotion)
        if emotion_id is None:
            log.warning("Unknown emotion %r -> neutral", emotion)
            emotion_id = P.EMOTION_ID["neutral"]
        clamped_intensity = _clamp_percent(intensity)
        self._send(P.ID_EMOTION, [emotion_id, clamped_intensity, source])

    def send_mouth(self, opening):
        self._send(P.ID_MOUTH, [_clamp_byte(opening)])

    def send_head(self, eye_x, neck):
        """eye_x and neck are 0-255, 128 = center (protocol.HEAD_CENTER)."""
        self._send(P.ID_HEAD, [_clamp_byte(eye_x), _clamp_byte(neck)])

    # ---------- internal ----------
    def _send(self, arb_id, data):
        payload = data if isinstance(data, list) else list(data)
        if self.fake:
            self.sent.append((arb_id, payload))
            if arb_id not in _QUIET_FAKE_IDS and log.isEnabledFor(logging.DEBUG):
                log.debug("CAN(fake) 0x%03X %s", arb_id, payload)
            return True

        bus = self._bus
        can_mod = self._can
        if bus is None or can_mod is None:
            return False

        msg = can_mod.Message(
            arbitration_id=arb_id,
            data=bytes(payload),
            is_extended_id=False,
        )
        with self._lock:
            try:
                bus.send(msg, timeout=config.CAN_SEND_TIMEOUT_S)
                return True
            except can_mod.CanError as e:
                log.error("CAN send failed: %s", e)
                return False

    def _heartbeat_loop(self):
        interval = config.CAN_HEARTBEAT_S
        while not self._halt.is_set():
            self._counter = (self._counter + 1) & 0xFF
            current_mode = self.mode
            self._send(P.ID_HEARTBEAT, [current_mode, self._counter])
            self._halt.wait(interval)