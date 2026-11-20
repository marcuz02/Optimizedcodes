# ==========================================
# hardware/speaker.py
# JOB: play sound. Nothing else.
# Audio = numpy int16 array (or float -1..1) + sample rate.
# Fake mode (config.SPEAKER_FAKE = True) stores the sounds in self.played.
# ==========================================
import logging
import time

import config

log = logging.getLogger("speaker")

_np = None
_sd = None


def _get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_sd():
    global _sd
    if _sd is None:
        import sounddevice as sd
        _sd = sd
    return _sd


class SpeakerError(Exception):
    """Raised when the speaker cannot be used."""


class Speaker:
    def __init__(self, fake=None):
        self.fake = bool(config.SPEAKER_FAKE if fake is None else fake)
        self.played = []          # fake mode: list of (audio, sample_rate)

    def open(self):
        """Checks that the speaker exists."""
        if self.fake:
            log.info("Speaker fake mode: nothing is played")
            return
        try:
            sd = _get_sd()
            info = sd.query_devices(config.SPEAKER_DEVICE, "output")
        except Exception as e:
            raise SpeakerError(f"Cannot use speaker {config.SPEAKER_DEVICE!r}: {e}") from e
        log.info("Speaker device: %s", info["name"])

    def close(self):
        self.stop()

    def _prepare(self, audio):
        np = _get_np()
        data = np.asarray(audio)
        if data.ndim > 1:
            data = data.reshape(data.shape[0], -1).mean(axis=1)
        if data.dtype != np.int16:
            data = np.clip(data, -1.0, 1.0) * 32767.0
        vol = float(config.SPEAKER_VOLUME)
        if vol != 1.0:
            data = data.astype(np.float32, copy=False) * vol
        return np.clip(data, -32768.0, 32767.0).astype(np.int16)

    def play(self, audio, sample_rate, blocking=False):
        """Starts playing. Returns the time (time.perf_counter) when playing started."""
        data = self._prepare(audio)
        if self.fake:
            self.played.append((data, sample_rate))
            return time.perf_counter()
        try:
            sd = _get_sd()
            sd.play(data, sample_rate, device=config.SPEAKER_DEVICE)
        except Exception as e:
            raise SpeakerError(f"Playback failed: {e}") from e
        start = time.perf_counter()
        if blocking:
            sd.wait()
        return start

    def wait(self):
        """Waits until the sound has finished."""
        if not self.fake:
            _get_sd().wait()

    def stop(self):
        if not self.fake:
            try:
                _get_sd().stop()
            except Exception:
                pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()