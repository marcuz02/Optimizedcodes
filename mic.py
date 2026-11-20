# ==========================================
# hardware/mic.py
# JOB: record one spoken sentence. Nothing else.
# Returns audio as a numpy int16 array (16 kHz, mono), or None if nobody spoke.
# Fake mode (config.MIC_FAKE = True) returns a made-up tone, no microphone needed.
# ==========================================
import collections
import logging

import config

log = logging.getLogger("mic")
CHUNK_S = 0.1     # the mic is read in pieces of 0.1 s

_np = None


def _get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


class MicError(Exception):
    """Raised when the microphone cannot be used."""


def _rms(chunk, np_mod=None):
    np = _get_np() if np_mod is None else np_mod
    if chunk.size == 0:
        return 0.0
    samples = chunk.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(samples * samples)))


def collect_speech(chunks, chunk_s=CHUNK_S, threshold=None, silence_timeout=None, max_duration=None,
                   preroll_s=None, min_speech_s=None, tail_keep_s=None, on_level=None, abort=None):
    """
    The recording brain (no hardware inside, so it can be tested).
    chunks = any iterator of int16 arrays, each chunk_s seconds long.
    Waits for speech, keeps it, stops after silence. Returns int16 array or None.
    """
    np = _get_np()

    threshold = config.MIC_THRESHOLD if threshold is None else threshold
    silence_timeout = config.MIC_SILENCE_TIMEOUT_S if silence_timeout is None else silence_timeout
    max_duration = config.MIC_MAX_DURATION_S if max_duration is None else max_duration
    preroll_s = config.MIC_PREROLL_S if preroll_s is None else preroll_s
    min_speech_s = config.MIC_MIN_SPEECH_S if min_speech_s is None else min_speech_s
    tail_keep_s = config.MIC_TAIL_KEEP_S if tail_keep_s is None else tail_keep_s

    max_chunks = int(round(max_duration / chunk_s))
    max_silent = int(round(silence_timeout / chunk_s))
    tail_keep = int(round(tail_keep_s / chunk_s))
    preroll_len = max(1, int(round(preroll_s / chunk_s)))
    preroll = collections.deque(maxlen=preroll_len)

    recorded = []
    started = False
    silent = 0
    loud = 0
    has_abort = abort is not None
    has_on_level = on_level is not None

    for i, chunk in enumerate(chunks):
        if i >= max_chunks:
            break
        if has_abort and abort():
            break

        level = _rms(chunk, np)
        if has_on_level:
            on_level(level)

        if not started:
            preroll.append(chunk)
            if level > threshold:
                started = True
                loud = 1
                recorded.extend(preroll)          # includes the sound just before the speech
        else:
            recorded.append(chunk)
            if level > threshold:
                silent = 0
                loud += 1
            else:
                silent += 1
                if silent >= max_silent:
                    break

    if not started or (loud * chunk_s) < (min_speech_s - 1e-9):
        return None
    if silent > tail_keep:                        # cut the long silence at the end
        del recorded[len(recorded) - (silent - tail_keep):]
    return np.concatenate(recorded)


def _build_fake_clip(sample_rate):
    np = _get_np()
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (3000.0 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.int16)
    silence = np.zeros(sample_rate // 3, dtype=np.int16)
    return np.concatenate((silence, tone))


class Microphone:
    def __init__(self, fake=None):
        self.fake = bool(config.MIC_FAKE if fake is None else fake)
        self._fake_clip = None
        self._fake_sr = None

    def open(self):
        """Checks that the microphone exists."""
        if self.fake:
            log.info("Mic fake mode: made-up voice")
            return
        try:
            import sounddevice as sd
            info = sd.query_devices(config.MIC_DEVICE, "input")
        except Exception as e:
            raise MicError(f"Cannot use microphone {config.MIC_DEVICE!r}: {e}") from e
        log.info("Mic device: %s", info["name"])

    def record_until_silence(self, on_level=None, abort=None, **overrides):
        """Blocks until one sentence is recorded. Returns int16 array or None."""
        if self.fake:
            sr = config.AUDIO_SAMPLE_RATE
            if self._fake_clip is None or self._fake_sr != sr:
                self._fake_clip = _build_fake_clip(sr)
                self._fake_sr = sr
            return self._fake_clip.copy()
        return self._record_real(on_level, abort, overrides)

    def _stream_chunks(self):
        import sounddevice as sd
        sr = config.AUDIO_SAMPLE_RATE
        n = int(sr * CHUNK_S)
        with sd.InputStream(device=config.MIC_DEVICE, samplerate=sr,
                            channels=1, dtype="int16") as stream:
            while True:
                data, _ = stream.read(n)
                yield data[:, 0].copy()

    def _record_real(self, on_level, abort, overrides):
        gen = self._stream_chunks()
        try:
            return collect_speech(gen, on_level=on_level, abort=abort, **overrides)
        except Exception as e:
            raise MicError(f"Recording failed: {e}") from e
        finally:
            gen.close()                            # closes the microphone stream
