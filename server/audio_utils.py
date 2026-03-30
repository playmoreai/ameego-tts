from __future__ import annotations

import base64
from pathlib import Path

import soundfile as sf


MAX_AUDIO_BYTES = 10 * 1024 * 1024
MIN_AUDIO_DURATION_MS = 500.0


def decode_audio_base64(audio_base64: str) -> bytes:
    try:
        normalized = "".join(audio_base64.split())
        audio_bytes = base64.b64decode(normalized, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64-encoded audio data.") from exc

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError("Reference audio exceeds 10MB limit.")

    return audio_bytes


def normalize_audio_to_wav(
    *,
    audio_bytes: bytes,
    audio_format: str,
    output_path: Path,
    min_duration_ms: float = MIN_AUDIO_DURATION_MS,
) -> float:
    fmt = audio_format.lower()
    tmp_path = output_path.with_suffix(f".tmp.{fmt}")
    try:
        try:
            if fmt == "wav":
                output_path.write_bytes(audio_bytes)
                data, sample_rate = sf.read(str(output_path))
            else:
                tmp_path.write_bytes(audio_bytes)
                data, sample_rate = sf.read(str(tmp_path))
                sf.write(str(output_path), data, sample_rate)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    finally:
        tmp_path.unlink(missing_ok=True)

    if sample_rate <= 0:
        raise ValueError("Invalid audio sample rate.")

    duration_ms = (len(data) / sample_rate) * 1000
    if duration_ms < min_duration_ms:
        raise ValueError("Reference audio is too short.")

    return duration_ms
