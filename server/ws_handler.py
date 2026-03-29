from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import struct
import threading
import time

from fastapi import WebSocket, WebSocketDisconnect

from server.config import SUPPORTED_LANGUAGES, settings
from server.models import (
    ALLOWED_AUDIO_FORMATS,
    ErrorMessage,
    PongMessage,
    SynthesisCancelled,
    SynthesisEnd,
    SynthesisStart,
    SynthesizeRequest,
    UploadRefAudioRequest,
    VoiceClonePromptReady,
)
from server.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Binary frame header: magic(4) + request_id_hash(4) + chunk_index(4) + sample_rate(4)
HEADER_MAGIC = b"AMEG"
HEADER_SIZE = 16


def _make_header(request_id: str, chunk_index: int, sample_rate: int) -> bytes:
    rid_hash = struct.unpack("<I", hashlib.sha256(request_id.encode()).digest()[:4])[0]
    return HEADER_MAGIC + struct.pack("<III", rid_hash, chunk_index, sample_rate)


async def _send_error(
    ws: WebSocket,
    code: str,
    message: str,
    request_id: str | None = None,
) -> None:
    """Send an error message to the client."""
    await ws.send_json(
        ErrorMessage(request_id=request_id, code=code, message=message).model_dump()
    )


async def _cancel_task(
    cancel_event: threading.Event | None,
    synthesis_task: asyncio.Task | None,
) -> None:
    """Cancel active synthesis cleanly."""
    if cancel_event:
        cancel_event.set()
    if synthesis_task and not synthesis_task.done():
        synthesis_task.cancel()
        try:
            await synthesis_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("Synthesis task error during cancel", exc_info=True)


async def handle_websocket(ws: WebSocket, engine: TTSEngine) -> None:
    logger.info("WebSocket connected: %s", ws.client)

    cancel_event: threading.Event | None = None
    synthesis_task: asyncio.Task | None = None

    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type")

            if msg_type == "ping":
                await ws.send_json(PongMessage().model_dump())

            elif msg_type == "cancel":
                logger.info("Cancel requested for %s", raw.get("request_id", ""))
                await _cancel_task(cancel_event, synthesis_task)
                cancel_event = None
                synthesis_task = None

            elif msg_type == "synthesize":
                await _cancel_task(cancel_event, synthesis_task)
                cancel_event = threading.Event()
                synthesis_task = asyncio.create_task(
                    _handle_synthesize(ws, engine, raw, cancel_event)
                )

            elif msg_type == "upload_ref_audio":
                await _handle_upload_ref_audio(ws, engine, raw)

            else:
                await _send_error(ws, "UNKNOWN_TYPE", f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", ws.client)
        await _cancel_task(cancel_event, synthesis_task)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        await _cancel_task(cancel_event, synthesis_task)
        try:
            await _send_error(ws, "INTERNAL_ERROR", str(e))
        except Exception:
            pass


async def _handle_synthesize(
    ws: WebSocket,
    engine: TTSEngine,
    raw: dict,
    cancel_event: threading.Event,
) -> None:
    try:
        req = SynthesizeRequest(**raw)
    except Exception as e:
        await _send_error(ws, "INVALID_REQUEST", str(e))
        return

    # Validate text
    if not req.text or not req.text.strip():
        await _send_error(ws, "INVALID_TEXT", "Text cannot be empty.", req.request_id)
        return

    if len(req.text) > settings.max_text_length:
        await _send_error(
            ws, "INVALID_TEXT",
            f"Text exceeds maximum length of {settings.max_text_length} characters.",
            req.request_id,
        )
        return

    if req.language not in SUPPORTED_LANGUAGES:
        await _send_error(
            ws, "INVALID_LANGUAGE",
            f"Unsupported language: {req.language}. Supported: {sorted(SUPPORTED_LANGUAGES)}",
            req.request_id,
        )
        return

    # Resolve voice clone prompt
    ref_audio_path = None
    voice_clone_prompt = None
    if req.voice_clone_prompt_id:
        voice_clone_prompt = engine.get_voice_prompt(req.voice_clone_prompt_id)
        if voice_clone_prompt is None:
            # Prompt cache miss — fall back to ref audio (slower, recomputes embedding)
            path = engine.get_ref_audio_path(req.voice_clone_prompt_id)
            if path is None:
                await _send_error(
                    ws, "PROMPT_NOT_FOUND",
                    f"Voice clone prompt not found: {req.voice_clone_prompt_id}",
                    req.request_id,
                )
                return
            ref_audio_path = str(path)

    # Send synthesis_start
    await ws.send_json(
        SynthesisStart(
            request_id=req.request_id,
            sample_rate=engine.sample_rate,
        ).model_dump()
    )

    # Stream audio
    t_start = time.perf_counter()
    t_first_chunk = None
    chunk_index = 0
    total_bytes = 0
    cancelled = False

    try:
        async for pcm_chunk in engine.stream_tts(
            text=req.text,
            language=req.language,
            chunk_size=req.chunk_size,
            ref_audio_path=ref_audio_path,
            voice_clone_prompt=voice_clone_prompt,
            cancel_event=cancel_event,
        ):
            if cancel_event.is_set():
                cancelled = True
                break

            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()

            header = _make_header(req.request_id, chunk_index, engine.sample_rate)
            await ws.send_bytes(header + pcm_chunk)
            total_bytes += len(pcm_chunk)
            chunk_index += 1

    except Exception as e:
        logger.error("Synthesis streaming error: %s", e, exc_info=True)
        try:
            await _send_error(ws, "SYNTHESIS_ERROR", str(e), req.request_id)
        except Exception:
            pass
        return

    # Send appropriate end message
    if cancelled:
        logger.info("Synthesis cancelled: %d chunks sent", chunk_index)
        try:
            await ws.send_json(
                SynthesisCancelled(
                    request_id=req.request_id,
                    chunks_sent=chunk_index,
                ).model_dump()
            )
        except Exception:
            pass
    else:
        t_end = time.perf_counter()
        total_samples = total_bytes // 2
        duration_ms = (total_samples / engine.sample_rate) * 1000 if engine.sample_rate else 0
        wall_time_ms = (t_end - t_start) * 1000
        ttfa_ms = (t_first_chunk - t_start) * 1000 if t_first_chunk else 0
        rtf = (wall_time_ms / duration_ms) if duration_ms > 0 else 0

        await ws.send_json(
            SynthesisEnd(
                request_id=req.request_id,
                total_chunks=chunk_index,
                total_samples=total_samples,
                duration_ms=duration_ms,
                ttfa_ms=round(ttfa_ms, 1),
                rtf=round(rtf, 3),
            ).model_dump()
        )

        logger.info(
            "Synthesis complete: %d chunks, %.0fms audio, TTFA=%.0fms, RTF=%.3f",
            chunk_index, duration_ms, ttfa_ms, rtf,
        )


async def _handle_upload_ref_audio(
    ws: WebSocket,
    engine: TTSEngine,
    raw: dict,
) -> None:
    try:
        req = UploadRefAudioRequest(**raw)
    except Exception as e:
        await _send_error(ws, "INVALID_REQUEST", str(e))
        return

    # Validate audio format
    fmt = req.audio_format.lower().strip()
    if fmt not in ALLOWED_AUDIO_FORMATS:
        await _send_error(
            ws, "INVALID_AUDIO",
            f"Unsupported audio format: {req.audio_format}. Supported: {sorted(ALLOWED_AUDIO_FORMATS)}",
            req.request_id,
        )
        return

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        await _send_error(
            ws, "INVALID_AUDIO", "Invalid base64-encoded audio data.", req.request_id,
        )
        return

    if len(audio_bytes) < 1000:
        await _send_error(
            ws, "INVALID_AUDIO",
            "Reference audio is too short. Please provide at least 3 seconds.",
            req.request_id,
        )
        return

    if len(audio_bytes) > 10 * 1024 * 1024:
        await _send_error(
            ws, "INVALID_AUDIO", "Reference audio exceeds 10MB limit.", req.request_id,
        )
        return

    # Save and convert to WAV
    t0 = time.perf_counter()
    try:
        prompt_id, _ = engine.save_ref_audio(audio_bytes, fmt)
    except ValueError as e:
        await _send_error(ws, "INVALID_AUDIO", str(e), req.request_id)
        return

    # Pre-compute speaker embedding on GPU
    try:
        precompute_ms = await engine.run_on_gpu(
            engine.precompute_voice_prompt, prompt_id,
        )
    except Exception as e:
        logger.error("Failed to precompute voice prompt: %s", e, exc_info=True)
        await _send_error(
            ws, "VOICE_CLONE_ERROR",
            f"Failed to process reference audio: {e}",
            req.request_id,
        )
        return

    elapsed = (time.perf_counter() - t0) * 1000

    await ws.send_json(
        VoiceClonePromptReady(
            request_id=req.request_id,
            prompt_id=prompt_id,
            processing_ms=round(elapsed, 1),
        ).model_dump()
    )

    logger.info(
        "Voice clone prompt ready: %s (%.0fms, precompute=%.0fms)",
        prompt_id, elapsed, precompute_ms,
    )
