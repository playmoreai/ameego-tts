from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import threading
import time

from fastapi import WebSocket, WebSocketDisconnect

from server.audio_utils import decode_audio_base64
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
from server.tts_engine import RuntimeStateError, TTSRuntime

logger = logging.getLogger(__name__)

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
    await ws.send_json(
        ErrorMessage(request_id=request_id, code=code, message=message).model_dump()
    )


async def _cancel_task(
    cancel_event: threading.Event | None,
    synthesis_task: asyncio.Task | None,
) -> None:
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


async def handle_websocket(ws: WebSocket, runtime: TTSRuntime) -> None:
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
                    _handle_synthesize(ws, runtime, raw, cancel_event)
                )

            elif msg_type == "upload_ref_audio":
                await _handle_upload_ref_audio(ws, runtime, raw)

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
    runtime: TTSRuntime,
    raw: dict,
    cancel_event: threading.Event,
) -> None:
    request_id = raw.get("request_id")
    chunk_index = 0
    try:
        req = SynthesizeRequest(**raw)
    except Exception as e:
        await _send_error(ws, "INVALID_REQUEST", str(e))
        return

    if not req.text or not req.text.strip():
        await _send_error(ws, "INVALID_TEXT", "Text cannot be empty.", req.request_id)
        return

    if len(req.text) > settings.max_text_length:
        await _send_error(
            ws,
            "INVALID_TEXT",
            f"Text exceeds maximum length of {settings.max_text_length} characters.",
            req.request_id,
        )
        return

    if req.language not in SUPPORTED_LANGUAGES:
        await _send_error(
            ws,
            "INVALID_LANGUAGE",
            f"Unsupported language: {req.language}. Supported: {sorted(SUPPORTED_LANGUAGES)}",
            req.request_id,
        )
        return

    if req.mode == "voice_design":
        if req.voice_clone_prompt_id or req.voice_id:
            await _send_error(
                ws,
                "INVALID_REQUEST",
                "voice_clone_prompt_id and voice_id are not allowed in voice design mode.",
                req.request_id,
            )
            return
        if not req.instruct or not req.instruct.strip():
            await _send_error(
                ws,
                "INVALID_INSTRUCT",
                "Voice Design requires a non-empty instruction.",
                req.request_id,
            )
            return
    elif req.voice_clone_prompt_id and req.voice_id:
        await _send_error(
            ws,
            "INVALID_REQUEST",
            "voice_clone_prompt_id and voice_id cannot be used together.",
            req.request_id,
        )
        return

    ref_audio_path = None
    voice_clone_prompt = None

    try:
        async with runtime.acquire_request_engine(
            mode=req.mode,
            clone_model_size=req.model,
            prompt_id=req.voice_clone_prompt_id,
        ) as lease:
            engine = lease.engine
            if req.mode == "voice_clone":
                if req.voice_id:
                    _, voice_clone_prompt = await runtime.ensure_voice_prompt_for_replica(
                        replica=lease.replica,
                        voice_id=req.voice_id,
                    )
                elif req.voice_clone_prompt_id:
                    local_prompt_id = lease.local_prompt_id
                    voice_clone_prompt = engine.get_voice_prompt(local_prompt_id)
                    if voice_clone_prompt is None:
                        path = engine.get_ref_audio_path(local_prompt_id)
                        if path is None:
                            await _send_error(
                                ws,
                                "PROMPT_NOT_FOUND",
                                f"Voice clone prompt not found: {req.voice_clone_prompt_id}",
                                req.request_id,
                            )
                            return
                        ref_audio_path = str(path)

            await ws.send_json(
                SynthesisStart(
                    request_id=req.request_id,
                    mode=req.mode,
                    model=engine.display_name,
                    sample_rate=engine.sample_rate,
                ).model_dump()
            )

            t_start = time.perf_counter()
            t_first_chunk = None
            total_bytes = 0
            cancelled = False

            try:
                async for pcm_chunk in engine.stream_tts(
                    mode=req.mode,
                    text=req.text,
                    language=req.language,
                    chunk_size=req.chunk_size,
                    ref_audio_path=ref_audio_path,
                    voice_clone_prompt=voice_clone_prompt,
                    instruct=req.instruct.strip() if req.instruct else None,
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
                await _send_error(ws, "SYNTHESIS_ERROR", str(e), req.request_id)
                return

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
                return

            t_end = time.perf_counter()
            total_samples = total_bytes // 2
            duration_ms = (total_samples / engine.sample_rate) * 1000 if engine.sample_rate else 0
            wall_time_ms = (t_end - t_start) * 1000
            ttfa_ms = (t_first_chunk - t_start) * 1000 if t_first_chunk else 0
            rtf = (wall_time_ms / duration_ms) if duration_ms > 0 else 0

            await ws.send_json(
                SynthesisEnd(
                    request_id=req.request_id,
                    mode=req.mode,
                    model=engine.display_name,
                    total_chunks=chunk_index,
                    total_samples=total_samples,
                    duration_ms=duration_ms,
                    ttfa_ms=round(ttfa_ms, 1),
                    rtf=round(rtf, 3),
                ).model_dump()
            )

            logger.info(
                "[%s] Synthesis complete: %d chunks, %.0fms audio, TTFA=%.0fms, RTF=%.3f",
                engine.display_name,
                chunk_index,
                duration_ms,
                ttfa_ms,
                rtf,
            )
    except asyncio.CancelledError:
        logger.info("Synthesis task cancelled: request_id=%s, chunks_sent=%d", request_id, chunk_index)
        if request_id:
            try:
                await ws.send_json(
                    SynthesisCancelled(
                        request_id=request_id,
                        chunks_sent=chunk_index,
                    ).model_dump()
                )
            except Exception:
                pass
    except RuntimeStateError as e:
        await _send_error(ws, e.code, e.message, req.request_id)


async def _handle_upload_ref_audio(
    ws: WebSocket,
    runtime: TTSRuntime,
    raw: dict,
) -> None:
    try:
        req = UploadRefAudioRequest(**raw)
    except Exception as e:
        await _send_error(ws, "INVALID_REQUEST", str(e))
        return

    fmt = req.audio_format.lower().strip()
    if fmt not in ALLOWED_AUDIO_FORMATS:
        await _send_error(
            ws,
            "INVALID_AUDIO",
            f"Unsupported audio format: {req.audio_format}. Supported: {sorted(ALLOWED_AUDIO_FORMATS)}",
            req.request_id,
        )
        return

    try:
        audio_bytes = decode_audio_base64(req.audio_base64)
    except ValueError as e:
        await _send_error(
            ws,
            "INVALID_AUDIO",
            str(e),
            req.request_id,
        )
        return

    t0 = time.perf_counter()

    try:
        async with runtime.acquire_request_engine(
            mode="voice_clone",
            clone_model_size=req.model,
            purpose="precompute",
        ) as lease:
            engine = lease.engine
            local_prompt_id, _ = engine.save_ref_audio(audio_bytes, fmt)
            precompute_ms = await engine.run_on_gpu(engine.precompute_voice_prompt, local_prompt_id)
            elapsed = (time.perf_counter() - t0) * 1000
            prompt_id = runtime.encode_prompt_id(lease.replica.replica_id, local_prompt_id)

            await ws.send_json(
                VoiceClonePromptReady(
                    request_id=req.request_id,
                    model=engine.display_name,
                    prompt_id=prompt_id,
                    processing_ms=round(elapsed, 1),
                    runtime_generation=runtime.runtime_generation,
                ).model_dump()
            )

            logger.info(
                "Voice clone prompt ready: %s (model=%s, %.0fms, precompute=%.0fms)",
                prompt_id,
                engine.display_name,
                elapsed,
                precompute_ms,
            )
    except RuntimeStateError as e:
        await _send_error(ws, e.code, e.message, req.request_id)
    except ValueError as e:
        await _send_error(ws, "INVALID_AUDIO", str(e), req.request_id)
    except Exception as e:
        logger.error("Failed to precompute voice prompt: %s", e, exc_info=True)
        await _send_error(
            ws,
            "VOICE_CLONE_ERROR",
            f"Failed to process reference audio: {e}",
            req.request_id,
        )
