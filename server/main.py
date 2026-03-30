from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.audio_utils import decode_audio_base64
from server.models import (
    ALLOWED_AUDIO_FORMATS,
    CreateVoiceRequest,
    ModeSwitchRequest,
    VoiceCreatedResponse,
    VoiceMetadataResponse,
)
from server.tts_engine import RuntimeStateError, TTSRuntime
from server.ws_handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _is_allowed_origin(origin: str | None) -> bool:
    if not settings.allowed_origin_list:
        return True
    if not origin:
        return True
    return origin in settings.allowed_origin_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ameego TTS server...")
    logger.info(
        "Base models: %s, initial clone model: %s, initial mode: %s, voice design enabled: %s",
        settings.model_sizes,
        settings.initial_clone_model_size,
        settings.initial_mode,
        settings.voice_design_enabled,
    )

    runtime = TTSRuntime(settings)
    await runtime.initialize()
    app.state.runtime = runtime
    app.state.active_connections = 0

    logger.info(
        "Server ready. Mode: %s, Active model: %s, Max connections: %d",
        runtime.active_mode,
        runtime.active_model_id,
        settings.max_connections,
    )
    try:
        yield
    finally:
        logger.info("Shutting down Ameego TTS server...")
        await runtime.shutdown()


app = FastAPI(title="Ameego TTS", lifespan=lifespan)

if settings.allowed_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    runtime: TTSRuntime = app.state.runtime
    payload = runtime.health_payload()
    payload.update(
        {
            "max_connections": settings.max_connections,
            "active_connections": app.state.active_connections,
        }
    )
    return JSONResponse(payload)


@app.post("/voices")
async def create_voice(req: CreateVoiceRequest):
    runtime: TTSRuntime = app.state.runtime

    fmt = req.audio_format.lower().strip()
    if fmt not in ALLOWED_AUDIO_FORMATS:
        return JSONResponse({"code": "INVALID_AUDIO", "message": f"Unsupported audio format: {req.audio_format}"}, status_code=400)

    try:
        audio_bytes = decode_audio_base64(req.audio_base64)
    except ValueError as e:
        return JSONResponse({"code": "INVALID_AUDIO", "message": str(e)}, status_code=400)

    try:
        stored_voice = runtime.voice_store.create_voice(
            audio_bytes=audio_bytes,
            audio_format=fmt,
            display_name=req.display_name,
        )
    except ValueError as e:
        return JSONResponse({"code": "INVALID_AUDIO", "message": str(e)}, status_code=400)
    except Exception as e:
        logger.error("Failed to create voice: %s", e, exc_info=True)
        return JSONResponse({"code": "VOICE_CREATE_ERROR", "message": str(e)}, status_code=500)

    return JSONResponse(
        VoiceCreatedResponse(
            voice_id=stored_voice.voice_id,
            audio_format=stored_voice.metadata.get("audio_format", "wav"),
            duration_ms=float(stored_voice.metadata.get("duration_ms", 0)),
            created_at=str(stored_voice.metadata.get("created_at", "")),
            display_name=stored_voice.metadata.get("display_name"),
        ).model_dump()
    )


@app.get("/voices/{voice_id}")
async def get_voice(voice_id: str):
    runtime: TTSRuntime = app.state.runtime
    stored_voice = runtime.voice_store.get_voice(voice_id)
    if stored_voice is None:
        return JSONResponse({"code": "VOICE_NOT_FOUND", "message": f"Voice not found: {voice_id}"}, status_code=404)

    return JSONResponse(
        VoiceMetadataResponse(
            voice_id=stored_voice.voice_id,
            audio_format=stored_voice.metadata.get("audio_format", "wav"),
            duration_ms=stored_voice.metadata.get("duration_ms"),
            created_at=stored_voice.metadata.get("created_at"),
            display_name=stored_voice.metadata.get("display_name"),
        ).model_dump()
    )


if settings.app_profile == "test":
    @app.post("/mode/switch")
    async def switch_mode(req: ModeSwitchRequest):
        runtime: TTSRuntime = app.state.runtime
        try:
            result = await runtime.start_mode_switch(
                target_mode=req.mode,
                clone_model_size=req.model,
            )
            return JSONResponse(result)
        except RuntimeStateError as e:
            return JSONResponse(
                {
                    "status": "error",
                    "code": e.code,
                    "message": e.message,
                }
            )


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    runtime: TTSRuntime = app.state.runtime

    if not _is_allowed_origin(ws.headers.get("origin")):
        await ws.close(code=1008, reason="Origin not allowed")
        return

    await ws.accept()

    if app.state.active_connections >= settings.max_connections:
        await ws.close(code=1013, reason="Server at max capacity")
        return

    app.state.active_connections += 1
    try:
        await handle_websocket(ws, runtime)
    finally:
        app.state.active_connections -= 1


if settings.app_profile == "test":
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
