from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.models import ModeSwitchRequest
from server.tts_engine import RuntimeStateError, TTSRuntime
from server.ws_handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


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
    yield

    logger.info("Shutting down Ameego TTS server...")


app = FastAPI(title="Ameego TTS", lifespan=lifespan)


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

    await ws.accept()

    if app.state.active_connections >= settings.max_connections:
        await ws.close(code=1013, reason="Server at max capacity")
        return

    app.state.active_connections += 1
    try:
        await handle_websocket(ws, runtime)
    finally:
        app.state.active_connections -= 1


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
