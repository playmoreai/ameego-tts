from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.tts_engine import TTSEngine
from server.ws_handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ameego TTS server...")
    logger.info("Model: %s (%s)", settings.model_id, settings.model_size)

    engine = TTSEngine.from_config(settings)
    engine.warm_up()
    app.state.tts_engine = engine
    app.state.active_connections = 0

    logger.info("Server ready. Max connections: %d", settings.max_connections)
    yield

    logger.info("Shutting down Ameego TTS server...")


app = FastAPI(title="Ameego TTS", lifespan=lifespan)


@app.get("/health")
async def health():
    engine: TTSEngine = app.state.tts_engine
    return JSONResponse({
        "status": "ok",
        "model_id": settings.model_id,
        "model_size": settings.model_size,
        "sample_rate": engine.sample_rate,
        "max_connections": settings.max_connections,
        "active_connections": app.state.active_connections,
    })


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    engine: TTSEngine = app.state.tts_engine

    await ws.accept()

    # Reject if at capacity (asyncio is single-threaded, no race)
    if app.state.active_connections >= settings.max_connections:
        await ws.close(code=1013, reason="Server at max capacity")
        return

    app.state.active_connections += 1
    try:
        await handle_websocket(ws, engine)
    finally:
        app.state.active_connections -= 1


# Mount static web app last (catch-all)
app.mount("/", StaticFiles(directory="web", html=True), name="web")
