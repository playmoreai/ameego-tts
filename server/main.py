from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.tts_engine import TTSEngineRegistry
from server.ws_handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ameego TTS server...")
    logger.info("Models: %s (default: %s)", settings.model_sizes, settings.default_model_size)

    registry = TTSEngineRegistry.from_config(settings)
    registry.warm_up_all()
    app.state.engine_registry = registry
    app.state.active_connections = 0

    logger.info(
        "Server ready. Models: %s, Max connections: %d",
        registry.available_models, settings.max_connections,
    )
    yield

    logger.info("Shutting down Ameego TTS server...")


app = FastAPI(title="Ameego TTS", lifespan=lifespan)


@app.get("/health")
async def health():
    registry: TTSEngineRegistry = app.state.engine_registry
    models = {}
    for size, engine in registry.items():
        models[size] = {
            "model_id": engine.model.model_name if hasattr(engine.model, 'model_name') else f"Qwen3-TTS-{size}",
            "sample_rate": engine.sample_rate,
        }
    return JSONResponse({
        "status": "ok",
        "available_models": registry.available_models,
        "default_model": registry.default_model,
        "models": models,
        "max_connections": settings.max_connections,
        "active_connections": app.state.active_connections,
    })


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    registry: TTSEngineRegistry = app.state.engine_registry

    await ws.accept()

    # Reject if at capacity (asyncio is single-threaded, no race)
    if app.state.active_connections >= settings.max_connections:
        await ws.close(code=1013, reason="Server at max capacity")
        return

    app.state.active_connections += 1
    try:
        await handle_websocket(ws, registry)
    finally:
        app.state.active_connections -= 1


# Mount static web app last (catch-all)
app.mount("/", StaticFiles(directory="web", html=True), name="web")
