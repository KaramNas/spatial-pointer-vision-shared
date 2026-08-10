"""
FastAPI + WebSocket inference server for spatial-pointer-vision.

Loads a Qwen2-VL model (optionally with a fine-tuned LoRA adapter, see
config.py / MODEL_CHECKPOINT_PATH) once at startup, then serves
object-identification requests over a WebSocket. This is the "tethered
mode" server the Quest app's InferenceClient.cs connects to over the local
network.

Run directly:
    python server/main.py

Or via Docker:
    docker compose up --build

Endpoints:
    GET  /health        liveness + whether a model is currently loaded
    WS   /ws/identify    send raw image bytes (one binary frame per request),
                         receive back {"label": str, "confidence": float}
                         or {"error": str, "message": str} on failure. The
                         connection stays open across multiple requests.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import settings
from model_service import InvalidImageError, ModelNotLoadedError, ModelService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

model_service = ModelService(
    checkpoint_path=settings.model_checkpoint_path,
    base_model_name=settings.base_model_name,
    device=settings.device,
)

# Serializes access to model_service.identify(): a single GPU model instance
# generating for two requests at once would either error or silently corrupt
# results, so concurrent WebSocket clients queue behind this lock rather than
# racing each other.
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except Exception:
        logger.exception(
            "Model failed to load at startup. The server will stay up so "
            "/health can report model_loaded=False, but /ws/identify will "
            "reject requests until this is fixed and the server restarted."
        )
    yield


app = FastAPI(title="spatial-pointer-vision inference server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded,
        "base_model": settings.base_model_name,
        "checkpoint": settings.model_checkpoint_path,
        "device": model_service.device,
    }


@app.websocket("/ws/identify")
async def ws_identify(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected: %s", websocket.client)

    try:
        while True:
            try:
                image_bytes = await websocket.receive_bytes()
            except WebSocketDisconnect:
                raise
            except (RuntimeError, KeyError):
                # Client sent a text frame or something else non-binary.
                await websocket.send_json(
                    {"error": "invalid_image", "message": "Expected a binary image frame."}
                )
                continue

            try:
                async with inference_lock:
                    result = await asyncio.to_thread(
                        model_service.identify, image_bytes, settings.max_new_tokens
                    )
                await websocket.send_json(result)
            except ModelNotLoadedError:
                await websocket.send_json(
                    {"error": "model_not_loaded", "message": "Model is not loaded on the server yet."}
                )
            except InvalidImageError as e:
                await websocket.send_json({"error": "invalid_image", "message": str(e)})
            except Exception:
                logger.exception("Unexpected error during inference")
                await websocket.send_json(
                    {"error": "internal_error", "message": "Unexpected server error during inference."}
                )
    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", websocket.client)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
