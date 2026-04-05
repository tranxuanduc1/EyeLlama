import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import model as m


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        m.load_model()
    except Exception as exc:
        print(f"[startup] Model loading failed: {exc}")
    yield


app = FastAPI(title="EyeLlama API", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]


@app.get("/health/")
async def health():
    return {"status": "ok", "models_loaded": m.is_loaded()}


@app.post("/generate/")
async def generate(request: GenerateRequest):
    if not m.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [msg.model_dump() for msg in request.messages]

    async def event_generator() -> AsyncGenerator[str, None]:
        async for token in m.generate_stream(messages):
            yield json.dumps({"delta": token})
        yield "[DONE]"

    return EventSourceResponse(event_generator())
