# EyeLlama

A FastAPI service that serves a TinyLlama-1.1B model fine-tuned with LoRA on eye disease Q&A data. The model acts as a medical assistant specialized in eye diseases and streams responses via Server-Sent Events (SSE).

## Architecture

```
TinyLlama/          # Base model weights (TinyLlama-1.1B-Chat-v1.0)
LoRA/eye_llm_lora/  # LoRA adapter (r=8, alpha=16, target: q_proj & v_proj)
project/
  main.py           # FastAPI app & route definitions
  model.py          # Model loading & streaming inference
```

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended (model loads with `device_map="auto"`)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the base model weights in `TinyLlama/` and ensure `LoRA/eye_llm_lora/` contains the adapter files.

## Running the server

```bash
cd project
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Reference

### `GET /health/`

Returns the service status and whether the model has been loaded.

**Response**

```json
{
  "status": "ok",
  "models_loaded": true
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when the server is running |
| `models_loaded` | boolean | `true` if the model and tokenizer are ready |

---

### `POST /generate/`

Streams a response for a conversation using Server-Sent Events (SSE). Accepts the full message history, enabling multi-turn conversations.

**Request body**

```json
{
  "messages": [
    {"role": "user", "content": "What are the symptoms of glaucoma?"}
  ]
}
```

For multi-turn conversations, include the full history:

```json
{
  "messages": [
    {"role": "user", "content": "What is glaucoma?"},
    {"role": "assistant", "content": "Glaucoma is a group of eye conditions..."},
    {"role": "user", "content": "How is it treated?"}
  ]
}
```

**Message roles**

| Role | Description |
|---|---|
| `user` | A message from the user |
| `assistant` | A previous response from the model |
| `system` | Optional system prompt override. If omitted, the default medical assistant prompt is injected automatically. |

**Response**

The endpoint returns a `text/event-stream`. Each SSE `data` field contains a JSON object with a `delta` key holding one decoded token. The final event is `[DONE]`.

```
data: {"delta": "Glaucoma"}

data: {"delta": " is"}

data: {"delta": " characterized"}

...

data: [DONE]
```

**Error responses**

| Status | Detail | Cause |
|---|---|---|
| `503` | `"Model not loaded"` | Model failed to load at startup |

**Example — curl**

```bash
curl -N -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are the symptoms of glaucoma?"}]}'
```

**Example — Python (httpx)**

```python
import httpx
import json

with httpx.stream("POST", "http://localhost:8000/generate/",
                  json={"messages": [{"role": "user", "content": "What are the symptoms of glaucoma?"}]}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                break
            print(json.loads(payload)["delta"], end="", flush=True)
```

## Generation parameters

| Parameter | Value |
|---|---|
| `max_new_tokens` | 512 |
| `temperature` | 0.7 |
| `top_p` | 0.95 |
| `do_sample` | true |

The default system prompt injected when no `system` message is provided:
> *"You are a medical assistant specialized in eye diseases."*

## Interactive docs

FastAPI auto-generates interactive documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
