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

Streams a response for a given prompt using Server-Sent Events (SSE).

**Request body**

```json
{
  "prompt": "What are the symptoms of glaucoma?"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | The user question or message |

**Response**

The endpoint returns a text/event-stream. Each SSE `data` field contains one decoded token. The final event is `[DONE]`.

```
data: Glaucoma

data:  is

data:  characterized

data:  by

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
  -d '{"prompt": "What are the symptoms of glaucoma?"}'
```

**Example — Python (httpx)**

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/generate/",
                  json={"prompt": "What are the symptoms of glaucoma?"}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            token = line[6:]
            if token == "[DONE]":
                break
            print(token, end="", flush=True)
```

## Generation parameters

| Parameter | Value |
|---|---|
| `max_new_tokens` | 512 |
| `temperature` | 0.7 |
| `top_p` | 0.95 |
| `do_sample` | true |

The system prompt injected into every request is:
> *"You are a medical assistant specialized in eye diseases."*

## Interactive docs

FastAPI auto-generates interactive documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
