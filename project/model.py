from pathlib import Path
from threading import Thread
import asyncio
from typing import AsyncGenerator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

BASE_MODEL_DIR = Path(__file__).resolve().parent.parent / "TinyLlama"
LORA_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "LoRA" / "eye_llm_lora"

SYSTEM_PROMPT = "You are a medical assistant specialized in eye diseases."

_model: PeftModel | None = None
_tokenizer: AutoTokenizer | None = None


def load_model() -> None:
    global _model, _tokenizer

    # Load tokenizer from base model dir — it has the chat_template
    _tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Re-enable cache (training disables it)
    base.config.use_cache = True

    _model = PeftModel.from_pretrained(base, str(LORA_ADAPTER_DIR))
    _model.eval()


def is_loaded() -> bool:
    return _model is not None and _tokenizer is not None


async def generate_stream(messages: list[dict]) -> AsyncGenerator[str, None]:
    if not is_loaded():
        raise RuntimeError("Model not loaded")

    if not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(formatted, return_tensors="pt").to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=_model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    iter_streamer = iter(streamer)
    sentinel = object()

    def _next_token():
        try:
            return next(iter_streamer)
        except StopIteration:
            return sentinel

    while True:
        token = await loop.run_in_executor(None, _next_token)
        if token is sentinel:
            break
        if token:
            yield token

    thread.join()
