"""Model loading and all GPU-decorated inference for the ZeroGPU Space.

This is the ONLY module that imports torch/transformers. Models are placed on
`cuda` at module level — the required ZeroGPU pattern (CUDA is emulated outside
`@spaces.GPU` functions and real inside them).

Presets (env `SHOWCASE_PRESET`):
  full   gemma-4-12b-it bf16 (~24 GB, multimodal) — for the ZeroGPU Space (48 GB slice)
  small  gemma-4-12b-it in bitsandbytes 4-bit (~8 GB, multimodal) — local rehearsal

The `small` preset is the same 12B multimodal checkpoint as `full`, only loaded
with 4-bit NF4 quantization so it fits a consumer GPU; vision tabs work in both.
Quantization can be forced/disabled independently of the preset with
`SHOWCASE_QUANTIZE=1`/`0`. It requires a CUDA GPU (bitsandbytes is CUDA-only).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

import numpy as np
import torch
from prompts import CHAT_SYSTEM, OCR_SYSTEM, build_translate_messages, language_label
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)
from zerogpu import GPU

PRESETS = {
    "full": "google/gemma-4-12b-it",
    "small": "google/gemma-4-12b-it",
}
PRESET = os.environ.get("SHOWCASE_PRESET", "full")
GEMMA_ID = os.environ.get("SHOWCASE_GEMMA_ID") or PRESETS.get(PRESET, PRESETS["full"])

# The `small` preset trades VRAM for the same checkpoint via 4-bit quantization.
# Honour an explicit override; otherwise quantize only for the small preset.
QUANTIZE_4BIT = os.environ.get("SHOWCASE_QUANTIZE", "1" if PRESET == "small" else "0") == "1"
WHISPER_ID = os.environ.get("SHOWCASE_WHISPER_ID", "openai/whisper-large-v3-turbo")

# On ZeroGPU, CUDA is emulated at module level and must be used; locally we
# fall back to CPU when no CUDA build of torch is installed (slow but works).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Some Gemma variants ship only a tokenizer — no AutoProcessor-recognizable
# processing class — so fall back to text-only mode. The 12B checkpoint used by
# both presets is multimodal, so this normally keeps the vision tabs enabled.
try:
    processor = AutoProcessor.from_pretrained(GEMMA_ID)
except (ValueError, OSError):
    processor = AutoTokenizer.from_pretrained(GEMMA_ID)
MULTIMODAL = hasattr(processor, "image_processor")
_tokenizer = getattr(processor, "tokenizer", processor)

if QUANTIZE_4BIT:
    if DEVICE != "cuda":
        raise RuntimeError(
            "4-bit (bitsandbytes) quantization requires a CUDA GPU. Install a CUDA "
            "build of torch, or set SHOWCASE_QUANTIZE=0 to load in bf16 instead."
        )
    # NF4 + double quant + bf16 compute: the standard high-quality 4-bit recipe.
    #
    # Two non-obvious requirements (transformers 5.11 + bitsandbytes 0.49):
    #   * Do NOT also pass `dtype=` to from_pretrained — combining an explicit
    #     dtype with a bnb config silently skips the 4-bit packing, so weights
    #     load unquantized and the first forward asserts inside bitsandbytes.
    #     `bnb_4bit_compute_dtype` already fixes the compute precision.
    #   * Keep the vision/audio towers and the (tied) lm_head in full precision.
    #     The Gemma vision encoder casts pixel_values to its patch-projection
    #     weight dtype; if that Linear is 4-bit its weight dtype is uint8 and the
    #     following LayerNorm dies with "not implemented for 'Byte'". And an
    #     explicit skip list *replaces* the auto-detected one, so lm_head must be
    #     re-listed or its tied embedding weight never gets packed either.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head", "model.embed_vision", "model.embed_audio"],
    )
    _model_kwargs: dict = {"quantization_config": quantization_config, "device_map": "auto"}
else:
    _model_kwargs = {"dtype": torch.bfloat16}

try:
    gemma = AutoModelForImageTextToText.from_pretrained(GEMMA_ID, **_model_kwargs)
except ValueError:
    gemma = AutoModelForCausalLM.from_pretrained(GEMMA_ID, **_model_kwargs)
# A quantized model is already placed on its device(s) via `device_map`, and
# moving a 4-bit model with `.to()` is unsupported — only place the bf16 model.
if not QUANTIZE_4BIT:
    gemma = gemma.to(DEVICE)

asr = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_ID,
    dtype=torch.bfloat16,
    device=DEVICE,
)

_generate_lock = threading.Lock()

# Greedy decoding can fall into repetition loops, and Whisper hallucinates a
# single token many times over near-silent trailing audio — the live "last
# chunk" (when you pause or cut the mic) becomes one word repeated dozens of
# times. A mild repetition penalty plus an n-gram block break the loop without
# distorting well-formed speech. Applied to the translation/ASR paths; chat and
# OCR stay pure-greedy so legitimately repeated text (tables, code) is preserved.
_ANTI_REPEAT = {"repetition_penalty": 1.3, "no_repeat_ngram_size": 3}


def _flatten_messages(messages: list[dict]) -> list[dict]:
    """Collapse content-part lists to plain strings for tokenizer-only templates."""
    flattened = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(
                part["text"] for part in content if part.get("type") == "text"
            )
        flattened.append({"role": message["role"], "content": content})
    return flattened


def _gemma_generate(
    messages: list[dict], *, max_new_tokens: int = 512, **gen_kwargs
) -> str:
    if not MULTIMODAL:
        messages = _flatten_messages(messages)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(gemma.device)
    with _generate_lock, torch.inference_mode():
        output = gemma.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, **gen_kwargs
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def _utterance_duration(audio: np.ndarray, sample_rate: int, *_args) -> int:
    """Dynamic @spaces.GPU duration: short hints improve visitor queue priority."""
    seconds = len(audio) / max(sample_rate, 1)
    return int(min(40, 10 + seconds * 1.5))


@GPU(duration=_utterance_duration)
def transcribe_and_translate(
    audio: np.ndarray,
    sample_rate: int,
    source_lang: str | None,
    target_lang: str,
    tone: str,
    context: list[tuple[str, str]],
) -> tuple[str, str]:
    """One live-interpreter utterance: Whisper STT + Gemma MT in a single GPU call."""
    asr_kwargs: dict = {"task": "transcribe", **_ANTI_REPEAT}
    if source_lang:
        asr_kwargs["language"] = language_label(source_lang).lower()
    transcript = asr(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=asr_kwargs,
    )["text"].strip()
    if not transcript:
        return "", ""

    translation = _gemma_generate(
        build_translate_messages(
            transcript,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            context=context,
        ),
        max_new_tokens=256,
        **_ANTI_REPEAT,
    )
    return transcript, translation


@GPU(duration=120)
def transcribe_file(audio: np.ndarray, sample_rate: int, source_lang: str | None) -> str:
    asr_kwargs: dict = {"task": "transcribe", **_ANTI_REPEAT}
    if source_lang:
        asr_kwargs["language"] = language_label(source_lang).lower()
    return asr(
        {"array": audio, "sampling_rate": sample_rate},
        chunk_length_s=30,
        generate_kwargs=asr_kwargs,
    )["text"].strip()


@GPU(duration=60)
def translate_text(
    text: str, source_lang: str | None, target_lang: str, tone: str
) -> str:
    return _gemma_generate(
        build_translate_messages(text, source_lang=source_lang, target_lang=target_lang, tone=tone),
        max_new_tokens=1024,
        **_ANTI_REPEAT,
    )


@GPU(duration=90)
def chat_stream(history: list[dict], image=None) -> Iterator[str]:
    """Streaming chat; `history` is [{'role','content'}] text turns, optional PIL image."""
    if image is not None and not MULTIMODAL:
        raise NotImplementedError(
            f"The current demo preset ({GEMMA_ID}) is text-only — image understanding "
            "needs the multimodal 12B preset (or the full local app)."
        )
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": CHAT_SYSTEM}]}
    ]
    for i, turn in enumerate(history):
        content: list[dict] = []
        if image is not None and i == len(history) - 1 and turn["role"] == "user":
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": turn["content"]})
        messages.append({"role": turn["role"], "content": content})
    if not MULTIMODAL:
        messages = _flatten_messages(messages)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(gemma.device)
    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(**inputs, max_new_tokens=1024, do_sample=False, streamer=streamer)
    with _generate_lock:
        thread = threading.Thread(target=lambda: gemma.generate(**kwargs))
        thread.start()
        reply = ""
        for token in streamer:
            reply += token
            yield reply
        thread.join()


@GPU(duration=120)
def ocr_images(images: list, instructions: str = "") -> str:
    """Gemma vision OCR over one or more page images."""
    if not MULTIMODAL:
        raise NotImplementedError(
            f"The current demo preset ({GEMMA_ID}) is text-only — OCR needs the "
            "multimodal 12B preset (or the full local app)."
        )
    pages: list[str] = []
    for image in images:
        user_text = "Extract all text from this image."
        if instructions.strip():
            user_text += f" Additional instructions: {instructions.strip()}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": OCR_SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        pages.append(_gemma_generate(messages, max_new_tokens=2048))
    return "\n\n---\n\n".join(pages)
