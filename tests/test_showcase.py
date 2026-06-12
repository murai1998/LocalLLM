"""Showcase distribution checks — GPU-free (models.py is deliberately untouched)."""

import sys
from pathlib import Path

import numpy as np
import pytest

SHOWCASE = Path(__file__).resolve().parents[1] / "showcase"
sys.path.insert(0, str(SHOWCASE))

import piper_voices as sc_voices  # noqa: E402
import prompts as sc_prompts  # noqa: E402

from localllm.pipelines import translate as ll_translate  # noqa: E402
from localllm.tts import piper as ll_piper  # noqa: E402


def test_vendored_tones_and_languages_in_sync():
    assert sc_prompts.TONE_PRESETS == dict(ll_translate.TONE_PRESETS)
    assert sc_prompts.LANGUAGE_LABELS == ll_translate.LANGUAGE_LABELS
    assert sc_prompts.DEFAULT_TONE == ll_translate.DEFAULT_TONE


def test_vendored_voices_in_sync():
    assert sc_voices.VOICE_OPTIONS == dict(ll_piper.VOICE_OPTIONS)
    assert not sc_voices.tts_supported("ja")
    with pytest.raises(ValueError):
        sc_voices.resolve_voice_name(language="ja")


def test_translate_prompt_parity_with_local_app():
    local = ll_translate.build_translate_messages(
        "hola mundo", source_lang="es", target_lang="en", tone="exact"
    )
    showcase = sc_prompts.build_translate_messages(
        "hola mundo", source_lang="es", target_lang="en", tone="exact"
    )
    assert local == showcase


def test_translate_prompt_context_block():
    msgs = sc_prompts.build_translate_messages(
        "more text",
        source_lang=None,
        target_lang="de",
        context=[("hola", "hallo")],
    )
    assert "hola → hallo" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Source text:\nmore text")


def test_zerogpu_decorator_falls_back_locally():
    pytest.importorskip("gradio", reason="showcase extra not installed")
    import zerogpu

    @zerogpu.GPU(duration=30)
    def f(x):
        return x + 1

    @zerogpu.GPU
    def g(x):
        return x * 2

    assert f(1) == 2
    assert g(2) == 4


def test_friendly_errors_maps_quota_to_gr_error():
    gr = pytest.importorskip("gradio", reason="showcase extra not installed")
    import zerogpu

    @zerogpu.friendly_errors
    def boom():
        raise RuntimeError("ZeroGPU quota exceeded for today")

    with pytest.raises(gr.Error) as exc_info:
        boom()
    assert "github.com/murai1998/LocalLLM" in str(exc_info.value)

    @zerogpu.friendly_errors
    def other():
        raise ValueError("unrelated")

    with pytest.raises(ValueError):
        other()


def test_friendly_errors_preserves_generator_functions():
    """Gradio streams generator outputs — wrapping must not collapse them
    into a single returned generator object (caused 'Error' pills in chat)."""
    import inspect

    import zerogpu

    @zerogpu.friendly_errors
    def streamer(n):
        for i in range(n):
            yield i

    assert inspect.isgeneratorfunction(streamer)
    assert list(streamer(3)) == [0, 1, 2]


def test_reply_handler_accumulates_private_history():
    pytest.importorskip("fastrtc", reason="showcase extra not installed")
    from interpreter import build_reply_handler

    seen = {}
    replies = iter([("hola", "hello"), ("adiós", "goodbye"), ("sí", "yes")])

    def fake_translate(samples, sr, source, target, tone, context):
        seen["context"] = context
        return next(replies)

    def fake_tts(text, *, language, voice_id):
        return 22050, np.zeros(2205, dtype=np.int16)

    handler = build_reply_handler(fake_translate, fake_tts)
    audio = (16000, np.ones(16000, dtype=np.int16) * 1000)
    events = list(handler(audio, "", "en", "professional", ""))

    # First event: AdditionalOutputs(sources_pane, translations_pane); then audio.
    assert len(events) == 2
    src_pane, tgt_pane = events[0].args
    assert src_pane == "hola" and tgt_pane == "hello"
    rate, voice = events[1]
    assert rate == 22050
    assert voice.shape[0] == 1  # mono channel-first for fastrtc

    # Second utterance: history is remembered inside the handler closure
    # (gr.State round-trips don't work through fastrtc's additional-outputs).
    events2 = list(handler(audio, "", "en", "professional", ""))
    src_pane, tgt_pane = events2[0].args
    assert src_pane == "hola\n\nadiós" and tgt_pane == "hello\n\ngoodbye"
    assert seen["context"] == [("hola", "hello")]  # rolling translate context

    # fastrtc 0.0.34 also passes the WebRTC component's own value as an extra
    # leading argument — the handler must tolerate both calling conventions.
    events3 = list(handler(audio, None, "", "en", "professional", ""))
    assert "sí" in events3[0].args[0]

    # A fresh handler (new connection) starts with an empty transcript.
    fresh = build_reply_handler(lambda *a: ("uno", "one"), fake_tts)
    assert list(fresh(audio, "", "en", "professional", ""))[0].args == ("uno", "one")


def test_to_mono_float32_scales_2d_int16():
    """Channel-averaging promotes int16 → float64; scaling must happen first.
    (Regression: ±32767 amplitudes shipped to the WAV encoder clipped into a
    square-wave buzz — the live interpreter heard 'silent' audio.)"""
    from interpreter import to_mono_float32

    pcm = (np.ones((1, 1600), dtype=np.int16) * 16384)  # fastrtc shape (1, n)
    mono = to_mono_float32(pcm)
    assert mono.dtype == np.float32
    assert mono.ndim == 1
    assert abs(float(mono.max()) - 0.5) < 0.01

    stereo = np.stack([np.ones(1600, np.int16) * 8192, np.ones(1600, np.int16) * 16384])
    mono2 = to_mono_float32(stereo)
    assert abs(float(mono2.max()) - 0.375) < 0.01  # mean of 0.25 and 0.5

    already_float = np.ones(100, dtype=np.float32) * 0.7
    assert float(to_mono_float32(already_float).max()) == pytest.approx(0.7)


def test_split_sentences_and_degenerate_detection():
    from interpreter import is_degenerate, split_sentences

    assert split_sentences("Hola. ¿Cómo estás? ¡Bien!") == ["Hola.", "¿Cómo estás?", "¡Bien!"]
    assert split_sentences("just one") == ["just one"]
    # Whisper/LLM repetition loops (the mangled trailing chunk) are flagged…
    assert is_degenerate("the the the the the the the the")
    assert is_degenerate("sí no " * 15)
    # …while ordinary speech is not.
    assert not is_degenerate("hello how are you today")
    assert not is_degenerate("I went to the store to buy bread and milk")


def test_reply_handler_streams_one_audio_chunk_per_sentence():
    pytest.importorskip("fastrtc", reason="showcase extra not installed")
    from interpreter import build_reply_handler

    def fake_translate(*_a):
        return "hola mundo", "Hello world. How are you?"

    def fake_tts(text, *, language, voice_id):
        return 22050, np.zeros(2205, dtype=np.int16)

    handler = build_reply_handler(fake_translate, fake_tts)
    audio = (16000, np.ones(16000, dtype=np.int16) * 1000)
    events = list(handler(audio, "", "en", "professional", ""))
    # One transcript event, then one audio chunk per translated sentence.
    assert len(events) == 3
    assert all(e[1].shape[0] == 1 for e in events[1:])


def test_reply_handler_drops_repetition_loops():
    pytest.importorskip("fastrtc", reason="showcase extra not installed")
    from interpreter import build_reply_handler

    handler = build_reply_handler(
        lambda *a: ("word word word word word word word", "x"),
        lambda *a, **k: (22050, np.zeros(10, np.int16)),
    )
    audio = (16000, np.ones(16000, dtype=np.int16) * 1000)
    assert list(handler(audio, "", "en", "professional", "")) == []


def test_reply_handler_skips_blips_and_unsupported_tts():
    pytest.importorskip("fastrtc", reason="showcase extra not installed")
    from interpreter import build_reply_handler

    handler = build_reply_handler(
        lambda *a: ("text", "translated"), lambda *a, **k: (22050, np.zeros(10, np.int16))
    )
    blip = (16000, np.ones(1600, dtype=np.int16))  # 0.1 s < threshold
    assert list(handler(blip, "", "en", "professional", "")) == []

    # Japanese has no Piper voice: transcript event yes, audio event no.
    utterance = (16000, np.ones(16000, dtype=np.int16))
    events = list(handler(utterance, "", "ja", "professional", ""))
    assert len(events) == 1


def test_llama_backend_wav_roundtrip():
    pytest.importorskip("httpx")
    import io
    import wave

    import llama_models as lm

    sr = 16000
    audio = (np.sin(np.linspace(0, 440 * 2 * np.pi, sr)) * 0.5).astype(np.float32)
    data = lm.wav_bytes(audio, sr)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == sr
        assert w.getnframes() == sr
    part = lm.audio_part(audio, sr)
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"


def test_llama_backend_image_part_is_png_data_url():
    pytest.importorskip("httpx")
    Image = pytest.importorskip("PIL.Image")
    import llama_models as lm

    part = lm.image_part(Image.new("RGB", (8, 8), "white"))
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_llama_backend_import_is_side_effect_free():
    """Importing the backend must not spawn llama-server or hit the network."""
    pytest.importorskip("httpx")
    import llama_models as lm

    assert lm._proc is None
    assert lm.MULTIMODAL is True


def test_vendored_endpointer_matches_local_logic():
    import re

    def class_body(path):
        text = path.read_text(encoding="utf-8")
        return re.search(r"class StreamingEndpointer.*", text, re.DOTALL).group(0)

    local = class_body(Path("localllm/live/endpointer.py"))
    vendored = class_body(SHOWCASE / "endpointer.py")
    assert local == vendored


def test_showcase_modules_do_not_import_localllm():
    import re

    for py in SHOWCASE.glob("*.py"):
        source = py.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+localllm", source, re.M), py.name


def test_demo_disclaimer_is_prominent():
    """The reduced-capability notice + GitHub link must appear in README and app UI."""
    readme = (SHOWCASE / "README.md").read_text(encoding="utf-8")
    app = (SHOWCASE / "app.py").read_text(encoding="utf-8")
    for text, name in ((readme, "README.md"), (app, "app.py")):
        assert "reduced-capability demo" in text.lower(), name
        assert "github.com/murai1998/LocalLLM" in text, name
