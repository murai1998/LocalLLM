"""ZeroGPU helpers: the @spaces.GPU decorator with a local fallback, and a
quota-friendly error wrapper so visitors see guidance instead of stack traces.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

from prompts import GITHUB_URL

try:
    import spaces

    GPU = spaces.GPU
    ON_SPACES = bool(os.environ.get("SPACE_ID"))
except ImportError:  # local rehearsal / tests without the `spaces` package

    def GPU(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

    ON_SPACES = False


QUOTA_MESSAGE = (
    "⏳ Your free daily GPU minutes are used up. Sign in to Hugging Face for a larger "
    "quota, come back tomorrow — or run the FULL app locally with no limits: "
    f"{GITHUB_URL}"
)


def _translate_exception(exc: Exception) -> Exception:
    import gradio as gr

    if isinstance(exc, NotImplementedError):
        # Capability gaps (e.g. text-only preset asked to do vision) — show the
        # explanation to the visitor instead of a generic error pill.
        return gr.Error(str(exc))
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "quota" in text or "zerogpu" in text:
        return gr.Error(QUOTA_MESSAGE)
    return exc


def friendly_errors(fn: Callable) -> Callable:
    """Map ZeroGPU quota errors to a clear gr.Error; pass other errors through.

    Generator functions must stay generator functions (Gradio streams them), so
    they get a yielding wrapper instead of a returning one.
    """
    import inspect

    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen_wrapper(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            except Exception as exc:
                raise _translate_exception(exc) from exc

        return gen_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise _translate_exception(exc) from exc

    return wrapper
