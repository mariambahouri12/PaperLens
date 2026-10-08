# extraction/gemini_client.py
"""
Gemini client setup and model selection.

Responsibilities (and ONLY these):
  - read the API key (env var or module override),
  - build the `genai.Client`,
  - discover / pick the flash models to try.

Every error from the underlying SDK is wrapped in `GeminiError` so
callers only have to catch one exception type. The actual Gemini
calls, prompt handling, retry logic and response parsing live in
`api.py`.
"""
from __future__ import annotations

import logging
import os

from google import genai

from app.domain.exceptions import GeminiError

logger = logging.getLogger("paperlens.infrastructure.gemini")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Explicit list of models to try. If empty, models are discovered
# automatically from the API.
MODELS_OVERRIDE: list[str] = []

# Optional API key override. Prefer the GEMINI_API_KEY environment
# variable in production; this constant is kept for local scripts.
API_KEY_OVERRIDE = ""

# Retry / throttling constants (kept here because they describe how we
# talk to the Gemini service).
MAX_ATTEMPTS = 8
PAUSE_BETWEEN_IMAGES = 5


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------


def get_client() -> genai.Client:
    """
    Create a Gemini client from the configured API key.

    Raises
    ------
    GeminiError
        If no API key is configured, or if the SDK fails to build
        the client (invalid key format, network issue, ...).
    """
    key = API_KEY_OVERRIDE or os.environ.get("GEMINI_API_KEY")

    if not key:
        raise GeminiError(
            "Missing Gemini API key: set GEMINI_API_KEY "
            "or fill API_KEY_OVERRIDE in gemini_client.py."
        )

    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        # Any SDK error becomes a GeminiError so callers have a single
        # exception type to handle.
        raise GeminiError(f"Could not build Gemini client: {exc}") from exc


def pick_models(client: genai.Client) -> list[str]:
    """
    Return the ordered list of models to try (primary first).

    If `MODELS_OVERRIDE` is set, it is returned as-is. Otherwise the
    models are discovered from the API, keeping flash models only and
    sorting newest first with `lite` variants last.

    Raises
    ------
    GeminiError
        If the API call fails or no suitable model is found.
    """
    if MODELS_OVERRIDE:
        return list(MODELS_OVERRIDE)

    try:
        raw_models = list(client.models.list())
    except Exception as exc:
        # Wrap SDK errors (auth, quota, network, ...) as GeminiError.
        raise GeminiError(
            f"Could not list Gemini models: {exc}"
        ) from exc

    names: list[str] = []

    for model in raw_models:
        name = model.name.replace("models/", "")
        low = name.lower()

        if "flash" in low and not any(
            excluded in low
            for excluded in ("image", "tts", "live", "audio", "embed")
        ):
            names.append(name)

    names.sort(key=lambda n: ("lite" in n.lower(), [-ord(c) for c in n]))

    if not names:
        raise GeminiError(
            "No flash model found. Put your own list in MODELS_OVERRIDE."
        )

    print("Candidate models:", ", ".join(names[:4]))
    return names[:4]