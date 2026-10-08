# app/infrastructure/extraction/api.py
"""
Gemini table extraction: call, retry, parse, normalize.

Responsibilities:
  - call Gemini with retries and model fallback,
  - parse and validate the JSON response,
  - normalize column names.

Client setup and prompts are handled by dedicated modules.
Table orchestration and MinerU fallback live in `table_enricher.py`.
"""
from __future__ import annotations

import json
import logging
import re
import time

from google.genai import errors, types

from app.domain.exceptions import GeminiError

from .gemini_client import MAX_ATTEMPTS
from .gemini_prompts import PROMPT

logger = logging.getLogger(
    "paperlens.infrastructure.gemini"
)


def call_gemini(
    client,
    models,
    img,
    extra_hint: str = "",
) -> str:
    """
    Call Gemini with retry and model fallback.

    404:
        Model is removed from the candidate list.

    429:
        Wait and retry.

    5xx:
        Wait and retry with backoff.

    Other Gemini client errors:
        Converted to GeminiError immediately.

    Raises
    ------
    GeminiError
        If Gemini cannot produce a response.
    """
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
    )

    prompt = PROMPT + (
        f"\nAdditional hint: {extra_hint}\n"
        if extra_hint
        else ""
    )

    dead: set[str] = set()

    for attempt in range(MAX_ATTEMPTS):
        alive = [
            model
            for model in models
            if model not in dead
        ]

        if not alive:
            raise GeminiError(
                "No usable Gemini model remains."
            )

        model = alive[attempt % len(alive)]
        wait = min(5 * (attempt + 1), 40)

        try:
            response = client.models.generate_content(
                model=model,
                contents=[img, prompt],
                config=cfg,
            )

            if not response.text:
                raise GeminiError(
                    f"Gemini returned an empty response "
                    f"with model {model}."
                )

            print(f"  OK with {model}")
            return response.text

        except errors.ServerError as error:
            print(
                f"  {model} unavailable ({error.code}). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

        except errors.ClientError as error:
            if error.code == 404:
                print(
                    f"  {model} rejected (404), "
                    "skipping it."
                )
                dead.add(model)

            elif error.code == 429:
                print(
                    f"  Quota reached (429). "
                    f"Pausing for {wait + 20}s..."
                )
                time.sleep(wait + 20)

            else:
                raise GeminiError(
                    f"Gemini client error "
                    f"({error.code}): {error}"
                ) from error

        except (OSError, ConnectionError) as error:
            # Network-level failures should also trigger the
            # MinerU fallback rather than crash the pipeline.
            print(
                f"  Network error with {model}: {error}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise GeminiError(
        "Gemini failed after several attempts. "
        "Please try again later."
    )


def parse_json(text: str) -> list[dict]:
    """
    Parse and validate a Gemini table response.

    Accepted forms:

        {"tables": [...]}

    or:

        [...]

    Returns
    -------
    list[dict]
        Validated table objects.

    Raises
    ------
    GeminiError
        If the response is invalid or has an unexpected structure.
    """
    cleaned = re.sub(
        r"^```(?:json)?|```$",
        "",
        text.strip(),
        flags=re.M,
    ).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiError(
            f"Gemini returned invalid JSON: {exc}"
        ) from exc

    if isinstance(data, list):
        tables = data
    elif isinstance(data, dict):
        tables = data.get("tables", [])
    else:
        raise GeminiError(
            "Gemini JSON must be an object or a list."
        )

    if not isinstance(tables, list):
        raise GeminiError(
            "Gemini JSON field 'tables' must be a list."
        )

    validated: list[dict] = []

    for index, table in enumerate(tables, 1):
        if not isinstance(table, dict):
            raise GeminiError(
                f"Gemini table #{index} is not an object."
            )

        columns = table.get("columns", [])
        rows = table.get("rows", [])

        if not isinstance(columns, list):
            raise GeminiError(
                f"Gemini table #{index} has invalid columns."
            )

        if not isinstance(rows, list):
            raise GeminiError(
                f"Gemini table #{index} has invalid rows."
            )

        for row_index, row in enumerate(rows, 1):
            if not isinstance(row, list):
                raise GeminiError(
                    f"Gemini table #{index}, row "
                    f"#{row_index} is not a list."
                )

        validated.append(
            {
                "title": table.get("title"),
                "columns": columns,
                "rows": rows,
            }
        )

    return validated


def unique(names: list) -> list[str]:
    """
    Make column names unique and non-empty.

    Empty / None names become `col_N`.
    Duplicates receive `_2`, `_3`, ... suffixes.
    """
    seen: dict[str, int] = {}
    out: list[str] = []

    for index, name in enumerate(names):
        cleaned = (
            str(name).strip()
            if name not in (None, "")
            else f"col_{index + 1}"
        )

        seen[cleaned] = (
            seen.get(cleaned, 0) + 1
        )

        out.append(
            cleaned
            if seen[cleaned] == 1
            else f"{cleaned}_{seen[cleaned]}"
        )

    return out