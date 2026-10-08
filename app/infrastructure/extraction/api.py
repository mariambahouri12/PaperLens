# extraction/api.py
"""
GENERIC table extraction from images using the Gemini API (free tier).
No column name is imposed: the model reads the headers by itself.

Usage:
    python api.py                  -> processes tab.jpg
    python api.py my_image.png     -> processes one image
    python api.py images_folder    -> processes every image in the folder

API key: GEMINI_API_KEY environment variable, or paste it into API_KEY below.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from google import genai
from google.genai import errors, types

# ----------------------------- CONFIGURATION -----------------------------
API_KEY = ""

# Explicit list of models to try. If empty, models are discovered automatically.
MODELS = []

MAX_ATTEMPTS = 8           # maximum number of Gemini calls per image
PAUSE_BETWEEN_IMAGES = 5   # seconds to wait between two images

PROMPT = """You are a table extraction engine. Extract every table visible in this image.

Return ONLY valid JSON with exactly this shape:
{"tables": [{"title": string or null,
             "columns": [string, ...],
             "rows": [[value, ...], ...]}]}

Rules:
1. Columns: read the headers exactly as printed. If headers span several levels,
   merge them into one label per column (e.g. "BLEU (dev)"). Write math symbols
   and subscripts in plain text (e.g. d_model, P_drop).
2. Every row must contain exactly as many values as there are columns, in
   left-to-right order. Keep every value in the column it is visually aligned with.
3. Empty cells must be null. Never fill an empty cell with a value from another
   row or column, and never guess missing values.
4. A cell that spans several rows (a row-group label): repeat its text in each
   row it covers. A cell that spans several columns: put its text in the first
   column it covers and null in the others.
5. Copy numbers, units, symbols and text exactly as printed. Do not round,
   translate, correct, reorder or summarize.
6. Include every row. Do not skip, merge or duplicate rows.
7. If the image contains no table, return {"tables": []}.
"""
# -------------------------------------------------------------------------

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_client():
    """Create a Gemini client from API_KEY or the GEMINI_API_KEY variable."""
    key = API_KEY or os.environ.get("GEMINI_API_KEY")

    if not key:
        sys.exit(
            "Missing API key: set GEMINI_API_KEY "
            "or fill API_KEY in this script."
        )

    return genai.Client(api_key=key)


def pick_models(client):
    """Return the list of models to try, from the primary to the fallbacks."""
    if MODELS:
        return MODELS

    names = []

    for model in client.models.list():
        name = model.name.replace("models/", "")
        low = name.lower()

        if "flash" in low and not any(
            excluded in low
            for excluded in ("image", "tts", "live", "audio", "embed")
        ):
            names.append(name)

    # Newest versions first, "lite" variants last.
    names.sort(key=lambda n: ("lite" in n.lower(), [-ord(c) for c in n]))

    if not names:
        sys.exit("No flash model found. Put your own list in MODELS.")

    print("Candidate models:", ", ".join(names[:4]))

    return names[:4]


def call_gemini(client, models, img, extra_hint: str = ""):
    """
    Call Gemini with retries and model fallback.

    `extra_hint` is appended to the base prompt for this call only; there
    is no module-level mutable state.
    """
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
    )

    prompt = PROMPT + (
        f"\nAdditional hint: {extra_hint}\n" if extra_hint else ""
    )

    dead = set()  # models rejected by the API (404)

    for attempt in range(MAX_ATTEMPTS):
        alive = [m for m in models if m not in dead]

        if not alive:
            sys.exit("No usable model.")

        model = alive[attempt % len(alive)]
        wait = min(5 * (attempt + 1), 40)

        try:
            response = client.models.generate_content(
                model=model,
                contents=[img, prompt],
                config=cfg,
            )
            print(f"  OK with {model}")
            return response.text

        except errors.ServerError as error:  # 500/503: server overloaded            
            print(
                f"  {model} unavailable ({error.code}). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

        except errors.ClientError as error:
            if error.code == 404:  # model does not exist / was removed
                print(f"  {model} rejected (404), skipping it.")
                dead.add(model)

            elif error.code == 429:  # quota exceeded
                print(f"  Quota reached (429). Pausing for {wait + 20}s...")
                time.sleep(wait + 20)

            else:
                raise

    sys.exit("Failed after several attempts. Please try again later.")


def parse_json(text):
    """Parse the model response and return the list of tables."""
    text = re.sub(
        r"^```(?:json)?|```$",
        "",
        text.strip(),
        flags=re.M,
    ).strip()

    data = json.loads(text)

    if isinstance(data, list):  # the model returned a bare list of tables
        data = {"tables": data}

    return data.get("tables", [])


def unique(names):
    """Make column names unique (col, col_2, ...) and non-empty."""
    seen, out = {}, []

    for index, name in enumerate(names):
        name = (
            str(name).strip()
            if name not in (None, "")
            else f"col_{index + 1}"
        )

        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    return out


def table_to_df(table):
    """Convert a table dict to a DataFrame; return it with the bad-row count."""
    cols = unique(table.get("columns", []))
    rows = table.get("rows", [])

    fixed, problems = [], 0

    for row in rows:
        if len(row) != len(cols):
            problems += 1
            row = (list(row) + [None] * len(cols))[:len(cols)]  # pad / truncate

        fixed.append(row)

    return pd.DataFrame(fixed, columns=cols), problems


def process_image(client, models, path, extra_hint: str = ""):
    """Extract the tables of one image and save them as JSON and CSV."""
    print(f"\n=== {path.name} ===")

    img = Image.open(path).convert("RGB")
    raw = call_gemini(client, models, img, extra_hint)

    # Keep the raw response in case the JSON is invalid.
    raw_file = path.with_suffix(".raw.txt")
    raw_file.write_text(raw, encoding="utf-8")

    try:
        tables = parse_json(raw)
    except json.JSONDecodeError as error:
        print(f"  Invalid JSON ({error}). See {raw_file.name}")
        return

    if not tables:
        print("  No table detected.")
        return

    path.with_suffix(".json").write_text(
        json.dumps(tables, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for number, table in enumerate(tables, 1):
        df, problems = table_to_df(table)

        suffix = f"_table{number}" if len(tables) > 1 else ""
        out_csv = path.with_name(path.stem + suffix + ".csv")

        # utf-8-sig so the file opens correctly in Excel.
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"\n  Table {number}: {table.get('title') or '(untitled)'}")
        print(df.to_string())
        print(f"\n  Rows: {len(df)} | Columns: {len(df.columns)}")

        if problems:
            print(
                f"  WARNING: {problems} row(s) had a number of values "
                f"different from the number of columns "
                f"(automatically fixed, please verify)."
            )

        print(f"  Empty cells per column:\n{df.isna().sum().to_string()}")
        print(f"  Saved: {out_csv.name}")


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tab.jpg")

    if target.is_dir():
        files = sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in IMAGE_EXT
        )
    elif target.is_file():
        files = [target]
    else:
        sys.exit(f"Not found: {target}")

    if not files:
        sys.exit("No image found.")

    client = get_client()
    models = pick_models(client)

    for index, file in enumerate(files):
        process_image(client, models, file)

        if index < len(files) - 1:
            time.sleep(PAUSE_BETWEEN_IMAGES)


if __name__ == "__main__":
    main()