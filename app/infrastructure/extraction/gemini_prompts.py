# extraction/gemini_prompts.py
"""
Prompts used with the Gemini table extractor.

Centralised here so they can be inspected, tweaked or translated
without touching any Python logic.
"""

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

GEMINI_TABLE_HINT = (
    "Extract only the table grid (header and data rows). "
    "Ignore the caption, title, footnotes and any text outside the grid. "
    "Every value must be a JSON string exactly as printed "
    '(for example "5.00" or "100K"), never a JSON number.'
)