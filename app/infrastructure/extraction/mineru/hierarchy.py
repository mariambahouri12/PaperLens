from __future__ import annotations

import re


def detect_logical_level(title: str) -> int:
    """
    Detect the logical hierarchy level of a heading.

    Rules:
        I. Introduction                  -> level 1
        II. Background                  -> level 1
        III. Methodology                -> level 1

        A. Dataset                      -> level 2
        B. Methodology                 -> level 2
        C. Results                     -> level 2

        1. Introduction                -> level 1
        2. Methodology                 -> level 1

        2.1 Dataset                    -> level 2
        2.2 Training                   -> level 2

        2.1.1 Preprocessing             -> level 3

    A single uppercase letter is checked before Roman numerals because
    C, D, I, V, X, L and M are also valid Roman numeral symbols.
    """

    title = str(title or "").strip()

    title = title.replace("\u00a0", " ")
    title = title.replace("\u2007", " ")
    title = title.replace("\u202f", " ")
    title = re.sub(r"\s+", " ", title)
    title = title.strip()

    # A. / B. / C. / D. -> level 2
    if re.match(r"^[A-Z]\.\s+", title):
        return 2

    # I. / II. / III. / IV. -> level 1
    if re.match(r"^[IVXLCDM]+\.\s+", title):
        return 1

    # 1. / 2. / 2.1 / 2.1.1 -> corresponding level
    decimal_match = re.match(
        r"^(\d+(?:\.\d+)*)\.?\s+",
        title,
    )

    if decimal_match:
        number = decimal_match.group(1)
        return len(number.split("."))

    # Unnumbered heading -> level 1
    return 1