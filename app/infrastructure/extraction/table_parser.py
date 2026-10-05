from __future__ import annotations

from html.parser import HTMLParser


class MinerUTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()

        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        if tag == "tr":
            self._row = []

        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []

            self._row.append(
                " ".join(
                    "".join(self._cell).split()
                )
            )

            self._cell = None

        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_to_rows(html: str) -> list[list[str]]:
    parser = MinerUTableParser()
    parser.feed(html or "")
    return parser.rows