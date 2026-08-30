"""Turn a fetched HTML page into the text Orion actually reasons over.

A regex tag-stripper cannot do this correctly, and the failure is not cosmetic.
Measured on the Spanish Wikipedia article for a football manager: 22% of the
extracted "text" was MediaWiki template markup, because Parsoid stores template
source in a `data-mw` attribute whose JSON contains `>` characters. `<[^>]+>`
stops at the first one and spills the rest of the attribute into the output as
if it were prose. That markup repeats the subject's name in every citation
title, so it then outscored the real content in the excerpt selection and
crowded the career table out of the model's context entirely.

A real parser also lets table structure survive. Career histories, squad lists
and statistics tables are the shape sports data arrives in; flattening a row
into a space-separated run of words destroys which year belongs to which club.
"""

from __future__ import annotations

from html.parser import HTMLParser

# Containers whose text is chrome or code, never page content.
SKIPPED_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
        "template",
        "select",
        "button",
    }
)

# Tags that end a line of text.
BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "li",
        "tr",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "figcaption",
        "dd",
        "dt",
        "caption",
    }
)

CELL_TAGS = frozenset({"td", "th"})

# Footnote markers ([1], [2]) that add nothing once the page is plain text.
REFERENCE_CLASS_HINTS = ("reference", "mw-editsection", "noprint")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._skipped_tag: str | None = None
        # Inside a cell, block elements must not break the line. Wikipedia wraps
        # cell contents in divs and spans, and a newline there splits "2013-2015"
        # away from the club it belongs to, which is the whole point of the row.
        self._cell_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            if tag == self._skipped_tag:
                self._skip_depth += 1
            return
        if tag in SKIPPED_TAGS or _is_reference_marker(attrs):
            self._skipped_tag = tag
            self._skip_depth = 1
            return
        if tag in CELL_TAGS:
            self._cell_depth += 1
            self._parts.append(" | ")
        elif tag == "tr":
            self._parts.append("\n")
        elif tag in BLOCK_TAGS and not self._cell_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skipped_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skipped_tag = None
            return
        if tag in CELL_TAGS:
            # A closing cell must not break the line: the row is the unit that
            # carries meaning, and the next cell's opening tag separates them.
            self._cell_depth = max(0, self._cell_depth - 1)
            return
        if tag == "table":
            self._cell_depth = 0
            self._parts.append("\n")
        elif tag in BLOCK_TAGS and not self._cell_depth:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skip_depth and tag == "br":
            self._parts.append(" " if self._cell_depth else "\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _is_reference_marker(attrs: list[tuple[str, str | None]]) -> bool:
    for name, value in attrs:
        if name == "class" and value:
            classes = value.lower()
            if any(hint in classes for hint in REFERENCE_CLASS_HINTS):
                return True
    return False


def visible_text(html: str) -> str:
    """Readable text from an HTML document, with table rows kept as rows.

    Never raises on malformed markup: a page Orion cannot parse still yields
    whatever text was recovered before the parser gave up, which is strictly
    better than dropping the source.
    """

    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:  # noqa: BLE001 - a broken page must not lose its readable part
        pass
    return _tidy(extractor.text())


def _tidy(value: str) -> str:
    lines: list[str] = []
    for raw_line in value.replace("\xa0", " ").splitlines():
        # Collapse runs of whitespace inside the line, but keep the cell markers
        # that carry the table's columns.
        line = " ".join(raw_line.split())
        line = line.strip(" |").strip()
        if len(line) < 2:
            continue
        if lines and lines[-1] == line:
            # Wikipedia repeats headings as navigation; one copy is enough.
            continue
        lines.append(line)
    return "\n".join(lines).strip()
