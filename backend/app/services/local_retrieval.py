from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from backend.app.services.knowledge_base import KnowledgeDocument


WORD_PATTERN = re.compile(r"[\wáéíóúüñ]+", re.IGNORECASE)
STOP_WORDS = {
    "para",
    "como",
    "esta",
    "este",
    "entre",
    "sobre",
    "desde",
    "tiene",
    "puede",
    "qué",
    "que",
    "una",
    "uno",
    "los",
    "las",
    "del",
    "por",
    "con",
    "sin",
    "sus",
    "son",
    "hay",
    "más",
    "muy",
    "cómo",
}


@dataclass(frozen=True, slots=True)
class RetrievedLocalChunk:
    document_name: str
    chunk_index: int
    content: str
    score: int
    truncated: bool


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(character)
    )


def _terms(value: str) -> set[str]:
    return {
        _fold(term)
        for term in WORD_PATTERN.findall(value)
        if len(term) > 2 and _fold(term) not in STOP_WORDS
    }


def _text_chunks(content: str, *, size: int = 1_500) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [content.strip()]:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                paragraph[start : start + size]
                for start in range(0, len(paragraph), size)
            )
            continue
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _csv_chunks(content: str, *, size: int = 1_500) -> list[str]:
    rows = list(csv.reader(io.StringIO(content)))
    if not rows:
        return []
    header_index = next(
        (index for index, row in enumerate(rows) if any(cell.strip() for cell in row)),
        0,
    )
    headers = [
        cell.strip() or f"columna_{index + 1}"
        for index, cell in enumerate(rows[header_index])
    ]
    header_text = "Cabeceras CSV: " + " | ".join(headers)
    chunks = [header_text[:size]]
    for row in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        fields = [
            f"{header}={value.strip()}"
            for header, value in zip(headers, row)
            if value.strip()
        ]
        if fields:
            chunks.append(("Fila CSV: " + " | ".join(fields))[:size])
    return chunks


def _split_document(document: KnowledgeDocument) -> list[str]:
    if document.name.lower().endswith(".csv"):
        return _csv_chunks(document.content)
    return _text_chunks(document.content)


def _representative_indexes(length: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    candidates = {0, length - 1, length // 2}
    return tuple(sorted(index for index in candidates if 0 <= index < length))


def retrieve_local_chunks(
    documents: Sequence[KnowledgeDocument],
    query: str,
    *,
    selected_names: Sequence[str] = (),
    max_characters: int = 12_000,
    max_chunks: int = 12,
) -> tuple[RetrievedLocalChunk, ...]:
    """Retrieve bounded candidate chunks from selected local documents.

    Lexical overlap is used only as a cheap candidate-retrieval signal after the LLM
    has already interpreted the user's intent and selected documents. It never decides
    what the question means or what Orion should answer. When overlap is weak, the
    retriever samples multiple positions instead of blindly returning the file prefix.
    """

    if max_characters <= 0 or max_chunks <= 0:
        return ()
    wanted = {name.casefold() for name in selected_names if name.strip()}
    selected = [
        document
        for document in documents
        if not wanted or document.name.casefold() in wanted
    ]
    query_terms = _terms(query)
    candidates: list[RetrievedLocalChunk] = []

    for document in selected:
        chunks = _split_document(document)
        if not chunks:
            continue
        scored = [
            (
                len(query_terms & _terms(chunk)) if query_terms else 0,
                index,
                chunk,
            )
            for index, chunk in enumerate(chunks)
        ]
        positive = [item for item in scored if item[0] > 0]
        chosen = sorted(positive, key=lambda item: (-item[0], item[1]))
        if not chosen:
            chosen = [scored[index] for index in _representative_indexes(len(scored))]
        for score, index, chunk in chosen[:max_chunks]:
            candidates.append(
                RetrievedLocalChunk(
                    document_name=document.name,
                    chunk_index=index,
                    content=chunk,
                    score=score,
                    truncated=len(chunk) < len(document.content),
                )
            )

    candidates.sort(key=lambda item: (-item.score, item.document_name, item.chunk_index))
    remaining = max_characters
    result: list[RetrievedLocalChunk] = []
    for candidate in candidates:
        if remaining <= 0 or len(result) >= max_chunks:
            break
        content = candidate.content[:remaining].strip()
        if not content:
            continue
        result.append(
            RetrievedLocalChunk(
                document_name=candidate.document_name,
                chunk_index=candidate.chunk_index,
                content=content,
                score=candidate.score,
                truncated=candidate.truncated or len(content) < len(candidate.content),
            )
        )
        remaining -= len(content)
    return tuple(result)
