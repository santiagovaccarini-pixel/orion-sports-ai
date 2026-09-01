from __future__ import annotations

import json
import csv
import io
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock


WORD_PATTERN = re.compile(r"[\wáéíóúüñ]+", re.IGNORECASE)
STOP_WORDS = {
    "para", "como", "esta", "este", "entre", "sobre", "desde", "tiene",
    "puede", "qué", "que", "una", "uno", "los", "las", "del", "por",
    "con", "sin", "sus", "son", "hay", "más", "muy", "cómo",
}


# Documents whose stored text is a table Orion can compute over. A workbook
# qualifies because extraction turns a single sheet into plain CSV precisely so
# it lands on this path: filtering and averaging a GPS export must not depend on
# whether the person exported it as .csv or handed over the .xlsx they were sent.
TABULAR_EXTENSIONS = (".csv", ".xlsx", ".xlsm")


def is_tabular_document(name: str) -> bool:
    return name.lower().endswith(TABULAR_EXTENSIONS)


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    id: str
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    document_id: str
    document_name: str
    index: int
    content: str


def csv_chart(content: str, query: str, name: str) -> dict[str, object] | None:
    lowered_query = _fold(query)
    if not any(marker in lowered_query for marker in ("grafic", "fico", "visualiz", "chart")):
        return None
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (index for index, row in enumerate(rows) if any("player name" in cell.lower() for cell in row)),
        None,
    )
    if header_index is None:
        return None
    headers = [cell.strip() for cell in rows[header_index]]
    player_index = next((index for index, header in enumerate(headers) if "player name" in header.lower()), None)
    metric_index = next((index for index, header in enumerate(headers) if "total distance" in header.lower()), None)
    period_index = next((index for index, header in enumerate(headers) if "period name" in header.lower()), None)
    if player_index is None or metric_index is None:
        return None
    query_terms = _terms(query)
    player_rows = [
        row for row in rows[header_index + 1 :]
        if len(row) > metric_index and query_terms & _terms(row[player_index])
    ]
    if not player_rows:
        return None
    points: list[dict[str, object]] = []
    for row in player_rows:
        try:
            value = float(row[metric_index])
        except (ValueError, IndexError):
            continue
        label = row[period_index].strip() if period_index is not None and len(row) > period_index else "Registro"
        points.append({"label": label, "value": value})
    if not points:
        return None
    return {
        "type": "bar",
        "title": f"{headers[metric_index]} por período",
        "unit": "m",
        "source": name,
        "metric": headers[metric_index],
        "points": points,
    }


def csv_chart_is_ambiguous(content: str, query: str) -> bool:
    lowered_query = _fold(query)
    if not any(marker in lowered_query for marker in ("grafic", "fico", "visualiz", "chart")):
        return False
    return not csv_chart(content, query, "archivo.csv")


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def list_documents(self) -> list[KnowledgeDocument]:
        with self._lock:
            return self._read_documents()

    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        with self._lock:
            documents = [item for item in self._read_documents() if item.id != document.id]
            documents.append(document)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(item) for item in documents], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return document

    def delete_document(self, document_id: str) -> bool:
        """Remove a document for good. Returns False when it was already gone.

        Until this existed, the × on an attachment only cleared the chip in the
        composer: the document stayed in the knowledge base and kept feeding every
        later answer. Someone who thought they had withdrawn a file had not.
        """

        with self._lock:
            documents = self._read_documents()
            remaining = [item for item in documents if item.id != document_id]
            if len(remaining) == len(documents):
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(item) for item in remaining], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return True

    def search(self, query: str, *, limit: int = 12) -> list[KnowledgeChunk]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        chunks: list[KnowledgeChunk] = []
        for document in self.list_documents():
            csv_mode = document.name.lower().endswith(".csv")
            for index, content in enumerate(
                _split_chunks(document.content, csv_mode=csv_mode)
            ):
                chunk_terms = _terms(content)
                score = sum(1 for term in query_terms if term in chunk_terms)
                if score:
                    chunks.append(KnowledgeChunk(document.id, document.name, index, content))
        return sorted(chunks, key=lambda chunk: (-_score(chunk, query_terms), chunk.document_name, chunk.index))[:limit]

    def _read_documents(self) -> list[KnowledgeDocument]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [KnowledgeDocument(**item) for item in payload if isinstance(item, dict)]


def _terms(value: str) -> set[str]:
    return {
        term.lower()
        for term in WORD_PATTERN.findall(value)
        if len(term) > 2 and term.lower() not in STOP_WORDS
    }


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(character)
    )


def _split_chunks(
    content: str,
    *,
    size: int = 1_200,
    csv_mode: bool = False,
) -> list[str]:
    if csv_mode:
        return _split_csv_rows(content, size=size)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
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


def _split_csv_rows(content: str, *, size: int) -> list[str]:
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any("player name" in cell.lower() for cell in row)
        ),
        None,
    )
    if header_index is None:
        return [line[:size] for line in content.splitlines() if line.strip()]

    headers = [cell.strip() or f"columna_{index + 1}" for index, cell in enumerate(rows[header_index])]
    chunks = ["Cabeceras CSV: " + " | ".join(headers)]
    for row in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        values = [cell.strip() for cell in row]
        fields = [
            f"{header}={value}"
            for header, value in zip(headers, values)
            if value
        ]
        if fields:
            chunks.append("Fila CSV: " + " | ".join(fields)[:size])
    return chunks


def _score(chunk: KnowledgeChunk, query_terms: set[str]) -> int:
    return len(query_terms & _terms(chunk.content))


def format_context(chunks: list[KnowledgeChunk], *, max_characters: int = 2_000) -> str:
    if not chunks:
        return ""
    entries: list[str] = []
    remaining = max_characters
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.index == 0, chunk.document_name, chunk.index),
    )
    for chunk in ordered_chunks:
        prefix = f"[Fuente local: {chunk.document_name}, fragmento {chunk.index + 1}]\n"
        available = remaining - len(prefix)
        if available <= 0:
            break
        content = chunk.content[:available].rstrip()
        entries.append(prefix + content)
        remaining -= len(prefix) + len(content) + 2
        if len(content) < len(chunk.content):
            break
    return (
        "Usá estas fuentes locales cuando sean pertinentes. No inventes datos que no "
        "aparezcan en ellas y mencioná el nombre de la fuente al respaldar una afirmación.\n\n"
        + "\n\n".join(entries)
    )


def csv_calculation(content: str, query: str) -> str:
    if not re.search(r"\b(sumá|suma|sumar|total|promedio|promediar)\b", query.lower()):
        return ""
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (index for index, row in enumerate(rows) if any("player name" in cell.lower() for cell in row)),
        None,
    )
    if header_index is None:
        return ""
    headers = [cell.strip() for cell in rows[header_index]]
    player_index = next((index for index, header in enumerate(headers) if "player name" in header.lower()), None)
    if player_index is None:
        return ""
    query_terms = _terms(query)
    matching_rows = [
        row for row in rows[header_index + 1 :]
        if len(row) > player_index and query_terms & _terms(row[player_index])
    ]
    if not matching_rows:
        return ""
    column_index = next(
        (index for index, header in enumerate(headers) if "total distance" in header.lower()),
        None,
    )
    if column_index is None:
        return ""
    period_index = next(
        (index for index, header in enumerate(headers) if "period name" in header.lower()),
        None,
    )
    parsed_rows: list[tuple[str, str, Decimal]] = []
    for row in matching_rows:
        if len(row) <= column_index:
            continue
        try:
            value = Decimal(row[column_index].strip())
        except (InvalidOperation, ValueError):
            continue
        period = row[period_index].strip() if period_index is not None and len(row) > period_index else "sin período"
        parsed_rows.append((row[player_index].strip(), period, value))
    if not parsed_rows:
        return ""
    detailed_rows = [
        row
        for row in parsed_rows
        if row[1].lower() not in {"session", "sesión", "sesion"}
    ]
    rows_to_sum = detailed_rows or parsed_rows
    rows_detail = [
        f"- jugador={player} | período={period} | Total Distance={value}"
        for player, period, value in parsed_rows
    ]
    total = sum((value for _, _, value in rows_to_sum), Decimal(0))
    aggregate_note = " (sin duplicar la fila agregada Session)" if detailed_rows and len(detailed_rows) < len(parsed_rows) else ""
    return (
        "RESULTADO ESTRUCTURADO Y DETERMINISTA DE ORION. Estos valores provienen "
        "directamente de las filas CSV y deben prevalecer sobre cualquier suposición del modelo.\n"
        + "\n".join(rows_detail)
        + f"\n- SUMA Total Distance de {len(rows_to_sum)} períodos{aggregate_note} = {total}\n"
    )


def csv_tool_result(content: str, query: str, name: str) -> str:
    """Run small, deterministic CSV operations before asking the model to explain them."""
    lowered = _fold(query)
    if not any(marker in lowered for marker in ("promedio", "media", "compar", "filtr", "atip", "atíp")):
        return ""
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (index for index, row in enumerate(rows) if any("player name" in cell.lower() for cell in row)),
        None,
    )
    if header_index is None:
        return ""
    headers = [cell.strip() for cell in rows[header_index]]
    player_index = next((index for index, header in enumerate(headers) if "player name" in header.lower()), None)
    metric_index = next((index for index, header in enumerate(headers) if "total distance" in header.lower()), None)
    if player_index is None or metric_index is None:
        return ""
    data = []
    for row in rows[header_index + 1 :]:
        if len(row) <= metric_index:
            continue
        try:
            value = Decimal(row[metric_index].strip())
        except (InvalidOperation, ValueError):
            continue
        data.append((row[player_index].strip(), value))
    if not data:
        return ""
    if "promedio" in lowered or "media" in lowered:
        average = sum((value for _, value in data), Decimal(0)) / len(data)
        return f"RESULTADO DETERMINISTA: promedio de {headers[metric_index]} en {len(data)} filas = {average.quantize(Decimal('0.01'))} ({name}).\n"
    if "atip" in lowered or "atíp" in lowered:
        values = [value for _, value in data]
        minimum, maximum = min(values), max(values)
        spread = maximum - minimum
        threshold = spread * Decimal("0.75")
        outliers = [(player, value) for player, value in data if value == minimum or value == maximum]
        return (
            f"RESULTADO DETERMINISTA: rango de {headers[metric_index]} = {minimum} a {maximum}; "
            f"amplitud = {spread}. Extremos detectados (no prueba causalidad): "
            + ", ".join(f"{player}={value}" for player, value in outliers)
            + f". Umbral exploratorio de amplitud: {threshold}. Fuente: {name}.\n"
        )
    return ""


def csv_overview(content: str, name: str) -> str:
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (index for index, row in enumerate(rows) if any("player name" in cell.lower() for cell in row)),
        None,
    )
    if header_index is None:
        return ""
    headers = [cell.strip() for cell in rows[header_index] if cell.strip()]
    data_rows = [row for row in rows[header_index + 1 :] if any(cell.strip() for cell in row)]
    players = sorted({row[0].strip() for row in data_rows if row and row[0].strip()})
    return (
        f"ARCHIVO CSV RECIBIDO: {name}\n"
        f"Columnas detectadas: {', '.join(headers)}\n"
        f"Filas de datos: {len(data_rows)}\n"
        f"Jugadores o entidades detectadas: {', '.join(players[:30]) or 'no identificadas'}\n"
        "Si la petición no indica jugador, columna, período o cálculo, pedí una aclaración "
        "antes de responder. Ofrecé opciones basadas en estas columnas y no inventes un objetivo.\n"
    )


def csv_query_is_ambiguous(content: str, query: str) -> bool:
    if not _query_targets_data(query):
        return False
    overview = csv_overview(content, "")
    if not overview:
        return False
    query_terms = _terms(query)
    meaningful_terms = {
        "distancia", "distance", "velocidad", "velocity", "aceleracion", "aceleración",
        "desaceleracion", "desaceleración", "duracion", "duración", "minutos", "carga",
        "periodo", "período", "session", "sesion", "sesión", "jugador", "promedio",
        "media", "suma", "total", "comparar", "compará", "comparacion", "comparación",
    }
    if query_terms & meaningful_terms:
        return False
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next(
        (index for index, row in enumerate(rows) if any("player name" in cell.lower() for cell in row)),
        None,
    )
    if header_index is None:
        return False
    player_index = next(
        (index for index, cell in enumerate(rows[header_index]) if "player name" in cell.lower()),
        None,
    )
    if player_index is None:
        return False
    players = {
        term
        for row in rows[header_index + 1 :]
        if len(row) > player_index
        for term in _terms(row[player_index])
    }
    return not query_terms.intersection(players)


def _query_targets_data(query: str) -> bool:
    query_terms = _terms(query)
    data_terms = {
        "archivo", "csv", "dato", "datos", "fila", "filas", "columna", "columnas",
        "jugador", "jugadores", "periodo", "período", "distancia", "velocidad",
        "aceleracion", "aceleración", "desaceleracion", "desaceleración", "métrica",
        "metrica", "gráfico", "grafico", "visualizar", "comparar", "suma", "promedio",
    }
    return bool(query_terms & data_terms)