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


# Column-name hints used to generalize CSV schema detection beyond the original
# GPS tracking export shape (Player Name / Total Distance / Period Name). The
# identifier hints deliberately keep "player name" first so GPS files keep
# resolving through the exact same path as before.
_ID_HEADER_HINTS = (
    "player name", "jugador", "jugadora", "player", "athlete", "atleta",
    "nombre", "name", "entidad", "entity", "equipo", "team", "empresa",
    "company", "código", "codigo", "id",
)
_PERIOD_HEADER_HINTS = (
    "period name", "período", "periodo", "period", "fecha", "date", "session",
    "sesión", "sesion", "semana", "week", "mes", "month", "jornada", "round",
    "ronda", "trimestre", "quarter",
)
_UNIT_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("distance", "distancia", "metros", "meters"), "m"),
    (("velocidad", "speed", "velocity"), "m/s"),
    (("duracion", "duración", "duration", "tiempo", "time", "minutos", "minutes"), "min"),
    (("porcentaje", "percent", "%"), "%"),
    (("potencia", "power", "watts"), "W"),
)


@dataclass(frozen=True, slots=True)
class CsvSchema:
    """Detected shape of a tabular CSV: which column identifies the entity,
    which ones are numeric metrics, and which one (if any) is a period/date."""

    header_index: int
    headers: list[str]
    id_index: int
    metric_indices: tuple[int, ...]
    period_index: int | None


def _looks_numeric(value: str) -> bool:
    try:
        Decimal(value.strip())
        return True
    except (InvalidOperation, ValueError):
        return False


def _header_words(value: str) -> set[str]:
    return set(re.findall(r"[a-záéíóúüñ0-9]+", value.lower()))


def _matches_any_hint(value: str, hints: tuple[str, ...]) -> bool:
    """Match column-name hints as whole words, not raw substrings.

    Short single-word hints like "id" would otherwise false-positive inside
    unrelated words (e.g. "Cantidad" contains "id"). Multi-word hints such as
    "player name" are specific enough to keep matching as a substring.
    """
    lowered = value.lower()
    words = _header_words(value)
    for hint in hints:
        if " " in hint:
            if hint in lowered:
                return True
        elif hint in words:
            return True
    return False


def _infer_unit(header: str) -> str:
    folded = _fold(header)
    for hints, unit in _UNIT_HINTS:
        if any(hint in folded for hint in hints):
            return unit
    return ""


def _find_header_row(rows: list[list[str]]) -> int | None:
    keyword_match = next(
        (
            index
            for index, row in enumerate(rows)
            if any(_matches_any_hint(cell, _ID_HEADER_HINTS) for cell in row)
        ),
        None,
    )
    if keyword_match is not None:
        return keyword_match
    # Generic fallback for any other tabular CSV: the header row is text-only
    # (no numeric cells) and is immediately followed by a row with at least
    # one numeric value.
    for index, row in enumerate(rows[:-1]):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) < 2 or any(_looks_numeric(cell) for cell in cells):
            continue
        next_values = [cell.strip() for cell in rows[index + 1] if cell.strip()]
        if next_values and any(_looks_numeric(cell) for cell in next_values):
            return index
    return None


def _infer_id_index(rows: list[list[str]], header_index: int, headers: list[str]) -> int:
    keyword_index = next(
        (index for index, header in enumerate(headers) if _matches_any_hint(header, _ID_HEADER_HINTS)),
        None,
    )
    if keyword_index is not None:
        return keyword_index
    data_rows = [row for row in rows[header_index + 1 :] if any(cell.strip() for cell in row)]
    for index in range(len(headers)):
        values = [row[index].strip() for row in data_rows if len(row) > index and row[index].strip()]
        if values and not any(_looks_numeric(value) for value in values):
            return index
    return 0


def _infer_period_index(headers: list[str]) -> int | None:
    return next(
        (index for index, header in enumerate(headers) if _matches_any_hint(header, _PERIOD_HEADER_HINTS)),
        None,
    )


def _infer_metric_indices(
    rows: list[list[str]], header_index: int, headers: list[str], *, exclude: set[int]
) -> tuple[int, ...]:
    data_rows = [row for row in rows[header_index + 1 :] if any(cell.strip() for cell in row)]
    indices: list[int] = []
    for index in range(len(headers)):
        if index in exclude:
            continue
        values = [row[index].strip() for row in data_rows if len(row) > index and row[index].strip()]
        if not values:
            continue
        numeric_ratio = sum(_looks_numeric(value) for value in values) / len(values)
        if numeric_ratio >= 0.8:
            indices.append(index)
    return tuple(indices)


def _detect_schema(rows: list[list[str]]) -> CsvSchema | None:
    header_index = _find_header_row(rows)
    if header_index is None:
        return None
    headers = [cell.strip() for cell in rows[header_index]]
    if not any(headers):
        return None
    id_index = _infer_id_index(rows, header_index, headers)
    period_index = _infer_period_index(headers)
    metric_indices = _infer_metric_indices(
        rows, header_index, headers, exclude={id_index, period_index} - {None}
    )
    return CsvSchema(header_index, headers, id_index, metric_indices, period_index)


def _select_metric_index(schema: CsvSchema, query: str) -> int | None:
    """Pick which numeric column the query refers to.

    A single detected metric is always unambiguous. With several candidates,
    the query must name one by its actual column header; otherwise Orion
    should ask instead of guessing.
    """
    if not schema.metric_indices:
        return None
    if len(schema.metric_indices) == 1:
        return schema.metric_indices[0]
    query_terms = _terms(query)
    scored = sorted(
        ((len(query_terms & _terms(schema.headers[index])), index) for index in schema.metric_indices),
        reverse=True,
    )
    best_score, best_index = scored[0]
    return best_index if best_score > 0 else None


def csv_chart(content: str, query: str, name: str) -> dict[str, object] | None:
    lowered_query = _fold(query)
    if not any(marker in lowered_query for marker in ("grafic", "fico", "visualiz", "chart")):
        return None
    rows = list(csv.reader(io.StringIO(content)))
    schema = _detect_schema(rows)
    if schema is None:
        return None
    metric_index = _select_metric_index(schema, query)
    if metric_index is None:
        return None
    query_terms = _terms(query)
    entity_rows = [
        row for row in rows[schema.header_index + 1 :]
        if len(row) > metric_index and query_terms & _terms(row[schema.id_index])
    ]
    if not entity_rows:
        return None
    points: list[dict[str, object]] = []
    for row in entity_rows:
        try:
            value = float(row[metric_index])
        except (ValueError, IndexError):
            continue
        label = (
            row[schema.period_index].strip()
            if schema.period_index is not None and len(row) > schema.period_index
            else "Registro"
        )
        points.append({"label": label, "value": value})
    if not points:
        return None
    metric_name = schema.headers[metric_index]
    return {
        "type": "bar",
        "title": f"{metric_name} por período",
        "unit": _infer_unit(metric_name),
        "source": name,
        "metric": metric_name,
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
    header_index = _find_header_row(rows)
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
    schema = _detect_schema(rows)
    if schema is None:
        return ""
    query_terms = _terms(query)
    matching_rows = [
        row for row in rows[schema.header_index + 1 :]
        if len(row) > schema.id_index and query_terms & _terms(row[schema.id_index])
    ]
    if not matching_rows:
        return ""
    metric_index = _select_metric_index(schema, query)
    if metric_index is None:
        return ""
    entity_label = schema.headers[schema.id_index]
    period_label = schema.headers[schema.period_index] if schema.period_index is not None else "período"
    metric_name = schema.headers[metric_index]
    parsed_rows: list[tuple[str, str, Decimal]] = []
    for row in matching_rows:
        if len(row) <= metric_index:
            continue
        try:
            value = Decimal(row[metric_index].strip())
        except (InvalidOperation, ValueError):
            continue
        period = (
            row[schema.period_index].strip()
            if schema.period_index is not None and len(row) > schema.period_index
            else "sin período"
        )
        parsed_rows.append((row[schema.id_index].strip(), period, value))
    if not parsed_rows:
        return ""
    detailed_rows = [
        row
        for row in parsed_rows
        if row[1].lower() not in {"session", "sesión", "sesion"}
    ]
    rows_to_sum = detailed_rows or parsed_rows
    rows_detail = [
        f"- {entity_label}={player} | {period_label}={period} | {metric_name}={value}"
        for player, period, value in parsed_rows
    ]
    total = sum((value for _, _, value in rows_to_sum), Decimal(0))
    aggregate_note = " (sin duplicar la fila agregada Session)" if detailed_rows and len(detailed_rows) < len(parsed_rows) else ""
    return (
        "RESULTADO ESTRUCTURADO Y DETERMINISTA DE ORION. Estos valores provienen "
        "directamente de las filas CSV y deben prevalecer sobre cualquier suposición del modelo.\n"
        + "\n".join(rows_detail)
        + f"\n- SUMA {metric_name} de {len(rows_to_sum)} períodos{aggregate_note} = {total}\n"
    )


def csv_tool_result(content: str, query: str, name: str) -> str:
    """Run small, deterministic CSV operations before asking the model to explain them."""
    lowered = _fold(query)
    if not any(marker in lowered for marker in ("promedio", "media", "compar", "filtr", "atip", "atíp")):
        return ""
    rows = list(csv.reader(io.StringIO(content)))
    schema = _detect_schema(rows)
    if schema is None:
        return ""
    metric_index = _select_metric_index(schema, query)
    if metric_index is None:
        return ""
    headers = schema.headers
    data = []
    for row in rows[schema.header_index + 1 :]:
        if len(row) <= metric_index:
            continue
        try:
            value = Decimal(row[metric_index].strip())
        except (InvalidOperation, ValueError):
            continue
        data.append((row[schema.id_index].strip(), value))
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
    schema = _detect_schema(rows)
    if schema is None:
        return ""
    headers = [header for header in schema.headers if header]
    data_rows = [row for row in rows[schema.header_index + 1 :] if any(cell.strip() for cell in row)]
    entities = sorted(
        {
            row[schema.id_index].strip()
            for row in data_rows
            if len(row) > schema.id_index and row[schema.id_index].strip()
        }
    )
    return (
        f"ARCHIVO CSV RECIBIDO: {name}\n"
        f"Columnas detectadas: {', '.join(headers)}\n"
        f"Filas de datos: {len(data_rows)}\n"
        f"Jugadores o entidades detectadas: {', '.join(entities[:30]) or 'no identificadas'}\n"
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
    schema = _detect_schema(rows)
    if schema is None:
        return False
    entities = {
        term
        for row in rows[schema.header_index + 1 :]
        if len(row) > schema.id_index
        for term in _terms(row[schema.id_index])
    }
    return not query_terms.intersection(entities)


def _query_targets_data(query: str) -> bool:
    query_terms = _terms(query)
    data_terms = {
        "archivo", "csv", "dato", "datos", "fila", "filas", "columna", "columnas",
        "jugador", "jugadores", "periodo", "período", "distancia", "velocidad",
        "aceleracion", "aceleración", "desaceleracion", "desaceleración", "métrica",
        "metrica", "gráfico", "grafico", "visualizar", "comparar", "suma", "promedio",
    }
    return bool(query_terms & data_terms)