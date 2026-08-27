from __future__ import annotations

import ast
import csv
import io
import math
import operator
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from backend.app.services.knowledge_base import KnowledgeDocument


VALID_AGGREGATIONS = frozenset({"none", "count", "sum", "average", "min", "max"})
VALID_CHART_TYPES = frozenset({"bar"})
MAX_CHART_POINTS = 40
MAX_MATCHED_ROWS = 500
MAX_ABS_EXPONENT = 12


class SemanticToolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CsvFilter:
    column: str
    value: str


@dataclass(frozen=True, slots=True)
class CsvOperationSpec:
    document_name: str
    filters: tuple[CsvFilter, ...] = ()
    value_column: str | None = None
    aggregation: str = "none"
    x_column: str | None = None
    chart_type: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticToolExecution:
    context: str = ""
    chart: dict[str, object] | None = None
    error: str | None = None


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.strip().lower())
        if not unicodedata.combining(character)
    )


def _number(value: str) -> float:
    cleaned = value.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("vacío")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression: str) -> float | int:
    """Evaluate arithmetic without eval(), names, calls or attribute access."""

    if not expression.strip() or len(expression) > 500:
        raise SemanticToolError("La expresión de cálculo está vacía o es demasiado larga.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SemanticToolError("La expresión de cálculo no es sintácticamente válida.") from exc

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool) or not math.isfinite(float(node.value)):
                raise SemanticToolError("El cálculo contiene un valor numérico inválido.")
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > MAX_ABS_EXPONENT:
                raise SemanticToolError("El exponente excede el límite seguro.")
            try:
                value = _BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, OverflowError) as exc:
                raise SemanticToolError("El cálculo produjo una operación numérica inválida.") from exc
            if not math.isfinite(float(value)):
                raise SemanticToolError("El cálculo produjo un resultado no finito.")
            return value
        raise SemanticToolError(
            "La calculadora solo acepta números, paréntesis y operadores aritméticos seguros."
        )

    return visit(tree)


def _resolve_column(headers: Sequence[str], requested: str | None) -> str | None:
    if requested is None:
        return None
    target = _fold(requested)
    matches = [header for header in headers if _fold(header) == target]
    if len(matches) != 1:
        raise SemanticToolError(f"No se pudo resolver de forma única la columna '{requested}'.")
    return matches[0]


def _document(documents: Sequence[KnowledgeDocument], name: str) -> KnowledgeDocument:
    target = name.casefold().strip()
    matches = [document for document in documents if document.name.casefold() == target]
    if len(matches) != 1:
        raise SemanticToolError(
            f"No se encontró de forma única el documento local '{name}'."
        )
    if not matches[0].name.lower().endswith(".csv"):
        raise SemanticToolError("La operación estructurada de datos requiere un CSV.")
    return matches[0]


def _read_csv(document: KnowledgeDocument) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.reader(io.StringIO(document.content))
    rows = list(reader)
    header_index = next(
        (index for index, row in enumerate(rows) if any(cell.strip() for cell in row)),
        None,
    )
    if header_index is None:
        raise SemanticToolError("El CSV no contiene cabeceras ni filas utilizables.")
    headers = [cell.strip() or f"columna_{index + 1}" for index, cell in enumerate(rows[header_index])]
    dict_rows: list[dict[str, str]] = []
    for raw in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in raw):
            continue
        padded = list(raw) + [""] * max(0, len(headers) - len(raw))
        dict_rows.append({header: padded[index].strip() for index, header in enumerate(headers)})
    return headers, dict_rows


def execute_csv_operation(
    documents: Sequence[KnowledgeDocument],
    spec: CsvOperationSpec,
) -> SemanticToolExecution:
    try:
        if spec.aggregation not in VALID_AGGREGATIONS:
            raise SemanticToolError(f"Agregación no soportada: {spec.aggregation}.")
        if spec.chart_type is not None and spec.chart_type not in VALID_CHART_TYPES:
            raise SemanticToolError(f"Tipo de gráfico no soportado: {spec.chart_type}.")

        document = _document(documents, spec.document_name)
        headers, rows = _read_csv(document)
        resolved_filters = [
            (_resolve_column(headers, item.column), item.value) for item in spec.filters
        ]
        value_column = _resolve_column(headers, spec.value_column)
        x_column = _resolve_column(headers, spec.x_column)

        matched = [
            row
            for row in rows
            if all(
                column is not None and _fold(row.get(column, "")) == _fold(value)
                for column, value in resolved_filters
            )
        ]
        if len(matched) > MAX_MATCHED_ROWS:
            matched = matched[:MAX_MATCHED_ROWS]
        if not matched:
            raise SemanticToolError("La operación estructurada no encontró filas con esos filtros.")

        values: list[float] = []
        if value_column is not None:
            for row in matched:
                try:
                    values.append(_number(row.get(value_column, "")))
                except ValueError:
                    continue

        aggregate_text = ""
        if spec.aggregation == "count":
            aggregate_text = f"count={len(matched)}"
        elif spec.aggregation != "none":
            if value_column is None:
                raise SemanticToolError("La agregación requiere value_column.")
            if not values:
                raise SemanticToolError(
                    f"No hay valores numéricos utilizables en '{value_column}'."
                )
            if spec.aggregation == "sum":
                aggregate = sum(values)
            elif spec.aggregation == "average":
                aggregate = sum(values) / len(values)
            elif spec.aggregation == "min":
                aggregate = min(values)
            else:
                aggregate = max(values)
            aggregate_text = (
                f"{spec.aggregation}({value_column})={aggregate:g}; "
                f"numeric_rows={len(values)}; matched_rows={len(matched)}"
            )

        filter_text = ", ".join(
            f"{column}={value}" for column, value in resolved_filters if column is not None
        ) or "sin filtros"
        context_parts = [
            "RESULTADO DETERMINÍSTICO DE CSV (no estimado por el modelo):",
            f"document={document.name}",
            f"filters={filter_text}",
            f"matched_rows={len(matched)}",
        ]
        if aggregate_text:
            context_parts.append(aggregate_text)

        chart: dict[str, object] | None = None
        if spec.chart_type is not None:
            if x_column is None or value_column is None:
                raise SemanticToolError("El gráfico requiere x_column y value_column.")
            points: list[dict[str, object]] = []
            for row in matched:
                try:
                    value = _number(row.get(value_column, ""))
                except ValueError:
                    continue
                label = row.get(x_column, "").strip()
                if label:
                    points.append({"label": label, "value": value})
                if len(points) >= MAX_CHART_POINTS:
                    break
            if not points:
                raise SemanticToolError("El gráfico no encontró puntos numéricos válidos.")
            chart = {
                "type": "bar",
                "title": spec.title or f"{value_column} por {x_column}",
                "unit": "",
                "source": document.name,
                "metric": value_column,
                "points": points,
            }
            context_parts.append(
                f"chart={spec.chart_type}; x={x_column}; y={value_column}; points={len(points)}"
            )

        return SemanticToolExecution(context="\n".join(context_parts), chart=chart)
    except SemanticToolError as exc:
        return SemanticToolExecution(error=str(exc))


def execute_calculation(expression: str | None) -> SemanticToolExecution:
    if expression is None:
        return SemanticToolExecution(error="El planner pidió calculadora sin una expresión ejecutable.")
    try:
        result = evaluate_expression(expression)
    except SemanticToolError as exc:
        return SemanticToolExecution(error=str(exc))
    return SemanticToolExecution(
        context=(
            "RESULTADO DETERMINÍSTICO DE CALCULADORA (no estimado por el modelo):\n"
            f"expression={expression}\nresult={result}"
        )
    )


_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_TOLERATED_SMALL_INT_MAX = 12
_TOLERATED_YEAR_MIN = 1900
_TOLERATED_YEAR_MAX = 2100


def _normalize_numeric_token(token: str) -> str | None:
    is_percent = token.endswith("%")
    core = token[:-1] if is_percent else token
    if "," in core and "." in core:
        if core.rfind(",") > core.rfind("."):
            core = core.replace(".", "").replace(",", ".")
        else:
            core = core.replace(",", "")
    elif "," in core:
        parts = core.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2) and len(parts[0]) <= 3:
            core = core.replace(",", ".")
        else:
            core = core.replace(",", "")
    elif "." in core:
        parts = core.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            core = core.replace(".", "")
    try:
        value = float(core)
    except ValueError:
        return None
    normalized = str(int(value)) if value == int(value) else repr(value)
    return f"{normalized}%" if is_percent else normalized


def _is_tolerated_numeric(normalized: str) -> bool:
    if normalized.endswith("%"):
        return False
    try:
        value = float(normalized)
    except ValueError:
        return False
    if value != int(value):
        return False
    int_value = int(value)
    if 0 <= int_value <= _TOLERATED_SMALL_INT_MAX:
        return True
    return _TOLERATED_YEAR_MIN <= int_value <= _TOLERATED_YEAR_MAX


def audit_numeric_support(
    answer: str, *, allowed_texts: Sequence[str]
) -> tuple[str, ...]:
    """Flag numeric tokens in the answer that cannot be traced to any allowed
    source (user messages, deterministic tool results, accepted evidence).

    Observability only: callers should record findings, not block on them,
    until false-positive rates are measured against real traffic.
    """

    allowed_normalized: set[str] = set()
    for text in allowed_texts:
        for match in _NUMERIC_TOKEN_RE.finditer(text):
            normalized = _normalize_numeric_token(match.group())
            if normalized is not None:
                allowed_normalized.add(normalized)

    unsupported: list[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_TOKEN_RE.finditer(answer):
        token = match.group()
        normalized = _normalize_numeric_token(token)
        if normalized is None or normalized in seen:
            continue
        if normalized in allowed_normalized or _is_tolerated_numeric(normalized):
            continue
        seen.add(normalized)
        unsupported.append(token)
    return tuple(unsupported)
