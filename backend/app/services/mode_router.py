from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.app.domain.models import SelectedMode


class MessageLike(Protocol):
    content: str


DEEP_MARKERS = (
    "analiz",
    "argument",
    "compar",
    "investig",
    "hipótesis",
    "hipotesis",
    "modelo",
    "estrateg",
    "riesgo",
    "valid",
    "explicá paso",
    "explica paso",
    "fuentes",
    "evidencia",
    "causal",
    "predec",
    "planific",
)


@dataclass(frozen=True, slots=True)
class ModeRecommendation:
    mode: SelectedMode
    reason: str


def recommend_mode(messages: Sequence[MessageLike]) -> ModeRecommendation:
    prompt = messages[-1].content.lower()
    word_count = len(prompt.split())
    marker_count = sum(marker in prompt for marker in DEEP_MARKERS)
    question_count = prompt.count("?")

    score = 0
    reasons: list[str] = []
    if word_count >= 90:
        score += 2
        reasons.append("la consulta contiene bastante contexto")
    elif word_count >= 45:
        score += 1
        reasons.append("la consulta es extensa")
    if marker_count >= 2:
        score += 2
        reasons.append("requiere análisis o validación")
    elif marker_count == 1:
        score += 1
        reasons.append("incluye una tarea analítica")
    if question_count >= 3:
        score += 1
        reasons.append("reúne varias preguntas")

    if score >= 2:
        return ModeRecommendation(
            mode=SelectedMode.DEEP,
            reason="Orion recomienda Profundo porque " + " y ".join(reasons) + ".",
        )
    return ModeRecommendation(
        mode=SelectedMode.QUICK,
        reason="Orion recomienda Rápido porque la consulta parece directa y acotada.",
    )
