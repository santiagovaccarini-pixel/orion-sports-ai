from __future__ import annotations

from dataclasses import dataclass
from unicodedata import normalize
from typing import Protocol, Sequence

from backend.app.domain.intent import SemanticPlan
from backend.app.domain.models import SelectedMode


class MessageLike(Protocol):
    content: str


DEEP_MARKERS = (
    "analiz",
    "argument",
    "compar",
    "investig",
    "hipotesis",
    "modelo",
    "estrateg",
    "riesgo",
    "valid",
    "explica paso",
    "fuentes",
    "evidencia",
    "causal",
    "predec",
    "planific",
)


def _fold_text(value: str) -> str:
    return "".join(
        character
        for character in normalize("NFD", value.lower())
        if character.isalnum() or character.isspace() or character in "¿?"
    )


@dataclass(frozen=True, slots=True)
class ModeRecommendation:
    mode: SelectedMode
    reason: str


def recommend_mode(
    messages: Sequence[MessageLike],
    semantic_plan: SemanticPlan | None = None,
) -> ModeRecommendation:
    if semantic_plan is not None:
        semantic_reasons: list[str] = []
        semantic_score = 0

        if semantic_plan.complexity >= 0.65:
            semantic_score += 2
            semantic_reasons.append("la intención requiere razonamiento complejo")
        elif semantic_plan.complexity >= 0.45:
            semantic_score += 1
            semantic_reasons.append("la consulta requiere interpretación")

        if semantic_plan.causal_claim_risk:
            semantic_score += 2
            semantic_reasons.append("hay una inferencia causal que conviene revisar")
        if semantic_plan.ambiguity >= 0.6:
            semantic_score += 1
            semantic_reasons.append("hay ambigüedad relevante")
        if semantic_plan.task_type in {
            "interpretation",
            "planning",
            "research",
            "debugging",
        }:
            semantic_score += 1
            semantic_reasons.append("el tipo de tarea se beneficia de análisis")

        if semantic_score >= 2:
            return ModeRecommendation(
                mode=SelectedMode.DEEP,
                reason="Orion recomienda Profundo porque "
                + " y ".join(dict.fromkeys(semantic_reasons))
                + ".",
            )
        if semantic_plan.confidence >= 0.65 and semantic_plan.complexity <= 0.35:
            return ModeRecommendation(
                mode=SelectedMode.QUICK,
                reason="Orion recomienda Rápido porque la intención fue identificada como directa y de baja complejidad.",
            )

    # Deterministic compatibility fallback for planning failures or uncertain plans.
    prompt = _fold_text(messages[-1].content)
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
