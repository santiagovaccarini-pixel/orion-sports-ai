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
    """Choose the expensive 8B path only when its expected value justifies latency.

    The current hardware benchmark is CPU-bound, so ordinary interpretation and
    comparison should remain on Quick after semantic normalization. Deep is reserved
    for genuinely difficult reasoning, not simply for queries containing analytical
    vocabulary.
    """
    if semantic_plan is not None:
        deep_reasons: list[str] = []

        if semantic_plan.complexity >= 0.82:
            deep_reasons.append("la intención requiere razonamiento de alta complejidad")

        if semantic_plan.causal_claim_risk and semantic_plan.complexity >= 0.68:
            deep_reasons.append("hay una inferencia causal compleja que conviene revisar")

        if (
            semantic_plan.task_type in {"research", "planning", "debugging"}
            and semantic_plan.complexity >= 0.58
        ):
            deep_reasons.append("el tipo de tarea necesita análisis más profundo")

        # Ambiguity alone is not a reason to spend 8B compute. If clarification is
        # required, Quick should ask the missing variable instead of reasoning longer.
        if deep_reasons and not semantic_plan.requires_clarification:
            return ModeRecommendation(
                mode=SelectedMode.DEEP,
                reason="Orion recomienda Profundo porque "
                + " y ".join(dict.fromkeys(deep_reasons))
                + ".",
            )

        quick_reason = "la intención puede resolverse con el planner semántico y el modelo rápido"
        if semantic_plan.requires_clarification:
            quick_reason = "falta una variable concreta y conviene pedirla sin ejecutar el modelo pesado"
        elif semantic_plan.task_type in {"definition", "direct_answer", "calculation", "data_query", "chart"}:
            quick_reason = "la tarea es directa y no justifica el costo del modelo profundo"
        elif semantic_plan.task_type in {"interpretation", "comparison"}:
            quick_reason = "la interpretación está acotada y la ontología semántica aporta el contexto necesario"

        return ModeRecommendation(
            mode=SelectedMode.QUICK,
            reason=f"Orion recomienda Rápido porque {quick_reason}.",
        )

    # Compatibility fallback if no semantic plan is available.
    prompt = _fold_text(messages[-1].content)
    word_count = len(prompt.split())
    marker_count = sum(marker in prompt for marker in DEEP_MARKERS)
    question_count = prompt.count("?")

    score = 0
    reasons: list[str] = []
    if word_count >= 100:
        score += 2
        reasons.append("la consulta contiene bastante contexto")
    elif word_count >= 55:
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

    if score >= 3:
        return ModeRecommendation(
            mode=SelectedMode.DEEP,
            reason="Orion recomienda Profundo porque " + " y ".join(reasons) + ".",
        )
    return ModeRecommendation(
        mode=SelectedMode.QUICK,
        reason="Orion recomienda Rápido porque la consulta parece directa o acotada.",
    )
