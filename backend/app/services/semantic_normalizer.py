from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import ChatMessage, SportContext


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _add_unique(values: list[str], *items: str) -> None:
    seen = {item.casefold() for item in values}
    for item in items:
        clean = " ".join(item.strip().split())
        key = clean.casefold()
        if clean and key not in seen:
            values.append(clean)
            seen.add(key)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def normalize_semantic_plan(
    plan: SemanticPlan,
    messages: Sequence[ChatMessage],
    sport: SportContext,
    *,
    has_local_documents: bool,
) -> SemanticPlan:
    """Apply deterministic domain invariants after semantic inference.

    The LLM remains responsible for open-ended intent inference. This layer only
    normalizes high-confidence linguistic and sports-domain relationships so a small
    local model cannot silently turn an obvious concept into a contradictory plan.
    It is deliberately conservative and does not answer the user's question.
    """
    query = messages[-1].content.strip()
    folded = _fold(query)
    prior_messages = list(messages[:-1])
    prior_text = _fold(" ".join(message.content for message in prior_messages[-5:]))
    conversation = f"{prior_text} {folded}".strip()

    # References to prior conversation or private operating conventions.
    if prior_messages and re.search(
        r"\b(eso|esto|esa|ese|aquello|lo mismo|como antes|anterior)\b", folded
    ):
        plan.referenced_previous_context = True
    if _has_any(
        folded,
        (
            "veniamos", "como lo usamos", "como usamos", "nuestro criterio",
            "nuestra metodologia", "nuestro protocolo", "mi protocolo",
            "mi metodologia", "en el club", "para nosotros",
        ),
    ):
        plan.needs_private_memory = True
        if _has_any(folded, ("veniamos", "como antes", "anterior")):
            plan.referenced_previous_context = True

    # Explicit task forms are deterministic and should not drift between runs.
    explicit_definition = bool(
        re.search(r"^\s*¿?\s*(que es|que significa|define|definime)\b", folded)
        or re.search(r"^\s*(explica|explicame|explica en|explicame en).*\bque es\b", folded)
    )
    if explicit_definition:
        plan.task_type = "definition"
        plan.requires_clarification = False
        plan.ambiguity = min(plan.ambiguity, 0.25)
        plan.complexity = min(plan.complexity, 0.35)

    if _has_any(folded, ("grafic", "visualiz", "chart")):
        plan.task_type = "chart"
        plan.needs_local_data = has_local_documents

    if _has_any(
        folded,
        (
            "busca fuentes", "busca estudios", "fuentes actuales", "estudios recientes",
            "investiga", "buscar en internet", "busca en internet",
        ),
    ):
        plan.task_type = "research"
        plan.needs_web = True
        plan.needs_global_knowledge = True

    # Comparison and causal-inference cues are conceptually important even when the
    # user never uses the words "comparar" or "causalidad".
    if _has_any(
        folded,
        (
            "mas que", "menos que", "quien trabajo mas", "quien corrio mas",
            "estuvo peor", "estuvo mejor", "diferencia entre", " vs ", "versus",
        ),
    ):
        plan.comparison = True

    causal_question = _has_any(
        folded,
        (
            "causo", "causa las", "provoco", "provoca", "fue porque", "debido a",
            "explica que", "por eso", "implica que", "significa que estuvo",
        ),
    )
    performance_inference = _has_any(folded, ("estuvo peor", "estuvo mejor")) and _has_any(
        conversation,
        ("corrio", "distancia", "hsr", "sprint", "aceler", "desaceler", "carga"),
    )
    if causal_question or performance_inference:
        plan.causal_claim_risk = True
        if plan.task_type not in {"research", "planning", "debugging"}:
            plan.task_type = "interpretation"
        plan.complexity = max(plan.complexity, 0.6)

    # "Who worked more?" is not answerable until work/load is operationally defined.
    if _has_any(folded, ("quien trabajo mas", "quien trabajó mas", "quien tuvo mas carga")):
        plan.comparison = True
        plan.needs_local_data = has_local_documents
        plan.requires_clarification = True
        plan.ambiguity = max(plan.ambiguity, 0.75)
        _add_unique(plan.missing_variables, "métrica de carga o trabajo a comparar")
        _add_unique(plan.concepts, "training load", "external load")
        if plan.task_type not in {"chart", "research"}:
            plan.task_type = "comparison"

    if sport is SportContext.FOOTBALL:
        _normalize_football(plan, folded, conversation)

    # Orion cannot require local data if none exists. Private memory is a separate
    # future store and therefore is intentionally not tied to local documents.
    if not has_local_documents:
        plan.needs_local_data = False

    # Canonical concepts also become retrieval reformulations without another LLM call.
    if plan.concepts:
        canonical_query = " ".join(plan.concepts[:8])
        _add_unique(plan.retrieval_queries, canonical_query)

    plan.confidence = max(0.0, min(plan.confidence, 1.0))
    return plan


def _normalize_football(plan: SemanticPlan, folded: str, conversation: str) -> None:
    physical_markers = (
        "hsr", "high speed", "sprint", "distancia", "corrio", "gps", "velocidad",
        "aceler", "desaceler", "carga externa", "metros por minuto",
    )
    if _has_any(conversation, physical_markers):
        if plan.domain in {"general", "sports", "football"} or _has_any(
            folded, ("hsr", "sprint", "corrio", "distancia", "rendimiento fisico")
        ):
            plan.domain = "physical_performance"
        _add_unique(plan.concepts, "external load", "match exposure")

    if "hsr" in conversation or "high-speed running" in conversation or "high speed running" in conversation:
        _add_unique(plan.concepts, "HSR", "high-speed running", "match exposure")

    if "sprint" in conversation:
        _add_unique(plan.concepts, "sprint", "speed threshold")

    if _has_any(folded, ("estuvo peor", "estuvo mejor", "rendimiento")) and _has_any(
        conversation, physical_markers
    ):
        _add_unique(plan.concepts, "physical performance", "external load", "match exposure")

    if "rpe" in conversation or "esfuerzo percibido" in conversation:
        if plan.domain in {"general", "sports", "football"}:
            plan.domain = "internal_load"
        _add_unique(plan.concepts, "RPE", "rating of perceived exertion", "internal load")

    tactical_markers = (
        "presion alta", "high press", "bloque alto", "linea alta", "salir largo",
        "salida larga", "juego largo", "build up", "build-up", "transicion ofensiva",
        "contraataque", "posesion",
    )
    if _has_any(conversation, tactical_markers):
        plan.domain = "tactical_analysis"

    if _has_any(conversation, ("presion alta", "high press")):
        _add_unique(plan.concepts, "high press", "pressing")
    if _has_any(folded, ("salir largo", "salida larga", "juego largo")):
        _add_unique(plan.concepts, "long ball", "build-up")
    if "transicion ofensiva" in conversation:
        _add_unique(plan.concepts, "offensive transition", "possession transition")

    if _has_any(conversation, ("lesion", "lesiones")) and _has_any(conversation, ("carga", "training load")):
        _add_unique(plan.concepts, "injury", "training load", "causality")
        if _has_any(folded, ("causo", "causa", "provoco", "provoca")):
            plan.causal_claim_risk = True
            plan.task_type = "interpretation"
            plan.complexity = max(plan.complexity, 0.7)
