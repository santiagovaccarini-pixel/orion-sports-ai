from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence

from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.core.sports_semantics import semantic_guide
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.providers.ollama import (
    ModelNotInstalledError,
    OllamaClient,
    OllamaUnavailableError,
)
from backend.app.services.knowledge_base import _query_targets_data
from backend.app.services.web_research import is_web_request


PLANNER_SYSTEM_PROMPT = """
Sos el planificador semántico interno de Orion. No respondas la pregunta del usuario.
Tu tarea es inferir qué intenta conseguir, qué conceptos técnicos están involucrados y
qué información necesita Orion antes de responder.

Reglas:
- Interpretá la conversación, no sólo coincidencias de palabras.
- Diferenciá pedido literal de objetivo real.
- No inventes datos, definiciones privadas ni hechos actuales.
- Si el mensaje depende de una referencia anterior ("eso", "lo mismo", "como antes"),
  resolvela usando únicamente la conversación suministrada y marcá
  referenced_previous_context=true.
- concepts debe usar términos técnicos canónicos y breves.
- retrieval_queries debe reformular la necesidad de información, incluyendo términos
  técnicos o sinónimos útiles aunque no aparezcan literalmente en la pregunta.
- needs_local_data=true sólo cuando la pregunta realmente requiera consultar datos o
  documentos cargados.
- needs_private_memory=true cuando el usuario se refiera a su protocolo, metodología,
  club, definición propia, preferencias o decisiones previas.
- needs_web=true sólo para hechos actuales, búsqueda explícita, fuentes recientes o
  información externa que deba verificarse.
- causal_claim_risk=true cuando la pregunta intente concluir que X causó Y o use una
  métrica como explicación causal.
- requires_clarification=true sólo cuando falta una variable indispensable y no pueda
  resolverse con el contexto disponible.
- ambiguity, complexity y confidence deben estar entre 0 y 1.
- Elegí task_type exclusivamente entre los valores permitidos por el esquema.
""".strip()


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _conversation_text(messages: Sequence[ChatMessage], *, limit: int = 6) -> str:
    selected = list(messages[-limit:])
    lines: list[str] = []
    for message in selected:
        role = "USUARIO" if message.role == "user" else "ORION"
        content = " ".join(message.content.strip().split())
        if len(content) > 1600:
            content = content[-1600:]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _fallback_plan(
    messages: Sequence[ChatMessage],
    *,
    has_local_documents: bool,
) -> SemanticPlan:
    query = messages[-1].content.strip()
    folded = _fold(query)
    previous_reference = bool(
        re.search(r"\b(eso|esto|lo mismo|como antes|esa|ese|aquello|anterior)\b", folded)
    )
    comparison = any(
        marker in folded
        for marker in ("compar", "versus", " vs ", "mas que", "menos que", "diferencia")
    )
    causal = any(
        marker in folded
        for marker in ("causa", "causo", "provoca", "provoco", "porque", "por que", "debido a")
    )
    chart = any(marker in folded for marker in ("grafic", "visualiz", "chart"))
    calculation = any(
        marker in folded
        for marker in ("calcular", "calcula", "promedio", "media", "suma", "porcentaje")
    )
    local = has_local_documents and _query_targets_data(query)
    web = is_web_request(query)

    if chart:
        task_type = "chart"
    elif calculation:
        task_type = "calculation"
    elif comparison:
        task_type = "comparison"
    elif causal or any(marker in folded for marker in ("interpret", "explica", "significa", "rindio", "rendimiento")):
        task_type = "interpretation"
    elif web:
        task_type = "research"
    elif local:
        task_type = "data_query"
    elif query.endswith("?") and len(query.split()) <= 12:
        task_type = "direct_answer"
    else:
        task_type = "direct_answer"

    private = any(
        marker in folded
        for marker in (
            "nosotros", "nuestro", "nuestra", "mi protocolo", "mi metodologia",
            "mi criterio", "en el club", "como usamos", "como venimos",
        )
    )
    complexity = 0.25
    if comparison:
        complexity += 0.2
    if causal:
        complexity += 0.25
    if previous_reference:
        complexity += 0.15
    if len(query.split()) > 45:
        complexity += 0.15
    complexity = min(complexity, 1.0)

    return SemanticPlan(
        literal_request=query[:400],
        user_goal=query[:500],
        domain="sports" if any(
            marker in folded
            for marker in (
                "jugador", "partido", "entren", "hsr", "sprint", "rpe", "gps",
                "carga", "fatiga", "lesion", "futbol", "basket", "rendimiento",
            )
        ) else "general",
        task_type=task_type,
        concepts=[],
        retrieval_queries=[query],
        missing_variables=[],
        needs_global_knowledge=not local or task_type in {"interpretation", "comparison", "research"},
        needs_private_memory=private,
        needs_local_data=local,
        needs_web=web,
        comparison=comparison,
        causal_claim_risk=causal,
        requires_clarification=False,
        referenced_previous_context=previous_reference,
        ambiguity=0.45 if previous_reference else 0.2,
        complexity=complexity,
        confidence=0.55,
    )


async def create_semantic_plan(
    settings: Settings,
    messages: Sequence[ChatMessage],
    sport: SportContext,
    *,
    has_local_documents: bool,
) -> SemanticPlan:
    """Infer user intent with a short structured LLM pass.

    The deterministic fallback is deliberately retained. A planning failure must
    never make the main chat unavailable.
    """
    fallback = _fallback_plan(messages, has_local_documents=has_local_documents)
    if not settings.semantic_planner_enabled:
        return fallback

    conversation = _conversation_text(messages)
    user_prompt = (
        f"DEPORTE SELECCIONADO: {sport.value}\n"
        f"HAY DOCUMENTOS LOCALES: {'sí' if has_local_documents else 'no'}\n\n"
        "GUÍA SEMÁNTICA DEL DOMINIO:\n"
        f"{semantic_guide(sport)}\n\n"
        "CONVERSACIÓN RECIENTE:\n"
        f"{conversation}\n\n"
        "Generá el plan semántico para el último mensaje. No lo respondas."
    )

    try:
        payload = await OllamaClient(settings).structured_json(
            model=settings.quick_model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=SemanticPlan.model_json_schema(),
            max_tokens=settings.semantic_planner_max_tokens,
        )
        plan = SemanticPlan.model_validate(payload)
    except (
        ModelNotInstalledError,
        OllamaUnavailableError,
        ValidationError,
        TypeError,
        ValueError,
    ):
        return fallback

    # System facts override model guesses where Orion can know them deterministically.
    if not has_local_documents:
        plan.needs_local_data = False
    if is_web_request(messages[-1].content):
        plan.needs_web = True
    return plan


def format_semantic_context(plan: SemanticPlan) -> str:
    """Compact plan injected into the answer model; no hidden chain-of-thought."""
    payload = {
        "objetivo_usuario": plan.user_goal,
        "dominio": plan.domain,
        "tipo_tarea": plan.task_type,
        "conceptos": plan.concepts,
        "variables_faltantes": plan.missing_variables,
        "riesgo_causal": plan.causal_claim_risk,
        "ambiguedad": round(plan.ambiguity, 2),
        "requiere_aclaracion": plan.requires_clarification,
    }
    return (
        "PLAN SEMÁNTICO DE ORION (usalo para responder la intención real, no para "
        "inventar información):\n" + json.dumps(payload, ensure_ascii=False)
    )
