from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.providers.model_provider import ModelProvider, ModelResult
from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.local_retrieval import retrieve_local_chunks
from backend.app.services.web_research import WebSource


MAX_REVIEW_INPUT_CHARACTERS = 19_000
VALID_EVIDENCE_POLICIES = frozenset({"model_knowledge", "external", "local", "mixed"})
ModelResultCallback = Callable[[str, ModelResult], None]


class SemanticOrchestrationError(RuntimeError):
    """Raised when a structured semantic decision cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    objective: str
    entities: tuple[str, ...]
    constraints: tuple[str, ...]
    references: tuple[str, ...]
    information_needed: tuple[str, ...]
    ambiguities: tuple[str, ...]
    evidence_policy: str
    use_web: bool
    use_local_data: bool
    use_calculator: bool
    use_chart: bool
    needs_clarification: bool
    clarifying_question: str | None
    web_query: str | None
    local_document_names: tuple[str, ...]
    recommended_mode: SelectedMode
    reason: str


@dataclass(frozen=True, slots=True)
class LocalEvidence:
    source_id: str
    document_name: str
    content: str
    truncated: bool
    chunk_index: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceReview:
    sufficient: bool
    relevant_source_ids: tuple[str, ...]
    discarded_source_ids: tuple[str, ...]
    missing_information: tuple[str, ...]
    follow_up_web_query: str | None
    needs_clarification: bool
    clarifying_question: str | None
    resolved_scope: str | None
    reason: str


PLANNER_PROMPT = """
Sos la etapa de comprensión y planificación de Orion. No respondas la pregunta del
usuario. Interpretá semánticamente la conversación completa y decidí qué información
y herramientas hacen falta antes de responder.

Reglas obligatorias:
- No clasifiques por coincidencia de palabras, listas de términos ni plantillas del
  tipo «si dice X significa Y». Inferí intención, referencias, alcance y restricciones
  por el significado de la conversación completa.
- Resolvé referencias conversacionales como «eso», «lo mismo», «hacelo con 5» usando
  el contexto previo cuando exista.
- Conservá el alcance que realmente pidió el usuario. No agregues por conveniencia un
  período, competición, población, filtro, unidad o recorte que no surja de la
  conversación. Si el usuario no fijó una restricción temporal, no la inventes para
  adaptar la pregunta a una fuente más fácil de encontrar.
- Elegí una política de evidencia por significado, no por palabras clave:
  * model_knowledge: definiciones, explicaciones, razonamiento o conocimiento general
    estable que no necesita un valor actual ni datos privados para responder.
  * external: hechos públicos actuales/cambiantes o afirmaciones que el usuario pide
    verificar externamente.
  * local: la respuesta depende de documentos o datos privados/locales disponibles.
  * mixed: hacen falta tanto evidencia externa como datos locales.
- No supongas que un dato del modelo está actualizado. Si el hecho puede haber
  cambiado con el tiempo y la respuesta necesita el valor actual, la política no puede
  ser model_knowledge.
- Los documentos locales y la web son herramientas complementarias. Podés elegir
  ninguna, una o ambas según lo que realmente necesite la pregunta.
- La política y las herramientas deben ser coherentes: external requiere web si está
  disponible; local requiere datos locales; mixed requiere ambos. model_knowledge no
  debe forzar búsquedas solo para demostrar algo que es conocimiento general estable.
- No inventes que un documento contiene un dato. Solo podés elegir documentos que
  aparezcan en el catálogo disponible.
- No interrumpas automáticamente una consulta breve solo porque admita más de un
  alcance razonable. Si el contexto, el uso ordinario o la investigación permiten
  adoptar una interpretación defendible, investigá y dejá que la respuesta explicite
  el alcance adoptado. Pedí aclaración únicamente cuando varias interpretaciones
  sigan siendo materialmente distintas y ninguna pueda resolverse razonablemente con
  contexto o evidencia; en acciones destructivas o sensibles, preferí aclarar.
- La consulta de búsqueda web debe surgir del objetivo entendido, no de una regla
  escrita para un jugador, deporte, métrica o frase particular.
- Recomendá modo quick para consultas directas y deep cuando haga falta análisis,
  comparación, varias etapas, incertidumbre sustancial o una explicación extensa.

Devolvé exclusivamente un objeto JSON válido con estas claves:
{
  "objective": "...",
  "entities": ["..."],
  "constraints": ["..."],
  "references": ["..."],
  "information_needed": ["..."],
  "ambiguities": ["..."],
  "evidence_policy": "model_knowledge" or "external" or "local" or "mixed",
  "use_web": true,
  "use_local_data": false,
  "use_calculator": false,
  "use_chart": false,
  "needs_clarification": false,
  "clarifying_question": null,
  "web_query": "..." or null,
  "local_document_names": ["..."],
  "recommended_mode": "quick" or "deep",
  "reason": "explicación breve de la decisión"
}
""".strip()


REVIEW_PROMPT = """
Sos la etapa crítica de revisión de Orion. No contestes todavía al usuario. Auditá
tres cosas de forma independiente: la conversación original, el plan interpretado y
la evidencia reunida. La conversación original es la fuente de verdad sobre lo que
pidió el usuario; el plan puede estar equivocado y no debe validarse por defecto.

Reglas obligatorias:
- Primero compará el plan con la conversación original. Si el plan agregó, quitó o
  cambió materialmente un período, competición, población, filtro, unidad, entidad o
  alcance que no surge de la conversación, consideralo una interpretación defectuosa.
- No uses cantidad fija de fuentes como criterio de verdad. Una fuente primaria y
  explícita puede ser suficiente; muchas fuentes irrelevantes no lo son.
- Una fuente solo puede ser relevante si responde a la misma entidad, métrica,
  alcance, período, competición/contexto y unidad que requiere la conversación. Un
  dato de un subconjunto no demuestra automáticamente el total del conjunto.
- Comprobá fecha y actualidad cuando el dato pueda cambiar con el tiempo.
- Si dos cifras parecen contradictorias, primero evaluá si en realidad miden cosas
  distintas. No las presentes como discrepancia del mismo dato sin demostrarlo.
- Priorizá evidencia primaria, explícita, reciente y directamente relacionada con la
  pregunta.
- Esta etapa se usa cuando el plan requiere evidencia externa/local. No completes
  huecos de esa evidencia con conocimiento de memoria del modelo.
- Si falta información y una búsqueda adicional puede resolverla, proponé UNA nueva
  consulta web semánticamente dirigida a lo que falta y al alcance original. No uses
  reglas particulares para nombres, frases, métricas o deportes.
- Si la evidencia permite sostener una interpretación razonable del alcance original,
  marcala en resolved_scope y continuá. Pedí aclaración solo cuando buscar más no
  resolvería una ambigüedad material o cuando elegir por cuenta propia pueda producir
  una acción sensible o destructiva.

Devolvé exclusivamente un objeto JSON válido con estas claves:
{
  "sufficient": true,
  "relevant_source_ids": ["W1", "L1"],
  "discarded_source_ids": ["W2"],
  "missing_information": ["..."],
  "follow_up_web_query": null,
  "needs_clarification": false,
  "clarifying_question": null,
  "resolved_scope": "alcance que realmente respalda la evidencia" or null,
  "reason": "explicación breve"
}
""".strip()


def _extract_json_object(value: str) -> dict[str, object]:
    clean = value.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end < start:
        raise SemanticOrchestrationError("El modelo no devolvió un objeto JSON.")
    try:
        payload = json.loads(clean[start : end + 1])
    except ValueError as exc:
        raise SemanticOrchestrationError("El plan semántico no es JSON válido.") from exc
    if not isinstance(payload, dict):
        raise SemanticOrchestrationError("El plan semántico debe ser un objeto JSON.")
    return payload


def _strings(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise SemanticOrchestrationError(f"{key} debe ser una lista.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _boolean(
    payload: dict[str, object],
    key: str,
    *,
    default: bool = False,
) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise SemanticOrchestrationError(f"{key} debe ser booleano, no {type(value).__name__}.")


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticOrchestrationError(f"{key} debe ser texto o null.")
    clean = value.strip()
    return clean or None


def _evidence_policy(payload: dict[str, object]) -> str:
    value = str(payload.get("evidence_policy") or "").strip().lower()
    if value not in VALID_EVIDENCE_POLICIES:
        allowed = ", ".join(sorted(VALID_EVIDENCE_POLICIES))
        raise SemanticOrchestrationError(
            f"evidence_policy debe ser uno de: {allowed}."
        )
    return value


def parse_semantic_plan(value: str) -> SemanticPlan:
    payload = _extract_json_object(value)
    objective = str(payload.get("objective") or "").strip()
    if not objective:
        raise SemanticOrchestrationError("El plan semántico no definió un objetivo.")
    try:
        recommended_mode = SelectedMode(str(payload.get("recommended_mode", "quick")))
    except ValueError as exc:
        raise SemanticOrchestrationError("El plan recomendó un modo inválido.") from exc
    return SemanticPlan(
        objective=objective,
        entities=_strings(payload, "entities"),
        constraints=_strings(payload, "constraints"),
        references=_strings(payload, "references"),
        information_needed=_strings(payload, "information_needed"),
        ambiguities=_strings(payload, "ambiguities"),
        evidence_policy=_evidence_policy(payload),
        use_web=_boolean(payload, "use_web"),
        use_local_data=_boolean(payload, "use_local_data"),
        use_calculator=_boolean(payload, "use_calculator"),
        use_chart=_boolean(payload, "use_chart"),
        needs_clarification=_boolean(payload, "needs_clarification"),
        clarifying_question=_optional_string(payload, "clarifying_question"),
        web_query=_optional_string(payload, "web_query"),
        local_document_names=_strings(payload, "local_document_names"),
        recommended_mode=recommended_mode,
        reason=str(payload.get("reason") or "Planificación semántica.").strip(),
    )


def parse_evidence_review(value: str) -> EvidenceReview:
    payload = _extract_json_object(value)
    return EvidenceReview(
        sufficient=_boolean(payload, "sufficient"),
        relevant_source_ids=_strings(payload, "relevant_source_ids"),
        discarded_source_ids=_strings(payload, "discarded_source_ids"),
        missing_information=_strings(payload, "missing_information"),
        follow_up_web_query=_optional_string(payload, "follow_up_web_query"),
        needs_clarification=_boolean(payload, "needs_clarification"),
        clarifying_question=_optional_string(payload, "clarifying_question"),
        resolved_scope=_optional_string(payload, "resolved_scope"),
        reason=str(payload.get("reason") or "Revisión semántica de evidencia.").strip(),
    )


def document_catalog(documents: Sequence[KnowledgeDocument]) -> str:
    if not documents:
        return "No hay documentos locales cargados."
    return "\n".join(
        f"- {document.name} ({len(document.content)} caracteres)" for document in documents
    )


def _capability_context(
    *,
    web_available: bool,
    documents: Sequence[KnowledgeDocument],
    sport: SportContext,
) -> str:
    return (
        f"Fecha actual del sistema: {date.today().isoformat()}\n"
        f"Contexto deportivo seleccionado: {sport.value}\n"
        f"Búsqueda web disponible: {'sí' if web_available else 'no'}\n"
        "Catálogo de documentos locales disponibles:\n"
        f"{document_catalog(documents)}"
    )


async def create_semantic_plan(
    provider: ModelProvider,
    messages: Sequence[ChatMessage],
    *,
    web_available: bool,
    documents: Sequence[KnowledgeDocument],
    sport: SportContext,
    on_model_result: ModelResultCallback | None = None,
) -> SemanticPlan:
    system_prompt = PLANNER_PROMPT + "\n\n" + _capability_context(
        web_available=web_available,
        documents=documents,
        sport=sport,
    )
    recent = list(messages[-12:])
    result = await provider.chat(
        mode=SelectedMode.QUICK,
        messages=recent,
        system_prompt=system_prompt,
        structured=True,
        reasoning_effort="low",
    )
    if on_model_result is not None:
        on_model_result("planning", result)
    return parse_semantic_plan(result.content)


def conservative_fallback_plan(
    messages: Sequence[ChatMessage],
    *,
    web_available: bool,
    documents: Sequence[KnowledgeDocument],
) -> SemanticPlan:
    """Safe fallback when structured planning fails without lexical classification."""

    question = messages[-1].content.strip()
    if documents and web_available:
        policy = "mixed"
    elif documents:
        policy = "local"
    elif web_available:
        policy = "external"
    else:
        policy = "model_knowledge"
    return SemanticPlan(
        objective=question,
        entities=(),
        constraints=(),
        references=(),
        information_needed=("Información suficiente para responder la pregunta original.",),
        ambiguities=(),
        evidence_policy=policy,
        use_web=web_available,
        use_local_data=bool(documents),
        use_calculator=False,
        use_chart=False,
        needs_clarification=False,
        clarifying_question=None,
        web_query=question if web_available else None,
        local_document_names=tuple(document.name for document in documents),
        recommended_mode=SelectedMode.QUICK,
        reason="Fallback conservador: recopilar evidencia sin clasificar por palabras.",
    )


def collect_local_evidence(
    documents: Sequence[KnowledgeDocument],
    plan: SemanticPlan,
    *,
    original_user_request: str = "",
    max_characters: int = 12_000,
) -> tuple[LocalEvidence, ...]:
    if not plan.use_local_data or not documents or max_characters <= 0:
        return ()
    retrieval_query = "\n".join(
        item
        for item in (
            original_user_request,
            plan.objective,
            *plan.entities,
            *plan.constraints,
            *plan.information_needed,
        )
        if item
    )
    chunks = retrieve_local_chunks(
        documents,
        retrieval_query,
        selected_names=plan.local_document_names,
        max_characters=max_characters,
        max_chunks=12,
    )
    return tuple(
        LocalEvidence(
            source_id=f"L{index}",
            document_name=chunk.document_name,
            content=chunk.content,
            truncated=chunk.truncated,
            chunk_index=chunk.chunk_index,
        )
        for index, chunk in enumerate(chunks, start=1)
    )


def merge_web_sources(
    existing: Sequence[WebSource], incoming: Sequence[WebSource]
) -> tuple[WebSource, ...]:
    merged: list[WebSource] = list(existing)
    seen = {source.url for source in merged}
    for source in incoming:
        if source.url in seen:
            continue
        seen.add(source.url)
        merged.append(source)
    return tuple(merged)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1].rstrip() + "…"


def _conversation_input(messages: Sequence[ChatMessage]) -> str:
    if not messages:
        return "(no disponible)"
    blocks = [
        f"{message.role.upper()}: {_clip(message.content, 1_500)}"
        for message in messages[-12:]
    ]
    return _clip("\n".join(blocks), 5_000)


def _review_input(
    plan: SemanticPlan,
    web_sources: Sequence[WebSource],
    local_evidence: Sequence[LocalEvidence],
    messages: Sequence[ChatMessage] = (),
) -> str:
    plan_payload = {
        "objective": plan.objective,
        "entities": list(plan.entities),
        "constraints": list(plan.constraints),
        "information_needed": list(plan.information_needed),
        "ambiguities": list(plan.ambiguities),
        "evidence_policy": plan.evidence_policy,
    }
    parts = [
        "CONVERSACIÓN ORIGINAL (fuente de verdad sobre el pedido):\n"
        + _conversation_input(messages),
        "PLAN INTERPRETADO (puede contener errores y debe auditarse):\n"
        + _clip(json.dumps(plan_payload, ensure_ascii=False), 3_500),
    ]

    if web_sources:
        web_blocks = [
            f"W{index} | {source.title}\nURL: {source.url}\n"
            f"Dominio: {source.domain}\nExtracto: {_clip(source.excerpt, 1_100)}"
            for index, source in enumerate(web_sources, start=1)
        ]
        parts.append("EVIDENCIA WEB:\n" + "\n\n".join(web_blocks))
    else:
        parts.append("EVIDENCIA WEB: ninguna.")

    if local_evidence:
        local_blocks = [
            f"{item.source_id} | {item.document_name}"
            f" | fragmento {item.chunk_index + 1 if item.chunk_index is not None else '?'}"
            f"{' | TRUNCADO' if item.truncated else ''}\n{_clip(item.content, 2_000)}"
            for item in local_evidence
        ]
        parts.append("EVIDENCIA LOCAL:\n" + "\n\n".join(local_blocks))
    else:
        parts.append("EVIDENCIA LOCAL: ninguna.")

    return _clip("\n\n".join(parts), MAX_REVIEW_INPUT_CHARACTERS)


async def review_evidence(
    provider: ModelProvider,
    plan: SemanticPlan,
    web_sources: Sequence[WebSource],
    local_evidence: Sequence[LocalEvidence],
    *,
    messages: Sequence[ChatMessage] = (),
    reasoning_effort: str = "low",
    on_model_result: ModelResultCallback | None = None,
    stage_name: str = "review",
) -> EvidenceReview:
    if plan.evidence_policy == "model_knowledge" and not web_sources and not local_evidence:
        return EvidenceReview(
            sufficient=True,
            relevant_source_ids=(),
            discarded_source_ids=(),
            missing_information=(),
            follow_up_web_query=None,
            needs_clarification=False,
            clarifying_question=None,
            resolved_scope="Conocimiento general estable; no requiere evidencia externa.",
            reason="La política semántica permite responder con conocimiento general del modelo.",
        )
    if not web_sources and not local_evidence:
        return EvidenceReview(
            sufficient=False,
            relevant_source_ids=(),
            discarded_source_ids=(),
            missing_information=plan.information_needed or ("Evidencia verificable.",),
            follow_up_web_query=plan.web_query if plan.use_web else None,
            needs_clarification=False,
            clarifying_question=None,
            resolved_scope=None,
            reason="No se reunió evidencia requerida para revisar.",
        )
    result = await provider.chat(
        mode=SelectedMode.QUICK,
        messages=[
            ChatMessage(
                role="user",
                content=_review_input(plan, web_sources, local_evidence, messages),
            )
        ],
        system_prompt=REVIEW_PROMPT + f"\n\nFecha actual: {date.today().isoformat()}",
        structured=True,
        reasoning_effort=reasoning_effort,
    )
    if on_model_result is not None:
        on_model_result(stage_name, result)
    return parse_evidence_review(result.content)


def format_reasoning_context(
    plan: SemanticPlan,
    review: EvidenceReview,
    web_sources: Sequence[WebSource],
    local_evidence: Sequence[LocalEvidence],
    *,
    original_user_request: str | None = None,
) -> str:
    header = {
        "original_user_request": original_user_request,
        "planned_objective": plan.objective,
        "entities": list(plan.entities),
        "constraints": list(plan.constraints),
        "evidence_policy": plan.evidence_policy,
        "resolved_scope": review.resolved_scope,
        "evidence_sufficient": review.sufficient,
        "missing_information": list(review.missing_information),
        "relevant_source_ids": list(review.relevant_source_ids),
        "discarded_source_ids": list(review.discarded_source_ids),
    }
    sections = [
        "CONTEXTO DE ORQUESTACIÓN SEMÁNTICA (no lo repitas al usuario):\n"
        + json.dumps(header, ensure_ascii=False),
        "La petición original del usuario manda sobre el plan. Si entran en conflicto, "
        "no adoptes silenciosamente el alcance del plan. Usá solo evidencia compatible "
        "con el pedido original y con la revisión. No conviertas fuentes descartadas en "
        "hechos.",
    ]
    if plan.evidence_policy == "model_knowledge":
        sections.append(
            "POLÍTICA DE EVIDENCIA: podés usar conocimiento general estable del modelo "
            "para responder. No inventes fuentes ni presentes como actual un dato que "
            "pueda haber cambiado; si durante la respuesta detectás que realmente hace "
            "falta actualidad o datos privados que no están disponibles, explicitá ese límite."
        )
    else:
        sections.append(
            "POLÍTICA DE EVIDENCIA: esta consulta requiere evidencia externa/local. Si "
            "evidence_sufficient es false, no presentes una cifra o conclusión como "
            "confirmada. No completes huecos con memoria del modelo. Si hay evidencia "
            "parcial útil, identificá exactamente qué alcance sí respalda. Citá evidencia "
            "web como [W1], [W2], etc."
        )
    if web_sources:
        sections.append(
            "FUENTES WEB:\n"
            + "\n\n".join(
                f"[W{index}] {source.title}\nURL: {source.url}\nExtracto: {source.excerpt}"
                for index, source in enumerate(web_sources, start=1)
            )
        )
    if local_evidence:
        sections.append(
            "DATOS LOCALES:\n"
            + "\n\n".join(
                f"[{item.source_id}] {item.document_name}"
                f" | fragmento {item.chunk_index + 1 if item.chunk_index is not None else '?'}"
                f"{' (extracto truncado)' if item.truncated else ''}\n{item.content}"
                for item in local_evidence
            )
        )
    return "\n\n".join(sections)
