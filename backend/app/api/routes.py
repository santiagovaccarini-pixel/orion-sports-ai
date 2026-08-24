from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.core.config import get_settings
from backend.app.core.prompt import build_system_prompt
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    RequestedMode,
    SportContext,
    StatusResponse,
    SystemSnapshotResponse,
)
from backend.app.domain.models import SelectedMode
from backend.app.providers.ollama import (
    ModelNotInstalledError,
    OllamaClient,
    OllamaUnavailableError,
)
from backend.app.services.mode_router import recommend_mode
from backend.app.services.resource_guard import (
    lower_ollama_priority,
    maintain_ollama_priority,
    read_snapshot,
)
from backend.app.services.resource_policy import evaluate_resources
from backend.app.services.response_policy import (
    response_style_instruction,
    response_token_budget,
)
from backend.app.services.knowledge_base import (
    KnowledgeBase,
    KnowledgeDocument,
    csv_calculation,
    csv_chart,
    csv_chart_is_ambiguous,
    csv_overview,
    csv_query_is_ambiguous,
    csv_tool_result,
    format_context,
)
from backend.app.services.semantic_planner import (
    create_semantic_plan,
    format_semantic_context,
)
from backend.app.services.semantic_retriever import search_with_intent
from backend.app.services.web_research import format_sources, research
from backend.app.services.orchestrator import OrchestrationPlan, create_plan


router = APIRouter()
chat_lock = asyncio.Lock()


def require_api_key(api_key: str | None = Header(default=None, alias="X-Orion-Api-Key")) -> None:
    configured_key = get_settings().api_key
    if configured_key is not None and api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_api_key", "message": "La clave de Orion no es válida."},
        )


@dataclass(frozen=True, slots=True)
class PreparedChat:
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    sport: SportContext
    semantic_plan: SemanticPlan


def _ndjson(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _model_is_installed(model: str, installed_models: tuple[str, ...]) -> bool:
    return any(
        installed == model
        or installed.startswith(f"{model}:")
        or model.startswith(f"{installed}:")
        for installed in installed_models
    )


def _knowledge_prompt(
    request: ChatRequest,
    prepared: PreparedChat,
    web_context: str = "",
) -> str:
    base_prompt = build_system_prompt(
        prepared.sport,
        prepared.selected_mode,
        request.messages[-1].content,
    )
    knowledge = KnowledgeBase(Path(get_settings().knowledge_path))
    query = request.messages[-1].content
    web_requested = bool(web_context)
    plan = _create_orchestration_plan(
        query,
        semantic_plan=prepared.semantic_plan,
        web_requested=web_requested,
    )
    csv_documents = [
        document
        for document in knowledge.list_documents()
        if document.name.lower().endswith(".csv")
    ]
    ambiguous_csv = plan.use_local_data and (
        prepared.semantic_plan.requires_clarification
        or any(csv_query_is_ambiguous(document.content, query) for document in csv_documents)
    )
    context = (
        []
        if ambiguous_csv
        else search_with_intent(
            knowledge,
            query,
            prepared.semantic_plan,
            limit=12,
        )
    )
    max_context = 2_000 if prepared.selected_mode is SelectedMode.QUICK else 5_000
    formatted = format_context(context, max_characters=max_context)
    semantic_context = format_semantic_context(prepared.semantic_plan)
    response_instruction = response_style_instruction(
        prepared.selected_mode,
        prepared.semantic_plan,
        query,
    )
    calculations = ""
    tool_results = ""
    overviews = ""
    for document in csv_documents:
        if plan.use_local_data:
            overviews += csv_overview(document.content, document.name)
        if plan.use_local_data and not ambiguous_csv:
            calculations += csv_calculation(document.content, query)
            tool_results += csv_tool_result(document.content, query, document.name)
    if ambiguous_csv:
        overviews += (
            "ACLARACIÓN OBLIGATORIA: la intención requiere datos locales pero falta "
            "definir una variable indispensable (jugador, columna, período, métrica o "
            "cálculo). No inventes la selección. Pedí únicamente el dato que falta.\n"
        )
    if plan.use_chart and any(csv_chart_is_ambiguous(document.content, query) for document in csv_documents):
        overviews += (
            "ACLARACIÓN PARA EL GRÁFICO: indicá qué jugador o entidad, qué columna "
            "y qué período querés visualizar. Orion puede generar un gráfico local "
            "cuando esos datos estén definidos.\n"
        )
    chart = _knowledge_chart(request) if plan.use_chart else None
    if chart:
        overviews += (
            "CAPACIDAD VISUAL ACTIVA: Orion ya generó un gráfico local verificable "
            "para esta consulta y lo mostrará en la interfaz. No digas que no podés "
            "generar gráficos. Describí el gráfico usando exclusivamente estos datos: "
            f"{json.dumps(chart, ensure_ascii=False)}\n"
        )
    extra = "\n".join(
        item
        for item in (
            semantic_context,
            response_instruction,
            web_context,
            overviews,
            tool_results,
            calculations,
            formatted,
        )
        if item
    )
    return f"{base_prompt}\n\n{extra}" if extra else base_prompt


def _create_orchestration_plan(
    query: str,
    *,
    semantic_plan: SemanticPlan | None = None,
    web_requested: bool = False,
) -> OrchestrationPlan:
    documents = KnowledgeBase(Path(get_settings().knowledge_path)).list_documents()
    plan = create_plan(
        query,
        has_local_documents=bool(documents),
        semantic_plan=semantic_plan,
    )
    if web_requested and not plan.use_web:
        return OrchestrationPlan(
            plan.intent,
            use_web=True,
            use_local_data=False,
            use_calculator=False,
            use_chart=False,
            needs_clarification=False,
            reason="La búsqueda web tiene prioridad para esta consulta.",
        )
    return plan


async def _web_context(request: ChatRequest, prepared: PreparedChat) -> str:
    settings = get_settings()
    query = request.messages[-1].content
    plan = _create_orchestration_plan(query, semantic_plan=prepared.semantic_plan)
    if not settings.web_enabled or not plan.use_web:
        return ""

    research_query = query
    if prepared.semantic_plan.referenced_previous_context:
        research_query = (
            prepared.semantic_plan.retrieval_queries[0]
            if prepared.semantic_plan.retrieval_queries
            else prepared.semantic_plan.user_goal
        )
    sources = await research(
        research_query,
        allowed_domains=settings.web_allowed_domains,
        minimum_sources=settings.web_minimum_sources,
    )
    return format_sources(sources, minimum_sources=settings.web_minimum_sources)


def _web_is_insufficient(context: str) -> bool:
    return context.startswith("INVESTIGACIÓN WEB INSUFICIENTE:")


def _insufficient_web_response(context: str) -> str:
    return (
        "Encontré información preliminar, pero todavía no puedo confirmarla con "
        "suficiente respaldo.\n\n"
        f"{context}\n\n"
        "Tomá los extractos como datos provisionales de las fuentes encontradas; "
        "no los presento como un hecho confirmado ni completo. No voy a calcular "
        "ni completar lo que falta con una suposición. Podés pedir una búsqueda "
        "más amplia, agregar una fuente específica o indicar un período exacto."
    )


def _knowledge_chart(request: ChatRequest) -> dict[str, object] | None:
    query = request.messages[-1].content
    base = KnowledgeBase(Path(get_settings().knowledge_path))
    for document in base.list_documents():
        if document.name.lower().endswith(".csv"):
            chart = csv_chart(document.content, query, document.name)
            if chart:
                return chart
    return None


async def _prepare_chat(request: ChatRequest, *, preflight_model: bool) -> PreparedChat:
    settings = get_settings()
    documents = KnowledgeBase(Path(settings.knowledge_path)).list_documents()
    semantic_plan = await create_semantic_plan(
        settings,
        request.messages,
        request.sport,
        has_local_documents=bool(documents),
    )
    recommendation = recommend_mode(request.messages, semantic_plan)
    selected_mode = (
        recommendation.mode
        if request.mode is RequestedMode.AUTO
        else SelectedMode(request.mode.value)
    )
    snapshot = read_snapshot()
    resource_decision = evaluate_resources(selected_mode, snapshot)

    if resource_decision.requires_confirmation and not request.allow_busy:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "resource_confirmation_required",
                "message": (
                    "Orion recomienda esperar o utilizar el modo Rápido para no afectar "
                    "las demás aplicaciones."
                ),
                "reasons": list(resource_decision.reasons),
                "selected_mode": selected_mode.value,
                "recommended_mode": recommendation.mode.value,
                "snapshot": asdict(snapshot),
            },
        )

    model = (
        settings.quick_model
        if selected_mode is SelectedMode.QUICK
        else settings.deep_model
    )
    if preflight_model:
        ollama = await OllamaClient(settings).status()
        if not ollama.online:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "ollama_unavailable",
                    "message": "No se pudo conectar con Ollama en esta computadora.",
                },
            )
        if not _model_is_installed(model, ollama.installed_models):
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail={
                    "code": "model_not_installed",
                    "message": (
                        f"Falta instalar {model}. Orion no descargará modelos sin tu autorización."
                    ),
                    "model": model,
                },
            )

    return PreparedChat(
        selected_mode=selected_mode,
        recommended_mode=recommendation.mode,
        recommendation_reason=recommendation.reason,
        model=model,
        sport=request.sport,
        semantic_plan=semantic_plan,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentResponse], dependencies=[Depends(require_api_key)])
async def list_knowledge_documents() -> list[KnowledgeDocumentResponse]:
    documents = KnowledgeBase(Path(get_settings().knowledge_path)).list_documents()
    return [KnowledgeDocumentResponse(id=item.id, name=item.name, characters=len(item.content)) for item in documents]


@router.post("/knowledge/documents", response_model=KnowledgeDocumentResponse, dependencies=[Depends(require_api_key)])
async def add_knowledge_document(request: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
    document_id = hashlib.sha256(
        f"{request.name}\0{request.content}".encode("utf-8")
    ).hexdigest()[:16]
    document = KnowledgeDocument(document_id, request.name.strip(), request.content.strip())
    KnowledgeBase(Path(get_settings().knowledge_path)).add_document(document)
    return KnowledgeDocumentResponse(id=document.id, name=document.name, characters=len(document.content))


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_api_key)])
async def system_status() -> StatusResponse:
    settings = get_settings()
    ollama = await OllamaClient(settings).status()
    return StatusResponse(
        version=settings.version,
        ollama_online=ollama.online,
        installed_models=list(ollama.installed_models),
        loaded_models=list(ollama.loaded_models),
        quick_model=settings.quick_model,
        deep_model=settings.deep_model,
        quick_threads=settings.quick_threads,
        deep_threads=settings.deep_threads,
        snapshot=SystemSnapshotResponse(**asdict(read_snapshot())),
        web_enabled=settings.web_enabled,
        web_minimum_sources=settings.web_minimum_sources,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    prepared = await _prepare_chat(request, preflight_model=False)
    web_context = await _web_context(request, prepared)
    generation_budget = response_token_budget(
        settings,
        prepared.selected_mode,
        prepared.semantic_plan,
        request.messages[-1].content,
    )
    if _web_is_insufficient(web_context):
        return ChatResponse(
            content=_insufficient_web_response(web_context),
            sport=prepared.sport,
            selected_mode=prepared.selected_mode,
            recommended_mode=prepared.recommended_mode,
            recommendation_reason=prepared.recommendation_reason,
            model=prepared.model,
            total_duration_ms=None,
            load_duration_ms=None,
            prompt_eval_duration_ms=None,
            eval_duration_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            tokens_per_second=None,
            thread_limit=0,
        )

    try:
        async with chat_lock:
            lower_ollama_priority()
            priority_stop = asyncio.Event()
            priority_task = asyncio.create_task(
                maintain_ollama_priority(priority_stop)
            )
            try:
                result = await OllamaClient(settings).chat(
                    model=prepared.model,
                    mode=prepared.selected_mode,
                    messages=request.messages,
                    system_prompt=_knowledge_prompt(request, prepared, web_context),
                    max_tokens_override=generation_budget,
                )
            finally:
                priority_stop.set()
                await priority_task
    except ModelNotInstalledError as exc:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail={
                "code": "model_not_installed",
                "message": f"Falta instalar {exc.model}. Orion no descargará modelos sin tu autorización.",
                "model": exc.model,
            },
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "ollama_unavailable", "message": str(exc)},
        ) from exc

    return ChatResponse(
        content=result.content,
        sport=prepared.sport,
        selected_mode=prepared.selected_mode,
        recommended_mode=prepared.recommended_mode,
        recommendation_reason=prepared.recommendation_reason,
        model=prepared.model,
        total_duration_ms=result.total_duration_ms,
        load_duration_ms=result.load_duration_ms,
        prompt_eval_duration_ms=result.prompt_eval_duration_ms,
        eval_duration_ms=result.eval_duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tokens_per_second=result.tokens_per_second,
        thread_limit=result.thread_limit,
    )


@router.post(
    "/chat/stream",
    dependencies=[Depends(require_api_key)],
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    settings = get_settings()
    prepared = await _prepare_chat(request, preflight_model=True)
    web_context = await _web_context(request, prepared)
    generation_budget = response_token_budget(
        settings,
        prepared.selected_mode,
        prepared.semantic_plan,
        request.messages[-1].content,
    )

    async def generate() -> AsyncIterator[bytes]:
        yield _ndjson(
            {
                "type": "meta",
                "selected_mode": prepared.selected_mode.value,
                "recommended_mode": prepared.recommended_mode.value,
                "recommendation_reason": prepared.recommendation_reason,
                "model": prepared.model,
                "sport": prepared.sport.value,
            }
        )
        if chart := _knowledge_chart(request):
            yield _ndjson({"type": "chart", "chart": chart})
        if _web_is_insufficient(web_context):
            yield _ndjson(
                {"type": "content", "content": _insufficient_web_response(web_context)}
            )
            yield _ndjson(
                {
                    "type": "done",
                    "total_duration_ms": None,
                    "load_duration_ms": None,
                    "prompt_eval_duration_ms": None,
                    "eval_duration_ms": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "tokens_per_second": None,
                    "thread_limit": 0,
                }
            )
            return

        try:
            async with chat_lock:
                lower_ollama_priority()
                priority_stop = asyncio.Event()
                priority_task = asyncio.create_task(
                    maintain_ollama_priority(priority_stop)
                )
                try:
                    async for event in OllamaClient(settings).chat_stream(
                        model=prepared.model,
                        mode=prepared.selected_mode,
                        messages=request.messages,
                        system_prompt=_knowledge_prompt(request, prepared, web_context),
                        max_tokens_override=generation_budget,
                    ):
                        if event.content:
                            yield _ndjson(
                                {"type": "content", "content": event.content}
                            )
                        if event.done:
                            yield _ndjson(
                                {
                                    "type": "done",
                                    "total_duration_ms": event.total_duration_ms,
                                    "load_duration_ms": event.load_duration_ms,
                                    "prompt_eval_duration_ms": (
                                        event.prompt_eval_duration_ms
                                    ),
                                    "eval_duration_ms": event.eval_duration_ms,
                                    "prompt_tokens": event.prompt_tokens,
                                    "completion_tokens": event.completion_tokens,
                                    "tokens_per_second": event.tokens_per_second,
                                    "thread_limit": event.thread_limit,
                                }
                            )
                finally:
                    priority_stop.set()
                    await priority_task
        except asyncio.CancelledError:
            raise
        except ModelNotInstalledError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "model_not_installed",
                    "message": f"Falta instalar {exc.model}.",
                }
            )
        except OllamaUnavailableError as exc:
            yield _ndjson(
                {
                    "type": "error",
                    "code": "ollama_unavailable",
                    "message": str(exc),
                }
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
