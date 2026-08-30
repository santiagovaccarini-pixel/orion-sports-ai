from __future__ import annotations

import asyncio
from datetime import date
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.app.core.config import get_settings
from backend.app.core.identity import direct_creator_answer
from backend.app.core.prompt import build_system_prompt
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    MemoryEntryRequest,
    MemoryEntryResponse,
    MemorySuggestionRequest,
    MemorySuggestionResponse,
    RequestedMode,
    SportContext,
    StatusResponse,
    SystemSnapshotResponse,
)
from backend.app.providers.model_provider import (
    ModelProvider,
    ModelProviderConfigurationError,
    ModelProviderModelError,
    ModelProviderUnavailableError,
    create_model_provider,
)
from backend.app.services.diagnostic_trace import DiagnosticTrace, diagnostic_traces
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
from backend.app.services.database import DatabaseUnavailableError
from backend.app.services.knowledge_repository import create_knowledge_base
from backend.app.services.memory_repository import (
    MemoryRepository,
    create_memory_repository,
)
from backend.app.services.memory_store import MemoryEntry, format_memory_context
from backend.app.services.memory_suggestions import suggest_memories
from backend.app.services.mode_router import recommend_mode
from backend.app.services.orchestrator import OrchestrationPlan, create_plan
from backend.app.services.rate_limit import (
    SlidingWindowRateLimiter,
    chat_rate_limiter,
    upload_rate_limiter,
)
from backend.app.services.reasoning_pipeline import ReasoningBundle, build_reasoning_bundle
from backend.app.services.semantic_tools import audit_numeric_support
from backend.app.services.resource_guard import (
    lower_ollama_priority,
    maintain_ollama_priority,
    read_snapshot,
)
from backend.app.services.resource_policy import evaluate_resources
from backend.app.services.web_research import format_sources, research


logger = logging.getLogger(__name__)

router = APIRouter()
chat_lock = asyncio.Lock()
# Bounds how many chat requests can be in flight at once regardless of provider.
# chat_lock below only ever applied to local (Ollama) inference to protect the
# developer's machine, which left the deployed cloud path with no cap at all.
_chat_slots = asyncio.Semaphore(get_settings().max_concurrent_chats)


def _client_id(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind a reverse proxy (Render) this is the proxy's address, not the end
    user's: uvicorn only honours X-Forwarded-For from `forwarded_allow_ips`,
    which defaults to 127.0.0.1. The limiter is therefore effectively a single
    shared bucket for the whole deployment.

    That is deliberate for now. Trusting X-Forwarded-For from any peer would
    make the limit per-user but also trivially bypassable by spoofing the
    header, and an unbypassable ceiling on total load is worth more here than
    per-user fairness while Orion has one operator. Revisit alongside real
    user accounts, where fairness starts to matter and identity is
    authenticated rather than inferred from an address.
    """

    client = request.client
    return client.host if client is not None else "unknown"


async def _enforce_rate_limit(
    request: Request,
    limiter: SlidingWindowRateLimiter,
    limit: int,
) -> None:
    decision = await limiter.check(_client_id(request), limit=limit)
    if decision.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "code": "rate_limited",
            "message": (
                "Demasiadas solicitudes seguidas. Esperá unos segundos antes "
                "de volver a intentar."
            ),
        },
        headers={"Retry-After": str(decision.retry_after_seconds)},
    )


async def limit_chat_rate(request: Request) -> None:
    await _enforce_rate_limit(
        request, chat_rate_limiter, get_settings().rate_limit_chat_per_minute
    )


async def limit_upload_rate(request: Request) -> None:
    await _enforce_rate_limit(
        request, upload_rate_limiter, get_settings().rate_limit_uploads_per_minute
    )


def require_api_key(
    api_key: str | None = Header(default=None, alias="X-Orion-Api-Key"),
) -> None:
    configured_key = get_settings().api_key
    if configured_key is None:
        # Fail closed: an unconfigured key must never mean "no auth required".
        # Silently allowing every request through here was a real, live
        # misconfiguration hazard - refuse everything until an operator sets
        # ORION_API_KEY, instead of exposing every endpoint by accident.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "server_misconfigured",
                "message": (
                    "Orion no tiene ORION_API_KEY configurada; el servicio "
                    "rechaza todas las solicitudes hasta que se configure."
                ),
            },
        )
    if api_key is None or not hmac.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "La clave de Orion no es válida.",
            },
        )


@dataclass(frozen=True, slots=True)
class PreparedChat:
    selected_mode: SelectedMode
    recommended_mode: SelectedMode
    recommendation_reason: str
    model: str
    sport: SportContext


def _identity_prepared(request: ChatRequest) -> PreparedChat:
    return PreparedChat(
        selected_mode=SelectedMode.QUICK,
        recommended_mode=SelectedMode.QUICK,
        recommendation_reason="Identidad institucional de Orion.",
        model="orion-institutional-identity",
        sport=request.sport,
    )


def _ndjson(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _provider_or_http_error() -> ModelProvider:
    try:
        return create_model_provider(get_settings())
    except ModelProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "provider_configuration_error",
                "message": str(exc),
            },
        ) from exc


def _provider_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelProviderModelError):
        return HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail={
                "code": "model_not_installed",
                "message": (
                    f"Falta instalar o habilitar {exc.model}. "
                    "Orion no descargará ni contratará modelos sin autorización."
                ),
                "model": exc.model,
            },
        )
    if isinstance(exc, ModelProviderConfigurationError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "provider_configuration_error",
                "message": str(exc),
            },
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "model_provider_unavailable",
            "message": str(exc),
        },
    )


@asynccontextmanager
async def _provider_runtime(provider: ModelProvider) -> AsyncIterator[None]:
    """Bound in-flight work, and apply PC protection for local inference only."""

    async with _chat_slots:
        if not provider.uses_local_resources:
            yield
            return
        async with chat_lock:
            lower_ollama_priority()
            priority_stop = asyncio.Event()
            priority_task = asyncio.create_task(maintain_ollama_priority(priority_stop))
            try:
                yield
            finally:
                priority_stop.set()
                await priority_task


async def _prepare_selected_chat(
    request: ChatRequest,
    *,
    provider: ModelProvider,
    selected_mode: SelectedMode,
    recommended_mode: SelectedMode,
    recommendation_reason: str,
    preflight_model: bool,
) -> PreparedChat:
    if provider.uses_local_resources:
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
                    "recommended_mode": recommended_mode.value,
                    "snapshot": asdict(snapshot),
                },
            )
    if preflight_model:
        try:
            await provider.preflight(selected_mode)
        except (
            ModelProviderConfigurationError,
            ModelProviderModelError,
            ModelProviderUnavailableError,
        ) as exc:
            raise _provider_http_exception(exc) from exc
    return PreparedChat(
        selected_mode=selected_mode,
        recommended_mode=recommended_mode,
        recommendation_reason=recommendation_reason,
        model=provider.model_for(selected_mode),
        sport=request.sport,
    )


async def _prepare_chat(
    request: ChatRequest,
    *,
    preflight_model: bool,
    provider: ModelProvider | None = None,
) -> PreparedChat:
    """Legacy preparation retained for rollback/local compatibility."""

    recommendation = recommend_mode(request.messages)
    selected_mode = (
        recommendation.mode
        if request.mode is RequestedMode.AUTO
        else SelectedMode(request.mode.value)
    )
    provider = provider or _provider_or_http_error()
    return await _prepare_selected_chat(
        request,
        provider=provider,
        selected_mode=selected_mode,
        recommended_mode=recommendation.mode,
        recommendation_reason=recommendation.reason,
        preflight_model=preflight_model,
    )


def _semantic_prompt(
    request: ChatRequest,
    prepared: PreparedChat,
    bundle: ReasoningBundle,
) -> str:
    base = build_system_prompt(
        prepared.sport,
        prepared.selected_mode,
        request.messages[-1].content,
    )
    return f"{base}\n\n{bundle.context}"


def _allowed_numeric_texts(
    request: ChatRequest,
    bundle: ReasoningBundle,
) -> list[str]:
    relevant_ids = {
        source_id.strip().upper() for source_id in bundle.review.relevant_source_ids
    }
    # Today's date reaches the model through the system prompt, not through any of
    # the sources below, so without it every answer that says "hoy es ..." looked
    # like an unsupported figure. Now that years are audited rather than tolerated
    # wholesale, that false positive would fire on almost every dated answer.
    today = date.today()
    texts = [
        f"{today.isoformat()} {today.day} {today.month} {today.year}",
        *(message.content for message in request.messages),
    ]
    if bundle.tool_context:
        texts.append(bundle.tool_context)
    texts.extend(item.content for item in bundle.local_evidence)
    texts.extend(
        source.excerpt
        for index, source in enumerate(bundle.web_sources, start=1)
        if f"W{index}" in relevant_ids or not bundle.review.audited
    )
    return texts


def _audit_final_answer(
    trace: DiagnosticTrace | None,
    request: ChatRequest,
    bundle: ReasoningBundle,
    answer: str,
) -> None:
    if trace is None or not answer:
        return
    unsupported = audit_numeric_support(
        answer, allowed_texts=_allowed_numeric_texts(request, bundle)
    )
    if unsupported:
        trace.record_guard(
            "unsupported_numbers_detected",
            "Cifras en la respuesta sin respaldo en mensajes, herramientas o "
            "evidencia aceptada: " + ", ".join(unsupported),
        )


def _legacy_knowledge_prompt(
    request: ChatRequest,
    prepared: PreparedChat,
    web_context: str = "",
) -> str:
    """Previous keyword-based tool path kept only as an explicit rollback mode."""

    base_prompt = build_system_prompt(
        prepared.sport,
        prepared.selected_mode,
        request.messages[-1].content,
    )
    knowledge = _knowledge_base()
    query = request.messages[-1].content
    web_requested = bool(web_context)
    plan = _create_orchestration_plan(query, web_requested=web_requested)
    csv_documents = [
        document
        for document in knowledge.list_documents()
        if document.name.lower().endswith(".csv")
    ]
    ambiguous_csv = plan.use_local_data and any(
        csv_query_is_ambiguous(document.content, query) for document in csv_documents
    )
    context = [] if ambiguous_csv else knowledge.search(query)
    max_context = 2_000 if prepared.selected_mode is SelectedMode.QUICK else 5_000
    formatted = format_context(context, max_characters=max_context)
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
            "ACLARACIÓN OBLIGATORIA: la petición no define qué jugador, columna, "
            "período o cálculo necesita el usuario. No analices ni resumas todavía. "
            "Preguntá qué dato desea consultar.\n"
        )
    if plan.use_chart and any(
        csv_chart_is_ambiguous(document.content, query) for document in csv_documents
    ):
        overviews += (
            "ACLARACIÓN PARA EL GRÁFICO: indicá qué entidad, columna y período querés "
            "visualizar.\n"
        )
    chart = _knowledge_chart(request) if plan.use_chart else None
    if chart:
        overviews += (
            "CAPACIDAD VISUAL ACTIVA: Orion ya generó un gráfico local verificable. "
            f"Describilo usando exclusivamente estos datos: {json.dumps(chart, ensure_ascii=False)}\n"
        )
    extra = "\n".join(
        item
        for item in (web_context, overviews, tool_results, calculations, formatted)
        if item
    )
    return f"{base_prompt}\n\n{extra}" if extra else base_prompt


# Backward-compatible alias used by existing tests and rollback tooling.
_knowledge_prompt = _legacy_knowledge_prompt


def _create_orchestration_plan(
    query: str,
    *,
    web_requested: bool = False,
) -> OrchestrationPlan:
    documents = _knowledge_base().list_documents()
    plan = create_plan(query, has_local_documents=bool(documents))
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


async def _web_context(request: ChatRequest) -> str:
    """Legacy web route retained for rollback mode."""

    settings = get_settings()
    query = request.messages[-1].content
    plan = _create_orchestration_plan(query)
    if not settings.web_enabled or not plan.use_web:
        return ""
    sources = await research(
        query,
        allowed_domains=settings.web_allowed_domains,
        minimum_sources=settings.web_minimum_sources,
        provider=settings.web_provider,
        tavily_api_key=settings.tavily_api_key,
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
        "no los presento como un hecho confirmado ni completo."
    )


def _knowledge_chart(request: ChatRequest) -> dict[str, object] | None:
    query = request.messages[-1].content
    base = _knowledge_base()
    for document in base.list_documents():
        if document.name.lower().endswith(".csv"):
            chart = csv_chart(document.content, query, document.name)
            if chart:
                return chart
    return None


def _knowledge_base() -> KnowledgeBase:
    settings = get_settings()
    return create_knowledge_base(
        database_url=settings.database_url,
        knowledge_path=settings.knowledge_path,
    )


def _documents() -> list[KnowledgeDocument]:
    return _knowledge_base().list_documents()


async def _memory_context() -> str:
    """Saved memory as prompt context, or empty when the user saved nothing.

    Runs in a thread: the store may be Postgres, and blocking the event loop on
    network I/O would stall every other in-flight request. A memory failure
    degrades to no memory rather than failing the whole chat - answering without
    saved context is far better than not answering.
    """

    def _load() -> str:
        return format_memory_context(_memory_store().list_entries())

    try:
        return await asyncio.to_thread(_load)
    except DatabaseUnavailableError:
        logger.warning("No se pudo leer la memoria; sigo sin ella", exc_info=True)
        return ""


def _direct_response(
    content: str,
    prepared: PreparedChat,
) -> ChatResponse:
    return ChatResponse(
        content=content,
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


def _done_event() -> bytes:
    return _ndjson(
        {
            "type": "done",
            "total_duration_ms": None,
            "load_duration_ms": None,
            "prompt_eval_duration_ms": None,
            "eval_duration_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "finish_reason": None,
            "reasoning_effort": None,
            "endpoint": None,
            "tokens_per_second": None,
            "thread_limit": 0,
        }
    )


def _start_trace(request: ChatRequest, enabled: bool) -> DiagnosticTrace | None:
    if not enabled:
        return None
    return diagnostic_traces.start(
        question=request.messages[-1].content,
        sport=request.sport.value,
        requested_mode=request.mode.value,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/knowledge/documents",
    response_model=list[KnowledgeDocumentResponse],
    dependencies=[Depends(require_api_key)],
)
async def list_knowledge_documents() -> list[KnowledgeDocumentResponse]:
    documents = _documents()
    return [
        KnowledgeDocumentResponse(
            id=item.id,
            name=item.name,
            characters=len(item.content),
        )
        for item in documents
    ]


@router.post(
    "/knowledge/documents",
    response_model=KnowledgeDocumentResponse,
    dependencies=[Depends(require_api_key), Depends(limit_upload_rate)],
)
async def add_knowledge_document(
    request: KnowledgeDocumentRequest,
) -> KnowledgeDocumentResponse:
    settings = get_settings()
    document_id = hashlib.sha256(
        f"{request.name}\0{request.content}".encode("utf-8")
    ).hexdigest()[:16]
    document = KnowledgeDocument(
        document_id,
        request.name.strip(),
        request.content.strip(),
    )
    knowledge = _knowledge_base()
    existing = knowledge.list_documents()
    # Re-uploading the same content keeps the same id and just replaces the
    # entry, so it must not count against the quota as a new document.
    replaces_existing = any(item.id == document.id for item in existing)
    if not replaces_existing and len(existing) >= settings.knowledge_max_documents:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "knowledge_document_limit_reached",
                "message": (
                    "Orion alcanzó el máximo de documentos guardados. Borrá "
                    "alguno antes de subir uno nuevo."
                ),
            },
        )
    projected_characters = sum(
        len(item.content) for item in existing if item.id != document.id
    ) + len(document.content)
    if projected_characters > settings.knowledge_max_total_characters:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "knowledge_storage_limit_reached",
                "message": (
                    "Orion alcanzó el límite total de almacenamiento de "
                    "documentos. Borrá alguno antes de subir uno nuevo."
                ),
            },
        )
    knowledge.add_document(document)
    return KnowledgeDocumentResponse(
        id=document.id,
        name=document.name,
        characters=len(document.content),
    )


@router.delete(
    "/knowledge/documents/{document_id}",
    dependencies=[Depends(require_api_key)],
)
async def delete_knowledge_document(document_id: str) -> dict[str, str]:
    """Withdraw a document from everything Orion will answer with.

    Before this existed the × on an attachment only cleared the chip in the
    composer. The document stayed in the knowledge base and kept feeding every
    later answer, so someone who believed they had withdrawn a file had not —
    and the quota error already told them to "borrá alguno" with no way to.
    """

    try:
        deleted = await asyncio.to_thread(
            _knowledge_base().delete_document, document_id
        )
    except DatabaseUnavailableError as exc:
        raise _memory_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "knowledge_document_not_found",
                "message": "Ese documento ya no está guardado.",
            },
        )
    return {"status": "deleted", "id": document_id}


@router.post(
    "/memory/suggestions",
    response_model=list[MemorySuggestionResponse],
    dependencies=[Depends(require_api_key), Depends(limit_chat_rate)],
)
async def suggest_memory_entries(
    request: MemorySuggestionRequest,
) -> list[MemorySuggestionResponse]:
    """Propose what to remember. Nothing is written here.

    Orion never saves on its own: this returns the exact sentences it would
    store, and the person decides whether any of them becomes memory.
    """

    provider = _provider_or_http_error()
    try:
        suggestions = await suggest_memories(
            provider,
            request.messages,
            request.answer,
            memory_context=await _memory_context(),
        )
    except (ModelProviderUnavailableError, ModelProviderConfigurationError):
        # A proposal is a convenience, never the point of the request: when the
        # model cannot be reached, offer nothing rather than failing the caller.
        logger.warning("No se pudieron proponer recuerdos", exc_info=True)
        return []
    return [
        MemorySuggestionResponse(content=item.content, reason=item.reason)
        for item in suggestions
    ]


def _memory_store() -> MemoryRepository:
    settings = get_settings()
    return create_memory_repository(
        database_url=settings.database_url,
        memory_path=settings.memory_path,
    )


def _memory_response(entry: MemoryEntry) -> MemoryEntryResponse:
    return MemoryEntryResponse(
        id=entry.id,
        content=entry.content,
        category=entry.category,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


def _memory_unavailable(exc: Exception) -> HTTPException:
    """Surface a storage outage as such instead of a generic 500.

    Memory is explicit user data: silently reporting success or an opaque
    failure would leave the user unsure whether their entry was saved.
    """

    logger.warning("La memoria no está disponible", exc_info=exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "memory_unavailable",
            "message": (
                "La memoria de Orion no está disponible en este momento. "
                "Intentá de nuevo en unos segundos."
            ),
        },
    )


@router.get(
    "/memory/entries",
    response_model=list[MemoryEntryResponse],
    dependencies=[Depends(require_api_key)],
)
async def list_memory_entries() -> list[MemoryEntryResponse]:
    try:
        entries = await asyncio.to_thread(_memory_store().list_entries)
    except DatabaseUnavailableError as exc:
        raise _memory_unavailable(exc) from exc
    return [_memory_response(entry) for entry in entries]


@router.post(
    "/memory/entries",
    response_model=MemoryEntryResponse,
    dependencies=[Depends(require_api_key), Depends(limit_upload_rate)],
)
async def add_memory_entry(request: MemoryEntryRequest) -> MemoryEntryResponse:
    settings = get_settings()
    store = _memory_store()

    def _save() -> tuple[MemoryEntry | None, bool]:
        if len(store.list_entries()) >= settings.memory_max_entries:
            return None, True
        entry_id = uuid4().hex[:16]
        return store.add_entry(entry_id, request.content, request.category), False

    try:
        entry, at_limit = await asyncio.to_thread(_save)
    except DatabaseUnavailableError as exc:
        raise _memory_unavailable(exc) from exc
    if at_limit or entry is None:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "memory_limit_reached",
                "message": (
                    "Orion alcanzó el máximo de recuerdos guardados. Borrá "
                    "alguno antes de agregar uno nuevo."
                ),
            },
        )
    return _memory_response(entry)


@router.delete(
    "/memory/entries/{entry_id}",
    dependencies=[Depends(require_api_key)],
)
async def delete_memory_entry(entry_id: str) -> dict[str, str]:
    try:
        deleted = await asyncio.to_thread(_memory_store().delete_entry, entry_id)
    except DatabaseUnavailableError as exc:
        raise _memory_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "memory_entry_not_found",
                "message": "Ese recuerdo ya no existe.",
            },
        )
    return {"status": "deleted", "id": entry_id}


@router.delete(
    "/memory/entries",
    dependencies=[Depends(require_api_key)],
)
async def delete_all_memory_entries() -> dict[str, str]:
    try:
        await asyncio.to_thread(_memory_store().delete_all)
    except DatabaseUnavailableError as exc:
        raise _memory_unavailable(exc) from exc
    return {"status": "deleted_all"}


@router.get(
    "/status",
    response_model=StatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def system_status() -> StatusResponse:
    settings = get_settings()
    provider = _provider_or_http_error()
    provider_status = await provider.status()
    return StatusResponse(
        version=settings.version,
        model_provider=provider.name,
        model_provider_online=provider_status.online,
        ollama_online=(provider.name == "ollama" and provider_status.online),
        installed_models=list(provider_status.installed_models),
        loaded_models=list(provider_status.loaded_models),
        quick_model=provider.model_for(SelectedMode.QUICK),
        deep_model=provider.model_for(SelectedMode.DEEP),
        quick_threads=settings.quick_threads if provider.uses_local_resources else 0,
        deep_threads=settings.deep_threads if provider.uses_local_resources else 0,
        snapshot=SystemSnapshotResponse(**asdict(read_snapshot())),
        memory_enabled=True,
        web_enabled=settings.web_enabled,
        web_minimum_sources=settings.web_minimum_sources,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key), Depends(limit_chat_rate)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    identity_answer = direct_creator_answer(request.messages[-1].content)
    if identity_answer is not None:
        trace = _start_trace(request, settings.diagnostics_enabled)
        if trace is not None:
            trace.set_model("orion-institutional-identity")
            trace.record_guard(
                "institutional_identity_direct",
                "Respuesta institucional de autoría de Orion; no se llamó a un modelo externo.",
            )
            trace.complete(identity_answer)
        return _direct_response(identity_answer, _identity_prepared(request))

    provider = _provider_or_http_error()
    trace: DiagnosticTrace | None = None

    try:
        if settings.semantic_orchestration:
            trace = _start_trace(request, settings.diagnostics_enabled)
            await provider.preflight(SelectedMode.QUICK)
            bundle = await build_reasoning_bundle(
                provider,
                request,
                settings,
                _documents(),
                trace=trace,
                memory_context=await _memory_context(),
            )
            prepared = await _prepare_selected_chat(
                request,
                provider=provider,
                selected_mode=bundle.selected_mode,
                recommended_mode=bundle.plan.recommended_mode,
                recommendation_reason=bundle.plan.reason,
                preflight_model=True,
            )
            if bundle.plan.needs_clarification and bundle.plan.clarifying_question:
                if trace is not None:
                    trace.complete(bundle.plan.clarifying_question)
                return _direct_response(bundle.plan.clarifying_question, prepared)
            if bundle.review.needs_clarification and bundle.review.clarifying_question:
                if trace is not None:
                    trace.complete(bundle.review.clarifying_question)
                return _direct_response(bundle.review.clarifying_question, prepared)
            system_prompt = _semantic_prompt(request, prepared, bundle)
            if trace is not None:
                trace.record_prompt_metadata(
                    system_prompt,
                    request.messages,
                    template_version=settings.version,
                )
            final_started = perf_counter()
            async with _provider_runtime(provider):
                result = await provider.chat(
                    mode=prepared.selected_mode,
                    messages=request.messages,
                    system_prompt=system_prompt,
                )
            if trace is not None:
                trace.record_model_call(
                    "final_answer",
                    result,
                    duration_ms=(perf_counter() - final_started) * 1000,
                )
                _audit_final_answer(trace, request, bundle, result.content)
                trace.complete(result.content)
        else:
            prepared = await _prepare_chat(
                request,
                preflight_model=False,
                provider=provider,
            )
            web_context = await _web_context(request)
            if _web_is_insufficient(web_context):
                return _direct_response(_insufficient_web_response(web_context), prepared)
            async with _provider_runtime(provider):
                result = await provider.chat(
                    mode=prepared.selected_mode,
                    messages=request.messages,
                    system_prompt=_legacy_knowledge_prompt(request, prepared, web_context),
                )
    except (
        ModelProviderConfigurationError,
        ModelProviderModelError,
        ModelProviderUnavailableError,
    ) as exc:
        if trace is not None:
            trace.fail(str(exc))
        raise _provider_http_exception(exc) from exc
    except Exception as exc:
        if trace is not None:
            trace.fail(str(exc))
        raise

    return ChatResponse(
        content=result.content,
        sport=prepared.sport,
        selected_mode=prepared.selected_mode,
        recommended_mode=prepared.recommended_mode,
        recommendation_reason=prepared.recommendation_reason,
        model=result.model,
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
    dependencies=[Depends(require_api_key), Depends(limit_chat_rate)],
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    settings = get_settings()
    identity_answer = direct_creator_answer(request.messages[-1].content)
    if identity_answer is not None:
        async def identity_generate() -> AsyncIterator[bytes]:
            trace = _start_trace(request, settings.diagnostics_enabled)
            if trace is not None:
                trace.set_model("orion-institutional-identity")
                trace.record_guard(
                    "institutional_identity_direct",
                    "Respuesta institucional de autoría de Orion; no se llamó a un modelo externo.",
                )
            prepared = _identity_prepared(request)
            yield _ndjson(
                {
                    "type": "meta",
                    "selected_mode": prepared.selected_mode.value,
                    "recommended_mode": prepared.recommended_mode.value,
                    "recommendation_reason": prepared.recommendation_reason,
                    "model": prepared.model,
                    "sport": prepared.sport.value,
                    "trace_id": trace.trace_id if trace is not None else None,
                }
            )
            yield _ndjson({"type": "content", "content": identity_answer})
            if trace is not None:
                trace.complete(identity_answer)
            yield _done_event()

        return StreamingResponse(
            identity_generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    provider = _provider_or_http_error()

    async def generate() -> AsyncIterator[bytes]:
        trace: DiagnosticTrace | None = None
        try:
            if settings.semantic_orchestration:
                trace = _start_trace(request, settings.diagnostics_enabled)
                # First visible byte before any slow work: the client sees activity
                # while planning/search/review still run inside the generator.
                last_stage = "planning"
                yield _ndjson({"type": "stage", "stage": last_stage})
                await provider.preflight(SelectedMode.QUICK)
                stage_events: asyncio.Queue[str] = asyncio.Queue()
                bundle_task = asyncio.create_task(
                    build_reasoning_bundle(
                        provider,
                        request,
                        settings,
                        _documents(),
                        trace=trace,
                        on_stage=stage_events.put_nowait,
                        memory_context=await _memory_context(),
                    )
                )
                try:
                    while not bundle_task.done():
                        stage_future = asyncio.ensure_future(stage_events.get())
                        await asyncio.wait(
                            {stage_future, bundle_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if stage_future.done():
                            stage = stage_future.result()
                            if stage != last_stage:
                                last_stage = stage
                                yield _ndjson({"type": "stage", "stage": stage})
                        else:
                            stage_future.cancel()
                    while not stage_events.empty():
                        stage = stage_events.get_nowait()
                        if stage != last_stage:
                            last_stage = stage
                            yield _ndjson({"type": "stage", "stage": stage})
                    bundle = await bundle_task
                except BaseException:
                    bundle_task.cancel()
                    raise
                prepared = await _prepare_selected_chat(
                    request,
                    provider=provider,
                    selected_mode=bundle.selected_mode,
                    recommended_mode=bundle.plan.recommended_mode,
                    recommendation_reason=bundle.plan.reason,
                    preflight_model=True,
                )
                yield _ndjson(
                    {
                        "type": "meta",
                        "selected_mode": prepared.selected_mode.value,
                        "recommended_mode": prepared.recommended_mode.value,
                        "recommendation_reason": prepared.recommendation_reason,
                        "model": prepared.model,
                        "sport": prepared.sport.value,
                        "trace_id": trace.trace_id if trace is not None else None,
                    }
                )
                clarification = None
                if bundle.plan.needs_clarification:
                    clarification = bundle.plan.clarifying_question
                if bundle.review.needs_clarification:
                    clarification = bundle.review.clarifying_question or clarification
                if clarification:
                    yield _ndjson({"type": "content", "content": clarification})
                    if trace is not None:
                        trace.complete(clarification)
                    yield _done_event()
                    return
                if bundle.chart is not None:
                    yield _ndjson({"type": "chart", "chart": bundle.chart})

                system_prompt = _semantic_prompt(request, prepared, bundle)
                if trace is not None:
                    trace.record_prompt_metadata(
                        system_prompt,
                        request.messages,
                        template_version=settings.version,
                    )
                visible_parts: list[str] = []
                final_started = perf_counter()
                async with _provider_runtime(provider):
                    async for event in provider.chat_stream(
                        mode=prepared.selected_mode,
                        messages=request.messages,
                        system_prompt=system_prompt,
                    ):
                        if event.content:
                            visible_parts.append(event.content)
                            yield _ndjson({"type": "content", "content": event.content})
                        if event.done:
                            if trace is not None:
                                trace.record_model_call(
                                    "final_answer",
                                    event,
                                    duration_ms=(perf_counter() - final_started) * 1000,
                                )
                                final_answer = "".join(visible_parts)
                                _audit_final_answer(
                                    trace, request, bundle, final_answer
                                )
                                trace.complete(final_answer)
                            yield _ndjson(
                                {
                                    "type": "done",
                                    "total_duration_ms": event.total_duration_ms,
                                    "load_duration_ms": event.load_duration_ms,
                                    "prompt_eval_duration_ms": event.prompt_eval_duration_ms,
                                    "eval_duration_ms": event.eval_duration_ms,
                                    "prompt_tokens": event.prompt_tokens,
                                    "completion_tokens": event.completion_tokens,
                                    "reasoning_tokens": event.reasoning_tokens,
                                    "finish_reason": event.finish_reason,
                                    "reasoning_effort": event.reasoning_effort,
                                    "endpoint": event.endpoint,
                                    "tokens_per_second": event.tokens_per_second,
                                    "thread_limit": event.thread_limit,
                                }
                            )
                return

            prepared = await _prepare_chat(
                request,
                preflight_model=True,
                provider=provider,
            )
            web_context = await _web_context(request)
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
                yield _done_event()
                return
            async with _provider_runtime(provider):
                async for event in provider.chat_stream(
                    mode=prepared.selected_mode,
                    messages=request.messages,
                    system_prompt=_legacy_knowledge_prompt(request, prepared, web_context),
                ):
                    if event.content:
                        yield _ndjson({"type": "content", "content": event.content})
                    if event.done:
                        yield _ndjson(
                            {
                                "type": "done",
                                "total_duration_ms": event.total_duration_ms,
                                "load_duration_ms": event.load_duration_ms,
                                "prompt_eval_duration_ms": event.prompt_eval_duration_ms,
                                "eval_duration_ms": event.eval_duration_ms,
                                "prompt_tokens": event.prompt_tokens,
                                "completion_tokens": event.completion_tokens,
                                "reasoning_tokens": event.reasoning_tokens,
                                "finish_reason": event.finish_reason,
                                "reasoning_effort": event.reasoning_effort,
                                "endpoint": event.endpoint,
                                "tokens_per_second": event.tokens_per_second,
                                "thread_limit": event.thread_limit,
                            }
                        )
        except asyncio.CancelledError:
            if trace is not None:
                trace.fail("Solicitud cancelada por el cliente.")
            raise
        except ModelProviderModelError as exc:
            if trace is not None:
                trace.fail(str(exc))
            yield _ndjson(
                {
                    "type": "error",
                    "code": "model_not_installed",
                    "message": f"Falta instalar o habilitar {exc.model}.",
                }
            )
        except ModelProviderConfigurationError as exc:
            if trace is not None:
                trace.fail(str(exc))
            yield _ndjson(
                {
                    "type": "error",
                    "code": "provider_configuration_error",
                    "message": str(exc),
                }
            )
        except ModelProviderUnavailableError as exc:
            if trace is not None:
                trace.fail(str(exc))
            yield _ndjson(
                {
                    "type": "error",
                    "code": "model_provider_unavailable",
                    "message": str(exc),
                }
            )
        except Exception as exc:
            if trace is not None:
                trace.fail(str(exc))
            # An unexpected (not one of the specific provider errors above)
            # exception must still surface to the client instead of silently
            # closing the connection: the caller needs something to show the
            # user and something to debug from, not a truncated empty stream.
            try:
                yield _ndjson(
                    {
                        "type": "error",
                        "code": "internal_error",
                        "message": str(exc) or exc.__class__.__name__,
                    }
                )
            except Exception:
                pass
            raise

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
