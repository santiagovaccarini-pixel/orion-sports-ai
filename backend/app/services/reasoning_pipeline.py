from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, RequestedMode
from backend.app.providers.model_provider import ModelProvider, ModelResult
from backend.app.services.diagnostic_trace import DiagnosticTrace, diagnostic_traces
from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.semantic_orchestrator import (
    EvidenceReview,
    LocalEvidence,
    SemanticOrchestrationError,
    SemanticPlan,
    collect_local_evidence,
    conservative_fallback_plan,
    create_semantic_plan,
    format_reasoning_context,
    merge_web_sources,
    review_evidence,
)
from backend.app.services.semantic_tools import execute_calculation, execute_csv_operation
from backend.app.services.web_reader import MAX_READ_PAGES, apply_page_reads, read_source_pages
from backend.app.services.web_research import WebSource, research


@dataclass(frozen=True, slots=True)
class ReasoningBundle:
    plan: SemanticPlan
    review: EvidenceReview
    web_sources: tuple[WebSource, ...]
    local_evidence: tuple[LocalEvidence, ...]
    context: str
    selected_mode: SelectedMode
    tool_evidence: tuple[LocalEvidence, ...] = ()
    tool_context: str = ""
    chart: dict[str, object] | None = None
    trace: DiagnosticTrace | None = None


def _record_model_result(
    trace: DiagnosticTrace | None,
    stage: str,
    result: ModelResult,
) -> None:
    if trace is not None:
        trace.record_model_call(stage, result)


def _notify_stage(on_stage: Callable[[str], None] | None, stage: str) -> None:
    if on_stage is not None:
        on_stage(stage)


async def _plan(
    provider: ModelProvider,
    request: ChatRequest,
    settings: Settings,
    documents: Sequence[KnowledgeDocument],
    trace: DiagnosticTrace | None = None,
) -> SemanticPlan:
    started = perf_counter()
    try:
        plan = await create_semantic_plan(
            provider,
            request.messages,
            web_available=settings.web_enabled,
            documents=documents,
            sport=request.sport,
            on_model_result=lambda stage, result: _record_model_result(
                trace, stage, result
            ),
        )
        if trace is not None:
            trace.record_plan(
                plan,
                fallback=False,
                duration_ms=(perf_counter() - started) * 1000,
            )
        return plan
    except SemanticOrchestrationError as exc:
        plan = conservative_fallback_plan(
            request.messages,
            web_available=settings.web_enabled,
            documents=documents,
        )
        if trace is not None:
            trace.record_plan(
                plan,
                fallback=True,
                duration_ms=(perf_counter() - started) * 1000,
                error=str(exc),
            )
        return plan


def _selected_mode(request: ChatRequest, plan: SemanticPlan) -> SelectedMode:
    if request.mode is RequestedMode.AUTO:
        return plan.recommended_mode
    return SelectedMode(request.mode.value)


def _fallback_review(plan: SemanticPlan) -> EvidenceReview:
    return EvidenceReview(
        sufficient=False,
        relevant_source_ids=(),
        discarded_source_ids=(),
        missing_information=("La revisión semántica de la evidencia no pudo completarse.",),
        follow_up_web_query=None,
        needs_clarification=False,
        clarifying_question=None,
        resolved_scope=None,
        reason=(
            "Fallback conservador de revisión: ninguna fuente queda validada "
            "automáticamente cuando falla el parseo estructurado."
        ),
    )


async def _review(
    provider: ModelProvider,
    request: ChatRequest,
    plan: SemanticPlan,
    web_sources: Sequence[WebSource],
    local_evidence: Sequence[LocalEvidence],
    *,
    round_number: int,
    trace: DiagnosticTrace | None = None,
) -> EvidenceReview:
    started = perf_counter()
    reasoning_effort = "low" if round_number <= 1 else "medium"
    try:
        review = await review_evidence(
            provider,
            plan,
            web_sources,
            local_evidence,
            messages=request.messages,
            reasoning_effort=reasoning_effort,
            stage_name=f"review_{round_number}",
            on_model_result=lambda stage, result: _record_model_result(
                trace, stage, result
            ),
        )
        if trace is not None:
            trace.record_review(
                review,
                round_number=round_number,
                fallback=False,
                duration_ms=(perf_counter() - started) * 1000,
                web_sources=web_sources,
            )
        return review
    except SemanticOrchestrationError as exc:
        review = _fallback_review(plan)
        if trace is not None:
            trace.record_review(
                review,
                round_number=round_number,
                fallback=True,
                duration_ms=(perf_counter() - started) * 1000,
                web_sources=web_sources,
                error=str(exc),
            )
        return review


async def _search(
    query: str,
    settings: Settings,
    *,
    round_number: int,
    trace: DiagnosticTrace | None = None,
) -> tuple[WebSource, ...]:
    started = perf_counter()
    sources = await research(
        query,
        allowed_domains=settings.web_allowed_domains,
        minimum_sources=settings.web_minimum_sources,
        result_limit=max(6, settings.web_minimum_sources * 2),
        provider=settings.web_provider,
        tavily_api_key=settings.tavily_api_key,
    )
    if trace is not None:
        trace.record_search(
            round_number=round_number,
            query=query,
            sources=sources,
            duration_ms=(perf_counter() - started) * 1000,
        )
    return sources


def _normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _web_read_candidates(
    review: EvidenceReview,
    *,
    attempted_source_ids: set[str],
    remaining: int,
) -> tuple[str, ...]:
    """Use the semantic reviewer's relevance decision to choose pages to open."""

    if remaining <= 0:
        return ()
    selected: list[str] = []
    for raw_source_id in review.relevant_source_ids:
        source_id = raw_source_id.strip().upper()
        if not source_id.startswith("W") or source_id in attempted_source_ids:
            continue
        suffix = source_id[1:]
        if not suffix.isdigit() or int(suffix) < 1:
            continue
        selected.append(source_id)
        if len(selected) >= remaining:
            break
    return tuple(selected)


async def _deepen_relevant_web_sources(
    request: ChatRequest,
    review: EvidenceReview,
    web_sources: tuple[WebSource, ...],
    *,
    attempted_source_ids: set[str],
    trace: DiagnosticTrace | None,
) -> tuple[tuple[WebSource, ...], bool]:
    remaining = MAX_READ_PAGES - len(attempted_source_ids)
    candidates = _web_read_candidates(
        review,
        attempted_source_ids=attempted_source_ids,
        remaining=remaining,
    )
    if not candidates:
        return web_sources, False

    attempted_source_ids.update(candidates)
    started = perf_counter()
    reads = await read_source_pages(
        web_sources,
        source_ids=candidates,
        query=request.messages[-1].content,
        max_pages=remaining,
    )
    duration_ms = (perf_counter() - started) * 1000
    if trace is not None:
        previous = trace.timings_ms.get("web_read_total", 0.0)
        trace.set_timing("web_read_total", previous + duration_ms)
        if reads:
            trace.record_guard(
                "web_read_completed",
                "Orion abrió directamente las páginas seleccionadas por el revisor: "
                + ", ".join(read.source_id for read in reads),
            )
        else:
            trace.record_guard(
                "web_read_unavailable",
                "No se pudo extraer contenido directo de las fuentes seleccionadas: "
                + ", ".join(candidates),
            )
    if not reads:
        return web_sources, False
    return apply_page_reads(web_sources, reads), True


def _execute_semantic_tools(
    plan: SemanticPlan,
    documents: Sequence[KnowledgeDocument],
    trace: DiagnosticTrace | None,
) -> tuple[tuple[LocalEvidence, ...], str, dict[str, object] | None]:
    evidence: list[LocalEvidence] = []
    context_blocks: list[str] = []
    chart: dict[str, object] | None = None
    tool_index = 0

    if plan.use_calculator and plan.calculation_expression is not None:
        tool_index += 1
        execution = execute_calculation(plan.calculation_expression)
        if execution.error:
            if trace is not None:
                trace.record_guard("semantic_calculator_error", execution.error)
        elif execution.context:
            source_id = f"T{tool_index}"
            evidence.append(
                LocalEvidence(
                    source_id=source_id,
                    document_name="Orion Calculator",
                    content=execution.context,
                    truncated=False,
                    chunk_index=None,
                )
            )
            context_blocks.append(f"[{source_id}] Orion Calculator\n{execution.context}")

    if plan.csv_operation is not None and (plan.use_calculator or plan.use_chart):
        tool_index += 1
        execution = execute_csv_operation(documents, plan.csv_operation)
        if execution.error:
            if trace is not None:
                trace.record_guard("semantic_csv_tool_error", execution.error)
        elif execution.context:
            source_id = f"T{tool_index}"
            evidence.append(
                LocalEvidence(
                    source_id=source_id,
                    document_name=plan.csv_operation.document_name,
                    content=execution.context,
                    truncated=False,
                    chunk_index=None,
                )
            )
            context_blocks.append(
                f"[{source_id}] Herramienta CSV: {plan.csv_operation.document_name}\n"
                f"{execution.context}"
            )
            chart = execution.chart

    return tuple(evidence), "\n\n".join(context_blocks), chart


async def build_reasoning_bundle(
    provider: ModelProvider,
    request: ChatRequest,
    settings: Settings,
    documents: Sequence[KnowledgeDocument],
    *,
    trace: DiagnosticTrace | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ReasoningBundle:
    """Understand, gather evidence, deepen relevant pages, execute tools and review."""

    if trace is None and settings.diagnostics_enabled:
        trace = diagnostic_traces.start(
            question=request.messages[-1].content,
            sport=request.sport.value,
            requested_mode=request.mode.value,
        )

    pipeline_started = perf_counter()
    _notify_stage(on_stage, "planning")
    plan = await _plan(provider, request, settings, documents, trace)
    selected_mode = _selected_mode(request, plan)
    if trace is not None:
        trace.set_model(provider.model_for(selected_mode))

    local_evidence = collect_local_evidence(
        documents,
        plan,
        original_user_request=request.messages[-1].content,
        max_characters=settings.semantic_local_context_characters,
    )
    tool_evidence, tool_context, chart = _execute_semantic_tools(
        plan, documents, trace
    )
    review_evidence_items = (*local_evidence, *tool_evidence)
    if trace is not None:
        trace.record_local_evidence(review_evidence_items)

    web_sources: tuple[WebSource, ...] = ()
    search_rounds = 0
    review_round = 0
    seen_queries: set[str] = set()
    attempted_page_reads: set[str] = set()

    if plan.use_web and settings.web_enabled:
        initial_query = plan.web_query or plan.objective
        normalized = _normalized_query(initial_query)
        if normalized:
            seen_queries.add(normalized)
            search_rounds = 1
            _notify_stage(on_stage, "searching")
            web_sources = await _search(
                initial_query,
                settings,
                round_number=search_rounds,
                trace=trace,
            )

    review_round += 1
    _notify_stage(on_stage, "reviewing")
    review = await _review(
        provider,
        request,
        plan,
        web_sources,
        review_evidence_items,
        round_number=review_round,
        trace=trace,
    )

    while not review.sufficient and not review.needs_clarification:
        # Before spending another search query, deepen only the web results that the
        # semantic reviewer itself marked relevant. This makes page opening model-led
        # while Python remains responsible for bounded/safe network execution.
        if web_sources and len(attempted_page_reads) < MAX_READ_PAGES:
            _notify_stage(on_stage, "reading")
            web_sources, deepened = await _deepen_relevant_web_sources(
                request,
                review,
                web_sources,
                attempted_source_ids=attempted_page_reads,
                trace=trace,
            )
            if deepened:
                review_round += 1
                _notify_stage(on_stage, "reviewing")
                review = await _review(
                    provider,
                    request,
                    plan,
                    web_sources,
                    review_evidence_items,
                    round_number=review_round,
                    trace=trace,
                )
                continue

        follow_up = review.follow_up_web_query
        if (
            not follow_up
            or not settings.web_enabled
            or search_rounds >= settings.semantic_max_tool_rounds
        ):
            break
        normalized = _normalized_query(follow_up)
        if not normalized or normalized in seen_queries:
            if trace is not None:
                trace.record_guard(
                    "duplicate_web_query_blocked",
                    "El revisor propuso una consulta idéntica a una ya ejecutada; "
                    "Orion no consumió otra ronda repitiendo la misma búsqueda.",
                )
            break
        seen_queries.add(normalized)
        search_rounds += 1
        _notify_stage(on_stage, "searching")
        incoming = await _search(
            follow_up,
            settings,
            round_number=search_rounds,
            trace=trace,
        )
        web_sources = merge_web_sources(web_sources, incoming)
        review_round += 1
        _notify_stage(on_stage, "reviewing")
        review = await _review(
            provider,
            request,
            plan,
            web_sources,
            review_evidence_items,
            round_number=review_round,
            trace=trace,
        )

    context = format_reasoning_context(
        plan,
        review,
        web_sources,
        local_evidence,
        original_user_request=request.messages[-1].content,
        tool_context=tool_context,
    )
    if trace is not None:
        trace.set_timing(
            "reasoning_bundle_total",
            (perf_counter() - pipeline_started) * 1000,
        )

    return ReasoningBundle(
        plan=plan,
        review=review,
        web_sources=web_sources,
        local_evidence=local_evidence,
        context=context,
        selected_mode=selected_mode,
        tool_evidence=tool_evidence,
        tool_context=tool_context,
        chart=chart,
        trace=trace,
    )
