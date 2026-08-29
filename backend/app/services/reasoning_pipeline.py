from __future__ import annotations

from dataclasses import dataclass, replace
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
    build_contract,
    collect_local_evidence,
    conservative_fallback_plan,
    create_semantic_plan,
    format_reasoning_context,
    merge_web_sources,
    partial_sum_context,
    review_evidence,
)
from backend.app.services.semantic_tools import execute_calculation, execute_csv_operation
from backend.app.services.web_reader import MAX_READ_PAGES, apply_page_reads, read_source_pages
from backend.app.services.web_research import WebSource, research


DEFAULT_FRESHNESS_WINDOW_DAYS = 30


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
    memory_context: str = "",
) -> SemanticPlan:
    started = perf_counter()
    try:
        plan = await create_semantic_plan(
            provider,
            request.messages,
            web_available=settings.web_enabled,
            documents=documents,
            sport=request.sport,
            memory_context=memory_context,
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
        audited=False,
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
    previous_review: EvidenceReview | None = None,
    memory_context: str = "",
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
            previous_review=previous_review,
            memory_context=memory_context,
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
    recency_days: int | None = None,
) -> tuple[WebSource, ...]:
    started = perf_counter()
    sources = await research(
        query,
        allowed_domains=settings.web_allowed_domains,
        minimum_sources=settings.web_minimum_sources,
        result_limit=max(6, settings.web_minimum_sources * 2),
        provider=settings.web_provider,
        tavily_api_key=settings.tavily_api_key,
        recency_days=recency_days,
    )
    if trace is not None:
        if recency_days is not None:
            trace.record_guard(
                "recency_window_applied",
                f"Búsqueda acotada a los últimos {recency_days} días porque el plan "
                "declaró información volátil.",
            )
        trace.record_search(
            round_number=round_number,
            query=query,
            sources=sources,
            duration_ms=(perf_counter() - started) * 1000,
        )
    return sources


def _sorted_by_recency(sources: tuple[WebSource, ...]) -> tuple[WebSource, ...]:
    """Stable sort: dated sources first (newest first), undated ones keep order."""

    return tuple(
        sorted(
            sources,
            key=lambda source: (
                source.published_age_days is None,
                source.published_age_days
                if source.published_age_days is not None
                else 0,
            ),
        )
    )


def _apply_freshness_backstop(
    plan: SemanticPlan,
    review: EvidenceReview,
    web_sources: tuple[WebSource, ...],
    *,
    already_triggered: bool,
    trace: DiagnosticTrace | None,
) -> tuple[EvidenceReview, bool]:
    """Deterministic check on structured dates: a volatile answer cannot be accepted
    without a dated, in-window accepted source. Demotes sufficiency at most once."""

    if (
        already_triggered
        or not plan.volatile_information
        or not review.audited
        or not review.sufficient
        or review.needs_clarification
    ):
        return review, already_triggered
    window = plan.recency_window_days or DEFAULT_FRESHNESS_WINDOW_DAYS
    accepted_ids = {
        source_id.strip().upper() for source_id in review.relevant_source_ids
    }
    fresh_accepted = any(
        source.published_age_days is not None
        and source.published_age_days <= window
        for index, source in enumerate(web_sources, start=1)
        if f"W{index}" in accepted_ids
    )
    if review.freshness_verified is True and fresh_accepted:
        return review, already_triggered
    demoted = replace(
        review,
        sufficient=False,
        missing_information=(
            *review.missing_information,
            "Confirmación con una fuente fechada dentro de la ventana de "
            "recencia requerida.",
        ),
    )
    if trace is not None:
        trace.record_guard(
            "freshness_backstop_triggered",
            "La revisión aceptó información volátil sin fuente fechada dentro de "
            f"la ventana de {window} días; Orion exige una verificación más antes "
            "de darla por suficiente.",
        )
    return demoted, True


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
    memory_context: str = "",
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
    plan = await _plan(
        provider, request, settings, documents, trace, memory_context=memory_context
    )
    selected_mode = _selected_mode(request, plan)
    if trace is not None:
        trace.set_model(provider.model_for(selected_mode))

    if plan.needs_clarification and not plan.missing_for_core:
        # Structural check, not lexical: without a core gap declared by the planner,
        # a clarification would block an answerable question.
        plan = replace(plan, needs_clarification=False)
        if trace is not None:
            trace.record_guard(
                "clarification_downgraded_to_precision",
                "El plan pidió aclaración sin declarar faltantes nucleares; Orion "
                "responde el núcleo y trata lo demás como precisión pendiente.",
            )
    if plan.needs_clarification and plan.clarifying_question:
        review = EvidenceReview(
            sufficient=False,
            relevant_source_ids=(),
            discarded_source_ids=(),
            missing_information=plan.missing_for_core,
            follow_up_web_query=None,
            needs_clarification=True,
            clarifying_question=plan.clarifying_question,
            resolved_scope=None,
            reason="El plan requiere una aclaración del usuario antes de investigar.",
            audited=False,
        )
        if trace is not None:
            trace.record_guard(
                "clarification_short_circuit",
                "Orion pidió la aclaración antes de gastar búsqueda web, "
                "herramientas y revisión.",
            )
            trace.record_contract(build_contract(plan, review))
            trace.set_timing(
                "reasoning_bundle_total",
                (perf_counter() - pipeline_started) * 1000,
            )
        return ReasoningBundle(
            plan=plan,
            review=review,
            web_sources=(),
            local_evidence=(),
            context="",
            selected_mode=selected_mode,
            trace=trace,
        )

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
    recency_days = plan.recency_window_days if plan.volatile_information else None
    freshness_backstop_used = False

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
                recency_days=recency_days,
            )
            if plan.volatile_information and web_sources:
                # Only before the first review: W ids become positional references
                # for reviewer, page reader and final answer after that.
                web_sources = _sorted_by_recency(web_sources)

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
        memory_context=memory_context,
    )
    review, freshness_backstop_used = _apply_freshness_backstop(
        plan,
        review,
        web_sources,
        already_triggered=freshness_backstop_used,
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
                    previous_review=review,
                    memory_context=memory_context,
                )
                review, freshness_backstop_used = _apply_freshness_backstop(
                    plan,
                    review,
                    web_sources,
                    already_triggered=freshness_backstop_used,
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
            recency_days=recency_days,
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
            previous_review=review,
            memory_context=memory_context,
        )
        review, freshness_backstop_used = _apply_freshness_backstop(
            plan,
            review,
            web_sources,
            already_triggered=freshness_backstop_used,
            trace=trace,
        )

    if trace is not None and review.source_checks:
        incomplete_checks = [
            check.source_id for check in review.source_checks if not check.complete
        ]
        if incomplete_checks:
            trace.record_guard(
                "source_check_incomplete",
                "Fuentes aceptadas sin coincidencia completa de entidad/métrica/"
                "período/competición/unidad: " + ", ".join(incomplete_checks),
            )

    partial_sum = partial_sum_context(review)
    if partial_sum:
        tool_context = f"{tool_context}\n\n{partial_sum}" if tool_context else partial_sum
        if trace is not None:
            trace.record_guard(
                "partial_sum_computed",
                f"{len(review.partial_values)} componentes verificados sumados "
                "determinísticamente porque ninguna fuente única confirmó el total.",
            )

    context = format_reasoning_context(
        plan,
        review,
        web_sources,
        local_evidence,
        original_user_request=request.messages[-1].content,
        tool_context=tool_context,
        memory_context=memory_context,
    )
    if trace is not None:
        trace.record_contract(build_contract(plan, review))
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
