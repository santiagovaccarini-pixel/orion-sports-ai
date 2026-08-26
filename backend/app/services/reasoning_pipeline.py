from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

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
from backend.app.services.web_research import WebSource, research


@dataclass(frozen=True, slots=True)
class ReasoningBundle:
    plan: SemanticPlan
    review: EvidenceReview
    web_sources: tuple[WebSource, ...]
    local_evidence: tuple[LocalEvidence, ...]
    context: str
    selected_mode: SelectedMode
    trace: DiagnosticTrace | None = None


def _record_model_result(
    trace: DiagnosticTrace | None,
    stage: str,
    result: ModelResult,
) -> None:
    if trace is not None:
        trace.record_model_call(stage, result)


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


async def build_reasoning_bundle(
    provider: ModelProvider,
    request: ChatRequest,
    settings: Settings,
    documents: Sequence[KnowledgeDocument],
    *,
    trace: DiagnosticTrace | None = None,
) -> ReasoningBundle:
    """Understand, gather evidence and review it before Orion answers."""

    if trace is None and settings.diagnostics_enabled:
        trace = diagnostic_traces.start(
            question=request.messages[-1].content,
            sport=request.sport.value,
            requested_mode=request.mode.value,
        )

    pipeline_started = perf_counter()
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
    if trace is not None:
        trace.record_local_evidence(local_evidence)

    web_sources: tuple[WebSource, ...] = ()
    rounds = 0
    seen_queries: set[str] = set()

    if plan.use_web and settings.web_enabled:
        initial_query = plan.web_query or plan.objective
        normalized = _normalized_query(initial_query)
        if normalized:
            seen_queries.add(normalized)
            rounds = 1
            web_sources = await _search(
                initial_query,
                settings,
                round_number=rounds,
                trace=trace,
            )

    review_round = max(rounds, 1)
    review = await _review(
        provider,
        request,
        plan,
        web_sources,
        local_evidence,
        round_number=review_round,
        trace=trace,
    )

    while (
        not review.sufficient
        and not review.needs_clarification
        and review.follow_up_web_query
        and settings.web_enabled
        and rounds < settings.semantic_max_tool_rounds
    ):
        follow_up = review.follow_up_web_query
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
        rounds += 1
        incoming = await _search(
            follow_up,
            settings,
            round_number=rounds,
            trace=trace,
        )
        web_sources = merge_web_sources(web_sources, incoming)
        review = await _review(
            provider,
            request,
            plan,
            web_sources,
            local_evidence,
            round_number=rounds,
            trace=trace,
        )

    context = format_reasoning_context(
        plan,
        review,
        web_sources,
        local_evidence,
        original_user_request=request.messages[-1].content,
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
        trace=trace,
    )
