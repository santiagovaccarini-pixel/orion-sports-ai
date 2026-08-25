from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, RequestedMode
from backend.app.providers.model_provider import (
    ModelProvider,
    ModelProviderUnavailableError,
)
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


async def _plan(
    provider: ModelProvider,
    request: ChatRequest,
    settings: Settings,
    documents: Sequence[KnowledgeDocument],
) -> SemanticPlan:
    try:
        return await create_semantic_plan(
            provider,
            request.messages,
            web_available=settings.web_enabled,
            documents=documents,
            sport=request.sport,
        )
    except (SemanticOrchestrationError, ModelProviderUnavailableError):
        # An auxiliary planning failure must not take the whole chat down. The
        # fallback deliberately avoids lexical classification and gathers broadly.
        return conservative_fallback_plan(
            request.messages,
            web_available=settings.web_enabled,
            documents=documents,
        )


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
            "automáticamente cuando falla la etapa crítica."
        ),
    )


async def _review(
    provider: ModelProvider,
    request: ChatRequest,
    plan: SemanticPlan,
    web_sources: Sequence[WebSource],
    local_evidence: Sequence[LocalEvidence],
) -> EvidenceReview:
    try:
        return await review_evidence(
            provider,
            plan,
            web_sources,
            local_evidence,
            messages=request.messages,
        )
    except (SemanticOrchestrationError, ModelProviderUnavailableError):
        # Preserve already retrieved evidence for the final model, but do not mark
        # any source as semantically validated if the critic itself failed.
        return _fallback_review(plan)


async def _search(
    query: str,
    settings: Settings,
) -> tuple[WebSource, ...]:
    return await research(
        query,
        allowed_domains=settings.web_allowed_domains,
        minimum_sources=settings.web_minimum_sources,
        result_limit=max(6, settings.web_minimum_sources * 2),
        provider=settings.web_provider,
        tavily_api_key=settings.tavily_api_key,
    )


async def build_reasoning_bundle(
    provider: ModelProvider,
    request: ChatRequest,
    settings: Settings,
    documents: Sequence[KnowledgeDocument],
) -> ReasoningBundle:
    """Understand, gather evidence and review it before Orion answers.

    The model decides tool use semantically. Python only executes those decisions and
    enforces bounded tool rounds; it does not map user words to meanings or answers.
    The critic always receives the original conversation so it can reject a plan that
    silently changed scope. Auxiliary reasoning failures degrade conservatively.
    """

    plan = await _plan(provider, request, settings, documents)
    selected_mode = _selected_mode(request, plan)

    local_evidence = collect_local_evidence(
        documents,
        plan,
        max_characters=settings.semantic_local_context_characters,
    )
    web_sources: tuple[WebSource, ...] = ()

    if plan.use_web and settings.web_enabled:
        initial_query = plan.web_query or plan.objective
        web_sources = await _search(initial_query, settings)

    review = await _review(provider, request, plan, web_sources, local_evidence)

    rounds = 1 if plan.use_web and settings.web_enabled else 0
    while (
        not review.sufficient
        and not review.needs_clarification
        and review.follow_up_web_query
        and settings.web_enabled
        and rounds < settings.semantic_max_tool_rounds
    ):
        incoming = await _search(review.follow_up_web_query, settings)
        web_sources = merge_web_sources(web_sources, incoming)
        rounds += 1
        review = await _review(provider, request, plan, web_sources, local_evidence)

    context = format_reasoning_context(
        plan,
        review,
        web_sources,
        local_evidence,
        original_user_request=request.messages[-1].content,
    )
    return ReasoningBundle(
        plan=plan,
        review=review,
        web_sources=web_sources,
        local_evidence=local_evidence,
        context=context,
        selected_mode=selected_mode,
    )
