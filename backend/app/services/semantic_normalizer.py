from __future__ import annotations

from collections.abc import Sequence

from backend.app.domain.intent import SemanticPlan
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.ontology_runtime import (
    canonical_labels,
    concept_flags,
    primary_domain,
    valid_concept_ids,
)


TASK_COMPLEXITY = {
    "direct_answer": 0.20,
    "definition": 0.22,
    "calculation": 0.28,
    "data_query": 0.30,
    "chart": 0.30,
    "interpretation": 0.48,
    "comparison": 0.52,
    "clarification": 0.25,
    "research": 0.66,
    "planning": 0.70,
    "debugging": 0.72,
}

INFERENCE_COMPLEXITY = {
    "descriptive": 0.00,
    "interpretive": 0.10,
    "comparative": 0.14,
    "causal": 0.25,
    "diagnostic": 0.22,
    "predictive": 0.25,
    "planning": 0.18,
}


def normalize_semantic_plan(
    plan: SemanticPlan,
    messages: Sequence[ChatMessage],
    sport: SportContext,
    *,
    has_local_documents: bool,
) -> SemanticPlan:
    """Validate a semantic plan using ontology structure, never phrase matching.

    Natural-language interpretation belongs to the planner model. Python only checks
    whether selected ontology IDs exist and derives deterministic consequences from
    their domains/flags and from the explicit inference type.
    """
    del messages  # Conversation text must not be reinterpreted through keyword rules.

    plan.concept_ids = valid_concept_ids(sport, plan.concept_ids)
    plan.concepts = canonical_labels(sport, plan.concept_ids)

    derived_domain = primary_domain(sport, plan.concept_ids)
    if derived_domain != "general":
        plan.domain = derived_domain

    flags = concept_flags(sport, plan.concept_ids)
    if "private_memory" in flags:
        plan.needs_private_memory = True
    if "causal_risk" in flags:
        plan.causal_claim_risk = True

    if plan.inference_type == "comparative" or plan.task_type == "comparison":
        plan.comparison = True
    if plan.inference_type == "causal":
        plan.causal_claim_risk = True
        if plan.task_type == "direct_answer":
            plan.task_type = "interpretation"

    if plan.requires_clarification and not plan.missing_variables:
        plan.missing_variables = ["variable indispensable no especificada"]

    if not has_local_documents:
        plan.needs_local_data = False

    # Private conventions are not global truth. Local calculations/charts normally do
    # not need global knowledge unless the planner explicitly frames an interpretation.
    plan.needs_global_knowledge = not (
        plan.needs_local_data
        and plan.task_type in {"calculation", "data_query", "chart"}
        and not plan.needs_private_memory
    )
    if plan.needs_private_memory and plan.task_type in {"direct_answer", "definition"}:
        plan.needs_global_knowledge = False
    if plan.needs_web or plan.task_type == "research":
        plan.needs_global_knowledge = True

    complexity = TASK_COMPLEXITY.get(plan.task_type, 0.35)
    complexity += INFERENCE_COMPLEXITY.get(plan.inference_type, 0.0)
    if plan.referenced_previous_context:
        complexity += 0.06
    if plan.requires_clarification:
        complexity -= 0.08
    if len(plan.concept_ids) >= 4:
        complexity += 0.05
    plan.complexity = max(0.10, min(complexity, 1.0))

    plan.ambiguity = max(
        0.0,
        min(
            1.0,
            0.75 if plan.requires_clarification else max(0.05, 1.0 - plan.confidence),
        ),
    )

    # Retrieval may use canonical labels later, but retrieval text never determines
    # the semantic plan. Keep this derived and compact.
    plan.retrieval_queries = [" ".join(plan.concepts)] if plan.concepts else []
    plan.confidence = max(0.0, min(plan.confidence, 1.0))
    return plan
