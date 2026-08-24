from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.app.domain.intent import SemanticPlan
from backend.app.services.knowledge_base import _query_targets_data
from backend.app.services.web_research import is_web_request


class Intent(str, Enum):
    GENERAL = "general"
    WEB_RESEARCH = "web_research"
    LOCAL_DATA = "local_data"
    CALCULATION = "calculation"
    CHART = "chart"
    CLARIFICATION = "clarification"


@dataclass(frozen=True, slots=True)
class OrchestrationPlan:
    intent: Intent
    use_web: bool
    use_local_data: bool
    use_calculator: bool
    use_chart: bool
    needs_clarification: bool
    reason: str


def _fold(query: str) -> str:
    return query.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def create_plan(
    query: str,
    *,
    has_local_documents: bool,
    semantic_plan: SemanticPlan | None = None,
) -> OrchestrationPlan:
    """Create an execution plan after semantic reasoning has completed.

    When a SemanticPlan exists it is authoritative. Raw wording must not override a
    resolved plan. Lexical routing is retained only as a legacy degradation path for
    callers that provide no semantic plan at all.
    """
    if semantic_plan is not None:
        use_web = semantic_plan.needs_web
        use_local = has_local_documents and semantic_plan.needs_local_data
        use_chart = use_local and semantic_plan.task_type == "chart"
        use_calculator = use_local and semantic_plan.task_type == "calculation"
        needs_clarification = semantic_plan.requires_clarification

        if needs_clarification:
            intent = Intent.CLARIFICATION
        elif use_web:
            intent = Intent.WEB_RESEARCH
        elif use_chart:
            intent = Intent.CHART
        elif use_calculator:
            intent = Intent.CALCULATION
        elif use_local:
            intent = Intent.LOCAL_DATA
        else:
            intent = Intent.GENERAL

        return OrchestrationPlan(
            intent=intent,
            use_web=use_web,
            use_local_data=use_local,
            use_calculator=use_calculator,
            use_chart=use_chart,
            needs_clarification=needs_clarification,
            reason=(
                "Plan construido desde el marco de razonamiento: "
                f"{semantic_plan.user_goal}"
            ),
        )

    # Legacy compatibility only. Normal chat requests always create a SemanticPlan,
    # even when the LLM planner is unavailable (neutral low-confidence fallback).
    folded = _fold(query)
    chart = any(marker in folded for marker in ("grafic", "visualiz", "chart"))
    calculation = any(
        marker in folded
        for marker in (
            "suma", "sumar", "sum", "promedio", "media", "compar", "filtr",
            "atip", "calcular", "calcula", "promedi",
        )
    )
    web = is_web_request(query)
    local = has_local_documents and _query_targets_data(query)

    if web:
        return OrchestrationPlan(
            Intent.WEB_RESEARCH,
            use_web=True,
            use_local_data=False,
            use_calculator=False,
            use_chart=False,
            needs_clarification=False,
            reason="Fallback legado: la consulta solicita información externa.",
        )
    if chart:
        intent = Intent.CHART
    elif calculation:
        intent = Intent.CALCULATION
    elif local:
        intent = Intent.LOCAL_DATA
    else:
        intent = Intent.GENERAL
    return OrchestrationPlan(
        intent,
        use_web=False,
        use_local_data=local,
        use_calculator=local and calculation,
        use_chart=local and chart,
        needs_clarification=False,
        reason="Fallback legado sin plan semántico disponible.",
    )
