from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


def create_plan(query: str, *, has_local_documents: bool) -> OrchestrationPlan:
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
            reason="La consulta depende de información actual o solicita fuentes web.",
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
        reason="La consulta se responde con conocimiento general o datos locales según su intención.",
    )