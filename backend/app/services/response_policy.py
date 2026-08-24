from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.domain.intent import SemanticPlan
from backend.app.domain.models import SelectedMode


VERBOSE_MARKERS = (
    "en detalle",
    "detalladamente",
    "profundiza",
    "profundizá",
    "completo",
    "completa",
    "exhaustivo",
    "exhaustiva",
    "paso a paso",
    "todo lo que",
    "muy desarrollado",
)


def response_token_budget(
    settings: Settings,
    mode: SelectedMode,
    plan: SemanticPlan,
    user_query: str,
) -> int:
    """Cap answer generation by task, not only by UI mode.

    On CPU, generation dominates latency. Deep mode therefore spends compute on
    reasoning when useful but does not automatically produce a long final answer.
    Explicit requests for detail keep the configured full budget.
    """
    lowered = user_query.casefold()
    explicit_verbose = any(marker in lowered for marker in VERBOSE_MARKERS)

    configured = (
        settings.quick_max_tokens
        if mode is SelectedMode.QUICK
        else settings.deep_max_tokens
    )
    if explicit_verbose:
        return configured

    if mode is SelectedMode.QUICK:
        if plan.task_type in {"definition", "direct_answer", "calculation", "data_query"}:
            return min(configured, 256)
        if plan.task_type in {"interpretation", "comparison", "chart"}:
            return min(configured, 384)
        return min(configured, 512)

    # Deep means deeper analysis, not unconditional verbosity. These caps matter
    # materially on the current CPU-bound 8B model (~6 tok/s in local benchmarks).
    if plan.task_type in {"definition", "direct_answer", "calculation", "data_query", "chart"}:
        return min(configured, 160)
    if plan.task_type in {"interpretation", "comparison", "clarification"}:
        return min(configured, 320)
    if plan.task_type in {"research", "planning", "debugging"}:
        return min(configured, 640)
    return min(configured, 384)


def response_style_instruction(
    mode: SelectedMode,
    plan: SemanticPlan,
    user_query: str,
) -> str:
    """Give the answer model a content-level latency policy alongside token caps."""
    lowered = user_query.casefold()
    explicit_verbose = any(marker in lowered for marker in VERBOSE_MARKERS)
    if explicit_verbose:
        return ""

    if mode is SelectedMode.DEEP:
        return (
            "POLÍTICA DE SALIDA: Profundo significa mayor calidad de razonamiento, no "
            "una respuesta innecesariamente larga. Razoná lo necesario internamente y "
            "entregá una síntesis compacta. Si el usuario pide N puntos, respetá N y usá "
            "una o dos frases por punto. Expandí sólo cuando cambie la decisión, el método "
            "o una limitación importante."
        )
    if plan.task_type in {"definition", "direct_answer"}:
        return (
            "POLÍTICA DE SALIDA: respuesta directa y corta; no conviertas una definición "
            "o dato puntual en un ensayo."
        )
    return ""
