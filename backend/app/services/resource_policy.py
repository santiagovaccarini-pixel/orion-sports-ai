from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.models import SelectedMode, SystemSnapshot


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    requires_confirmation: bool
    reasons: tuple[str, ...]


def evaluate_resources(
    mode: SelectedMode,
    snapshot: SystemSnapshot,
) -> ResourceDecision:
    reasons: list[str] = []

    if mode is SelectedMode.DEEP:
        estimated_model_gb = 12.5
        reserve_for_other_apps_gb = 8.0
        if snapshot.memory_available_gb - estimated_model_gb < reserve_for_other_apps_gb:
            reasons.append(
                "el modo Profundo podría dejar menos de 8 GB libres para Windows y tus aplicaciones"
            )
        if snapshot.cpu_percent >= 55:
            reasons.append(f"la CPU ya está usando {snapshot.cpu_percent:.0f}%")
        if snapshot.plugged_in is False and (snapshot.battery_percent or 100) < 35:
            reasons.append("la batería está baja y el equipo no está conectado")
    else:
        if snapshot.memory_available_gb < 7:
            reasons.append("hay menos de 7 GB de memoria disponible")
        if snapshot.cpu_percent >= 70:
            reasons.append(f"la CPU ya está usando {snapshot.cpu_percent:.0f}%")

    return ResourceDecision(
        requires_confirmation=bool(reasons),
        reasons=tuple(reasons),
    )
