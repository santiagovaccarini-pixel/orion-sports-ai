from __future__ import annotations

import asyncio
import os

import psutil

from backend.app.domain.models import SystemSnapshot


GIB = 1024**3


def read_snapshot(cpu_interval: float | None = None) -> SystemSnapshot:
    """Read current resources without pausing the API event loop by default."""
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    return SystemSnapshot(
        cpu_percent=round(psutil.cpu_percent(interval=cpu_interval), 1),
        memory_available_gb=round(memory.available / GIB, 2),
        memory_total_gb=round(memory.total / GIB, 2),
        battery_percent=round(battery.percent, 1) if battery else None,
        plugged_in=battery.power_plugged if battery else None,
    )


def lower_process_priority() -> None:
    process = psutil.Process(os.getpid())
    try:
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(5)
    except (psutil.AccessDenied, OSError):
        return


def lower_ollama_priority() -> None:
    for process in psutil.process_iter(["name"]):
        try:
            name = (process.info.get("name") or "").lower()
            if not name.startswith("ollama"):
                continue
            if os.name == "nt":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(5)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue


async def maintain_ollama_priority(
    stop_event: asyncio.Event,
    interval_seconds: float = 1.5,
) -> None:
    """Keep current and newly spawned Ollama processes below normal priority."""
    while not stop_event.is_set():
        lower_ollama_priority()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
