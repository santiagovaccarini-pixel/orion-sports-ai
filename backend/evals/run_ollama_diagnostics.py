from __future__ import annotations

import argparse
import json
from typing import Any

import httpx


def _gb(value: object) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return round(float(value) / (1024**3), 2)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in value.strip().lstrip("v").split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostica versión y offload CPU/GPU del runtime local de Ollama."
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    base_url = args.ollama_url.rstrip("/")

    with httpx.Client(timeout=10.0) as client:
        version_response = client.get(f"{base_url}/api/version")
        version_response.raise_for_status()
        ps_response = client.get(f"{base_url}/api/ps")
        ps_response.raise_for_status()

    version_payload = version_response.json()
    ps_payload = ps_response.json()
    version = str(version_payload.get("version", "desconocida"))

    print(f"Ollama: {version}")
    if _version_tuple(version) and _version_tuple(version) < (0, 30):
        print(
            "ADVERTENCIA: la versión es anterior a 0.30; las versiones actuales de "
            "Ollama incorporan soporte Vulkan más amplio para AMD/Intel."
        )

    models = ps_payload.get("models", [])
    if not isinstance(models, list) or not models:
        print("No hay modelos cargados. Ejecutá una respuesta en Orion y repetí este diagnóstico.")
        return 0

    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or model.get("model") or "modelo")
        size = model.get("size")
        size_vram = model.get("size_vram")
        size_number = float(size) if isinstance(size, (int, float)) else 0.0
        vram_number = float(size_vram) if isinstance(size_vram, (int, float)) else 0.0
        ratio = (vram_number / size_number * 100.0) if size_number > 0 else 0.0

        print()
        print(f"Modelo: {name}")
        print(f"  Tamaño cargado: {_gb(size)} GB")
        print(f"  En VRAM:        {_gb(size_vram)} GB")
        print(f"  Offload GPU:    {ratio:.1f}%")

        if ratio < 1:
            print("  Diagnóstico: ejecución prácticamente CPU-only.")
        elif ratio < 95:
            print("  Diagnóstico: offload parcial; CPU y GPU comparten el trabajo.")
        else:
            print("  Diagnóstico: modelo prácticamente completo en GPU/VRAM.")

        details: Any = model.get("details")
        if isinstance(details, dict):
            family = details.get("family") or details.get("families")
            quant = details.get("quantization_level")
            if family:
                print(f"  Familia:        {family}")
            if quant:
                print(f"  Cuantización:   {quant}")

    print()
    print(
        "Interpretación: si el modelo profundo permanece CPU-only, la mejora principal "
        "vendrá de menor generación, un modelo más eficiente o aceleración GPU; aumentar "
        "el contexto o agregar más hilos no garantiza mayor throughput."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
