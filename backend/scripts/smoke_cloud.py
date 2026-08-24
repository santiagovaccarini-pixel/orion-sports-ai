from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class SmokeResult:
    model: str
    provider: str
    first_token_ms: float | None
    total_ms: float
    content_characters: int


def _base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("ORION_CLOUD_URL está vacío.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("ORION_CLOUD_URL debe ser una URL http(s) válida.")
    if not value.endswith("/api/v1"):
        value = f"{value}/api/v1"
    return value


def _event(line: str) -> dict[str, object] | None:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Orion devolvió NDJSON inválido durante el smoke test.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Orion devolvió un evento de streaming inesperado.")
    return payload


def run_smoke(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float = 120.0,
) -> SmokeResult:
    headers = {"X-Orion-Api-Key": api_key}
    timeout = httpx.Timeout(timeout_seconds)

    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise RuntimeError("El health check de Orion no devolvió status=ok.")

        status = client.get(f"{base_url}/status", headers=headers)
        status.raise_for_status()
        status_payload = status.json()
        provider = str(status_payload.get("model_provider") or "unknown")
        if not status_payload.get("model_provider_online"):
            raise RuntimeError(
                f"El proveedor configurado ({provider}) no figura como disponible."
            )

        started = time.perf_counter()
        first_token_ms: float | None = None
        model = "unknown"
        content_characters = 0
        saw_meta = False
        saw_done = False

        with client.stream(
            "POST",
            f"{base_url}/chat/stream",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Respondé únicamente: ORION CLOUD OK. "
                            "No agregues ninguna explicación."
                        ),
                    }
                ],
                "mode": "quick",
                "sport": "general",
                "allow_busy": False,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                payload = _event(line)
                if payload is None:
                    continue
                event_type = payload.get("type")
                if event_type == "meta":
                    saw_meta = True
                    model = str(payload.get("model") or "unknown")
                elif event_type == "content":
                    content = payload.get("content")
                    if isinstance(content, str) and content:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        content_characters += len(content)
                elif event_type == "done":
                    saw_done = True
                elif event_type == "error":
                    message = str(payload.get("message") or "Error cloud sin detalle")
                    raise RuntimeError(f"Orion devolvió un error: {message}")

        total_ms = (time.perf_counter() - started) * 1000

    if not saw_meta:
        raise RuntimeError("Faltó el evento meta del streaming.")
    if not saw_done:
        raise RuntimeError("Faltó el evento done del streaming.")
    if content_characters == 0:
        raise RuntimeError("El modelo no produjo contenido.")

    return SmokeResult(
        model=model,
        provider=provider,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        content_characters=content_characters,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test seguro para Orion Cloud. No imprime credenciales."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("ORION_CLOUD_URL", ""),
        help="URL pública de Orion Cloud. También puede usarse ORION_CLOUD_URL.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ORION_API_KEY", "")
    if not api_key:
        print("ERROR: falta ORION_API_KEY en el entorno local.", file=sys.stderr)
        return 2

    try:
        base_url = _base_url(args.url)
        result = run_smoke(base_url=base_url, api_key=api_key)
    except Exception as exc:
        print(f"SMOKE CLOUD: FALLÓ — {exc}", file=sys.stderr)
        return 1

    first = (
        f"{result.first_token_ms:.0f} ms"
        if result.first_token_ms is not None
        else "sin medir"
    )
    print("SMOKE CLOUD: OK")
    print(f"Proveedor: {result.provider}")
    print(f"Modelo: {result.model}")
    print(f"Primer texto: {first}")
    print(f"Tiempo total: {result.total_ms:.0f} ms")
    print(f"Caracteres recibidos: {result.content_characters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
