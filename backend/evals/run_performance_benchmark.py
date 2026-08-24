from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import httpx


DEFAULT_PROMPT = "Explicá en tres puntos qué es la carga interna en fútbol."


def run_once(
    client: httpx.Client,
    *,
    base_url: str,
    prompt: str,
    mode: str,
) -> dict[str, float | int | None]:
    started_at = time.perf_counter()
    first_text_ms: float | None = None
    final_event: dict[str, object] | None = None

    with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/chat/stream",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "mode": mode,
            "allow_busy": True,
        },
    ) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "content" and first_text_ms is None:
                first_text_ms = (time.perf_counter() - started_at) * 1_000
            elif event.get("type") == "error":
                raise RuntimeError(str(event.get("message", "Error de Orion")))
            elif event.get("type") == "done":
                final_event = event

    if final_event is None:
        raise RuntimeError("La respuesta terminó sin métricas finales.")
    return {
        "first_text_ms": round(first_text_ms, 2) if first_text_ms else None,
        "wall_duration_ms": round((time.perf_counter() - started_at) * 1_000, 2),
        "ollama_duration_ms": final_event.get("total_duration_ms"),
        "load_duration_ms": final_event.get("load_duration_ms"),
        "tokens_per_second": final_event.get("tokens_per_second"),
        "completion_tokens": final_event.get("completion_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mide tiempo al primer texto y velocidad local de Orion."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765/api/v1",
    )
    parser.add_argument("--mode", choices=("quick", "deep"), default="quick")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    results: list[dict[str, float | int | None]] = []
    headers = {}
    if api_key := os.getenv("ORION_API_KEY"):
        headers["X-Orion-Api-Key"] = api_key

    with httpx.Client(timeout=600.0, headers=headers) as client:
        for index in range(max(1, args.runs)):
            result = run_once(
                client,
                base_url=args.base_url,
                prompt=args.prompt,
                mode=args.mode,
            )
            results.append(result)
            print(f"Ejecución {index + 1}: {json.dumps(result, ensure_ascii=False)}")

    first_text_values = [
        float(result["first_text_ms"])
        for result in results
        if result["first_text_ms"] is not None
    ]
    wall_values = [float(result["wall_duration_ms"]) for result in results]
    if first_text_values:
        print(f"Mediana al primer texto: {statistics.median(first_text_values):.0f} ms")
    print(f"Mediana total: {statistics.median(wall_values):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
