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
    meta_ms: float | None = None
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
            event_type = event.get("type")
            if event_type == "meta" and meta_ms is None:
                # /chat/stream only yields meta after _prepare_chat(), therefore this
                # isolates semantic planning + preflight overhead before generation.
                meta_ms = (time.perf_counter() - started_at) * 1_000
            elif event_type == "content" and first_text_ms is None:
                first_text_ms = (time.perf_counter() - started_at) * 1_000
            elif event_type == "error":
                raise RuntimeError(str(event.get("message", "Error de Orion")))
            elif event_type == "done":
                final_event = event

    if final_event is None:
        raise RuntimeError("La respuesta terminó sin métricas finales.")

    first_text_after_meta_ms = None
    if first_text_ms is not None and meta_ms is not None:
        first_text_after_meta_ms = max(0.0, first_text_ms - meta_ms)

    return {
        "reasoning_preflight_ms": round(meta_ms, 2) if meta_ms is not None else None,
        "first_text_ms": round(first_text_ms, 2) if first_text_ms is not None else None,
        "first_text_after_preflight_ms": (
            round(first_text_after_meta_ms, 2)
            if first_text_after_meta_ms is not None
            else None
        ),
        "wall_duration_ms": round((time.perf_counter() - started_at) * 1_000, 2),
        "ollama_duration_ms": final_event.get("total_duration_ms"),
        "load_duration_ms": final_event.get("load_duration_ms"),
        "prompt_eval_duration_ms": final_event.get("prompt_eval_duration_ms"),
        "prompt_tokens": final_event.get("prompt_tokens"),
        "tokens_per_second": final_event.get("tokens_per_second"),
        "completion_tokens": final_event.get("completion_tokens"),
    }


def _median(results: list[dict[str, float | int | None]], key: str) -> float | None:
    values = [float(item[key]) for item in results if item.get(key) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mide por separado el razonamiento previo, tiempo al primer texto y "
            "throughput de generación de Orion."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/api/v1")
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

    if (value := _median(results, "reasoning_preflight_ms")) is not None:
        print(f"Mediana razonamiento/preflight: {value:.0f} ms")
    if (value := _median(results, "first_text_ms")) is not None:
        print(f"Mediana al primer texto: {value:.0f} ms")
    if (value := _median(results, "first_text_after_preflight_ms")) is not None:
        print(f"Mediana preflight → primer texto: {value:.0f} ms")
    if (value := _median(results, "wall_duration_ms")) is not None:
        print(f"Mediana total: {value:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
