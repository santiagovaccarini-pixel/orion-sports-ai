from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

from backend.evals.run_local_evaluation import (
    CASES_PATH,
    FOOTBALL_PATH,
    FOUNDATIONS_PATH,
    precheck,
)


DEFAULT_CLOUD_URL = "https://orion-core-prototype.onrender.com/api/v1"
DATASETS = {
    "quality": CASES_PATH,
    "foundations": FOUNDATIONS_PATH,
    "football": FOOTBALL_PATH,
}


def _load_cases(dataset: str) -> list[dict[str, object]]:
    return json.loads(DATASETS[dataset].read_text(encoding="utf-8"))


def _base_url(value: str) -> str:
    clean = value.strip().rstrip("/")
    if not clean.startswith(("http://", "https://")):
        raise ValueError("La URL cloud debe empezar con http:// o https://.")
    if not clean.endswith("/api/v1"):
        clean += "/api/v1"
    return clean


def _stream_case(
    client: httpx.Client,
    *,
    base_url: str,
    case: dict[str, object],
) -> dict[str, object]:
    prompt = str(case["prompt"])
    request_payload = {
        "messages": [{"role": "user", "content": prompt}],
        "mode": case["mode"],
        "sport": case.get("sport", "football"),
        "allow_busy": False,
    }
    started = time.perf_counter()
    first_text_ms: float | None = None
    meta: dict[str, object] | None = None
    done: dict[str, object] | None = None
    answer_parts: list[str] = []
    charts: list[dict[str, object]] = []
    stream_error: dict[str, object] | None = None

    with client.stream(
        "POST",
        f"{base_url}/chat/stream",
        json=request_payload,
    ) as response:
        status_code = response.status_code
        if status_code != 200:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "http_status": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": body[:2000],
            }

        for line in response.iter_lines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                stream_error = {
                    "type": "invalid_ndjson",
                    "message": line[:1000],
                }
                break
            if not isinstance(event, dict):
                stream_error = {
                    "type": "invalid_event",
                    "message": repr(event)[:1000],
                }
                break
            event_type = event.get("type")
            if event_type == "meta":
                meta = event
            elif event_type == "content":
                content = event.get("content")
                if isinstance(content, str) and content:
                    if first_text_ms is None:
                        first_text_ms = round(
                            (time.perf_counter() - started) * 1000,
                            2,
                        )
                    answer_parts.append(content)
            elif event_type == "chart":
                chart = event.get("chart")
                if isinstance(chart, dict):
                    charts.append(chart)
            elif event_type == "done":
                done = event
            elif event_type == "error":
                stream_error = event
                break

    return {
        "http_status": 200,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "first_text_ms": first_text_ms,
        "meta": meta,
        "done": done,
        "answer": "".join(answer_parts),
        "charts": charts,
        "stream_error": stream_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta las baterías de Orion contra el deployment cloud real mediante "
            "el mismo stream NDJSON de la interfaz y adjunta la traza exacta de cada caso."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("ORION_CLOUD_URL", DEFAULT_CLOUD_URL),
        help="URL pública de Orion Core (con o sin /api/v1).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="0 ejecuta todos los casos de cada dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("orion-cloud-eval.json"),
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Devuelve código 1 si falla cualquier prechequeo de calidad.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ORION_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta ORION_API_KEY en esta terminal. No pegues la clave en el código ni en Git."
        )

    base_url = _base_url(args.base_url)
    headers = {"X-Orion-Api-Key": api_key}
    report: dict[str, object] = {
        "base_url": base_url,
        "started_at_epoch": time.time(),
        "datasets": {},
    }
    infrastructure_errors = 0
    quality_failures = 0

    with httpx.Client(timeout=300.0, headers=headers) as client:
        status_response = client.get(f"{base_url}/status")
        status_response.raise_for_status()
        status_payload = status_response.json()
        if status_payload.get("model_provider") != "cloudflare":
            raise SystemExit(
                "La evaluación cloud se negó a continuar porque /status no reporta cloudflare."
            )
        report["status"] = {
            "version": status_payload.get("version"),
            "model_provider": status_payload.get("model_provider"),
            "quick_model": status_payload.get("quick_model"),
            "deep_model": status_payload.get("deep_model"),
        }

        datasets_payload: dict[str, list[dict[str, object]]] = {}
        for dataset in args.datasets:
            cases = _load_cases(dataset)
            if args.limit_per_dataset > 0:
                cases = cases[: args.limit_per_dataset]
            dataset_results: list[dict[str, object]] = []

            for index, case in enumerate(cases, start=1):
                prompt = str(case["prompt"])
                streamed = _stream_case(
                    client,
                    base_url=base_url,
                    case=case,
                )
                item: dict[str, object] = {
                    "id": case["id"],
                    "dataset": dataset,
                    "index": index,
                    **streamed,
                }

                if streamed.get("http_status") != 200 or streamed.get("stream_error"):
                    infrastructure_errors += 1
                    dataset_results.append(item)
                    continue

                answer = str(streamed.get("answer") or "")
                missing_groups, forbidden_hits = precheck(answer, case)
                quality_ok = not missing_groups and not forbidden_hits
                if not quality_ok:
                    quality_failures += 1
                item.update(
                    {
                        "quality_precheck_ok": quality_ok,
                        "missing_groups": missing_groups,
                        "forbidden_hits": forbidden_hits,
                    }
                )

                meta = streamed.get("meta")
                trace_id = meta.get("trace_id") if isinstance(meta, dict) else None
                if isinstance(trace_id, str) and trace_id:
                    trace_response = client.get(
                        f"{base_url}/diagnostics/traces/{trace_id}"
                    )
                    if trace_response.status_code == 200:
                        trace = trace_response.json()
                        if trace.get("question") == prompt:
                            item["trace"] = trace
                        else:
                            item["trace_warning"] = (
                                "La traza pedida por ID no coincide con el prompt del caso."
                            )
                    else:
                        item["trace_warning"] = (
                            f"Traza {trace_id} no disponible: HTTP "
                            f"{trace_response.status_code}."
                        )
                else:
                    item["trace_warning"] = "El stream no devolvió trace_id."

                dataset_results.append(item)

            datasets_payload[dataset] = dataset_results

    report["datasets"] = datasets_payload
    report["finished_at_epoch"] = time.time()
    report["infrastructure_errors"] = infrastructure_errors
    report["quality_failures"] = quality_failures
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_cases = sum(len(items) for items in datasets_payload.values())
    print(f"Evaluación cloud terminada: {total_cases} casos.")
    print(f"Errores de infraestructura: {infrastructure_errors}.")
    print(f"Prechequeos de calidad fallidos: {quality_failures}.")
    print(f"Reporte: {args.output}")

    if infrastructure_errors:
        return 2
    if args.strict_quality and quality_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
