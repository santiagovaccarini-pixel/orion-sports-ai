from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

from backend.evals.run_local_evaluation import (
    FOOTBALL_PATH,
    FOUNDATIONS_PATH,
    CASES_PATH,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta las baterías de Orion contra el deployment cloud real y adjunta "
            "diagnóstico end-to-end por caso."
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

        for dataset in args.datasets:
            cases = _load_cases(dataset)
            if args.limit_per_dataset > 0:
                cases = cases[: args.limit_per_dataset]
            dataset_results: list[dict[str, object]] = []

            for index, case in enumerate(cases, start=1):
                prompt = str(case["prompt"])
                started = time.perf_counter()
                response = client.post(
                    f"{base_url}/chat",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "mode": case["mode"],
                        "allow_busy": False,
                    },
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                item: dict[str, object] = {
                    "id": case["id"],
                    "dataset": dataset,
                    "index": index,
                    "http_status": response.status_code,
                    "latency_ms": latency_ms,
                }

                if response.status_code != 200:
                    infrastructure_errors += 1
                    item["error"] = response.text[:2000]
                    dataset_results.append(item)
                    continue

                payload = response.json()
                answer = str(payload.get("content") or "")
                missing_groups, forbidden_hits = precheck(answer, case)
                quality_ok = not missing_groups and not forbidden_hits
                if not quality_ok:
                    quality_failures += 1
                item.update(
                    {
                        "quality_precheck_ok": quality_ok,
                        "missing_groups": missing_groups,
                        "forbidden_hits": forbidden_hits,
                        "answer": answer,
                        "model": payload.get("model"),
                        "prompt_tokens": payload.get("prompt_tokens"),
                        "completion_tokens": payload.get("completion_tokens"),
                    }
                )

                trace_response = client.get(f"{base_url}/diagnostics/traces/latest")
                if trace_response.status_code == 200:
                    trace = trace_response.json()
                    if trace.get("question") == prompt:
                        item["trace"] = trace
                    else:
                        item["trace_warning"] = (
                            "La última traza no coincide con la pregunta del caso; "
                            "posible concurrencia externa durante el benchmark."
                        )
                else:
                    item["trace_warning"] = (
                        f"Diagnóstico no disponible: HTTP {trace_response.status_code}."
                    )

                dataset_results.append(item)

            report["datasets"][dataset] = dataset_results  # type: ignore[index]

    report["finished_at_epoch"] = time.time()
    report["infrastructure_errors"] = infrastructure_errors
    report["quality_failures"] = quality_failures
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_cases = sum(
        len(items) for items in report["datasets"].values()  # type: ignore[union-attr]
    )
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
