from __future__ import annotations

import argparse
import json
import os
import unicodedata
from pathlib import Path

import httpx


CASES_PATH = Path(__file__).with_name("sports_quality_cases.json")
FOUNDATIONS_PATH = Path(__file__).with_name("sports_foundations_cases.json")
FOOTBALL_PATH = Path(__file__).with_name("football_cases.json")


def normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.lower())
    return "".join(character for character in folded if not unicodedata.combining(character))


def precheck(answer: str, case: dict[str, object]) -> tuple[list[list[str]], list[str]]:
    if not answer.strip():
        return [["respuesta con contenido"]], []
    normalized_answer = normalize(answer)
    missing_groups: list[list[str]] = []
    for raw_group in case["required_any"]:  # type: ignore[index]
        group = [str(term) for term in raw_group]
        if not any(normalize(term) in normalized_answer for term in group):
            missing_groups.append(group)

    forbidden_hits = [
        str(term)
        for term in case["forbidden"]  # type: ignore[index]
        if normalize(str(term)) in normalized_answer
    ]
    return missing_groups, forbidden_hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta la batería local de calidad de Orion."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765/api/v1",
        help="URL del núcleo local de Orion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Cantidad de casos a ejecutar. Usá 8 para la batería completa.",
    )
    parser.add_argument(
        "--dataset",
        choices=("quality", "foundations", "football"),
        default="quality",
        help="Batería a ejecutar. Usá foundations para las 10 preguntas básicas.",
    )
    args = parser.parse_args()

    cases_path = {
        "quality": CASES_PATH,
        "foundations": FOUNDATIONS_PATH,
        "football": FOOTBALL_PATH,
    }[args.dataset]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    selected_cases = cases[: max(1, min(args.limit, len(cases)))]
    passed = 0

    print("Evaluación local de Orion · prechequeo automático")
    print("Cada respuesta requiere revisión humana aunque el prechequeo apruebe.\n")

    headers = {}
    if api_key := os.getenv("ORION_API_KEY"):
        headers["X-Orion-Api-Key"] = api_key

    with httpx.Client(timeout=300.0, headers=headers) as client:
        for index, case in enumerate(selected_cases, start=1):
            print(f"[{index}/{len(selected_cases)}] {case['id']}")
            response = client.post(
                f"{args.base_url.rstrip('/')}/chat",
                json={
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "mode": case["mode"],
                    "allow_busy": False,
                },
            )
            if response.status_code != 200:
                print(f"  NO EJECUTADO · HTTP {response.status_code}: {response.text}\n")
                continue

            answer = response.json()["content"]
            missing_groups, forbidden_hits = precheck(answer, case)
            if not missing_groups and not forbidden_hits:
                passed += 1
                print("  PRECHEQUEO OK")
            else:
                print("  REVISAR")
                if missing_groups:
                    print(f"  Conceptos no detectados: {missing_groups}")
                if forbidden_hits:
                    print(f"  Afirmaciones riesgosas detectadas: {forbidden_hits}")
                if correction := case.get("correction"):
                    print(f"  Corrección esperada: {correction}")
            print(f"  Respuesta: {answer}\n")

    print(f"Resultado preliminar: {passed}/{len(selected_cases)} casos.")
    return 0 if passed == len(selected_cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
