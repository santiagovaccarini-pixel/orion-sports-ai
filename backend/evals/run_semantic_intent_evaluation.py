from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.semantic_planner import create_semantic_plan


DATASETS = {
    "core": Path(__file__).with_name("semantic_intent_cases.json"),
    "generalization": Path(__file__).with_name("semantic_generalization_cases.json"),
}


def _fold(value: str) -> str:
    return value.casefold().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def _contains_any(values: list[str], expected: list[str]) -> bool:
    haystack = " | ".join(values)
    folded = _fold(haystack)
    return any(_fold(item) in folded for item in expected)


def _contains_all(values: list[str], groups: list[list[str]]) -> bool:
    return all(_contains_any(values, group) for group in groups)


def _score(plan: Any, expected: dict[str, Any]) -> tuple[int, int, list[str]]:
    passed = 0
    total = 0
    failures: list[str] = []

    for field in (
        "task_type",
        "inference_type",
        "comparison",
        "causal_claim_risk",
        "needs_private_memory",
        "needs_local_data",
        "needs_web",
        "needs_global_knowledge",
        "requires_clarification",
        "referenced_previous_context",
    ):
        if field not in expected:
            continue
        total += 1
        actual = getattr(plan, field)
        if actual == expected[field]:
            passed += 1
        else:
            failures.append(f"{field}: esperado={expected[field]!r}, actual={actual!r}")

    if expected_domains := expected.get("domain_contains"):
        total += 1
        if _contains_any([plan.domain], expected_domains):
            passed += 1
        else:
            failures.append(
                f"domain: esperado alguno de {expected_domains!r}, actual={plan.domain!r}"
            )

    semantic_values = [
        *plan.concept_ids,
        *plan.concepts,
        *plan.retrieval_queries,
        plan.user_goal,
    ]
    if expected_concepts := expected.get("concepts_any"):
        total += 1
        if _contains_any(semantic_values, expected_concepts):
            passed += 1
        else:
            failures.append(
                "concepts_any: no apareció ninguno de "
                f"{expected_concepts!r}; ids={plan.concept_ids!r}; labels={plan.concepts!r}"
            )

    if concept_groups := expected.get("concepts_all"):
        total += 1
        if _contains_all(semantic_values, concept_groups):
            passed += 1
        else:
            failures.append(
                "concepts_all: no se cubrieron todos los grupos "
                f"{concept_groups!r}; ids={plan.concept_ids!r}; labels={plan.concepts!r}"
            )

    if minimum := expected.get("complexity_min"):
        total += 1
        if plan.complexity >= float(minimum):
            passed += 1
        else:
            failures.append(
                f"complexity: esperado>={minimum}, actual={plan.complexity:.2f}"
            )

    if maximum := expected.get("complexity_max"):
        total += 1
        if plan.complexity <= float(maximum):
            passed += 1
        else:
            failures.append(
                f"complexity: esperado<={maximum}, actual={plan.complexity:.2f}"
            )

    return passed, total, failures


async def run(dataset: str = "core", limit: int | None = None) -> int:
    cases_path = DATASETS[dataset]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]

    settings = replace(get_settings(), semantic_planner_enabled=True)
    total_passed = 0
    total_checks = 0
    failed_cases = 0

    for case in cases:
        messages = [ChatMessage(**message) for message in case["messages"]]
        plan = await create_semantic_plan(
            settings,
            messages,
            SportContext(case.get("sport", "football")),
            has_local_documents=bool(case.get("has_local_documents", False)),
        )
        passed, checks, failures = _score(plan, case["expect"])
        total_passed += passed
        total_checks += checks
        if failures:
            failed_cases += 1
            status = "FAIL"
        else:
            status = "OK"

        print(
            f"[{status}] {case['id']}: {passed}/{checks} | "
            f"task={plan.task_type} inference={plan.inference_type} domain={plan.domain} "
            f"complexity={plan.complexity:.2f} ambiguity={plan.ambiguity:.2f} "
            f"confidence={plan.confidence:.2f}"
        )
        for failure in failures:
            print(f"    - {failure}")
        print(f"    objetivo: {plan.user_goal}")
        print(f"    concept_ids: {', '.join(plan.concept_ids) or '-'}")
        print(f"    conceptos: {', '.join(plan.concepts) or '-'}")

    accuracy = (total_passed / total_checks * 100.0) if total_checks else 0.0
    print(
        f"\nDataset: {dataset} | Resultado: {total_passed}/{total_checks} checks "
        f"({accuracy:.1f}%), {failed_cases}/{len(cases)} casos con fallos."
    )
    return 1 if failed_cases else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa intención, inferencia y selección conceptual de Orion."
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="core")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.dataset, args.limit)))


if __name__ == "__main__":
    main()
