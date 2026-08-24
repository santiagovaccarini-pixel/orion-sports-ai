from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.domain.schemas import ChatMessage, SportContext
from backend.app.services.semantic_planner import create_semantic_plan


CASES_PATH = Path(__file__).with_name("reasoning_paraphrase_cases.json")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


async def main_async() -> int:
    groups = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = replace(get_settings(), semantic_planner_enabled=True)
    passed_groups = 0

    for group in groups:
        plans = []
        sport = SportContext(group.get("sport", "football"))
        for text in group["variants"]:
            plan = await create_semantic_plan(
                settings,
                [ChatMessage(role="user", content=text)],
                sport,
                has_local_documents=False,
            )
            plans.append(plan)

        required = set(group["required_concepts"])
        required_ok = all(required <= set(plan.concept_ids) for plan in plans)
        inference_types = {plan.inference_type for plan in plans}
        task_types = {plan.task_type for plan in plans}
        pairwise = [
            _jaccard(set(left.concept_ids), set(right.concept_ids))
            for left, right in combinations(plans, 2)
        ]
        mean_jaccard = sum(pairwise) / len(pairwise) if pairwise else 1.0

        # We do not demand identical optional concepts. The invariant is that the core
        # ontology concepts survive every paraphrase and that the reasoning form remains
        # reasonably stable. Jaccard >= .5 allows useful contextual additions.
        stable = (
            required_ok
            and len(inference_types) == 1
            and len(task_types) <= 2
            and mean_jaccard >= 0.5
        )
        passed_groups += int(stable)
        status = "OK" if stable else "FAIL"
        print(
            f"[{status}] {group['id']} | required={required_ok} | "
            f"inference={sorted(inference_types)} | tasks={sorted(task_types)} | "
            f"concept_jaccard={mean_jaccard:.2f}"
        )
        for index, plan in enumerate(plans, start=1):
            print(
                f"    v{index}: inference={plan.inference_type} task={plan.task_type} "
                f"ids={plan.concept_ids} confidence={plan.confidence:.2f}"
            )
            if plan.claim_to_evaluate:
                print(f"        claim={plan.claim_to_evaluate}")

    total = len(groups)
    accuracy = passed_groups / total * 100.0 if total else 0.0
    print(
        f"\nConsistencia de razonamiento: {passed_groups}/{total} grupos "
        f"({accuracy:.1f}%)."
    )
    return 0 if passed_groups == total else 1


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
