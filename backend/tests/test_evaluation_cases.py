from __future__ import annotations

import json
import unittest

from backend.evals.run_cloud_evaluation import DIAGNOSTIC_PATH
from backend.evals.run_local_evaluation import (
    CASES_PATH,
    FOOTBALL_PATH,
    FOUNDATIONS_PATH,
    precheck,
)


class EvaluationCaseTests(unittest.TestCase):
    def test_case_ids_are_unique_and_rubrics_are_present(self) -> None:
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        ids = [case["id"] for case in cases]
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(ids), len(set(ids)))
        for case in cases:
            self.assertTrue(case["prompt"].strip())
            self.assertTrue(case["required_any"])
            self.assertIn(case["mode"], {"quick", "deep"})

    def test_precheck_handles_accents_and_forbidden_claims(self) -> None:
        case = {
            "required_any": [["frecuencia cardíaca"], ["RPE"]],
            "forbidden": ["previene lesiones"],
        }
        missing, forbidden = precheck(
            "La frecuencia cardiaca y el RPE son útiles; no garantiza resultados.",
            case,
        )
        self.assertEqual(missing, [])
        self.assertEqual(forbidden, [])

    def test_precheck_rejects_empty_answers(self) -> None:
        missing, forbidden = precheck(
            "   ",
            {"required_any": [], "forbidden": []},
        )
        self.assertEqual(missing, [["respuesta con contenido"]])
        self.assertEqual(forbidden, [])

    def test_foundation_curriculum_has_ten_correctable_cases(self) -> None:
        cases = json.loads(FOUNDATIONS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case["id"] for case in cases}), 10)
        for case in cases:
            self.assertTrue(case["correction"].strip())
            self.assertTrue(case["required_any"])
            self.assertTrue(case["forbidden"])

    def test_football_curriculum_has_ten_distinct_cases(self) -> None:
        cases = json.loads(FOOTBALL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case["id"] for case in cases}), 10)
        self.assertTrue(all(case["correction"].strip() for case in cases))
        self.assertIn("balón fuera del campo", cases[0]["forbidden"])


    def test_factual_cases_check_truth_not_only_reasoning(self) -> None:
        """The other datasets grade how Orion reasons; this one grades whether the
        facts it states are true, which is the failure class live testing found.
        """

        import json
        from pathlib import Path as _Path

        cases = json.loads(
            (_Path(__file__).resolve().parents[1] / "evals" / "factual_accuracy_cases.json")
            .read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(cases), 10)
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for case in cases:
            with self.subTest(case=case["id"]):
                # A factual case is only meaningful with both halves: what the
                # answer must contain, and the wrong answers it must not contain.
                self.assertTrue(case.get("required_any"))
                self.assertTrue(case.get("forbidden"))
                self.assertTrue(case.get("prompt", "").strip())
        categories = {case["category"] for case in cases}
        # The specific ways Orion has been wrong, each with a case that catches it.
        self.assertIn("factual_entity", categories)
        self.assertIn("factual_date", categories)
        self.assertIn("factual_honesty", categories)


    def test_a_case_that_never_answered_is_never_scored_as_passing(self) -> None:
        """A failed request must read as a failure, not as a blank.

        Hammering the deployment tripped its own rate limiter mid-battery. The two
        cases that got no answer carried no quality fields at all, so a reader
        scanning the report saw nothing where a failure belonged.
        """

        from pathlib import Path as _Path

        source = (
            _Path(__file__).resolve().parents[1] / "evals" / "run_cloud_evaluation.py"
        ).read_text(encoding="utf-8")
        marker = source.index("infrastructure_errors += 1")
        block = source[marker : marker + 800]
        self.assertIn('"quality_precheck_ok": False', block)
        self.assertIn('"not_evaluated": True', block)

    def test_diagnostic_cases_are_well_formed_and_unique(self) -> None:
        cases = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for case in cases:
            self.assertTrue(case.get("category", "").strip())
            self.assertIn(case.get("mode"), {"quick", "deep"})
            self.assertTrue(case.get("expected_behavior", "").strip())
            has_prompt = bool(str(case.get("prompt", "")).strip())
            has_messages = bool(case.get("messages"))
            self.assertTrue(
                has_prompt or has_messages,
                f"Caso {case.get('id')} no tiene prompt ni conversación.",
            )

    def test_diagnostic_cases_cover_evidence_synthesis_and_stale_affiliation(
        self,
    ) -> None:
        # Regression coverage for two live bugs found and fixed on 2026-08-27:
        # the reviewer refusing to sum disjoint verified partial totals, and
        # the planner assuming a player's club from stale training knowledge
        # instead of treating current affiliation as something to verify.
        cases = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
        ids = {case["id"] for case in cases}
        self.assertIn("web_partial_goal_totals_sum", ids)
        self.assertIn("web_player_current_club_not_assumed", ids)


if __name__ == "__main__":
    unittest.main()
