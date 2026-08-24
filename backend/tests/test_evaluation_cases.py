from __future__ import annotations

import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
