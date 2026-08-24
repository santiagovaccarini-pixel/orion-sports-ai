from __future__ import annotations

import unittest

from backend.app.services.orchestrator import Intent, create_plan


class OrchestratorTests(unittest.TestCase):
    def test_current_question_uses_web_and_excludes_local_csv(self) -> None:
        plan = create_plan("¿Cuántos goles hizo un jugador esta temporada?", has_local_documents=True)
        self.assertEqual(plan.intent, Intent.WEB_RESEARCH)
        self.assertTrue(plan.use_web)
        self.assertFalse(plan.use_local_data)

    def test_csv_calculation_uses_local_calculator(self) -> None:
        plan = create_plan("Sumá la distancia de Ruan en el CSV", has_local_documents=True)
        self.assertEqual(plan.intent, Intent.CALCULATION)
        self.assertTrue(plan.use_local_data)
        self.assertTrue(plan.use_calculator)
        self.assertFalse(plan.use_web)

    def test_calculation_variants_are_recognized(self) -> None:
        for query in ("Calculá el promedio del CSV", "Sumá la distancia del archivo"):
            plan = create_plan(query, has_local_documents=True)
            self.assertEqual(plan.intent, Intent.CALCULATION)

    def test_general_question_ignores_loaded_documents(self) -> None:
        plan = create_plan("¿Qué cambia si reduzco el descanso entre series?", has_local_documents=True)
        self.assertEqual(plan.intent, Intent.GENERAL)
        self.assertFalse(plan.use_local_data)
        self.assertFalse(plan.use_web)

    def test_chart_request_uses_local_chart_only_when_data_is_targeted(self) -> None:
        plan = create_plan("Graficá la distancia de Ruan por período", has_local_documents=True)
        self.assertEqual(plan.intent, Intent.CHART)
        self.assertTrue(plan.use_chart)


if __name__ == "__main__":
    unittest.main()