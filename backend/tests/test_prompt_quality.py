from __future__ import annotations

import unittest

from backend.app.core.prompt import ORION_SYSTEM_PROMPT, build_system_prompt
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, SportContext


class PromptQualityTests(unittest.TestCase):
    def test_prompt_rejects_simplistic_load_dichotomy(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no la presentes como exclusivamente subjetiva", prompt)
        self.assertIn("no reduzcas carga externa a volumen", prompt)

    def test_prompt_requires_cautious_injury_language(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no previene ni predice lesiones por sí sola", prompt)
        self.assertIn("no inventes fuentes", prompt)

    def test_prompt_answers_question_before_sport_context(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("respondé primero la pregunta central", prompt)
        self.assertIn("nunca fuerces una relación deportiva", prompt)

    def test_prompt_adapts_length_to_question(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("estimá cuánto texto necesita", prompt)
        self.assertIn("eliminá repeticiones", prompt)
        quick_prompt = build_system_prompt(
            SportContext.GENERAL, SelectedMode.QUICK
        ).lower()
        self.assertIn("nunca rellenes el presupuesto disponible", quick_prompt)

    def test_prompt_chooses_visual_explanations_by_purpose(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("reduzca mejor el esfuerzo de comprensión", prompt)
        self.assertIn("no inventes imágenes", prompt)
        self.assertIn("responsabilidad de la interfaz", prompt)

    def test_prompt_verifies_entity_period_scope_and_units(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("entidad, la columna, el período y la", prompt)
        self.assertIn("unidad coincidan", prompt)
        self.assertIn("no las sumes juntas", prompt)
        self.assertIn("misma entidad, métrica, alcance, período y unidad", prompt)

    def test_prompt_rejects_unrequested_precision(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no agregues cifras, rangos, tiempos", prompt)
        self.assertIn("nunca uses números", prompt)
        self.assertIn("inventados para dar apariencia de precisión", prompt)

    def test_prompt_contains_corrective_sports_foundations(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("el rpe es una valoración subjetiva", prompt)
        self.assertIn("el calentamiento prepara progresivamente", prompt)
        self.assertIn("causalidad sin considerar confusores", prompt)

    def test_prompt_distinguishes_offside_from_ball_out_of_play(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("fuera de juego no significa que el balón haya salido", prompt)

    def test_prompt_does_not_turn_failed_research_into_nonexistence(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no concluyas que una", prompt)
        self.assertIn("no se pudo confirmar", prompt)

    def test_prompt_does_not_inject_factual_cards_from_query_words(self) -> None:
        rpe_prompt = build_system_prompt(
            SportContext.GENERAL,
            SelectedMode.QUICK,
            "¿Qué mide el RPE?",
        )
        unrelated_prompt = build_system_prompt(
            SportContext.GENERAL,
            SelectedMode.QUICK,
            "Explicame una idea general",
        )
        self.assertEqual(rpe_prompt, unrelated_prompt)
        self.assertNotIn("FICHA FACTUAL PRIORITARIA", rpe_prompt)

    def test_prompt_uses_stable_markdown_without_latex(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("toda tabla debe tener encabezado", prompt)
        self.assertIn("no uses latex", prompt)
        self.assertIn("símbolos unicode", prompt)
        self.assertIn("no uses tablas dibujadas", prompt)

    def test_prompt_does_not_claim_live_research_without_evidence(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("si el sistema no entrega investigación web", prompt)
        self.assertIn("no afirmes haber buscado", prompt)

    def test_prompt_rejects_lexical_substitution_for_semantic_plan(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no sustituyas esa interpretación por coincidencias de palabras", prompt)
        self.assertIn("fuentes descartadas", prompt)

    def test_sport_and_mode_are_added_to_each_request(self) -> None:
        prompt = build_system_prompt(SportContext.CYCLING, SelectedMode.QUICK)
        self.assertIn("Contexto seleccionado: Ciclismo", prompt)
        self.assertIn("W/kg", prompt)
        self.assertIn("Modo Rápido", prompt)

    def test_football_is_initial_context_but_can_be_changed(self) -> None:
        default_request = ChatRequest(
            messages=[{"role": "user", "content": "Hola"}]
        )
        general_request = ChatRequest(
            messages=[{"role": "user", "content": "Hola"}],
            sport="general",
        )
        self.assertEqual(default_request.sport, SportContext.FOOTBALL)
        self.assertEqual(general_request.sport, SportContext.GENERAL)


if __name__ == "__main__":
    unittest.main()
