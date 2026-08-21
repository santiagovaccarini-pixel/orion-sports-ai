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

    def test_prompt_answers_the_question_before_applying_sport_context(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("respondé primero la pregunta central", prompt)
        self.assertIn("nunca fuerces una relación deportiva", prompt)

    def test_prompt_uses_stable_markdown_without_latex(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("toda tabla debe tener una fila", prompt)
        self.assertIn("no uses latex", prompt)
        self.assertIn("símbolos unicode", prompt)

    def test_prompt_does_not_claim_live_research(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("este módulo no posee búsqueda web", prompt)
        self.assertIn("no afirmes haber buscado", prompt)

    def test_sport_and_mode_are_added_to_each_request(self) -> None:
        prompt = build_system_prompt(SportContext.CYCLING, SelectedMode.QUICK)
        self.assertIn("Contexto seleccionado: Ciclismo", prompt)
        self.assertIn("W/kg", prompt)
        self.assertIn("Modo Rápido", prompt)

    def test_football_is_the_initial_context_but_can_be_changed(self) -> None:
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
