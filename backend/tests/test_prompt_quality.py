from __future__ import annotations

import unittest

from backend.app.core.prompt import ORION_SYSTEM_PROMPT, build_system_prompt
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatRequest, SportContext


class PromptQualityTests(unittest.TestCase):
    def test_prompt_uses_validated_reasoning_frame(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("marco de razonamiento validado", prompt)
        self.assertIn("no vuelvas a decidir la intención por coincidencias de palabras", prompt)
        self.assertIn("respondé primero la pregunta central", prompt)

    def test_prompt_preserves_scientific_reasoning_invariants(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("observación, dato, cálculo, hipótesis e inferencia", prompt)
        self.assertIn("no conviertas correlación", prompt)
        self.assertIn("explicaciones alternativas", prompt)
        self.assertIn("variables faltantes", prompt)

    def test_prompt_preserves_data_integrity_and_source_boundaries(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("entidad, columna, período, unidad", prompt)
        self.assertIn("duplicados y faltantes", prompt)
        self.assertIn("no afirmes haber leído un archivo", prompt)
        self.assertIn("no afirmes haber buscado", prompt)
        self.assertIn("no inventes fuentes", prompt)

    def test_prompt_keeps_private_conventions_scoped(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("criterios propios del usuario o club", prompt)
        self.assertIn("no deben presentarse como", prompt)
        self.assertIn("verdad universal", prompt)

    def test_prompt_does_not_expose_chain_of_thought(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no muestres cadena de pensamiento interna", prompt)
        self.assertIn("conclusión", prompt)
        self.assertIn("supuestos relevantes", prompt)

    def test_user_query_no_longer_selects_prompt_fragments_by_keywords(self) -> None:
        rpe_prompt = build_system_prompt(
            SportContext.FOOTBALL,
            SelectedMode.QUICK,
            "¿Qué mide el RPE?",
        )
        unrelated_prompt = build_system_prompt(
            SportContext.FOOTBALL,
            SelectedMode.QUICK,
            "¿Qué es una transición ofensiva?",
        )
        self.assertEqual(rpe_prompt, unrelated_prompt)
        self.assertNotIn("FICHA FACTUAL PRIORITARIA", rpe_prompt)

    def test_football_context_keeps_high_value_domain_invariants(self) -> None:
        prompt = build_system_prompt(SportContext.FOOTBALL, SelectedMode.QUICK).lower()
        self.assertIn("posición/rol", prompt)
        self.assertIn("minutos o", prompt)
        self.assertIn("exposición", prompt)
        self.assertIn("métrica gps aislada no equivale", prompt)
        self.assertIn("asociación con lesión no prueba causalidad", prompt)
        self.assertIn("fuera de juego no significa balón fuera", prompt)

    def test_mode_and_sport_context_are_added(self) -> None:
        quick = build_system_prompt(SportContext.CYCLING, SelectedMode.QUICK)
        deep = build_system_prompt(SportContext.CYCLING, SelectedMode.DEEP)
        self.assertIn("Contexto: Ciclismo", quick)
        self.assertIn("W/kg", quick)
        self.assertIn("Modo Rápido", quick)
        self.assertIn("Modo Profundo", deep)

    def test_prompt_stays_compact(self) -> None:
        # The previous prompt had accumulated multiple factual cards and duplicated
        # reasoning rules. Keep a regression ceiling so latency does not silently grow.
        prompt = build_system_prompt(SportContext.FOOTBALL, SelectedMode.QUICK)
        self.assertLess(len(prompt), 4500)

    def test_markdown_contract_remains_simple(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("markdown simple", prompt)
        self.assertIn("no uses html ni latex", prompt)

    def test_football_is_default_but_context_can_change(self) -> None:
        default_request = ChatRequest(messages=[{"role": "user", "content": "Hola"}])
        general_request = ChatRequest(
            messages=[{"role": "user", "content": "Hola"}],
            sport="general",
        )
        self.assertEqual(default_request.sport, SportContext.FOOTBALL)
        self.assertEqual(general_request.sport, SportContext.GENERAL)


if __name__ == "__main__":
    unittest.main()
