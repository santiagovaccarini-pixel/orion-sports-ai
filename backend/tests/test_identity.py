from __future__ import annotations

import unittest
from datetime import date

from backend.app.core.identity import (
    ORION_CREATOR_NAME,
    creator_age,
    creator_context,
    current_engine_fact,
    direct_creator_answer,
    institutional_identity_brief,
)


class OrionIdentityTests(unittest.TestCase):
    def test_creator_profile_is_available_as_validated_model_context(self) -> None:
        context = creator_context()
        self.assertIn(ORION_CREATOR_NAME, context)
        self.assertIn("16/01/2007", context)
        self.assertIn("Buenos Aires, Argentina", context)
        self.assertIn("Ciencia de Datos en ISTEA", context)
        self.assertIn("Atlético Mineiro", context)
        self.assertIn("Estudiantes de La Plata", context)
        self.assertIn("Excel", context)
        self.assertIn("Inteligencia Artificial", context)
        self.assertIn("PowerPoint", context)
        self.assertIn("VBA", context)
        self.assertIn("Licencia Pro", context)
        self.assertNotIn("Universidad de la Empresa", context)
        self.assertNotIn("Club Atlético Los Andes", context)

    def test_creator_age_is_derived_from_birth_date_instead_of_hardcoded(self) -> None:
        self.assertEqual(creator_age(date(2026, 1, 15)), 18)
        self.assertEqual(creator_age(date(2026, 1, 16)), 19)
        self.assertEqual(creator_age(date(2026, 8, 26)), 19)

    def test_no_creator_question_is_answered_by_python_phrase_matching(self) -> None:
        examples = (
            "¿Quién creó Orion?",
            "¿Quién te creó?",
            "Contame sobre la persona detrás de Orion",
            "¿Quién creó el motor de Orion?",
            "¿Quién creó el modelo gpt-oss que usa Orion?",
        )
        for query in examples:
            with self.subTest(query=query):
                self.assertIsNone(direct_creator_answer(query))

    def test_identity_context_distinguishes_orion_from_external_models(self) -> None:
        context = creator_context()
        self.assertIn("Santiago Vaccarini es el creador de Orion", context)
        self.assertIn("no creó gpt-oss", context)
        self.assertIn("Cloudflare Workers AI", context)
        self.assertIn("Ollama", context)
        self.assertIn("Interpretá semánticamente", context)

    def test_institutional_brief_resolves_default_referent_semantically(self) -> None:
        brief = institutional_identity_brief()
        self.assertIn(ORION_CREATOR_NAME, brief)
        self.assertIn("producto Orion", brief)
        self.assertIn("motor", brief)
        self.assertIn("no requiere búsqueda externa", brief)
        # Es contexto semántico, no una respuesta enlatada ni routing por frases.
        self.assertNotIn("¿", brief)

    def test_current_engine_fact_is_sourced_from_settings_not_hardcoded(self) -> None:
        # Live testing found Orion answering "GPT-4 by OpenAI" when asked which
        # engine powers it — a self-identity hallucination, since the real engine
        # is gpt-oss served via Cloudflare Workers AI. This fact must come from
        # settings (so it can't drift from the real deployment) and be stated
        # affirmatively so the model has a grounded answer instead of guessing.
        fact = current_engine_fact()
        self.assertIn("Cloudflare Workers AI", fact)
        self.assertIn("gpt-oss", fact)
        self.assertIn("No es GPT-4", fact)
        self.assertIn(fact, creator_context())
        self.assertIn(fact, institutional_identity_brief())


if __name__ == "__main__":
    unittest.main()
