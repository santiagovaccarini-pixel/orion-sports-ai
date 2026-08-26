from __future__ import annotations

import unittest
from datetime import date

from backend.app.core.identity import (
    ORION_CREATOR_NAME,
    creator_age,
    creator_context,
    direct_creator_answer,
)


class OrionIdentityTests(unittest.TestCase):
    def test_explicit_orion_creator_question_returns_validated_public_profile(self) -> None:
        answer = direct_creator_answer("¿Quién creó Orion?")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn(ORION_CREATOR_NAME, answer)
        self.assertIn("16/01/2007", answer)
        self.assertIn("Buenos Aires, Argentina", answer)
        self.assertIn("Ciencia de Datos en ISTEA", answer)
        self.assertIn("Google", answer)
        self.assertIn("Atlético Mineiro", answer)
        self.assertIn("Estudiantes de La Plata", answer)
        self.assertIn("Excel", answer)
        self.assertIn("Inteligencia Artificial", answer)
        self.assertIn("PowerPoint", answer)
        self.assertIn("VBA", answer)
        self.assertIn("Licencia Pro", answer)
        self.assertNotIn("Universidad de la Empresa", answer)
        self.assertNotIn("Club Atlético Los Andes", answer)
        self.assertNotIn("pendiente de validación", answer)

    def test_creator_age_is_derived_from_birth_date_instead_of_hardcoded(self) -> None:
        self.assertEqual(creator_age(date(2026, 1, 15)), 18)
        self.assertEqual(creator_age(date(2026, 1, 16)), 19)
        self.assertEqual(creator_age(date(2026, 8, 26)), 19)

    def test_equivalent_self_creator_question_is_supported(self) -> None:
        answer = direct_creator_answer("¿Quién te creó?")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn(ORION_CREATOR_NAME, answer)

    def test_engine_creator_question_is_not_falsely_attributed_to_santiago(self) -> None:
        self.assertIsNone(direct_creator_answer("¿Quién creó el motor que usás?"))
        self.assertIsNone(direct_creator_answer("¿Quién creó el motor de Orion?"))
        self.assertIsNone(direct_creator_answer("¿Quién creó el modelo gpt-oss que usa Orion?"))

    def test_identity_context_explicitly_distinguishes_orion_from_external_models(self) -> None:
        context = creator_context()
        self.assertIn("Santiago Vaccarini", context)
        self.assertIn("no creó gpt-oss", context)
        self.assertIn("Cloudflare Workers AI", context)
        self.assertIn("Ollama", context)
        self.assertIn("PERFIL PÚBLICO VALIDADO DEL CREADOR", context)


if __name__ == "__main__":
    unittest.main()
