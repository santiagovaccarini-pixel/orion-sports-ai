from __future__ import annotations

import unittest

from backend.app.core.identity import (
    ORION_CREATOR_NAME,
    creator_context,
    direct_creator_answer,
)


class OrionIdentityTests(unittest.TestCase):
    def test_explicit_orion_creator_question_returns_creator_without_unvalidated_career(self) -> None:
        answer = direct_creator_answer("¿Quién creó Orion?")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn(ORION_CREATOR_NAME, answer)
        self.assertIn("pendiente de validación", answer)
        self.assertNotIn("Universidad de la Empresa", answer)
        self.assertNotIn("Club Atlético Los Andes", answer)

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
        self.assertIn("Santiago Vaccarini creó Orion", context)
        self.assertIn("no creó gpt-oss", context)
        self.assertIn("Cloudflare Workers AI", context)
        self.assertIn("Ollama", context)


if __name__ == "__main__":
    unittest.main()
