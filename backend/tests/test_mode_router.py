from __future__ import annotations

import unittest

from backend.app.domain.models import ConversationMessage, SelectedMode
from backend.app.services.mode_router import recommend_mode


class ModeRouterTests(unittest.TestCase):
    def test_direct_question_uses_quick_mode(self) -> None:
        result = recommend_mode(
            [ConversationMessage(content="¿Qué es la carga interna?")]
        )
        self.assertEqual(result.mode, SelectedMode.QUICK)

    def test_complex_analysis_uses_deep_mode(self) -> None:
        result = recommend_mode(
            [
                ConversationMessage(
                    content=(
                        "Analizá y compará dos modelos para predecir lesiones. "
                        "Explicá riesgos, validación, evidencia y supuestos."
                    ),
                )
            ]
        )
        self.assertEqual(result.mode, SelectedMode.DEEP)


if __name__ == "__main__":
    unittest.main()
