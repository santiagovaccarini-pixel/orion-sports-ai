from __future__ import annotations

import unittest

from backend.app.core.prompt import ORION_SYSTEM_PROMPT


class PromptQualityTests(unittest.TestCase):
    def test_prompt_rejects_simplistic_load_dichotomy(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no la presentes como exclusivamente subjetiva", prompt)
        self.assertIn("no reduzcas carga externa a volumen", prompt)

    def test_prompt_requires_cautious_injury_language(self) -> None:
        prompt = ORION_SYSTEM_PROMPT.lower()
        self.assertIn("no previene ni predice lesiones por sí sola", prompt)
        self.assertIn("no inventes fuentes", prompt)


if __name__ == "__main__":
    unittest.main()
