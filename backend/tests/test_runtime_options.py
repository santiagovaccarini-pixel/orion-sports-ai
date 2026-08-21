from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.providers.ollama import runtime_options


class RuntimeOptionsTests(unittest.TestCase):
    def test_quick_mode_uses_conservative_thread_budget(self) -> None:
        options = runtime_options(Settings(), SelectedMode.QUICK)
        self.assertEqual(options["num_thread"], 6)
        self.assertEqual(options["num_ctx"], 8192)

    def test_deep_mode_has_separate_thread_budget(self) -> None:
        options = runtime_options(Settings(), SelectedMode.DEEP)
        self.assertEqual(options["num_thread"], 8)
        self.assertEqual(options["num_ctx"], 16384)


if __name__ == "__main__":
    unittest.main()
