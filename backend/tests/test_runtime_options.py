from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.ollama import runtime_options, select_history


class RuntimeOptionsTests(unittest.TestCase):
    def test_quick_mode_uses_performance_budget(self) -> None:
        options = runtime_options(Settings(), SelectedMode.QUICK)
        self.assertEqual(options["num_thread"], 8)
        self.assertEqual(options["num_ctx"], 4096)
        self.assertEqual(options["num_predict"], 384)

    def test_deep_mode_has_separate_thread_budget(self) -> None:
        options = runtime_options(Settings(), SelectedMode.DEEP)
        self.assertEqual(options["num_thread"], 8)
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 1024)

    def test_quick_history_keeps_recent_turns_only(self) -> None:
        messages = [
            ChatMessage(
                role="user" if index % 2 == 0 else "assistant",
                content=f"mensaje {index}",
            )
            for index in range(12)
        ]
        selected = select_history(Settings(), SelectedMode.QUICK, messages)
        self.assertLessEqual(len(selected), 8)
        self.assertEqual(selected[-1].content, "mensaje 11")
        self.assertEqual(selected[0].role, "user")


if __name__ == "__main__":
    unittest.main()
