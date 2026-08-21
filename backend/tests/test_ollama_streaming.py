from __future__ import annotations

import unittest

from backend.app.providers.ollama import parse_stream_payload


class OllamaStreamingTests(unittest.TestCase):
    def test_parses_content_chunk(self) -> None:
        event = parse_stream_payload(
            {"message": {"content": "Hola"}, "done": False},
            thread_limit=8,
        )
        self.assertEqual(event.content, "Hola")
        self.assertFalse(event.done)

    def test_parses_final_performance_metrics(self) -> None:
        event = parse_stream_payload(
            {
                "message": {"content": ""},
                "done": True,
                "total_duration": 2_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 120,
                "prompt_eval_duration": 300_000_000,
                "eval_count": 30,
                "eval_duration": 1_000_000_000,
            },
            thread_limit=8,
        )
        self.assertTrue(event.done)
        self.assertEqual(event.total_duration_ms, 2000.0)
        self.assertEqual(event.load_duration_ms, 500.0)
        self.assertEqual(event.prompt_tokens, 120)
        self.assertEqual(event.completion_tokens, 30)
        self.assertEqual(event.tokens_per_second, 30.0)
        self.assertEqual(event.thread_limit, 8)


if __name__ == "__main__":
    unittest.main()
