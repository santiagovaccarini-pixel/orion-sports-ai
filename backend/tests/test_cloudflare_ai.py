from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.providers.cloudflare_ai import (
    CloudAIConfigurationError,
    CloudflareAIClient,
    _selected_model,
    parse_sse_data,
)


class CloudflareAIProviderTests(unittest.TestCase):
    def test_sse_content_is_parsed(self) -> None:
        payload = parse_sse_data(
            'data: {"choices":[{"delta":{"content":"Hola"}}]}'
        )
        self.assertIsInstance(payload, dict)
        assert payload is not None
        choices = payload["choices"]
        self.assertEqual(choices[0]["delta"]["content"], "Hola")

    def test_sse_done_is_parsed(self) -> None:
        self.assertEqual(parse_sse_data("data: [DONE]"), {"done": True})

    def test_non_data_line_is_ignored(self) -> None:
        self.assertIsNone(parse_sse_data("event: ping"))

    def test_client_requires_cloudflare_credentials(self) -> None:
        with self.assertRaises(CloudAIConfigurationError):
            CloudflareAIClient(Settings())

    def test_models_are_selected_by_mode(self) -> None:
        settings = Settings(
            cloudflare_quick_model="quick-model",
            cloudflare_deep_model="deep-model",
        )
        self.assertEqual(
            _selected_model(settings, SelectedMode.QUICK),
            "quick-model",
        )
        self.assertEqual(
            _selected_model(settings, SelectedMode.DEEP),
            "deep-model",
        )


if __name__ == "__main__":
    unittest.main()
