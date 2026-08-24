from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.providers.model_provider import (
    CloudflareModelProvider,
    ModelProviderConfigurationError,
    ModelProviderModelError,
    OllamaModelProvider,
    _model_is_installed,
    create_model_provider,
)


class ModelProviderTests(unittest.TestCase):
    def test_factory_keeps_ollama_as_default(self) -> None:
        provider = create_model_provider(Settings())
        self.assertIsInstance(provider, OllamaModelProvider)
        self.assertTrue(provider.uses_local_resources)
        self.assertEqual(provider.model_for(SelectedMode.QUICK), "qwen3:4b-instruct")
        self.assertEqual(provider.model_for(SelectedMode.DEEP), "qwen3:8b")

    def test_factory_builds_cloudflare_without_using_local_resources(self) -> None:
        provider = create_model_provider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
                cloudflare_quick_model="cloud-quick",
                cloudflare_deep_model="cloud-deep",
            )
        )
        self.assertIsInstance(provider, CloudflareModelProvider)
        self.assertFalse(provider.uses_local_resources)
        self.assertEqual(provider.model_for(SelectedMode.QUICK), "cloud-quick")
        self.assertEqual(provider.model_for(SelectedMode.DEEP), "cloud-deep")

    def test_cloudflare_requires_credentials_before_chat(self) -> None:
        with self.assertRaises(ModelProviderConfigurationError):
            create_model_provider(Settings(model_provider="cloudflare"))

    def test_cloudflare_preflight_does_not_spend_inference_quota(self) -> None:
        provider = create_model_provider(
            Settings(
                model_provider="cloudflare",
                cloudflare_account_id="account-test",
                cloudflare_api_token="token-test",
            )
        )
        provider.client.chat = AsyncMock(side_effect=AssertionError("No debe inferir"))
        asyncio.run(provider.preflight(SelectedMode.QUICK))
        provider.client.chat.assert_not_called()

    def test_local_preflight_rejects_missing_model(self) -> None:
        provider = OllamaModelProvider(Settings(quick_model="missing-model"))
        provider.client.status = AsyncMock()
        provider.client.status.return_value.online = True
        provider.client.status.return_value.installed_models = ("other-model:latest",)
        provider.client.status.return_value.loaded_models = ()

        with self.assertRaises(ModelProviderModelError):
            asyncio.run(provider.preflight(SelectedMode.QUICK))

    def test_model_name_matching_accepts_ollama_tags(self) -> None:
        self.assertTrue(_model_is_installed("qwen3:4b", ("qwen3:4b",)))
        self.assertTrue(_model_is_installed("qwen3", ("qwen3:4b",)))
        self.assertFalse(_model_is_installed("qwen3:8b", ("qwen3:4b",)))


if __name__ == "__main__":
    unittest.main()
