from __future__ import annotations

import unittest

from backend.app.core.config import Settings
from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.cloudflare import (
    CloudflareClient,
    CloudflareConfigurationError,
    select_cloud_history,
)


class CloudflareProviderTests(unittest.TestCase):
    def test_base_url_is_built_from_account_id(self) -> None:
        settings = Settings(
            cloudflare_account_id="account-123",
            cloudflare_api_token="secret",
        )
        self.assertEqual(
            settings.cloudflare_base_url,
            "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1",
        )

    def test_credentials_are_required_only_when_cloud_client_is_used(self) -> None:
        client = CloudflareClient(Settings())
        with self.assertRaises(CloudflareConfigurationError):
            client._credentials()

    def test_cloud_history_keeps_latest_relevant_messages(self) -> None:
        settings = Settings(quick_history_characters=50)
        messages = [
            ChatMessage(role="user", content="mensaje antiguo que debe quedar afuera"),
            ChatMessage(role="assistant", content="respuesta anterior"),
            ChatMessage(role="user", content="consulta actual"),
        ]
        selected = select_cloud_history(settings, SelectedMode.QUICK, messages)

        self.assertEqual(selected[-1].content, "consulta actual")
        self.assertEqual(selected[0].role, "user")


if __name__ == "__main__":
    unittest.main()
