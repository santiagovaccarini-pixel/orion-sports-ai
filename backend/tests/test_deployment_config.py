from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RENDER_YAML = ROOT / "render.yaml"


class DeploymentConfigTests(unittest.TestCase):
    def test_render_blueprint_is_explicitly_free(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn("plan: free", content)
        self.assertIn("branch: cloud-prototype", content)
        self.assertIn("autoDeployTrigger: checksPass", content)

    def test_cloud_provider_is_selected_without_hardcoded_credentials(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn("value: cloudflare", content)

        for secret in (
            "ORION_CLOUDFLARE_ACCOUNT_ID",
            "ORION_CLOUDFLARE_API_TOKEN",
            "ORION_API_KEY",
        ):
            marker = f"- key: {secret}\n        sync: false"
            self.assertIn(marker, content)

    def test_render_starts_the_fastapi_core_and_has_health_check(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn("backend.app.main:app", content)
        self.assertIn("--host 0.0.0.0", content)
        self.assertIn("--port $PORT", content)
        self.assertIn("healthCheckPath: /api/v1/health", content)


if __name__ == "__main__":
    unittest.main()
