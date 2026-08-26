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
            "ORION_TAVILY_API_KEY",
        ):
            marker = f"- key: {secret}\n        sync: false"
            self.assertIn(marker, content)

    def test_cloud_prototype_uses_stronger_120b_brain_for_controlled_benchmark(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        expected = 'value: "@cf/openai/gpt-oss-120b"'
        self.assertEqual(content.count(expected), 2)
        self.assertNotIn('@cf/openai/gpt-oss-20b', content)

    def test_render_starts_fastapi_core_and_has_health_check(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn("backend.app.main:app", content)
        self.assertIn("--host 0.0.0.0", content)
        self.assertIn("--port $PORT", content)
        self.assertIn("healthCheckPath: /api/v1/health", content)

    def test_cloud_prototype_enables_semantic_bounded_research(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn("- key: ORION_WEB_ENABLED\n        value: \"true\"", content)
        self.assertIn("- key: ORION_WEB_PROVIDER\n        value: \"auto\"", content)
        self.assertIn(
            "- key: ORION_SEMANTIC_ORCHESTRATION\n        value: \"true\"",
            content,
        )
        self.assertIn(
            "- key: ORION_SEMANTIC_MAX_TOOL_ROUNDS\n        value: \"2\"",
            content,
        )

    def test_cloud_prototype_enables_ephemeral_diagnostics(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn(
            "- key: ORION_DIAGNOSTICS_ENABLED\n        value: \"true\"",
            content,
        )


if __name__ == "__main__":
    unittest.main()
