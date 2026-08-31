from __future__ import annotations

import re
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
        self.assertIn("- key: ORION_MODEL_PROVIDER\n        value: cerebras", content)
        for secret in (
            "ORION_CLOUDFLARE_ACCOUNT_ID",
            "ORION_CLOUDFLARE_API_TOKEN",
            "ORION_CEREBRAS_API_KEY",
            "ORION_GROQ_API_KEY",
            "ORION_API_KEY",
            "ORION_TAVILY_API_KEY",
        ):
            marker = f"- key: {secret}\n        sync: false"
            self.assertIn(marker, content)

    def test_cloudflare_stays_credentialed_as_the_rollback(self) -> None:
        """The blueprint and the dashboard must agree on the active provider.

        A dashboard change alone is not enough: the blueprint re-applies every
        `value:` it declares, so leaving this file on the old provider would quietly
        undo the switch on the next sync. Cloudflare keeps its credentials so
        rolling back stays a one-value edit rather than a redeploy.
        """

        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertNotIn("value: cloudflare", content)
        self.assertIn('value: "@cf/openai/gpt-oss-120b"', content)
        self.assertIn('- key: ORION_ENDPOINT_QUICK_REASONING_EFFORT\n        value: "low"', content)
        # The endpoint URL and model id come from the provider's own defaults;
        # pinning them here would mean editing two places to change one thing.
        self.assertNotIn("ORION_ENDPOINT_BASE_URL", content)
        self.assertNotIn("ORION_ENDPOINT_QUICK_MODEL", content)

    def test_cloud_prototype_uses_stronger_120b_brain_for_controlled_benchmark(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        expected = 'value: "@cf/openai/gpt-oss-120b"'
        self.assertEqual(content.count(expected), 2)
        self.assertNotIn('@cf/openai/gpt-oss-20b', content)

    def test_gpt_oss_reasoning_and_output_budgets_are_explicit(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn(
            '- key: ORION_CLOUDFLARE_QUICK_REASONING_EFFORT\n        value: "low"',
            content,
        )
        self.assertIn(
            '- key: ORION_CLOUDFLARE_DEEP_REASONING_EFFORT\n        value: "medium"',
            content,
        )
        self.assertIn(
            '- key: ORION_CLOUDFLARE_QUICK_MAX_TOKENS\n        value: "1536"',
            content,
        )
        self.assertIn(
            '- key: ORION_CLOUDFLARE_DEEP_MAX_TOKENS\n        value: "3072"',
            content,
        )

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
            "- key: ORION_SEMANTIC_MAX_TOOL_ROUNDS\n        value: \"3\"",
            content,
        )

    def test_cloud_prototype_enables_ephemeral_diagnostics(self) -> None:
        content = RENDER_YAML.read_text(encoding="utf-8")
        self.assertIn(
            "- key: ORION_DIAGNOSTICS_ENABLED\n        value: \"true\"",
            content,
        )


class DeployGateTests(unittest.TestCase):
    """A live monitoring battery must not be able to block a deploy.

    Render waits on checksPass, so a red mark on any workflow stops the release.
    The cloud diagnostic talks to the running service, which means it fails when
    production is slow, rate-limited or briefly down - none of which is a reason
    to refuse shipping, least of all a fix for production. Several commits sat
    undeployed while the endpoints they added returned 404.
    """

    def test_the_live_diagnostic_cannot_block_a_release(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "cloud-diagnostic.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("continue-on-error: true", workflow)
        # Still runs and still uploads its report: non-blocking is not the same
        # as not looking.
        self.assertIn("Run 24-case Orion Cloud diagnostic", workflow)
        self.assertIn("upload-artifact", workflow)

    def test_the_local_check_runs_under_the_same_environment_as_ci(self) -> None:
        """"Green locally" has to mean "green in CI", or the gate blocks blindly.

        The suite was usually run here with no ORION_* variables set, while CI
        exports a provider. An assertion that read the ambient provider passed
        here and failed there, and that red mark held every deploy back for a
        day. scripts/verify-like-ci.sh exists to reproduce CI's environment; if
        the workflow changes and the script does not, it stops being a check and
        goes back to being a guess.
        """

        workflow = (ROOT / ".github" / "workflows" / "backend-tests.yml").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "scripts" / "verify-like-ci.sh").read_text(encoding="utf-8")
        declared = re.findall(r"^\s+(ORION_[A-Z_]+):\s*\"?([^\"\n]*)\"?", workflow, re.M)
        self.assertTrue(declared, "the workflow no longer pins any ORION_ variable")
        for name, value in declared:
            self.assertIn(f"{name}={value}", script)
        # Same runner, too: unittest and pytest do not collect the same tests.
        self.assertIn("unittest discover -s backend/tests", script)
        self.assertIn("unittest discover -s backend/tests", workflow)

    def test_the_local_check_also_covers_the_frontend_gate(self) -> None:
        """Every gating workflow has to be reachable from one local command.

        The frontend job blocks a deploy exactly like the backend one, and its
        lint step had been failing for a day without anyone running it here. A
        check nobody can run locally is a check that only reports damage.
        """

        workflow = (ROOT / ".github" / "workflows" / "frontend-tests.yml").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "scripts" / "verify-like-ci.sh").read_text(encoding="utf-8")
        for step in ("npm audit --audit-level=high", "npm run lint", "npm test"):
            self.assertIn(step, workflow)
            self.assertIn(step, script)

    def test_only_shipped_dependencies_can_block_on_an_advisory(self) -> None:
        """Blocking on dev tooling would stop real work for a hole nobody can reach.

        Every current advisory is in something that builds or serves the app on
        one machine - vite, ws, undici - and none of it is sent to a browser.
        The workflow already treats the audit as a report; the local script must
        make the same call, or it refuses changes CI would accept. What ships is
        held to the stricter rule instead, and today it is clean.
        """

        workflow = (ROOT / ".github" / "workflows" / "frontend-tests.yml").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "scripts" / "verify-like-ci.sh").read_text(encoding="utf-8")
        audit_step = workflow.split("- name: Audit dependencies", 1)[1].split(
            "- name:", 1
        )[0]
        self.assertIn("continue-on-error: true", audit_step)
        self.assertIn("npm audit --omit=dev --audit-level=high", script)
        self.assertIn("npm audit --audit-level=high || true", script)

    def test_the_battery_paces_itself_against_the_rate_limiter(self) -> None:
        runner = (ROOT / "backend" / "evals" / "run_cloud_evaluation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("PAUSE_BETWEEN_CASES_SECONDS", runner)
        self.assertIn("time.sleep(PAUSE_BETWEEN_CASES_SECONDS)", runner)

if __name__ == "__main__":
    unittest.main()
