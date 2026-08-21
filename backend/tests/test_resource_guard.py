from __future__ import annotations

import unittest

from backend.app.domain.models import SelectedMode, SystemSnapshot
from backend.app.services.resource_policy import evaluate_resources


class ResourceGuardTests(unittest.TestCase):
    def test_deep_mode_warns_when_it_would_starve_other_apps(self) -> None:
        snapshot = SystemSnapshot(
            cpu_percent=20,
            memory_available_gb=13.5,
            memory_total_gb=31.2,
            battery_percent=100,
            plugged_in=True,
        )
        decision = evaluate_resources(SelectedMode.DEEP, snapshot)
        self.assertTrue(decision.requires_confirmation)

    def test_quick_mode_runs_with_healthy_resources(self) -> None:
        snapshot = SystemSnapshot(
            cpu_percent=20,
            memory_available_gb=18,
            memory_total_gb=31.2,
            battery_percent=100,
            plugged_in=True,
        )
        decision = evaluate_resources(SelectedMode.QUICK, snapshot)
        self.assertFalse(decision.requires_confirmation)

    def test_quick_mode_warns_when_cpu_is_already_busy(self) -> None:
        snapshot = SystemSnapshot(
            cpu_percent=75,
            memory_available_gb=18,
            memory_total_gb=31.2,
            battery_percent=100,
            plugged_in=True,
        )
        decision = evaluate_resources(SelectedMode.QUICK, snapshot)
        self.assertTrue(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
