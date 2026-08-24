from __future__ import annotations

import unittest

from backend.scripts.smoke_cloud import _base_url, _event


class SmokeCloudTests(unittest.TestCase):
    def test_base_url_adds_api_prefix_once(self) -> None:
        self.assertEqual(
            _base_url("https://orion.example.com"),
            "https://orion.example.com/api/v1",
        )
        self.assertEqual(
            _base_url("https://orion.example.com/api/v1/"),
            "https://orion.example.com/api/v1",
        )

    def test_base_url_rejects_invalid_values(self) -> None:
        for value in ("", "orion.example.com", "ftp://orion.example.com"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _base_url(value)

    def test_event_parses_ndjson_and_rejects_invalid_json(self) -> None:
        self.assertEqual(_event(""), None)
        self.assertEqual(_event('{"type":"done"}'), {"type": "done"})
        with self.assertRaises(RuntimeError):
            _event("not-json")


if __name__ == "__main__":
    unittest.main()
