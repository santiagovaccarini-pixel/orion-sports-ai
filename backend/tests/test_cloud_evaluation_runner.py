from __future__ import annotations

import json
import unittest

import httpx

from backend.evals.run_cloud_evaluation import _base_url, _stream_case


class CloudEvaluationRunnerTests(unittest.TestCase):
    def test_base_url_normalizes_render_root(self) -> None:
        self.assertEqual(
            _base_url("https://orion.example.com"),
            "https://orion.example.com/api/v1",
        )
        self.assertEqual(
            _base_url("https://orion.example.com/api/v1/"),
            "https://orion.example.com/api/v1",
        )

    def test_stream_case_collects_meta_content_done_and_chart(self) -> None:
        events = [
            {
                "type": "meta",
                "selected_mode": "quick",
                "recommended_mode": "quick",
                "recommendation_reason": "test",
                "model": "@cf/openai/gpt-oss-120b",
                "sport": "football",
                "trace_id": "trace-123",
            },
            {"type": "content", "content": "Respuesta "},
            {
                "type": "chart",
                "chart": {
                    "type": "bar",
                    "title": "HSR",
                    "unit": "m",
                    "source": "gps.csv",
                    "metric": "HSR",
                    "points": [{"label": "A", "value": 100}],
                },
            },
            {"type": "content", "content": "correcta."},
            {
                "type": "done",
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "reasoning_tokens": 12,
                "finish_reason": "completed",
                "reasoning_effort": "low",
                "endpoint": "responses",
            },
        ]
        body = "".join(json.dumps(event) + "\n" for event in events)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/v1/chat/stream")
            return httpx.Response(
                200,
                request=request,
                text=body,
                headers={"content-type": "application/x-ndjson"},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = _stream_case(
                client,
                base_url="https://orion.example.com/api/v1",
                case={
                    "id": "x",
                    "prompt": "Pregunta",
                    "mode": "quick",
                },
            )

        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["answer"], "Respuesta correcta.")
        self.assertEqual(result["meta"]["trace_id"], "trace-123")  # type: ignore[index]
        self.assertEqual(result["done"]["finish_reason"], "completed")  # type: ignore[index]
        self.assertEqual(len(result["charts"]), 1)  # type: ignore[arg-type]
        self.assertIsNotNone(result["first_text_ms"])
        self.assertIsNone(result["stream_error"])

    def test_stream_case_surfaces_protocol_error_event(self) -> None:
        body = (
            json.dumps(
                {
                    "type": "meta",
                    "selected_mode": "quick",
                    "recommended_mode": "quick",
                    "recommendation_reason": "test",
                    "model": "x",
                    "sport": "football",
                    "trace_id": "t",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "error",
                    "code": "model_provider_unavailable",
                    "message": "quota",
                }
            )
            + "\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request, text=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = _stream_case(
                client,
                base_url="https://orion.example.com/api/v1",
                case={"id": "x", "prompt": "Pregunta", "mode": "quick"},
            )

        self.assertEqual(result["http_status"], 200)
        self.assertEqual(
            result["stream_error"]["code"],  # type: ignore[index]
            "model_provider_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
