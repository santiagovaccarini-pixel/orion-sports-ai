from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.app.services.web_reader import (
    _public_ip,
    _safe_url_syntax,
    apply_page_reads,
    read_source_pages,
)
from backend.app.services.web_research import WebSource


class WebReaderTests(unittest.TestCase):
    def test_private_and_local_network_targets_are_rejected(self) -> None:
        self.assertFalse(_public_ip("127.0.0.1"))
        self.assertFalse(_public_ip("10.0.0.5"))
        self.assertFalse(_public_ip("169.254.1.1"))
        self.assertTrue(_public_ip("8.8.8.8"))
        self.assertIsNone(_safe_url_syntax("http://localhost/private"))
        self.assertIsNone(_safe_url_syntax("file:///etc/passwd"))
        self.assertIsNone(_safe_url_syntax("https://user:pass@example.com/a"))

    def test_reader_opens_page_directly_and_keeps_relevant_body_and_date(self) -> None:
        html = """
        <html><head>
          <title>Inter Miami official statistic</title>
          <meta property="article:published_time" content="2026-08-25T10:00:00Z">
        </head><body>
          <nav>menu menu menu</nav>
          <article>
            <p>Contexto general del partido.</p>
            <p>Lionel Messi alcanzó 95 goles oficiales con Inter Miami.</p>
            <p>Más información institucional.</p>
          </article>
        </body></html>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                text=html,
                headers={"content-type": "text/html; charset=utf-8"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sources = (
            WebSource(
                "Resultado corto",
                "https://example.com/messi",
                "snippet sin el total",
                "example.com",
            ),
        )
        with patch(
            "backend.app.services.web_reader._host_is_public",
            new=AsyncMock(return_value=True),
        ):
            reads = asyncio.run(
                read_source_pages(
                    sources,
                    source_ids=["W1"],
                    query="goles oficiales Messi Inter Miami",
                    client=client,
                )
            )
        asyncio.run(client.aclose())

        self.assertEqual(len(reads), 1)
        self.assertIn("95 goles oficiales", reads[0].excerpt)
        self.assertIn("2026-08-25", reads[0].excerpt)
        enriched = apply_page_reads(sources, reads)
        self.assertEqual(len(enriched), 1)
        self.assertIn("95 goles oficiales", enriched[0].excerpt)

    def test_reader_validates_redirect_destination_before_following(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(
                302,
                request=request,
                headers={"location": "http://127.0.0.1/private"},
            )

        async def public_host(host: str) -> bool:
            return host != "127.0.0.1"

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sources = (
            WebSource("A", "https://example.com/start", "snippet", "example.com"),
        )
        with patch(
            "backend.app.services.web_reader._host_is_public",
            new=AsyncMock(side_effect=public_host),
        ):
            reads = asyncio.run(
                read_source_pages(
                    sources,
                    source_ids=["W1"],
                    query="dato",
                    client=client,
                )
            )
        asyncio.run(client.aclose())

        self.assertEqual(reads, ())
        self.assertEqual(requests, ["https://example.com/start"])

    def test_only_reviewer_selected_source_ids_are_opened(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(
                200,
                request=request,
                text="<html><body><article>Contenido suficientemente largo para ser una fuente válida y verificable.</article></body></html>",
                headers={"content-type": "text/html"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sources = (
            WebSource("A", "https://a.example/1", "a", "a.example"),
            WebSource("B", "https://b.example/2", "b", "b.example"),
            WebSource("C", "https://c.example/3", "c", "c.example"),
        )
        with patch(
            "backend.app.services.web_reader._host_is_public",
            new=AsyncMock(return_value=True),
        ):
            reads = asyncio.run(
                read_source_pages(
                    sources,
                    source_ids=["W2"],
                    query="contenido",
                    client=client,
                )
            )
        asyncio.run(client.aclose())

        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].source_id, "W2")
        self.assertEqual(seen, ["https://b.example/2"])


if __name__ == "__main__":
    unittest.main()
