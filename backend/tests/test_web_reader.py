from __future__ import annotations

import asyncio
import unittest

import httpx

from backend.app.services.web_reader import (
    _safe_url_syntax,
    apply_page_reads,
    read_source_pages,
)
from backend.app.services.web_research import WebSource


class WebReaderTests(unittest.TestCase):
    def test_local_and_credentialed_url_syntax_is_rejected(self) -> None:
        # IP-level private/loopback/link-local blocking now happens in the
        # SSRF-safe transport itself (backend/tests/test_safe_http.py); this
        # only covers the cheap, network-free syntax checks that still live
        # here (scheme, embedded credentials, the literal "localhost" name).
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
        self.assertEqual(enriched[0].published_date, "2026-08-25T10:00:00Z")
        self.assertIsInstance(enriched[0].published_age_days, int)
        self.assertTrue(enriched[0].deepened)

    def test_apply_page_reads_keeps_richer_snippet_over_worse_deepened_read(
        self,
    ) -> None:
        rich_snippet = (
            "Con este tanto, alcanzó los 60 goles con la camiseta de Boca en 175"
            " partidos."
        )
        sources = (
            WebSource("Post original", "https://facebook.com/post", rich_snippet, "facebook.com"),
        )
        from backend.app.services.web_reader import PageRead

        reads = (
            PageRead(
                source_id="W1",
                title="Post original",
                url="https://facebook.com/post",
                domain="facebook.com",
                excerpt="Sign up\n\nLog in",
                published_date=None,
            ),
        )

        enriched = apply_page_reads(sources, reads)

        self.assertEqual(enriched[0].excerpt, rich_snippet)
        self.assertFalse(enriched[0].deepened)

    def test_reader_follows_a_single_redirect_to_the_final_page(self) -> None:
        # SSRF blocking of a malicious redirect target is a property of the
        # transport itself (backend/tests/test_safe_http.py: every connection
        # it makes is validated, including redirect hops) - this only checks
        # that _download_page's own redirect-following loop works for an
        # ordinary, safe redirect.
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            if str(request.url) == "https://example.com/start":
                return httpx.Response(
                    302,
                    request=request,
                    headers={"location": "https://example.com/final"},
                )
            return httpx.Response(
                200,
                request=request,
                text="<html><body><article>Contenido final tras seguir la redirección correctamente.</article></body></html>",
                headers={"content-type": "text/html"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sources = (
            WebSource("A", "https://example.com/start", "snippet", "example.com"),
        )
        reads = asyncio.run(
            read_source_pages(
                sources,
                source_ids=["W1"],
                query="dato",
                client=client,
            )
        )
        asyncio.run(client.aclose())

        self.assertEqual(len(reads), 1)
        self.assertIn("Contenido final", reads[0].excerpt)
        self.assertEqual(
            requests, ["https://example.com/start", "https://example.com/final"]
        )

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
