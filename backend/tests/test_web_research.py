from __future__ import annotations

import asyncio
import time
import unittest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import httpx

from backend.app.core.config import get_settings
from backend.app.services.web_research import (
    DEFAULT_ALLOWED_DOMAINS,
    WebSource,
    _allowed,
    _discover_duckduckgo_urls,
    _extract_search_urls,
    _fallback_excerpt,
    FALLBACK_EXCERPT_CHARACTERS,
    _search_domains,
    _tavily_sources,
    _tavily_time_range,
    format_sources,
    is_web_request,
    parse_published_date,
    research,
)


class WebResearchTests(unittest.TestCase):
    def test_default_allowlist_remains_available_for_legacy_fallback(self) -> None:
        settings = get_settings()
        self.assertIsInstance(settings.web_allowed_domains, tuple)
        self.assertIn("fifa.com", settings.web_allowed_domains)

    def test_legacy_fallback_rejects_domains_outside_allowlist(self) -> None:
        self.assertTrue(_allowed("https://www.fifa.com/laws", DEFAULT_ALLOWED_DOMAINS))
        self.assertFalse(_allowed("https://example.com/article", DEFAULT_ALLOWED_DOMAINS))

    def test_legacy_formatter_still_requires_configured_source_count(self) -> None:
        sources = tuple(
            WebSource(
                f"Fuente {index}",
                f"https://fifa.com/{index}",
                "Contenido verificable",
                "fifa.com",
            )
            for index in range(3)
        )
        result = format_sources(sources, minimum_sources=4)
        self.assertIn("INSUFICIENTE", result)
        self.assertNotIn("VERIFICADA", result)

    def test_legacy_heuristic_is_retained_only_for_rollback_mode(self) -> None:
        self.assertTrue(is_web_request("¿Cuántos goles hizo un jugador esta temporada?"))
        self.assertIn("Legacy-only heuristic", is_web_request.__doc__ or "")

    def test_legacy_duckduckgo_prioritization_is_unchanged(self) -> None:
        domains = _search_domains(
            "¿Cuántos goles tiene Messi?", DEFAULT_ALLOWED_DOMAINS
        )
        self.assertEqual(domains[0], "bocajuniors.com.ar")

    def test_the_fallback_excerpt_represents_the_whole_page(self) -> None:
        """It fits the page into the budget; it does not choose what matters.

        The old version scored windows by how many of the question's words they
        contained and kept the winner - keyword selection, which this project
        forbids, and which fails exactly where it counts: a table answering
        "¿en qué club juega?" names the club and never repeats "club". Now
        the page is sampled end to end and the reviewer decides.
        """

        html = (
            "<html><body>INICIO_DEL_DOCUMENTO "
            + ("relleno " * 4_000)
            + " MEDIO_DEL_DOCUMENTO "
            + ("relleno " * 4_000)
            + " FINAL_DEL_DOCUMENTO</body></html>"
        )
        excerpt = _fallback_excerpt(html)
        self.assertLessEqual(len(excerpt), FALLBACK_EXCERPT_CHARACTERS)
        # Both ends survive: a squad table or a results list usually sits at the
        # bottom of a page, where a head-only clip would never reach it.
        self.assertIn("INICIO_DEL_DOCUMENTO", excerpt)
        self.assertIn("FINAL_DEL_DOCUMENTO", excerpt)

    def test_a_short_page_passes_through_whole(self) -> None:
        excerpt = _fallback_excerpt("<html><body>Un dato corto y completo.</body></html>")
        self.assertEqual(excerpt, "Un dato corto y completo.")

    def test_duckduckgo_parser_does_not_depend_on_result_css_class(self) -> None:
        html = """
        <html><body>
          <a href="https://example.com/no">ignorar</a>
          <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.espn.com.ar%2Ffutbol%2Fnota">ESPN</a>
          <a class="anything" href="https://www.transfermarkt.es/lionel-messi/profil/spieler/28003">Messi</a>
        </body></html>
        """
        urls = _extract_search_urls(html, DEFAULT_ALLOWED_DOMAINS)
        self.assertEqual(len(urls), 2)
        self.assertTrue(any("espn.com.ar" in url for url in urls))
        self.assertTrue(any("transfermarkt.es" in url for url in urls))

    def test_tavily_open_web_results_are_not_hard_filtered_by_domain(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Fuente primaria nueva",
                    "url": "https://club-nuevo.example/noticia",
                    "content": "Una fuente que el código nunca conoció de antemano pero puede ser relevante.",
                },
                {
                    "title": "ESPN",
                    "url": "https://www.espn.com.ar/futbol/a",
                    "content": "Contenido deportivo suficientemente extenso para revisión semántica.",
                },
                {
                    "title": "Duplicado exacto",
                    "url": "https://www.espn.com.ar/futbol/a",
                    "content": "La misma URL no debe repetirse dentro de la evidencia.",
                },
            ]
        }
        sources = _tavily_sources(payload, limit=6)
        self.assertEqual(len(sources), 2)
        self.assertIn("club-nuevo.example", {source.domain for source in sources})
        self.assertIn("espn.com.ar", {source.domain for source in sources})

    def test_auto_provider_falls_back_when_tavily_returns_no_sources(self) -> None:
        fallback = (
            WebSource(
                "Fuente fallback",
                "https://fifa.com/fallback",
                "Contenido recuperado por el proveedor alternativo.",
                "fifa.com",
            ),
        )
        with (
            patch(
                "backend.app.services.web_research._research_tavily",
                new=AsyncMock(return_value=()),
            ) as tavily,
            patch(
                "backend.app.services.web_research._research_duckduckgo",
                new=AsyncMock(return_value=fallback),
            ) as duckduckgo,
        ):
            result = asyncio.run(
                research(
                    "consulta actual",
                    provider="auto",
                    tavily_api_key="secret-for-test",
                )
            )
        self.assertEqual(result, fallback)
        tavily.assert_awaited_once()
        duckduckgo.assert_awaited_once()

    def test_parse_published_date_accepts_common_formats(self) -> None:
        self.assertEqual(
            parse_published_date("2026-08-25T18:30:00Z"), date(2026, 8, 25)
        )
        self.assertEqual(parse_published_date("2026-08-25"), date(2026, 8, 25))
        self.assertEqual(parse_published_date("25/08/2026"), date(2026, 8, 25))
        self.assertIsNone(parse_published_date("hace dos días"))
        self.assertIsNone(parse_published_date(None))
        self.assertIsNone(parse_published_date("   "))

    def test_tavily_sources_capture_published_date_and_age(self) -> None:
        recent = (date.today() - timedelta(days=3)).isoformat()
        payload = {
            "results": [
                {
                    "title": "Crónica del partido",
                    "url": "https://diario.example/cronica",
                    "content": "Contenido del partido con longitud suficiente para revisión.",
                    "published_date": recent,
                },
                {
                    "title": "Página sin fecha",
                    "url": "https://otro.example/nota",
                    "content": "Contenido alternativo suficientemente extenso para revisar.",
                },
            ]
        }
        sources = _tavily_sources(payload, limit=6)
        self.assertEqual(sources[0].published_date, recent)
        self.assertEqual(sources[0].published_age_days, 3)
        self.assertIsNone(sources[1].published_date)
        self.assertIsNone(sources[1].published_age_days)

    def test_tavily_time_range_maps_window_days(self) -> None:
        self.assertEqual(_tavily_time_range(1), "day")
        self.assertEqual(_tavily_time_range(7), "week")
        self.assertEqual(_tavily_time_range(30), "month")
        self.assertEqual(_tavily_time_range(200), "year")

    def test_research_propagates_recency_window_to_tavily(self) -> None:
        with patch(
            "backend.app.services.web_research._research_tavily",
            new=AsyncMock(return_value=(WebSource("A", "https://a.test", "x" * 50, "a.test"),)),
        ) as tavily:
            asyncio.run(
                research(
                    "consulta volátil",
                    provider="tavily",
                    tavily_api_key="secret-for-test",
                    recency_days=14,
                )
            )
        self.assertEqual(tavily.await_args.kwargs["recency_days"], 14)

    def test_discover_duckduckgo_urls_respects_time_budget(self) -> None:
        # Regression: a query the search backends handle poorly (e.g. site:/
        # boolean operators) previously ran the full sequential 11-query x
        # 2-endpoint discovery loop even when every attempt was slow/empty,
        # producing multi-minute tail latency. A time budget caps that.
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return httpx.Response(200, request=request, text="<html></html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async def run() -> tuple[str, ...]:
            try:
                with patch(
                    "backend.app.services.web_research.DISCOVERY_TIME_BUDGET_SECONDS",
                    0.03,
                ):
                    return await _discover_duckduckgo_urls(
                        client,
                        "site:example.com \"consulta\" muy especifica 2026",
                        allowed_domains=DEFAULT_ALLOWED_DOMAINS,
                        minimum_sources=50,
                    )
            finally:
                await client.aclose()

        urls = asyncio.run(run())
        self.assertEqual(urls, ())
        # Worst case without a budget: 11 queries x 2 endpoints = 22 requests.
        self.assertLess(call_count, 22)

    def test_explicit_tavily_does_not_fall_back_silently(self) -> None:
        with (
            patch(
                "backend.app.services.web_research._research_tavily",
                new=AsyncMock(return_value=()),
            ),
            patch(
                "backend.app.services.web_research._research_duckduckgo",
                new=AsyncMock(),
            ) as duckduckgo,
        ):
            result = asyncio.run(
                research(
                    "consulta actual",
                    provider="tavily",
                    tavily_api_key="secret-for-test",
                )
            )
        self.assertEqual(result, ())
        duckduckgo.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
