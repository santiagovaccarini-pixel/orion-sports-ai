from __future__ import annotations

import unittest

from backend.app.core.config import get_settings
from backend.app.services.web_research import (
    DEFAULT_ALLOWED_DOMAINS,
    WebSource,
    _allowed,
    _extract_search_urls,
    _relevant_excerpt,
    _search_domains,
    _tavily_sources,
    format_sources,
    is_web_request,
)


class WebResearchTests(unittest.TestCase):
    def test_default_allowlist_is_a_concrete_domain_tuple(self) -> None:
        settings = get_settings()
        self.assertIsInstance(settings.web_allowed_domains, tuple)
        self.assertIn("fifa.com", settings.web_allowed_domains)

    def test_rejects_domains_outside_allowlist(self) -> None:
        self.assertTrue(_allowed("https://www.fifa.com/laws", DEFAULT_ALLOWED_DOMAINS))
        self.assertFalse(_allowed("https://example.com/article", DEFAULT_ALLOWED_DOMAINS))

    def test_requires_four_sources_before_claiming_verification(self) -> None:
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
        self.assertIn("URL:", result)
        self.assertIn("provisional", result)

    def test_formats_four_sources_with_urls_and_date(self) -> None:
        sources = tuple(
            WebSource(
                f"Fuente {index}",
                f"https://{domain}/{index}",
                "Contenido verificable",
                domain,
            )
            for index, domain in enumerate(
                ("fifa.com", "theifab.com", "uefa.com", "espn.com.ar")
            )
        )

        result = format_sources(sources, minimum_sources=4)

        self.assertIn("VERIFICADA", result)
        self.assertIn("URL: https://fifa.com/0", result)
        self.assertIn("fecha", result)

    def test_four_sources_must_be_independent_domains(self) -> None:
        sources = tuple(
            WebSource(
                f"Fuente {index}",
                f"https://fifa.com/{index}",
                "Contenido verificable",
                "fifa.com",
            )
            for index in range(4)
        )

        result = format_sources(sources, minimum_sources=4)

        self.assertIn("INSUFICIENTE", result)

    def test_detects_current_sports_questions_without_explicit_web_word(self) -> None:
        self.assertTrue(is_web_request("¿Cuántos goles hizo un jugador esta temporada?"))
        self.assertTrue(is_web_request("¿Qué cantidad de goles lleva Miguel Merentiel?"))

    def test_current_goal_questions_prioritize_football_sources(self) -> None:
        domains = _search_domains(
            "¿Cuántos goles tiene Messi?", DEFAULT_ALLOWED_DOMAINS
        )
        self.assertEqual(domains[0], "bocajuniors.com.ar")
        self.assertLess(domains.index("espn.com.ar"), domains.index("fifa.com"))

    def test_relevant_excerpt_keeps_statistic_far_from_page_header(self) -> None:
        html = (
            "<html><body>"
            + ("Menú navegación patrocinadores " * 120)
            + "Miguel Merentiel registra 50 goles en Boca Juniors en partidos oficiales. "
            + ("Noticias relacionadas " * 120)
            + "</body></html>"
        )

        excerpt = _relevant_excerpt(
            html, "¿Cuántos goles tiene Miguel Merentiel en Boca?"
        )

        self.assertIn("Miguel Merentiel", excerpt)
        self.assertIn("50 goles", excerpt)
        self.assertIn("Boca Juniors", excerpt)

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

    def test_tavily_results_are_allowlisted_and_deduplicated_by_domain(self) -> None:
        payload = {
            "results": [
                {
                    "title": "ESPN 1",
                    "url": "https://www.espn.com.ar/futbol/a",
                    "content": "Lionel Messi registra una cantidad actualizada de goles oficiales.",
                },
                {
                    "title": "ESPN 2",
                    "url": "https://www.espn.com.ar/futbol/b",
                    "content": "Otra nota extensa del mismo dominio sobre Lionel Messi.",
                },
                {
                    "title": "Transfermarkt",
                    "url": "https://www.transfermarkt.es/lionel-messi/leistungsdaten/spieler/28003",
                    "content": "Estadísticas detalladas y actuales de Lionel Messi por competición.",
                },
                {
                    "title": "No permitido",
                    "url": "https://example.com/messi",
                    "content": "Este contenido no debe entrar aunque parezca relevante.",
                },
            ]
        }

        sources = _tavily_sources(
            payload,
            allowed_domains=DEFAULT_ALLOWED_DOMAINS,
            limit=4,
        )

        self.assertEqual(len(sources), 2)
        self.assertEqual({source.domain for source in sources}, {"espn.com.ar", "transfermarkt.es"})


if __name__ == "__main__":
    unittest.main()
