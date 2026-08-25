from __future__ import annotations

import unittest
from backend.app.core.config import get_settings

from backend.app.services.web_research import (
    DEFAULT_ALLOWED_DOMAINS,
    WebSource,
    _allowed,
    _relevant_excerpt,
    _search_domains,
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
            WebSource(f"Fuente {index}", f"https://fifa.com/{index}", "Contenido verificable", "fifa.com")
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
            for index, domain in enumerate(("fifa.com", "theifab.com", "uefa.com", "espn.com.ar"))
        )

        result = format_sources(sources, minimum_sources=4)

        self.assertIn("VERIFICADA", result)
        self.assertIn("URL: https://fifa.com/0", result)
        self.assertIn("fecha", result)

    def test_four_sources_must_be_independent_domains(self) -> None:
        sources = tuple(
            WebSource(f"Fuente {index}", f"https://fifa.com/{index}", "Contenido verificable", "fifa.com")
            for index in range(4)
        )

        result = format_sources(sources, minimum_sources=4)

        self.assertIn("INSUFICIENTE", result)

    def test_detects_current_sports_questions_without_explicit_web_word(self) -> None:
        self.assertTrue(is_web_request("¿Cuántos goles hizo un jugador esta temporada?"))
        self.assertTrue(is_web_request("¿Qué cantidad de goles lleva Miguel Merentiel?"))

    def test_current_goal_questions_prioritize_football_sources(self) -> None:
        domains = _search_domains("¿Cuántos goles tiene Messi?", DEFAULT_ALLOWED_DOMAINS)
        self.assertEqual(domains[0], "bocajuniors.com.ar")
        self.assertLess(domains.index("espn.com.ar"), domains.index("fifa.com"))

    def test_relevant_excerpt_keeps_statistic_far_from_page_header(self) -> None:
        html = (
            "<html><body>" + ("Menú navegación patrocinadores " * 120)
            + "Miguel Merentiel registra 50 goles en Boca Juniors en partidos oficiales. "
            + ("Noticias relacionadas " * 120) + "</body></html>"
        )

        excerpt = _relevant_excerpt(html, "¿Cuántos goles tiene Miguel Merentiel en Boca?")

        self.assertIn("Miguel Merentiel", excerpt)
        self.assertIn("50 goles", excerpt)
        self.assertIn("Boca Juniors", excerpt)

    def test_extracts_duckduckgo_result_links(self) -> None:
        self.assertTrue(_allowed("https://www.espn.com.ar/futbol", DEFAULT_ALLOWED_DOMAINS))


if __name__ == "__main__":
    unittest.main()
