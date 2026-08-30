from __future__ import annotations

import unittest

from backend.app.services.page_text import visible_text
from backend.app.services.web_reader import (
    MAX_EXCERPT_CHARACTERS,
    _relevant_excerpt,
)


class PageTextTests(unittest.TestCase):
    def test_attribute_markup_containing_angle_brackets_never_becomes_text(self) -> None:
        """The failure that cost Orion a fifth of every Wikipedia page.

        Parsoid stores template source in a data-mw attribute whose JSON embeds
        `<ref>` tags. A `<[^>]+>` stripper stops at the first `>` inside the
        attribute and spills the rest into the output as if it were prose.
        """

        html = (
            '<p data-mw=\'{"parts":[{"template":{"params":{"apodo":{"wt":'
            '"Barba<ref>{{cita web|url=http://x.test}}</ref>"}}}}]}\'>'
            "Dirigió a Huracán.</p>"
        )
        text = visible_text(html)
        self.assertIn("Dirigió a Huracán.", text)
        for marker in ("cita web", '"wt"', "{{", "data-mw"):
            self.assertNotIn(marker, text)

    def test_a_table_row_stays_one_line_so_the_year_keeps_its_club(self) -> None:
        html = (
            "<table><tr><th>Club</th><th>Desde</th><th>Hasta</th></tr>"
            "<tr><td><div><a>Colón</a></div></td><td>2016</td><td>2018</td></tr>"
            "<tr><td><div><a>Huracán</a></div></td><td>2019</td><td>2020</td></tr>"
            "</table>"
        )
        lines = [line for line in visible_text(html).splitlines() if "|" in line]
        self.assertIn("Colón | 2016 | 2018", lines)
        self.assertIn("Huracán | 2019 | 2020", lines)

    def test_script_and_navigation_never_reach_the_model(self) -> None:
        html = (
            "<nav>Ir al contenido</nav>"
            "<script>var x = 1;</script>"
            "<sup class='reference'>[1]</sup>"
            "<p>El dato que importa.</p>"
        )
        text = visible_text(html)
        self.assertEqual(text, "El dato que importa.")

    def test_broken_markup_still_yields_the_text_it_recovered(self) -> None:
        text = visible_text("<p>Antes<div><span>del corte")
        self.assertIn("Antes", text)


class ExcerptTests(unittest.TestCase):
    def test_a_page_within_budget_is_never_cut(self) -> None:
        page = "Dato importante. " * 100
        self.assertEqual(_relevant_excerpt(page, "cualquier pregunta"), page)

    def test_selection_does_not_depend_on_the_words_of_the_question(self) -> None:
        """The rule this project enforces, measured rather than asserted.

        The old scorer kept whatever repeated the question's words. On a real
        article that was the citation markup, and the career table was dropped.
        """

        page = "".join(f"seccion {index} contenido. " for index in range(4_000))
        with_terms = _relevant_excerpt(page, "carrera de Eduardo Dominguez")
        without_terms = _relevant_excerpt(page, "otra pregunta completamente distinta")
        self.assertEqual(with_terms, without_terms)

    def test_an_oversized_page_is_represented_end_to_end(self) -> None:
        page = "INICIO" + ("x" * (MAX_EXCERPT_CHARACTERS * 3)) + "FINAL"
        excerpt = _relevant_excerpt(page, "pregunta")
        self.assertLessEqual(len(excerpt), MAX_EXCERPT_CHARACTERS)
        # Both ends survive: a career table near the bottom of a long article is
        # exactly the content the old keyword scoring used to throw away.
        self.assertIn("INICIO", excerpt)
        self.assertIn("FINAL", excerpt)


if __name__ == "__main__":
    unittest.main()
