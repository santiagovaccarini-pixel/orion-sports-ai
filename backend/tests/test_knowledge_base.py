from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.services.knowledge_base import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    csv_calculation,
    csv_chart,
    csv_chart_is_ambiguous,
    csv_overview,
    csv_query_is_ambiguous,
    csv_tool_result,
    format_context,
)


class KnowledgeBaseTests(unittest.TestCase):
    def test_documents_persist_and_search_relevant_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "documents.json")
            base.add_document(
                KnowledgeDocument(
                    id="gps-1",
                    name="sesion-gps.txt",
                    content="La distancia de alta velocidad debe compararse por exposición.\n\nEl umbral debe documentarse.",
                )
            )

            results = base.search("distancia alta velocidad exposición")

            self.assertEqual(len(base.list_documents()), 1)
            self.assertEqual(results[0].document_name, "sesion-gps.txt")
            self.assertIn("exposición", format_context(results))

    def test_context_respects_character_budget(self) -> None:
        document = KnowledgeDocument("one", "datos.txt", "palabra " * 500)
        context = format_context(
            [
                KnowledgeChunk("one", document.name, 0, document.content),
            ],
            max_characters=120,
        )
        self.assertLessEqual(len(context.split("\n\n", 1)[-1]), 120)

    def test_unknown_query_returns_no_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "documents.json")
            base.add_document(KnowledgeDocument("one", "datos.txt", "Carga externa y minutos."))

            self.assertEqual(base.search("astronomía"), [])

    def test_long_csv_like_lines_are_split_into_bounded_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "documents.json")
            base.add_document(KnowledgeDocument("csv", "datos.csv", "Ruan," + "0," * 2_000))

            results = base.search("Ruan")

            self.assertTrue(results)
            self.assertLessEqual(len(results[0].content), 1_200)

    def test_csv_search_returns_all_matching_player_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "documents.json")
            base.add_document(
                KnowledgeDocument(
                    "csv",
                    "sesion.csv",
                    "Player Name,Total Distance\nRUAN .,6330.83\nOTRO .,5000\nRUAN .,1200.50",
                )
            )

            results = base.search("distancia Ruan")

            self.assertEqual(len(results), 2)
            self.assertTrue(all("RUAN" in result.content for result in results))

    def test_csv_rows_are_labeled_with_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(Path(directory) / "documents.json")
            base.add_document(
                KnowledgeDocument(
                    "csv",
                    "sesion.csv",
                    "Player Name,Total Distance\nRUAN .,6330.83",
                )
            )

            result = base.search("Ruan distancia")[0]

            self.assertIn("Player Name=RUAN .", result.content)
            self.assertIn("Total Distance=6330.83", result.content)

    def test_csv_chart_returns_verified_points_for_a_specific_player(self) -> None:
        content = "Player Name,Period Name,Total Distance\nRUAN .,PRIMER TIEMPO,4689.53\nRUAN .,SEGUNDO TIEMPO,1641.30"

        chart = csv_chart(content, "gráfico de distancia de Ruan", "sesion.csv")

        self.assertIsNotNone(chart)
        self.assertEqual(chart["source"], "sesion.csv")
        self.assertEqual(len(chart["points"]), 2)
        self.assertTrue(csv_chart_is_ambiguous(content, "haceme un gráfico del CSV"))

    def test_csv_sum_is_calculated_deterministically_for_all_matching_rows(self) -> None:
        content = "Player Name,Period Name,Total Distance\nRUAN .,Session,6330.83\nOTRO .,Session,5000\nRUAN .,PRIMER TIEMPO,4689.53\nRUAN .,SEGUNDO TIEMPO,1641.30"

        calculation = csv_calculation(content, "sumá la distancia total de Ruan")

        self.assertIn("RESULTADO ESTRUCTURADO Y DETERMINISTA", calculation)
        self.assertIn("6330.83", calculation)
        self.assertIn("SUMA Total Distance de 2 períodos", calculation)
        self.assertIn("sin duplicar", calculation)

    def test_csv_overview_exposes_schema_for_ambiguous_requests(self) -> None:
        overview = csv_overview(
            "Player Name,Period Name,Total Distance\nRUAN .,Session,6330.83",
            "sesion.csv",
        )

        self.assertIn("Columnas detectadas", overview)
        self.assertIn("Total Distance", overview)
        self.assertIn("pedí una aclaración", overview)

    def test_csv_query_scope_distinguishes_ambiguous_and_specific_requests(self) -> None:
        content = "Player Name,Period Name,Total Distance\nRUAN .,Session,6330.83"

        self.assertTrue(csv_query_is_ambiguous(content, "Analizá este CSV"))
        self.assertFalse(csv_query_is_ambiguous(content, "¿Qué distancia recorrió Ruan?"))
        self.assertFalse(csv_query_is_ambiguous(content, "¿Qué cambia si reduzco el descanso entre series?"))

    def test_csv_context_prioritizes_all_rows_before_header(self) -> None:
        chunks = [
            KnowledgeChunk("csv", "sesion.csv", 0, "Player Name,Total Distance"),
            KnowledgeChunk("csv", "sesion.csv", 1, "RUAN .,6330.83"),
            KnowledgeChunk("csv", "sesion.csv", 2, "RUAN .,1200.50"),
        ]

        context = format_context(chunks, max_characters=300)

        self.assertLess(context.index("6330.83"), context.index("Player Name"))

    def test_csv_tool_calculates_average_without_model_estimation(self) -> None:
        content = "Player Name,Total Distance\nA,100\nB,200\nC,300"

        result = csv_tool_result(content, "calculá el promedio de distancia", "datos.csv")

        self.assertIn("promedio", result)
        self.assertIn("200.00", result)

    # -- Generalization beyond the GPS tracking schema (Fase 2) --------------

    BASKETBALL_ROSTER = (
        "Jugador,Equipo,Puntos,Rebotes,Fecha\n"
        "Fede,Halcones,20,5,2024-01-10\n"
        "Nico,Halcones,15,10,2024-01-10\n"
        "Fede,Halcones,24,6,2024-01-17"
    )

    def test_csv_overview_detects_a_non_gps_multi_metric_schema(self) -> None:
        overview = csv_overview(self.BASKETBALL_ROSTER, "roster.csv")

        self.assertIn("Jugador", overview)
        self.assertIn("Equipo", overview)
        self.assertIn("Puntos", overview)
        self.assertIn("Rebotes", overview)
        self.assertIn("Fede", overview)
        self.assertIn("Nico", overview)

    def test_csv_tool_result_is_ambiguous_with_several_metrics_and_none_named(self) -> None:
        result = csv_tool_result(self.BASKETBALL_ROSTER, "calculá el promedio", "roster.csv")

        self.assertEqual(result, "")

    def test_csv_tool_result_picks_the_metric_named_in_the_query(self) -> None:
        points_average = csv_tool_result(self.BASKETBALL_ROSTER, "calculá el promedio de puntos", "roster.csv")
        rebounds_average = csv_tool_result(self.BASKETBALL_ROSTER, "calculá el promedio de rebotes", "roster.csv")

        self.assertIn("promedio de Puntos", points_average)
        self.assertIn("19.67", points_average)
        self.assertIn("promedio de Rebotes", rebounds_average)
        self.assertIn("7.00", rebounds_average)

    def test_csv_chart_generalizes_to_a_non_gps_schema(self) -> None:
        chart = csv_chart(self.BASKETBALL_ROSTER, "graficá los puntos de Fede", "roster.csv")

        self.assertIsNotNone(chart)
        self.assertEqual(chart["metric"], "Puntos")
        self.assertEqual(len(chart["points"]), 2)
        self.assertEqual(chart["unit"], "")

    def test_csv_calculation_generalizes_to_a_non_sports_spreadsheet(self) -> None:
        content = "Empresa,Trimestre,Ingresos\nAcme,Q1,1000\nGlobex,Q1,1500\nAcme,Q2,1200"

        result = csv_calculation(content, "sumá los ingresos de Acme")

        self.assertIn("Empresa=Acme", result)
        self.assertIn("Trimestre=Q1", result)
        self.assertIn("Ingresos=1000", result)
        self.assertIn("SUMA Ingresos de 2 períodos = 2200", result)
        self.assertNotIn("sin duplicar", result)

    def test_csv_overview_detects_header_without_any_identifier_keyword(self) -> None:
        content = "Producto,Cantidad\nManzana,10\nBanana,20"

        overview = csv_overview(content, "inventario.csv")

        self.assertIn("Producto", overview)
        self.assertIn("Cantidad", overview)
        self.assertIn("Manzana", overview)
        self.assertIn("Banana", overview)


if __name__ == "__main__":
    unittest.main()