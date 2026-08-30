from __future__ import annotations

import unittest

from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.semantic_tools import (
    CsvFilter,
    CsvOperationSpec,
    audit_numeric_support,
    evaluate_expression,
    execute_calculation,
    execute_csv_operation,
)


CSV = """Player,Period,HSR\nA,PT,100\nA,ST,140\nB,PT,90\n"""


class SemanticToolsTests(unittest.TestCase):
    def test_safe_calculator_evaluates_arithmetic(self) -> None:
        self.assertEqual(evaluate_expression("(12 + 8) / 2"), 10)

    def test_safe_calculator_rejects_calls_and_names(self) -> None:
        result = execute_calculation("__import__('os').system('echo no')")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.context, "")

    def test_structured_csv_average_uses_exact_validated_columns(self) -> None:
        execution = execute_csv_operation(
            [KnowledgeDocument("1", "gps.csv", CSV)],
            CsvOperationSpec(
                document_name="gps.csv",
                filters=(CsvFilter(column="Player", value="A"),),
                value_column="HSR",
                aggregation="average",
            ),
        )
        self.assertIsNone(execution.error)
        self.assertIn("average(HSR)=120", execution.context)
        self.assertIn("matched_rows=2", execution.context)

    def test_structured_csv_rejects_invented_column(self) -> None:
        execution = execute_csv_operation(
            [KnowledgeDocument("1", "gps.csv", CSV)],
            CsvOperationSpec(
                document_name="gps.csv",
                filters=(CsvFilter(column="Jugador inventado", value="A"),),
                value_column="HSR",
                aggregation="average",
            ),
        )
        self.assertIsNotNone(execution.error)
        self.assertIn("No se pudo resolver", execution.error or "")

    def test_structured_chart_returns_verified_points_only(self) -> None:
        execution = execute_csv_operation(
            [KnowledgeDocument("1", "gps.csv", CSV)],
            CsvOperationSpec(
                document_name="gps.csv",
                filters=(CsvFilter(column="Player", value="A"),),
                value_column="HSR",
                aggregation="none",
                x_column="Period",
                chart_type="bar",
                title="HSR de A",
            ),
        )
        self.assertIsNone(execution.error)
        self.assertIsNotNone(execution.chart)
        assert execution.chart is not None
        self.assertEqual(execution.chart["source"], "gps.csv")
        self.assertEqual(
            execution.chart["points"],
            [{"label": "PT", "value": 100.0}, {"label": "ST", "value": 140.0}],
        )


    def test_audit_flags_number_absent_from_allowed_texts(self) -> None:
        unsupported = audit_numeric_support(
            "El jugador acumuló 143 goles en su carrera.",
            allowed_texts=["Preguntó por la carrera del jugador."],
        )
        self.assertEqual(unsupported, ("143",))

    def test_audit_accepts_numbers_traceable_to_allowed_sources(self) -> None:
        unsupported = audit_numeric_support(
            "El promedio ponderado dio 5.4.",
            allowed_texts=["RESULTADO DETERMINÍSTICO DE CALCULADORA: result=5.4"],
        )
        self.assertEqual(unsupported, ())

    def test_audit_matches_equivalent_number_formats(self) -> None:
        unsupported = audit_numeric_support(
            "El total fue de 1234.5 unidades.",
            allowed_texts=["La fuente reporta 1.234,5 unidades acumuladas."],
        )
        self.assertEqual(unsupported, ())

    def test_audit_tolerates_small_enumeration_integers(self) -> None:
        unsupported = audit_numeric_support(
            "Podés seguir estos 3 pasos para mejorar en 5 sesiones.",
            allowed_texts=[],
        )
        self.assertEqual(unsupported, ())

    def test_audit_flags_a_year_the_evidence_never_mentions(self) -> None:
        """A date is a claim, and it was the one claim the audit could not see.

        Years used to be tolerated across 1900-2100, so Orion could state that a
        manager played somewhere "1998-2002" — wrong on both counts — and the
        audit stayed silent, because both numbers looked like years.
        """

        unsupported = audit_numeric_support(
            "Jugó en Huracán entre 1998 y 2002.",
            allowed_texts=["Jugó en Vélez Sarsfield entre 1996 y 2006."],
        )
        self.assertEqual(unsupported, ("1998", "2002"))

    def test_audit_accepts_a_year_the_evidence_supports(self) -> None:
        unsupported = audit_numeric_support(
            "El torneo se disputó en 2026.",
            allowed_texts=["La edición 2026 del torneo terminó en agosto."],
        )
        self.assertEqual(unsupported, ())

    def test_audit_still_ignores_counts_and_ordinals(self) -> None:
        """Small integers come from sentence structure, not from evidence."""

        unsupported = audit_numeric_support(
            "Dirigió 3 clubes y quedó 2 veces en el puesto 5.",
            allowed_texts=["Historial del entrenador."],
        )
        self.assertEqual(unsupported, ())

    def test_audit_flags_unsupported_percentage(self) -> None:
        unsupported = audit_numeric_support(
            "La posesión fue del 67%.",
            allowed_texts=["Preguntó por la posesión del partido."],
        )
        self.assertEqual(unsupported, ("67%",))

    def test_audit_matches_percentage_from_allowed_text(self) -> None:
        unsupported = audit_numeric_support(
            "La posesión fue del 67%.",
            allowed_texts=["[W1] La fuente reporta 67% de posesión."],
        )
        self.assertEqual(unsupported, ())

    def test_audit_does_not_duplicate_repeated_unsupported_numbers(self) -> None:
        unsupported = audit_numeric_support(
            "El valor fue 250. Repito: 250.",
            allowed_texts=[],
        )
        self.assertEqual(unsupported, ("250",))


if __name__ == "__main__":
    unittest.main()
