from __future__ import annotations

import unittest

from backend.app.services.knowledge_base import KnowledgeDocument
from backend.app.services.semantic_tools import (
    CsvFilter,
    CsvOperationSpec,
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


if __name__ == "__main__":
    unittest.main()
