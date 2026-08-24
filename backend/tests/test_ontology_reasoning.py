from __future__ import annotations

import unittest

from backend.app.core.sports_ontology import ontology_for
from backend.app.domain.schemas import SportContext
from backend.app.services.ontology_runtime import (
    planner_ontology_context,
    selected_concept_context,
)


class OntologyReasoningTests(unittest.TestCase):
    def test_ontology_is_a_graph_not_similarity_examples(self) -> None:
        concepts = ontology_for(SportContext.FOOTBALL)
        self.assertTrue(concepts)
        external = next(item for item in concepts if item.concept_id == "external_load")
        self.assertFalse(hasattr(external, "semantic_examples"))
        self.assertFalse(hasattr(external, "embedding_text"))
        relations = {(item.relation, item.target_id) for item in external.relations}
        self.assertIn(("not_equivalent_to", "physical_performance"), relations)
        self.assertIn(("contextualized_by", "match_exposure"), relations)

    def test_hsr_encodes_context_and_comparability_dependencies(self) -> None:
        hsr = next(
            item
            for item in ontology_for(SportContext.FOOTBALL)
            if item.concept_id == "hsr"
        )
        relations = {(item.relation, item.target_id) for item in hsr.relations}
        self.assertIn(("contextualized_by", "match_exposure"), relations)
        self.assertIn(("comparability_checked_by", "gps_comparability"), relations)

    def test_transition_and_counterattack_are_explicitly_non_equivalent(self) -> None:
        transition = next(
            item
            for item in ontology_for(SportContext.FOOTBALL)
            if item.concept_id == "offensive_transition"
        )
        relations = {(item.relation, item.target_id) for item in transition.relations}
        self.assertIn(("not_equivalent_to", "counterattack"), relations)

    def test_planner_receives_relations_not_phrase_examples(self) -> None:
        context = planner_ontology_context(SportContext.FOOTBALL)
        self.assertIn("relations=", context)
        self.assertIn("not_equivalent_to:physical_performance", context)
        self.assertNotIn("examples=", context)

    def test_answer_context_keeps_selected_graph_edges(self) -> None:
        context = selected_concept_context(
            SportContext.FOOTBALL,
            ["external_load", "physical_performance"],
        )
        self.assertIn("external_load", context)
        self.assertIn("not_equivalent_to:physical_performance", context)
        self.assertIn("physical_performance", context)


if __name__ == "__main__":
    unittest.main()
