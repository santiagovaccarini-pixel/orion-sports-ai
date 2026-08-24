from __future__ import annotations

from collections import Counter

from backend.app.core.sports_ontology import OntologyConcept, ontology_for
from backend.app.domain.schemas import SportContext


def ontology_index(sport: SportContext) -> dict[str, OntologyConcept]:
    return {concept.concept_id: concept for concept in ontology_for(sport)}


def planner_ontology_context(sport: SportContext) -> str:
    """Compact concept graph exposed to the planner.

    The planner receives concept identities and meanings, not phrase examples to match.
    It must select IDs by interpreting the user's goal and inference structure.
    """
    concepts = ontology_for(sport)
    if not concepts:
        return "No hay ontología específica para este deporte. Usá conceptos generales."

    lines = []
    for concept in concepts:
        labels = ", ".join(concept.canonical[:3])
        flags = f"; flags={','.join(concept.flags)}" if concept.flags else ""
        lines.append(
            f"- {concept.concept_id} | domain={concept.domain} | labels={labels} | "
            f"meaning={concept.description}{flags}"
        )
    return "\n".join(lines)


def selected_concept_context(sport: SportContext, concept_ids: list[str]) -> str:
    index = ontology_index(sport)
    lines: list[str] = []
    for concept_id in concept_ids:
        concept = index.get(concept_id)
        if concept is None:
            continue
        lines.append(
            f"- {concept.concept_id}: {concept.description} "
            f"[dominio={concept.domain}]"
        )
    return "\n".join(lines)


def primary_domain(sport: SportContext, concept_ids: list[str]) -> str:
    index = ontology_index(sport)
    domains = [index[item].domain for item in concept_ids if item in index]
    if not domains:
        return "general"
    return Counter(domains).most_common(1)[0][0]


def canonical_labels(sport: SportContext, concept_ids: list[str]) -> list[str]:
    index = ontology_index(sport)
    labels: list[str] = []
    for concept_id in concept_ids:
        concept = index.get(concept_id)
        if concept and concept.canonical:
            labels.append(concept.canonical[0])
    return labels


def concept_flags(sport: SportContext, concept_ids: list[str]) -> set[str]:
    index = ontology_index(sport)
    flags: set[str] = set()
    for concept_id in concept_ids:
        concept = index.get(concept_id)
        if concept:
            flags.update(concept.flags)
    return flags


def valid_concept_ids(sport: SportContext, concept_ids: list[str]) -> list[str]:
    index = ontology_index(sport)
    result: list[str] = []
    seen: set[str] = set()
    for item in concept_ids:
        if item in index and item not in seen:
            result.append(item)
            seen.add(item)
    return result
