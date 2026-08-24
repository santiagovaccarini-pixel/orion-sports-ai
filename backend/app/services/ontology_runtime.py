from __future__ import annotations

from collections import Counter

from backend.app.core.sports_ontology import OntologyConcept, ontology_for
from backend.app.domain.schemas import SportContext


def ontology_index(sport: SportContext) -> dict[str, OntologyConcept]:
    return {concept.concept_id: concept for concept in ontology_for(sport)}


def _all_ontology_index() -> dict[str, OntologyConcept]:
    result: dict[str, OntologyConcept] = {}
    for sport in SportContext:
        for concept in ontology_for(sport):
            result.setdefault(concept.concept_id, concept)
    return result


def _relation_text(concept: OntologyConcept) -> str:
    if not concept.relations:
        return ""
    return ",".join(
        f"{relation.relation}:{relation.target_id}" for relation in concept.relations
    )


def planner_ontology_context(sport: SportContext) -> str:
    """Expose a compact concept graph, not examples to imitate.

    IDs, definitions and graph edges let the planner reason about equivalence,
    dependency and contextual requirements without a similarity-ranking stage.
    """
    concepts = ontology_for(sport)
    if not concepts:
        return "No hay ontología específica para este deporte. Usá conceptos generales."

    lines: list[str] = []
    for concept in concepts:
        labels = ", ".join(concept.canonical[:2])
        relations = _relation_text(concept)
        relation_field = f" | relations={relations}" if relations else ""
        flags = f" | flags={','.join(concept.flags)}" if concept.flags else ""
        lines.append(
            f"- {concept.concept_id} | domain={concept.domain} | labels={labels} | "
            f"definition={concept.definition}{relation_field}{flags}"
        )
    return "\n".join(lines)


def selected_concept_context(sport: SportContext, concept_ids: list[str]) -> str:
    index = ontology_index(sport)
    if any(item not in index for item in concept_ids):
        index = {**_all_ontology_index(), **index}

    lines: list[str] = []
    for concept_id in concept_ids:
        concept = index.get(concept_id)
        if concept is None:
            continue
        relations = _relation_text(concept)
        relation_field = f"; relaciones={relations}" if relations else ""
        lines.append(
            f"- {concept.concept_id}: {concept.definition} "
            f"[dominio={concept.domain}{relation_field}]"
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
    seen: set[str] = set()
    for concept_id in concept_ids:
        concept = index.get(concept_id)
        if not concept:
            continue
        for label in concept.canonical[:2]:
            key = label.casefold()
            if key not in seen:
                labels.append(label)
                seen.add(key)
    return labels[:12]


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
