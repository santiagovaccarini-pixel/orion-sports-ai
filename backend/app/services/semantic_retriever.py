from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.intent import SemanticPlan
from backend.app.services.knowledge_base import (
    KnowledgeBase,
    KnowledgeChunk,
    _fold,
    _split_chunks,
    _terms,
)


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: KnowledgeChunk
    score: float


def search_with_intent(
    knowledge: KnowledgeBase,
    original_query: str,
    semantic_plan: SemanticPlan | None,
    *,
    limit: int = 12,
) -> list[KnowledgeChunk]:
    """Search local documents only after intent has already been resolved.

    Lexical overlap is permitted here as a retrieval implementation detail, but it is
    never allowed to decide what the user means. If the reasoning plan does not require
    local data/documents, retrieval is skipped entirely so unrelated files cannot steer
    the answer.
    """
    if semantic_plan is None:
        return []
    if not semantic_plan.needs_local_data:
        return []

    queries = semantic_plan.retrieval_texts(original_query)
    query_terms = [_terms(query) for query in queries]
    concept_terms = [_terms(concept) for concept in semantic_plan.concepts]
    folded_concepts = [_fold(concept) for concept in semantic_plan.concepts if concept.strip()]

    ranked: list[RankedChunk] = []
    for document in knowledge.list_documents():
        csv_mode = document.name.lower().endswith(".csv")
        for index, content in enumerate(_split_chunks(document.content, csv_mode=csv_mode)):
            terms = _terms(content)
            if not terms:
                continue

            score = 0.0
            for query_index, terms_for_query in enumerate(query_terms):
                if not terms_for_query:
                    continue
                overlap = len(terms & terms_for_query)
                if overlap:
                    weight = 3.0 if query_index == 0 else 2.0 if query_index == 1 else 1.25
                    score += weight * overlap / max(len(terms_for_query), 1)

            for terms_for_concept in concept_terms:
                if terms_for_concept and terms_for_concept <= terms:
                    score += 1.4
                elif terms_for_concept and terms_for_concept & terms:
                    score += 0.45

            folded_content = _fold(content)
            for concept in folded_concepts:
                if concept and concept in folded_content:
                    score += 1.8

            if score > 0:
                ranked.append(
                    RankedChunk(
                        KnowledgeChunk(document.id, document.name, index, content),
                        score,
                    )
                )

    ranked.sort(
        key=lambda item: (-item.score, item.chunk.document_name, item.chunk.index)
    )
    return [item.chunk for item in ranked[:limit]]
