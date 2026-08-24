from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


TaskType = Literal[
    "direct_answer",
    "definition",
    "interpretation",
    "comparison",
    "calculation",
    "data_query",
    "chart",
    "research",
    "planning",
    "debugging",
    "clarification",
]


class SemanticPlan(BaseModel):
    """Structured interpretation of what the user is actually trying to achieve."""

    literal_request: str = Field(min_length=1, max_length=400)
    user_goal: str = Field(min_length=1, max_length=500)
    domain: str = Field(default="general", min_length=1, max_length=120)
    task_type: TaskType = "direct_answer"
    concepts: list[str] = Field(default_factory=list, max_length=12)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=6)
    missing_variables: list[str] = Field(default_factory=list, max_length=10)
    needs_global_knowledge: bool = True
    needs_private_memory: bool = False
    needs_local_data: bool = False
    needs_web: bool = False
    comparison: bool = False
    causal_claim_risk: bool = False
    requires_clarification: bool = False
    referenced_previous_context: bool = False
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("concepts", "retrieval_queries", "missing_variables")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        clean: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = " ".join(value.strip().split())
            key = item.casefold()
            if item and key not in seen:
                clean.append(item)
                seen.add(key)
        return clean

    def retrieval_texts(self, original_query: str) -> list[str]:
        """Queries used by retrieval after intent resolution, ordered by relevance."""
        candidates = [
            original_query,
            self.user_goal,
            *self.retrieval_queries,
            " ".join(self.concepts),
        ]
        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            clean = " ".join(candidate.strip().split())
            key = clean.casefold()
            if clean and key not in seen:
                result.append(clean)
                seen.add(key)
        return result
