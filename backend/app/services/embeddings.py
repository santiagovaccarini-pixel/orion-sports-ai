"""Find things by what they mean, when there are too many to send them all.

Two places in Orion had outgrown "send everything and let the model decide".
Memory ships every saved entry into three prompts per question, so a working
history of cases would eventually crowd out the evidence it is supposed to
support. Local retrieval picked document chunks by counting shared words, which
is the one thing this project forbids everywhere else - and it fails exactly
where it matters, because a chunk answering "¿cuánto corrió Pérez?" says
"9.800 m", not "cuánto".

An embedding turns a piece of text into a list of numbers positioned by
meaning, so two texts that say the same thing in different words end up near
each other. Comparing those positions is arithmetic, which keeps the decision
deterministic and out of the model's mood - the same principle as counting
sources for a confidence label.

This is a ranking aid, never a gate. It reorders candidates; the model still
decides what any of it means, and when embeddings are unavailable - not
configured, provider down, request failed - every caller falls back to the
behaviour it had before, so a search outage costs relevance, never an answer.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from typing import Protocol, Sequence

import httpx

from backend.app.core.config import Settings

# Embedding one query and a page of chunks is a single small request, but a
# runaway caller should not be able to turn a chat into a bulk job.
MAX_TEXTS_PER_CALL = 96
MAX_CHARACTERS_PER_TEXT = 2_000
CACHE_MAX_ENTRIES = 512

Vector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Vectors in the same order as `texts`, or () when unavailable."""


class NullEmbeddings:
    """What Orion uses when no embedding model is configured.

    Returning nothing rather than raising is the point: every caller reads an
    empty result as "rank the old way", so turning this feature off is a
    configuration change and never a failure.
    """

    available = False

    async def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return ()


class CloudflareEmbeddings:
    """Embeddings from Workers AI, over the same OpenAI-compatible endpoint.

    Cloudflare is where Orion already has credentials, and the multilingual
    model matters here: the text is Spanish, and an English-only model would
    place "carga interna" and "internal load" far apart while placing "carga
    interna" and "carga externa" close together.
    """

    available = True

    def __init__(self, settings: Settings) -> None:
        self._model = settings.embeddings_model
        self._base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/v1"
        )
        self._headers = {
            "Authorization": f"Bearer {settings.cloudflare_api_token}",
            "Content-Type": "application/json",
        }
        # Deliberately short. Ranking is an optimisation: waiting on it longer
        # than the answer itself would take is worse than ranking the old way.
        self._timeout = httpx.Timeout(settings.embeddings_timeout_seconds)
        self._cache: OrderedDict[str, Vector] = OrderedDict()

    def _cached(self, text: str) -> Vector | None:
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector

    def _store(self, text: str, vector: Vector) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    async def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        clipped = [text[:MAX_CHARACTERS_PER_TEXT] for text in texts[:MAX_TEXTS_PER_CALL]]
        if not clipped:
            return ()

        # A saved memory entry or a document chunk is embedded on every
        # question it might answer, and its text does not change between them.
        missing = [text for text in dict.fromkeys(clipped) if self._cached(text) is None]
        if missing:
            fresh = await self._request(missing)
            if fresh is None:
                return ()
            for text, vector in zip(missing, fresh):
                self._store(text, vector)

        vectors = [self._cached(text) for text in clipped]
        if any(vector is None for vector in vectors):
            return ()
        return tuple(vector for vector in vectors if vector is not None)

    async def _request(self, texts: list[str]) -> list[Vector] | None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers=self._headers,
                    json={"model": self._model, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            # Ranking is best-effort: a provider outage must degrade relevance,
            # never break the question being asked.
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            return None
        vectors: list[Vector] = []
        for item in data:
            if not isinstance(item, dict):
                return None
            values = item.get("embedding")
            if not isinstance(values, list) or not values:
                return None
            try:
                vectors.append(tuple(float(value) for value in values))
            except (TypeError, ValueError):
                return None
        return vectors


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """The configured provider, or the null one when it cannot be used.

    Credentials are checked here rather than at the call site so no caller has
    to remember that ranking is optional.
    """

    if not settings.embeddings_enabled:
        return NullEmbeddings()
    if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
        return NullEmbeddings()
    return CloudflareEmbeddings(settings)


def cosine_similarity(left: Vector, right: Vector) -> float:
    """How close two meanings are, from -1 (opposite) to 1 (the same).

    Length is divided out, so a long chunk is not favoured over a short one
    just for having more words in it.
    """

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


async def rank_by_meaning(
    provider: EmbeddingProvider,
    query: str,
    candidates: Sequence[str],
    *,
    limit: int,
) -> tuple[int, ...] | None:
    """Indexes of the `limit` candidates closest in meaning to `query`.

    Returns None - not an empty tuple - when ranking could not happen at all,
    so a caller can tell "nothing is relevant" apart from "ranking was
    unavailable" and fall back rather than silently dropping everything.
    """

    if not query.strip() or not candidates or limit <= 0:
        return None
    vectors = await provider.embed([query, *candidates])
    if len(vectors) != len(candidates) + 1:
        return None
    query_vector, *candidate_vectors = vectors
    scored = sorted(
        (
            (cosine_similarity(query_vector, vector), index)
            for index, vector in enumerate(candidate_vectors)
        ),
        # Ties break on the original order, so the result is stable: the same
        # question twice must not reshuffle the evidence under the answer.
        key=lambda item: (-item[0], item[1]),
    )
    return tuple(index for _score, index in scored[:limit])


async def rank_with_timeout(
    provider: EmbeddingProvider,
    query: str,
    candidates: Sequence[str],
    *,
    limit: int,
    timeout_seconds: float,
) -> tuple[int, ...] | None:
    """Ranking, or None if it does not arrive in time.

    The wall clock belongs to the person waiting for an answer. A slow ranking
    call must not extend a question; it must be abandoned.
    """

    try:
        return await asyncio.wait_for(
            rank_by_meaning(provider, query, candidates, limit=limit),
            timeout=timeout_seconds,
        )
    except (TimeoutError, asyncio.TimeoutError):
        return None
