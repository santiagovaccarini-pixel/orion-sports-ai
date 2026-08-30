"""Ask the model what, if anything, from an exchange is worth remembering.

Orion never writes to memory on its own. It proposes, showing the exact sentence
it would store, and the person decides: save it, edit the wording first, or throw
it away. Memory that describes someone has to stay theirs to review, and a
proposal they can read is the difference between a tool that remembers and one
that keeps a file on them.

This runs after the answer is delivered, so it costs the user no waiting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from backend.app.domain.models import SelectedMode
from backend.app.domain.schemas import ChatMessage
from backend.app.providers.model_provider import ModelProvider

MAX_SUGGESTIONS = 2
MAX_SUGGESTION_CHARACTERS = 300

SUGGESTION_PROMPT = """
Sos la etapa de memoria de Orion. Leés un intercambio ya terminado y decidís si
apareció algo que valga la pena recordar para conversaciones futuras.

Devolvés SOLO un objeto JSON con esta forma:
{"suggestions": [{"content": "el hecho, redactado como una oración completa",
                  "reason": "por qué serviría más adelante"}]}

Reglas:
- Proponé como máximo dos cosas, y lo normal es proponer NINGUNA. La lista vacía
  es la respuesta correcta la mayoría de las veces.
- Solo proponé hechos duraderos sobre el usuario, su trabajo, su equipo, sus
  jugadores o sus preferencias: cosas que sigan siendo ciertas dentro de meses y
  que cambien cómo responderías la próxima vez.
- Nunca propongas: datos que se pueden volver a buscar en internet, detalles de
  esta consulta puntual, resúmenes de tu propia respuesta, ni nada que el usuario
  no haya afirmado él mismo.
- Si el usuario preguntó algo pero no afirmó nada sobre sí mismo o su contexto,
  la lista va vacía.
- Redactá cada propuesta en primera persona del usuario o en tercera neutra, como
  una oración corta y verificable. El usuario va a leer ese texto exacto antes de
  decidir, así que tiene que entenderse solo, sin el resto de la conversación.
- No repitas nada que ya esté en la memoria actual que se te muestra abajo.
""".strip()


@dataclass(frozen=True, slots=True)
class MemorySuggestion:
    content: str
    reason: str


def _extract_json(value: str) -> dict[str, object]:
    clean = value.strip()
    if clean.startswith("```"):
        lines = [line for line in clean.splitlines() if not line.startswith("```")]
        clean = "\n".join(lines).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(clean[start : end + 1])
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_suggestions(raw: str) -> tuple[MemorySuggestion, ...]:
    """Read the model's proposals, dropping anything malformed.

    A suggestion that cannot be parsed is simply not offered: the cost of losing
    one is a fact the user can still type by hand, while the cost of guessing is
    a sentence they never approved.
    """

    payload = _extract_json(raw)
    items = payload.get("suggestions")
    if not isinstance(items, list):
        return ()
    suggestions: list[MemorySuggestion] = []
    for item in items[:MAX_SUGGESTIONS]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        suggestions.append(
            MemorySuggestion(
                content=content[:MAX_SUGGESTION_CHARACTERS],
                reason=str(item.get("reason") or "").strip()[:200],
            )
        )
    return tuple(suggestions)


async def suggest_memories(
    provider: ModelProvider,
    messages: Sequence[ChatMessage],
    answer: str,
    *,
    memory_context: str = "",
) -> tuple[MemorySuggestion, ...]:
    exchange = "\n\n".join(
        f"{message.role.upper()}: {message.content[:2_000]}" for message in messages[-4:]
    )
    prompt = (
        f"{SUGGESTION_PROMPT}\n\n"
        f"MEMORIA ACTUAL:\n{memory_context or '(vacía)'}\n\n"
        f"INTERCAMBIO:\n{exchange}\n\nRESPUESTA DE ORION:\n{answer[:4_000]}"
    )
    result = await provider.chat(
        mode=SelectedMode.QUICK,
        messages=[ChatMessage(role="user", content=prompt)],
        system_prompt="Devolvés únicamente JSON válido.",
        structured=True,
        reasoning_effort="low",
    )
    return parse_suggestions(result.content)
