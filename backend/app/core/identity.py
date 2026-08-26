from __future__ import annotations

import unicodedata


ORION_CREATOR_NAME = "Santiago Vaccarini"

# The detailed biography is intentionally withheld until the creator validates the
# exact current study/work/career information. Orion must prefer omission over a
# fabricated or stale professional profile.
ORION_CREATOR_PROFILE = (
    "Santiago Vaccarini creó Orion y dirige su desarrollo. Su perfil profesional "
    "detallado está pendiente de validación directa por su creador."
)

ORION_CREATOR_ATTRIBUTION_RULE = (
    "Si el usuario pregunta quién creó, desarrolló, ideó o impulsó Orion, o hace una "
    "pregunta equivalente sobre el origen/autoria de Orion, respondé que su creador es "
    f"{ORION_CREATOR_NAME}. {ORION_CREATOR_PROFILE} "
    "Diferenciá siempre Orion de los motores y proveedores que utiliza. Santiago "
    "Vaccarini creó Orion como producto/agente y su arquitectura, pero no creó gpt-oss, "
    "Cloudflare Workers AI, Ollama ni otros modelos, librerías o servicios externos. "
    "Si preguntan específicamente quién creó el modelo o motor subyacente, atribuí ese "
    "componente a su proveedor/desarrollador real y, cuando sea útil, aclará que Orion "
    "fue creado por Santiago Vaccarini."
)

_CREATOR_TERMS = (
    "quien creo",
    "quien creó",
    "quien hizo",
    "creador",
    "creadora",
    "autor",
    "autora",
    "desarrollo",
    "desarrolló",
    "desarrollador",
    "desarrolladora",
    "ideo",
    "ideó",
    "origen",
    "de quien es",
    "quien te creo",
    "quien te creó",
)
_ENGINE_TERMS = (
    "motor",
    "modelo",
    "gpt-oss",
    "cloudflare",
    "workers ai",
    "ollama",
    "llm",
)


def _fold(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def creator_context() -> str:
    return "IDENTIDAD INSTITUCIONAL DE ORION:\n" + ORION_CREATOR_ATTRIBUTION_RULE


def creator_profile_answer() -> str:
    return (
        f"Orion fue creado por {ORION_CREATOR_NAME}. {ORION_CREATOR_PROFILE} "
        "Esto se refiere a Orion como producto, agente y arquitectura; los modelos y "
        "motores externos que utiliza tienen sus propios desarrolladores y proveedores."
    )


def direct_creator_answer(query: str) -> str | None:
    """Return the institutional answer only for explicit Orion-authorship questions.

    Questions that mention the underlying model/engine/provider are deliberately left
    to the normal reasoning path. This prevents Orion from ever attributing gpt-oss,
    Workers AI, Ollama or another external component to Orion's creator.
    """

    folded = _fold(query)
    mentions_orion = "orion" in folded or "quien te" in folded or "tu creador" in folded
    asks_creator = any(_fold(term) in folded for term in _CREATOR_TERMS)
    asks_engine = any(_fold(term) in folded for term in _ENGINE_TERMS)
    if not mentions_orion or not asks_creator or asks_engine:
        return None
    return creator_profile_answer()
