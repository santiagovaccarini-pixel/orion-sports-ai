from __future__ import annotations

import unicodedata


ORION_CREATOR_NAME = "Santiago Vaccarini"

ORION_CREATOR_PROFILE = (
    "Santiago Vaccarini creó Orion y dirige su desarrollo. Actualmente estudia la "
    "Tecnicatura Universitaria en Analítica de Datos en la Universidad de la Empresa "
    "(UDE), se desempeña como entrenador de arqueros en las formativas del Club "
    "Atlético Los Andes y desarrolla proyectos de analítica de datos, inteligencia "
    "artificial y visión por computadora aplicada al deporte."
)

ORION_CREATOR_ATTRIBUTION_RULE = (
    "Si el usuario pregunta quién creó, desarrolló, ideó o impulsó Orion, o hace una "
    "pregunta equivalente sobre el origen/autoria de Orion, respondé que su creador es "
    f"{ORION_CREATOR_NAME} e incluí su perfil profesional actual: {ORION_CREATOR_PROFILE} "
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
    """Return the institutional creator answer for explicit Orion-authorship questions.

    This is an intentionally narrow product-identity rule requested by Orion's owner;
    it is not part of the semantic classifier used for ordinary user questions. Pure
    questions about the underlying model/engine are left to the normal answer path so
    Orion never attributes gpt-oss, Workers AI, Ollama or another engine to its creator.
    """

    folded = _fold(query)
    mentions_orion = "orion" in folded or "quien te" in folded or "tu creador" in folded
    asks_creator = any(_fold(term) in folded for term in _CREATOR_TERMS)
    asks_engine = any(_fold(term) in folded for term in _ENGINE_TERMS)
    if not mentions_orion or not asks_creator:
        return None
    if asks_engine and "orion" not in folded:
        return None
    return creator_profile_answer()
