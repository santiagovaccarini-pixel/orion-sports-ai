from __future__ import annotations

import unicodedata
from datetime import date


ORION_CREATOR_NAME = "Santiago Vaccarini"
ORION_CREATOR_BIRTH_DATE = date(2007, 1, 16)
ORION_CREATOR_BASE = "Buenos Aires, Argentina"

ORION_CREATOR_STUDIES = (
    "Es estudiante de Ciencia de Datos en ISTEA y actualmente también realiza una "
    "formación de nivel junior en análisis de datos de Google."
)

ORION_CREATOR_CAREER = (
    "Actualmente trabaja como científico de datos y analista de datos en Atlético "
    "Mineiro, dentro del ámbito del fútbol. Anteriormente trabajó en Estudiantes de "
    "La Plata entre 2025 y 2026."
)

ORION_CREATOR_SKILLS = (
    "Entre sus principales herramientas y áreas de trabajo se encuentran Excel, "
    "Inteligencia Artificial, PowerPoint y VBA."
)

ORION_CREATOR_SPORT_TRAINING = (
    "También participó de un curso para preparadores físicos en España en un programa "
    "en el que se impartía formación vinculada a la Licencia Pro. Esta descripción no "
    "afirma que haya obtenido personalmente dicha licencia."
)


def creator_age(on_date: date | None = None) -> int:
    reference = on_date or date.today()
    years = reference.year - ORION_CREATOR_BIRTH_DATE.year
    birthday_passed = (
        reference.month,
        reference.day,
    ) >= (
        ORION_CREATOR_BIRTH_DATE.month,
        ORION_CREATOR_BIRTH_DATE.day,
    )
    return years if birthday_passed else years - 1


def creator_public_profile() -> str:
    birth = ORION_CREATOR_BIRTH_DATE.strftime("%d/%m/%Y")
    return " ".join(
        (
            f"{ORION_CREATOR_NAME} nació el {birth}, tiene {creator_age()} años y es de "
            f"{ORION_CREATOR_BASE}.",
            ORION_CREATOR_STUDIES,
            ORION_CREATOR_CAREER,
            ORION_CREATOR_SKILLS,
            ORION_CREATOR_SPORT_TRAINING,
            "Es el creador de Orion.",
        )
    )


ORION_CREATOR_ATTRIBUTION_RULE = (
    "Si el usuario pregunta quién creó, desarrolló, ideó o impulsó Orion, o hace una "
    "pregunta equivalente sobre el origen o autoría de Orion, respondé que su creador "
    f"es {ORION_CREATOR_NAME} e incluí su perfil público completo provisto por el "
    "sistema. Diferenciá siempre Orion de los motores y proveedores que utiliza. "
    "Santiago Vaccarini creó Orion, pero no creó gpt-oss, Cloudflare Workers AI, "
    "Ollama ni otros modelos, librerías o servicios externos. Si preguntan "
    "específicamente quién creó el modelo o motor subyacente, atribuí ese componente "
    "a su proveedor o desarrollador real y, cuando sea útil, aclará que Orion fue "
    "creado por Santiago Vaccarini."
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
    return "\n".join(
        (
            "IDENTIDAD INSTITUCIONAL DE ORION:",
            ORION_CREATOR_ATTRIBUTION_RULE,
            "PERFIL PÚBLICO VALIDADO DEL CREADOR:",
            creator_public_profile(),
        )
    )


def creator_profile_answer() -> str:
    return (
        f"Orion fue creado por {ORION_CREATOR_NAME}. {creator_public_profile()} "
        "Esto se refiere a Orion como producto y agente; los modelos y motores "
        "externos que utiliza tienen sus propios desarrolladores y proveedores."
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
