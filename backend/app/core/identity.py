from __future__ import annotations

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
    "Hecho institucional: Santiago Vaccarini es el creador de Orion. Orion es el "
    "producto/agente creado por Santiago; los modelos, motores, librerías y servicios "
    "externos que Orion utiliza conservan la autoría de sus desarrolladores reales. "
    "Santiago Vaccarini no creó gpt-oss, Cloudflare Workers AI, Ollama ni otros "
    "componentes externos. Interpretá semánticamente la pregunta del usuario y usá "
    "estos hechos solo cuando sean pertinentes a lo que realmente está preguntando."
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


def direct_creator_answer(query: str) -> str | None:
    """Deprecated compatibility shim: lexical identity routing is disabled.

    Orion's creator information is supplied as institutional context to the model.
    The model must interpret the user's intent semantically; Python must not decide
    authorship questions through keywords, similar wording or phrase matching.
    """

    _ = query
    return None
