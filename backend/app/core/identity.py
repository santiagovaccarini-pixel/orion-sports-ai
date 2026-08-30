from __future__ import annotations

from datetime import date

from backend.app.core.config import (
    OPENAI_ENDPOINT_PROVIDERS,
    Settings,
    get_settings,
)


ORION_CREATOR_NAME = "Santiago Vaccarini"
ORION_CREATOR_BIRTH_DATE = date(2007, 1, 16)
ORION_CREATOR_BASE = "Buenos Aires, Argentina"

ORION_CREATOR_STUDIES = (
    "Es estudiante de Ciencia de Datos en ISTEA y tambien tiene un titulo como analista de datos junior con Google."
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
    "en el que se impartía formación vinculada a la Licencia Pro."
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


def current_engine_fact() -> str:
    """State, as a positive first-person fact, which model is actually answering.

    Sourced from settings (not hardcoded) so it can't drift from the real
    deployment, and phrased affirmatively so the model has a grounded answer
    instead of guessing its own identity from training-time self-belief (a
    well-known LLM failure mode: confidently claiming to be a different,
    unrelated model).
    """

    settings = get_settings()
    quick, deep, host = _engine_for_provider(settings)
    if quick == deep:
        models = f"el modelo {quick}"
    else:
        models = f"los modelos {quick} (modo rápido) y {deep} (modo profundo)"
    return (
        f"HECHO SOBRE VOS MISMO: el motor de lenguaje que estás usando ahora mismo "
        f"para generar esta respuesta es {models}, servido a través de {host}. "
        "No es GPT-4 ni ningún otro modelo: si te preguntan qué modelo o "
        "motor te hace funcionar, respondé con este dato exacto en vez de adivinar "
        "a partir de lo que creas recordar sobre vos mismo."
    )


def _engine_for_provider(settings: Settings) -> tuple[str, str, str]:
    """The models and the company actually serving them, for the selected provider.

    The same weights are available from several companies, so naming the host has
    to follow the configuration. Reading it off one provider's settings would make
    Orion state a false fact about itself the moment the provider changes.
    """

    if settings.model_provider == "ollama":
        return settings.quick_model, settings.deep_model, "Ollama en esta computadora"
    if settings.model_provider in OPENAI_ENDPOINT_PROVIDERS:
        return (
            settings.endpoint_quick_model,
            settings.endpoint_deep_model,
            settings.model_provider.capitalize(),
        )
    return (
        settings.cloudflare_quick_model,
        settings.cloudflare_deep_model,
        "Cloudflare Workers AI",
    )


def creator_context() -> str:
    return "\n".join(
        (
            "IDENTIDAD INSTITUCIONAL DE ORION:",
            ORION_CREATOR_ATTRIBUTION_RULE,
            current_engine_fact(),
            "PERFIL PÚBLICO VALIDADO DEL CREADOR:",
            creator_public_profile(),
        )
    )


def institutional_identity_brief() -> str:
    """Compact product-vs-engine identity for internal stages (planner/reviewer).

    This is context, not routing: the model resolves the referent semantically.
    """

    return (
        "IDENTIDAD DEL PRODUCTO: Orion es un producto/agente creado por "
        f"{ORION_CREATOR_NAME}. El motor de lenguaje y los servicios externos que "
        "Orion utiliza (gpt-oss de OpenAI, Cloudflare Workers AI, Ollama, etc.) "
        "tienen sus propios autores; Santiago no los creó. Cuando la conversación "
        "se dirige a «vos», «te» o «tu creador», el referente por defecto es el "
        "producto Orion, salvo que la conversación indique que se habla del motor "
        "subyacente. La autoría e identidad de Orion es un hecho institucional ya "
        "provisto y estable: no requiere búsqueda externa ni verificación web.\n"
        + current_engine_fact()
    )


def direct_creator_answer(query: str) -> str | None:
    """Deprecated compatibility shim: lexical identity routing is disabled.

    Orion's creator information is supplied as institutional context to the model.
    The model must interpret the user's intent semantically; Python must not decide
    authorship questions through keywords, similar wording or phrase matching.
    """

    _ = query
    return None
