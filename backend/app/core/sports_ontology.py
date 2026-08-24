from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.schemas import SportContext


@dataclass(frozen=True, slots=True)
class OntologyConcept:
    concept_id: str
    domain: str
    canonical: tuple[str, ...]
    description: str
    semantic_examples: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def embedding_text(self) -> str:
        return (
            f"concept={self.concept_id}; domain={self.domain}; "
            f"labels={', '.join(self.canonical)}; meaning={self.description}; "
            f"examples={' | '.join(self.semantic_examples)}"
        )


FOOTBALL_CONCEPTS: tuple[OntologyConcept, ...] = (
    OntologyConcept(
        "external_load", "physical_performance",
        ("external load", "carga externa"),
        "Trabajo físico realizado; no equivale por sí solo a rendimiento, fitness o fatiga.",
        ("interpretar distancia, HSR, sprint o aceleraciones", "comparar demandas físicas registradas por GPS"),
        ("carga externa", "external load"),
    ),
    OntologyConcept(
        "internal_load", "internal_load",
        ("internal load", "carga interna"),
        "Respuesta individual al entrenamiento o competición, con medidas como RPE o frecuencia cardíaca.",
        ("qué significa carga interna", "RPE como respuesta a una sesión"),
        ("carga interna", "internal load", "rpe"),
    ),
    OntologyConcept(
        "training_load_monitoring", "physical_performance",
        ("training load", "load monitoring", "monitoreo de carga", "monitoramento de carga"),
        "Seguimiento de demandas y respuestas al entrenamiento y competición.",
        ("estudios sobre monitorización de carga", "monitoramento de carga no futebol profissional"),
        ("monitoreo de carga", "monitoramento de carga", "training load", "load monitoring"),
    ),
    OntologyConcept(
        "match_exposure", "physical_performance",
        ("match exposure", "exposición", "minutes played", "minutos jugados"),
        "Tiempo o exposición que contextualiza métricas absolutas y comparaciones.",
        ("jugó menos minutos y quiero comparar HSR", "comparar jugadores con distinta exposición"),
        ("minutos", "exposición", "exposicion"),
    ),
    OntologyConcept(
        "physical_performance", "physical_performance",
        ("physical performance", "rendimiento físico"),
        "Constructo amplio que no debe inferirse automáticamente desde una sola métrica externa.",
        ("corrió menos entonces rindió peor", "una métrica GPS demuestra que estuvo mejor"),
        ("rendimiento físico", "rendimiento fisico"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "hsr", "physical_performance",
        ("HSR", "high-speed running"),
        "Carrera a alta velocidad definida mediante un umbral absoluto o relativo.",
        ("comparar HSR entre partidos", "metros recorridos a alta velocidad"),
        ("hsr", "high speed running", "high-speed running"),
    ),
    OntologyConcept(
        "sprint_threshold", "physical_performance",
        ("sprint", "speed threshold", "umbral de velocidad"),
        "Sprint depende de una definición operacional o umbral y no tiene un valor universal.",
        ("nuestro club llama sprint a más de 25 km/h", "comparar umbrales de sprint"),
        ("sprint", "umbral", "threshold", "zona 5", "speed zone"),
    ),
    OntologyConcept(
        "mechanical_events", "physical_performance",
        ("acceleration", "deceleration", "aceleración", "desaceleración"),
        "Aceleraciones y desaceleraciones describen demanda externa y no miden directamente frescura o fatiga.",
        ("muchas aceleraciones significa que llegó fresco", "más desaceleraciones demuestra fatiga"),
        ("aceleraciones", "desaceleraciones", "accelerations", "decelerations"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "fatigue_state", "physical_performance",
        ("fatigue", "fatiga", "freshness", "frescura"),
        "Estado fisiológico o perceptual que no se demuestra con una sola métrica externa.",
        ("inferir frescura desde GPS", "demostrar fatiga por una métrica"),
        ("fatiga", "fatigado", "fresco", "freshness"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "metres_per_minute", "physical_performance",
        ("metres per minute", "metros por minuto"),
        "Indicador relativo de locomoción que debe contextualizarse por posición, rol y fase del juego.",
        ("comparar metros por minuto de posiciones distintas", "quién trabajó más usando m/min"),
        ("metros por minuto", "m/min", "metres per minute"),
    ),
    OntologyConcept(
        "position_role", "physical_performance",
        ("position", "posición", "role", "rol"),
        "Posición y rol táctico condicionan las demandas esperables y las comparaciones entre jugadores.",
        ("comparar un lateral con un nueve", "el rol explica diferencias de demanda física"),
        ("posición", "posicion", "rol", "role", "position"),
    ),
    OntologyConcept(
        "gps_comparability", "physical_performance",
        ("provider", "proveedor", "device", "dispositivo", "GPS comparability"),
        "Métricas con el mismo nombre pueden no ser directamente comparables entre proveedores o definiciones.",
        ("Catapult y otro GPS reportan HSR", "mismo nombre de métrica en sistemas diferentes"),
        ("catapult", "proveedor", "provider", "dispositivo", "gps"),
    ),
    OntologyConcept(
        "injury_causality", "physical_performance",
        ("injury", "lesión", "causality", "causalidad", "training load"),
        "Una asociación temporal entre carga y lesión no prueba causalidad.",
        ("subimos la carga y aparecieron lesiones, entonces fue la causa", "más lesiones después de aumentar volumen"),
        ("lesión", "lesion", "lesiones", "injury", "causa", "causalidad"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "return_to_play_readiness", "physical_performance",
        ("return to play", "readiness", "rehabilitation", "rehabilitación"),
        "La preparación para competir es multidimensional y no se establece sólo por igualar una métrica previa.",
        ("en rehabilitación completó la misma distancia y quiero saber si ya está listo", "criterios de return to play"),
        ("rehabilitación", "rehabilitacion", "return to play", "readiness", "listo para competir"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "possession", "tactical_analysis",
        ("possession", "posesión"),
        "La posesión describe control temporal del balón pero no equivale automáticamente a jugar mejor.",
        ("tener 65 por ciento de posesión significa jugar mejor", "más posesión que el rival"),
        ("posesión", "posesion", "possession"),
        ("causal_risk",),
    ),
    OntologyConcept(
        "high_press", "tactical_analysis",
        ("high press", "presión alta", "pressing"),
        "Comportamiento colectivo de presión en zonas altas; una recuperación aislada no demuestra todo el comportamiento previo.",
        ("recuperamos cerca del área rival, eso prueba presión alta", "analizar pressing alto del equipo"),
        ("presión alta", "presion alta", "high press", "pressing"),
    ),
    OntologyConcept(
        "mid_block", "tactical_analysis",
        ("mid block", "bloque medio"),
        "Organización defensiva intermedia cuya interpretación depende de distancias, roles, saltos y disparadores colectivos.",
        ("defendemos en bloque medio", "el nueve salta solo al central desde un bloque medio"),
        ("bloque medio", "mid block"),
    ),
    OntologyConcept(
        "offensive_transition", "tactical_analysis",
        ("offensive transition", "transición ofensiva"),
        "Fase posterior a recuperar la posesión; no es sinónimo obligatorio de contraataque.",
        ("recuperamos y atacamos inmediatamente", "toda transición ofensiva es contraataque"),
        ("transición ofensiva", "transicion ofensiva", "offensive transition"),
    ),
    OntologyConcept(
        "counterattack", "tactical_analysis",
        ("counterattack", "contraataque"),
        "Ataque rápido que explota el desequilibrio rival; es una posibilidad dentro de la transición ofensiva.",
        ("recuperar y atacar rápido es siempre contraataque", "diferencia entre transición ofensiva y contraataque"),
        ("contraataque", "counterattack"),
    ),
    OntologyConcept(
        "ball_recovery", "tactical_analysis",
        ("ball recovery", "recuperación", "recuperación de balón"),
        "Evento de recuperar la posesión; su ubicación no determina por sí sola todo el comportamiento defensivo previo.",
        ("recuperaciones altas cerca del área rival", "recuperar la pelota no prueba cómo se presionó"),
        ("recuperamos", "recuperación", "recuperacion", "recovery", "recoveries"),
    ),
    OntologyConcept(
        "private_operational_definition", "sports",
        ("private protocol", "definición operacional privada", "club methodology"),
        "Regla, umbral o protocolo propio que debe permanecer en memoria privada y no convertirse en verdad global.",
        ("en nuestro club usamos esta definición", "para nosotros zona 5 empieza a determinada velocidad"),
        ("nuestro club", "para nosotros", "nuestro criterio", "nuestro protocolo"),
        ("private_memory",),
    ),
)


def ontology_for(sport: SportContext) -> tuple[OntologyConcept, ...]:
    return FOOTBALL_CONCEPTS if sport is SportContext.FOOTBALL else ()
