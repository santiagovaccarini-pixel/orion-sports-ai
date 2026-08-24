from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.schemas import SportContext


@dataclass(frozen=True, slots=True)
class OntologyRelation:
    relation: str
    target_id: str


@dataclass(frozen=True, slots=True)
class OntologyConcept:
    concept_id: str
    domain: str
    canonical: tuple[str, ...]
    definition: str
    relations: tuple[OntologyRelation, ...] = ()
    flags: tuple[str, ...] = ()


R = OntologyRelation


FOOTBALL_CONCEPTS: tuple[OntologyConcept, ...] = (
    OntologyConcept(
        "external_load",
        "physical_performance",
        ("external load", "carga externa"),
        "Trabajo físico realizado o prescripto.",
        (
            R("not_equivalent_to", "physical_performance"),
            R("not_equivalent_to", "fatigue_state"),
            R("contextualized_by", "match_exposure"),
        ),
    ),
    OntologyConcept(
        "internal_load",
        "internal_load",
        ("internal load", "carga interna"),
        "Respuesta individual al trabajo realizado.",
        (R("measured_by", "rpe"),),
    ),
    OntologyConcept(
        "rpe",
        "internal_load",
        ("RPE", "rating of perceived exertion"),
        "Valoración subjetiva del esfuerzo percibido.",
        (R("measure_of", "internal_load"),),
    ),
    OntologyConcept(
        "training_load_monitoring",
        "physical_performance",
        ("training load", "load monitoring"),
        "Seguimiento conjunto de demandas, respuestas y contexto de entrenamiento/competición.",
        (
            R("includes", "external_load"),
            R("includes", "internal_load"),
        ),
    ),
    OntologyConcept(
        "match_exposure",
        "physical_performance",
        ("match exposure", "minutes played"),
        "Tiempo o exposición que contextualiza métricas absolutas y comparaciones.",
    ),
    OntologyConcept(
        "physical_performance",
        "physical_performance",
        ("physical performance", "rendimiento físico"),
        "Constructo amplio; no se demuestra mediante una única métrica de demanda.",
        (R("not_equivalent_to", "external_load"),),
        ("causal_risk",),
    ),
    OntologyConcept(
        "total_distance",
        "physical_performance",
        ("total distance", "distancia total"),
        "Volumen locomotor total registrado durante una exposición.",
        (
            R("is_a", "external_load"),
            R("contextualized_by", "match_exposure"),
        ),
    ),
    OntologyConcept(
        "hsr",
        "physical_performance",
        ("HSR", "high-speed running"),
        "Distancia o tiempo de carrera a alta velocidad según una definición operacional.",
        (
            R("is_a", "external_load"),
            R("depends_on", "sprint_threshold"),
            R("contextualized_by", "match_exposure"),
            R("comparability_checked_by", "gps_comparability"),
        ),
    ),
    OntologyConcept(
        "sprint_threshold",
        "physical_performance",
        ("sprint", "speed threshold"),
        "Definición operacional del umbral de velocidad; no existe un valor universal único.",
        (
            R("comparability_checked_by", "gps_comparability"),
            R("may_be_private", "private_operational_definition"),
        ),
    ),
    OntologyConcept(
        "mechanical_events",
        "physical_performance",
        ("accelerations/decelerations", "aceleraciones/desaceleraciones"),
        "Cambios de velocidad que describen demanda mecánica externa.",
        (
            R("is_a", "external_load"),
            R("not_equivalent_to", "fatigue_state"),
        ),
        ("causal_risk",),
    ),
    OntologyConcept(
        "fatigue_state",
        "physical_performance",
        ("fatigue/freshness", "fatiga/frescura"),
        "Estado fisiológico o perceptual multidimensional.",
        (R("not_equivalent_to", "external_load"),),
        ("causal_risk",),
    ),
    OntologyConcept(
        "metres_per_minute",
        "physical_performance",
        ("metres per minute", "metros por minuto"),
        "Indicador relativo de locomoción por unidad de tiempo.",
        (
            R("is_a", "external_load"),
            R("contextualized_by", "position_role"),
        ),
    ),
    OntologyConcept(
        "position_role",
        "physical_performance",
        ("position/role", "posición/rol"),
        "Función posicional y táctica que condiciona demandas esperables.",
    ),
    OntologyConcept(
        "gps_comparability",
        "physical_performance",
        ("GPS comparability", "comparabilidad entre proveedores"),
        "La misma etiqueta de métrica puede diferir por dispositivo, algoritmo o definición.",
        (R("requires_definition_check", "sprint_threshold"),),
    ),
    OntologyConcept(
        "time_period",
        "data_context",
        ("period/session", "período/sesión"),
        "Unidad temporal necesaria para agrupar, calcular o graficar datos.",
    ),
    OntologyConcept(
        "injury_causality",
        "medical_performance",
        ("injury causality", "causalidad de lesión"),
        "Una asociación temporal o estadística no demuestra que una carga haya causado una lesión.",
        (
            R("related_to", "training_load_monitoring"),
            R("requires_alternatives", "fatigue_state"),
        ),
        ("causal_risk",),
    ),
    OntologyConcept(
        "return_to_play_readiness",
        "medical_performance",
        ("return to play readiness", "preparación para competir"),
        "Decisión multidimensional; igualar una sola métrica previa no demuestra preparación competitiva.",
        (
            R("not_equivalent_to", "total_distance"),
            R("contextualized_by", "physical_performance"),
        ),
        ("causal_risk",),
    ),
    OntologyConcept(
        "possession",
        "tactical_analysis",
        ("possession", "posesión"),
        "Proporción de control del balón; no equivale automáticamente a calidad global de juego.",
        (R("not_equivalent_to", "match_performance"),),
        ("causal_risk",),
    ),
    OntologyConcept(
        "match_performance",
        "tactical_analysis",
        ("match performance", "rendimiento de partido"),
        "Evaluación multidimensional del desempeño competitivo.",
    ),
    OntologyConcept(
        "high_press",
        "tactical_analysis",
        ("high press", "presión alta"),
        "Comportamiento colectivo de presión en zonas altas.",
        (R("not_proven_by", "ball_recovery"),),
    ),
    OntologyConcept(
        "mid_block",
        "tactical_analysis",
        ("mid block", "bloque medio"),
        "Organización defensiva intermedia dependiente de alturas, distancias, roles y disparadores.",
        (R("related_to", "pressing_action"),),
    ),
    OntologyConcept(
        "pressing_action",
        "tactical_analysis",
        ("pressing action", "acción de presión"),
        "Acción individual o colectiva de presión que debe interpretarse dentro de la organización del equipo.",
    ),
    OntologyConcept(
        "offensive_transition",
        "tactical_analysis",
        ("offensive transition", "transición ofensiva"),
        "Fase posterior a recuperar la posesión.",
        (R("not_equivalent_to", "counterattack"),),
    ),
    OntologyConcept(
        "counterattack",
        "tactical_analysis",
        ("counterattack", "contraataque"),
        "Ataque rápido que explota el desequilibrio rival; puede ocurrir dentro de una transición ofensiva.",
        (R("possible_form_of", "offensive_transition"),),
    ),
    OntologyConcept(
        "ball_recovery",
        "tactical_analysis",
        ("ball recovery", "recuperación de balón"),
        "Evento de recuperar la posesión; por sí solo no demuestra el comportamiento defensivo previo.",
        (R("not_equivalent_to", "high_press"),),
    ),
    OntologyConcept(
        "private_operational_definition",
        "sports",
        ("private operational definition", "definición operacional privada"),
        "Regla, umbral o protocolo propio del usuario/club que no debe convertirse en verdad global.",
        (),
        ("private_memory",),
    ),
)


def ontology_for(sport: SportContext) -> tuple[OntologyConcept, ...]:
    return FOOTBALL_CONCEPTS if sport is SportContext.FOOTBALL else ()
