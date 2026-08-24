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
        "gps_comparability", "physical_performance",
        ("provider", "proveedor", "device", "dispositivo", "GPS comparability"),
        "Métricas con el mismo nombre pueden no ser directamente comparables entre proveedores o definiciones.",
        ("Catapult y otro GPS reportan HSR", "mismo nombre de métrica en sistemas diferentes"),
        ("catapult", "proveedor", "provider", "dispositivo", "gps"),
    ),
)


def ontology_for(sport: SportContext) -> tuple[OntologyConcept, ...]:
    return FOOTBALL_CONCEPTS if sport is SportContext.FOOTBALL else ()
