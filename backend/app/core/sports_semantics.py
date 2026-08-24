from __future__ import annotations

from backend.app.domain.schemas import SportContext


GENERAL_SEMANTIC_GUIDE = """
Interpretá la intención antes de buscar términos. Diferenciá el dato literal que pide
el usuario del objetivo que intenta resolver. No conviertas asociación en causalidad,
ni una métrica aislada en una conclusión de rendimiento. Si una palabra puede tener
varios significados técnicos, identificá el dominio y las variables que faltan.
""".strip()


SPORT_SEMANTIC_GUIDES: dict[SportContext, str] = {
    SportContext.GENERAL: GENERAL_SEMANTIC_GUIDE,
    SportContext.FOOTBALL: """
En fútbol separá siempre el rendimiento físico, técnico, táctico, médico y contextual.
Distancia total, HSR, sprint, aceleraciones y desaceleraciones son medidas de demanda
externa; por sí solas no equivalen a jugar mejor, estar más en forma ni estar más
fatigado. Para compararlas considerá minutos/exposición, posición, rol, fase del juego,
marcador, rival, modelo de juego, definición operacional y proveedor/dispositivo.
RPE y frecuencia cardíaca pertenecen a la respuesta interna y tampoco deben
interpretarse sin contexto individual. Una transición ofensiva no es sinónimo de
contraataque; una recuperación no garantiza progresión; presión alta, bloque alto y
línea defensiva alta son conceptos relacionados pero no equivalentes. En análisis de
partido distinguí qué ocurrió, cuánto ocurrió y por qué pudo haber ocurrido. Si el
usuario pregunta por una causa, buscá primero explicaciones alternativas y variables
confusoras. Términos frecuentes pueden aparecer en español, portugués o inglés:
extremo/ponta/winger, volante/meio-campista/midfielder, HSR/high-speed running,
sprint, carga externa/external load, carga interna/internal load.
""".strip(),
    SportContext.BASKETBALL: """
En básquet separá producción estadística, eficiencia, rol, posesiones, minutos,
demanda física y contexto táctico. Más volumen no implica automáticamente mejor
rendimiento. Considerá ritmo de juego, posición, quintetos, rival y exposición.
""".strip(),
    SportContext.VOLLEYBALL: """
En vóley interpretá acciones según posición, rotación, set y exposición. Cantidad de
saltos, altura, aterrizajes y carga no equivalen por sí solos a rendimiento técnico.
""".strip(),
    SportContext.RUGBY: """
En rugby diferenciá carrera, contactos, colisiones, rol posicional y exposición. Las
demandas varían mucho por posición, fase y código; una métrica aislada no describe el
rendimiento completo.
""".strip(),
    SportContext.TENNIS: """
En tenis diferenciá resultado, calidad de golpe, carga, duración y contexto. Superficie,
rival, marcador, servicio, rallies y fatiga modifican la interpretación de métricas.
""".strip(),
    SportContext.ATHLETICS: """
En atletismo identificá primero la disciplina y la variable objetivo. No mezcles
rendimiento, carga y técnica; viento, superficie, parciales y fase de temporada pueden
cambiar materialmente la interpretación.
""".strip(),
    SportContext.SWIMMING: """
En natación identificá estilo, distancia, longitud de pileta y fase de la prueba.
Tiempo, frecuencia y longitud de brazada necesitan contexto antes de inferir técnica o
fatiga.
""".strip(),
    SportContext.CYCLING: """
En ciclismo diferenciá potencia absoluta, W/kg, carga, intensidad y rendimiento.
Duración, desnivel, viento, terreno, disciplina y estado competitivo condicionan la
interpretación.
""".strip(),
}


def semantic_guide(sport: SportContext) -> str:
    return SPORT_SEMANTIC_GUIDES.get(sport, GENERAL_SEMANTIC_GUIDE)
