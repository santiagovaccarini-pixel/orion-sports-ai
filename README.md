# Orion

Agente personal de inteligencia deportiva con modelo local, control de recursos
y memoria privada por consentimiento.

## Estado actual: Módulo 1.4 — Semantic Intelligence

Este módulo incorpora:

- interfaz inicial de chat;
- backend local en FastAPI;
- conexión con Ollama;
- modos Automático, Rápido y Profundo;
- respuesta progresiva: el texto aparece mientras Ollama lo genera;
- el mismo botón del cuadro de escritura permite enviar o detener una generación;
- selector de contexto local para General, Fútbol, Básquet, Vóley, Rugby,
  Tenis, Atletismo, Natación y Ciclismo;
- `qwen3:4b-instruct` para Rápido y `qwen3:8b` para Profundo;
- presupuesto configurable de 8 hilos físicos en ambos modos;
- contexto, historial y salida acotados por modo para evitar trabajo innecesario;
- servidor Python oculto, con verificación de arranque y registros de errores;
- conversación con desplazamiento independiente, lectura libre durante la
  generación y un control para volver al final;
- reconstrucción final en Markdown seguro y métricas de rendimiento por respuesta;
- reglas de prudencia científica y baterías reproducibles de calidad deportiva;
- modelo activo en memoria durante diez minutos para acelerar preguntas seguidas;
- base de conocimiento local opcional para importar TXT, Markdown, CSV y JSON;
- composición visual adaptativa para texto, ejemplo, tabla o gráfico;
- búsqueda web controlada por allowlist y mínimo de fuentes;
- **planificador semántico previo a la respuesta**: separa pedido literal, objetivo
  real, dominio, tipo de tarea, conceptos, variables faltantes, ambigüedad,
  complejidad y riesgo causal;
- **routing por intención**: web, datos locales, cálculo, gráficos y conocimiento
  general se eligen después de interpretar la pregunta, no sólo por palabras clave;
- **retrieval expandido por intención**: Orion reformula la necesidad del usuario en
  conceptos técnicos y consultas canónicas antes de buscar fragmentos relevantes;
- **contexto conversacional para el planner**: referencias como "eso", "lo mismo" o
  "como antes" pueden resolverse usando los turnos recientes;
- guía semántica deportiva con foco inicial en fútbol para separar rendimiento físico,
  técnico, táctico y contextual y evitar inferencias inválidas desde una métrica aislada;
- thinking de Qwen3 opcional en modo Profundo mediante `ORION_DEEP_THINKING_ENABLED`;
- fallbacks deterministas: si el planner semántico falla, Orion conserva el routing
  anterior en lugar de bloquear el chat;
- CI de backend en GitHub Actions para ejecutar la suite de regresión en cada PR/push.

### Qué cambió conceptualmente en 1.4

Antes, gran parte del routing se resolvía buscando marcadores como `promedio`,
`comparar`, `gráfico` o términos presentes en los documentos. Eso sigue existiendo
como fallback, pero ya no es la vía principal cuando el planner está disponible.

El flujo actual es:

```text
Mensaje + conversación reciente
        |
        v
Semantic Planner (Structured Output)
        |
        +-> objetivo real
        +-> dominio
        +-> conceptos
        +-> variables faltantes
        +-> ambigüedad / complejidad
        +-> riesgo causal
        +-> fuentes necesarias
        |
        v
Orquestación por intención
        |
        v
Retrieval expandido + herramientas
        |
        v
Context Builder
        |
        v
Qwen3 Rápido / Profundo
        |
        v
Respuesta
```

El planner utiliza Structured Outputs de Ollama con temperatura 0 y un timeout
corto. No responde al usuario: produce un objeto validado que guía las etapas
posteriores. Si Ollama no puede completar ese pre-pass, Orion vuelve al plan
determinista anterior.

La guía semántica de fútbol incluye relaciones que ayudan a evitar errores comunes:
HSR, sprint, distancia, aceleraciones y desaceleraciones describen demanda externa y
no equivalen por sí solas a rendimiento; RPE y frecuencia cardíaca representan
respuestas internas diferentes; exposición, posición, rol, marcador, rival, modelo de
juego y definición operacional pueden cambiar una comparación. También evita tratar
como sinónimos conceptos tácticos relacionados, por ejemplo transición ofensiva y
contraataque, o presión alta y línea defensiva alta.

## Conocimiento y memoria

La memoria persistente todavía no está implementada. La conversación existe únicamente
en la sesión actual y la base de conocimiento local contiene sólo los documentos que
se importan de forma explícita.

La arquitectura de intención ya prepara la separación futura entre:

1. **memoria/conocimiento privado por usuario o club**, con protocolos, definiciones,
   archivos, decisiones y metodologías propias;
2. **Orion Sports Core**, conocimiento deportivo global validado y versionado que podrá
   compartirse entre instalaciones sin exponer datos privados de un club.

`needs_private_memory` y `needs_global_knowledge` ya forman parte del plan semántico,
pero este módulo no persiste ni comparte memoria todavía.

## Base de conocimiento local

La base local guarda documentos en `.orion-runtime/knowledge`. Desde la interfaz se
pueden agregar TXT, Markdown, CSV o JSON. Los fragmentos se incorporan como fuentes
identificables.

En 1.4 la recuperación no depende exclusivamente del texto literal de la pregunta. El
planner genera reformulaciones y conceptos canónicos; `semantic_retriever.py` combina
esas señales con el índice local. La próxima evolución prevista es sumar embeddings
persistentes y reranking manteniendo el mismo contrato del retriever.

Las herramientas de CSV (`csv_overview`, `csv_calculation`, `csv_chart`,
`csv_tool_result`) ya no asumen la planilla de seguimiento GPS. Detectan la columna
identificadora, las columnas numéricas y la columna de período de cualquier planilla
tabular, primero por nombre de columna reconocible y, si no encuentran ninguno, por
estructura. Cuando una planilla tiene varias columnas numéricas y la consulta no nombra
cuál usar, Orion pide una aclaración en lugar de elegir una al azar.

## Web

La búsqueda web controlada se activa con `ORION_WEB_ENABLED=true`, utiliza la allowlist
`ORION_WEB_ALLOWED_DOMAINS` y exige al menos `ORION_WEB_MINIMUM_SOURCES` fuentes para
una síntesis confirmada. Con menos respaldo, Orion devuelve la evidencia como
preliminar y evita convertirla en un hecho confirmado.

El planner semántico puede detectar que una pregunta requiere información actual, pero
la búsqueda sigue estando gobernada por la configuración y las reglas de fuentes.

## Arquitectura

```text
Interfaz React/Vinext <- NDJSON progresivo <- FastAPI
                                                |
                                                +-> Semantic Planner
                                                +-> Intent Router
                                                +-> Knowledge Retriever
                                                +-> Web Research
                                                +-> CSV tools
                                                +-> Ollama local
                                                +-> Monitor de CPU y RAM
```

Todo el núcleo escucha únicamente en `127.0.0.1` durante el prototipo local.

## Uso de recursos

Generar texto localmente requiere cálculo. Un aumento temporal de CPU mientras Orion
responde es normal. Rápido usa el modelo 4B y Profundo el 8B; ambos tienen presupuestos
de contexto, salida e hilos independientes.

El planificador semántico agrega una inferencia corta antes de la respuesta y utiliza
el modelo Rápido, Structured Outputs, temperatura 0 y un máximo configurable mediante
`ORION_SEMANTIC_PLANNER_MAX_TOKENS`. Si el planner no responde dentro del timeout corto,
se usa el fallback determinista.

El modo Profundo puede activar el canal de thinking de Qwen3. Se controla con:

```text
ORION_DEEP_THINKING_ENABLED=true
```

Si la prioridad es latencia en una máquina local, puede desactivarse sin perder la capa
semántica previa.

## Preparación en Windows

Desde PowerShell, ubicado en la carpeta del proyecto:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Setup-Orion.ps1 -InstallOllama -DownloadQuickModel -DownloadDeepModel
```

El script crea un entorno Python aislado, instala las dependencias, prepara la interfaz
y descarga modelos sólo cuando se solicitan esos parámetros.

Para iniciar Orion:

```powershell
.\scripts\windows\Start-Orion.ps1
```

El núcleo Python se ejecuta oculto y los registros quedan en `.orion-runtime`.

## Pruebas

Backend:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```

El módulo 1.4 incorpora además `.github/workflows/backend-tests.yml`, que ejecuta esta
suite automáticamente en GitHub Actions.

Interfaz:

```powershell
npm run test:windows
```

Evaluación deportiva local, con Orion iniciado:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_local_evaluation --limit 3
```

Medición reproducible de rendimiento:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_performance_benchmark --mode quick --runs 2
```

Evaluación del planificador semántico, con Orion y Ollama iniciados (usa el modelo real,
no el fallback determinista, y cada caso puede tardar hasta el timeout del planner si
Ollama no responde):

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_semantic_intent_evaluation --dataset core
.\.venv\Scripts\python.exe -m backend.evals.run_semantic_intent_evaluation --dataset generalization
```

`core` cubre los tipos de intención fundamentales (incluye continuidad conversacional
encadenada y cambio de deporte a mitad de conversación); `generalization` evalúa robustez
frente a paráfrasis, consultas largas, inglés/portugués y jerga coloquial. Es esperable
que algunos casos de `generalization` sólo pasen con el modelo real: documentan el techo
que el fallback determinista todavía no alcanza por diseño.

Diagnóstico de Ollama (versión y si el modelo corre en CPU, GPU o mixto):

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_ollama_diagnostics
```

Para validar 1.4 no alcanza con medir la respuesta final. También deben evaluarse:

- precisión de intención;
- dominio detectado;
- conceptos recuperados;
- variables faltantes;
- detección de causalidad;
- aclaraciones innecesarias vs. necesarias;
- calidad del retrieval;
- latencia adicional del planner;
- diferencia de calidad y costo entre thinking activado/desactivado.

## Configuración

Los valores disponibles están documentados en `backend/.env.example`. Para un
despliegue remoto, definí `ORION_API_KEY` en el backend y la misma clave en
`NEXT_PUBLIC_ORION_API_KEY` para la interfaz. `/api/v1/health` queda disponible para
comprobaciones de disponibilidad.
