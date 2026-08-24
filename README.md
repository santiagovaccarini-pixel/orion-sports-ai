# Orion

Agente personal de inteligencia deportiva con modelo local, control de recursos
y memoria privada por consentimiento.

## Estado actual: Módulo 1.3

Este módulo incorpora:

- interfaz inicial de chat;
- backend local en FastAPI;
- conexión con Ollama;
- modos Automático, Rápido y Profundo;
- respuesta progresiva: el texto aparece mientras Ollama lo genera;
- el mismo botón del cuadro de escritura permite enviar o detener una generación;
- selector de contexto local para General, Fútbol, Básquet, Vóley, Rugby,
  Tenis, Atletismo, Natación y Ciclismo;
- recomendación transparente del modo de respuesta;
- advertencia antes de una operación pesada cuando la PC está exigida;
- prioridad reducida continua para los procesos de Ollama;
- `qwen3:4b-instruct` para Rápido y `qwen3:8b` para Profundo;
- presupuesto configurable de 8 hilos físicos en ambos modos;
- contexto, historial y salida acotados por modo para evitar trabajo innecesario;
- servidor Python oculto, con verificación de arranque y registros de errores;
- conversación con desplazamiento independiente, lectura libre durante la
  generación y un control para volver al final;
- reconstrucción final en Markdown seguro, fórmulas legibles sin LaTeX crudo y
  métricas de primer texto, carga, tiempo total, velocidad,
  tokens y pico de CPU por respuesta;
- reglas de prudencia científica y una batería reproducible de calidad deportiva;
- modelo activo en memoria durante diez minutos para acelerar preguntas seguidas;
- atajos de túnel desactivados en el iniciador local;
- cero memoria permanente y cero proveedores externos.
- base de conocimiento local opcional para importar TXT, Markdown, CSV y JSON;
  los fragmentos relevantes se incorporan como fuentes identificables en cada respuesta.
- composición visual adaptativa: Orion elige entre texto, ejemplo, tabla o gráfico
  según la tarea, con diseño orientado a comprensión y evidencia.
- orquestador de intención: separa conocimiento general, datos locales, cálculo,
  gráficos y búsqueda web antes de construir la respuesta.

El deporte seleccionado especializa el vocabulario, las variables y los
ejemplos del modelo local, pero no reemplaza la pregunta central ni activa una
búsqueda en Internet. Si una consulta depende de información reciente, Orion
debe reconocer esa limitación. La búsqueda web controlada ya está disponible de
forma opcional: activala con `ORION_WEB_ENABLED=true`, mantené una allowlist en
`ORION_WEB_ALLOWED_DOMAINS` y exigí al menos `ORION_WEB_MINIMUM_SOURCES` fuentes
permitidas. El valor predeterminado es cuatro y la búsqueda permanece
desactivada en local.

Con una a tres fuentes, Orion muestra directamente los extractos y enlaces como
información preliminar, sin pedirle al modelo que invente una síntesis. Con cuatro
o más dominios independientes, el modelo puede sintetizar y citar el resultado.

La memoria con Supabase se implementará en el Módulo 2. Hasta entonces, la
conversación existe únicamente en la pestaña abierta.

La base de conocimiento local no es memoria conversacional: guarda únicamente
los documentos que importes de forma explícita en `.orion-runtime/knowledge`.
Desde la interfaz podés usar `Documento` para agregar archivos de texto, CSV,
Markdown o JSON. Orion solo recupera fragmentos relacionados con la pregunta y
los marca como `Fuente local`; si no encuentra coincidencias, no agrega contexto.

Antes de responder, el orquestador clasifica la consulta. Las preguntas actuales
priorizan web; las preguntas sobre archivos priorizan las herramientas locales;
las preguntas generales no reciben el CSV cargado por accidente. Los cálculos y
gráficos se ejecutan con código y el modelo se limita a explicarlos.

El currículo de fundamentos deportivos de diez preguntas está en
`backend/evals/sports_foundations_cases.json`. Se puede ejecutar con
`--dataset foundations`; los fallos muestran la corrección esperada para convertir
cada error en un caso de regresión.

## Arquitectura

```text
Interfaz React/Vinext <- NDJSON progresivo <- FastAPI <- Ollama local
                                                |
                                                +-> Monitor de CPU y RAM
```

Todo el núcleo escucha únicamente en `127.0.0.1`. No queda expuesto a otros
equipos de la red durante este módulo.

## Uso de recursos

Generar texto localmente requiere cálculo. Un aumento temporal de CPU mientras
Orion responde es normal; no significa por sí solo que exista un problema. El
modo Rápido usa un modelo de 4B, contexto de 4096 tokens y un máximo de 384
tokens de salida. Profundo reutiliza el modelo 8B ya instalado, con más contexto
y una salida mayor. Ambos usan ocho hilos físicos y los procesos de Ollama se
mantienen con prioridad reducida para que Windows y las demás aplicaciones
tengan preferencia.

La interfaz ya no espera la respuesta completa: muestra cada fragmento apenas
llega y renderiza Markdown progresivamente. Esto reduce mucho el tiempo
percibido, aunque el tiempo total seguirá dependiendo del modelo, la longitud
de la respuesta y la carga del equipo.

El modo Rápido permite hasta 768 tokens de salida y Profundo hasta 1536 por
defecto. Estos límites pueden ajustarse con `ORION_QUICK_MAX_TOKENS` y
`ORION_DEEP_MAX_TOKENS`. Markdown admite negrita, tablas, listas y emojis;
HTML y estilos de color generados por el modelo no se ejecutan por seguridad.

El presupuesto de hilos reduce el impacto, pero no constituye un límite rígido
de porcentaje: controladores, GPU integrada y tareas auxiliares también pueden
usar CPU. Puede ajustarse con `ORION_QUICK_THREADS` y `ORION_DEEP_THREADS`.

## Preparación en Windows

Desde PowerShell, ubicado en la carpeta del proyecto:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Setup-Orion.ps1 -InstallOllama -DownloadQuickModel -DownloadDeepModel
```

El script crea un entorno Python aislado, instala las dependencias de Orion,
prepara la interfaz y, solamente cuando se incluyen esos parámetros, instala
Ollama y descarga `qwen3:4b-instruct` para Rápido y `qwen3:8b` para Profundo.

Si `qwen3:8b` ya está instalado, alcanza con ejecutar:

```powershell
.\scripts\windows\Setup-Orion.ps1 -DownloadQuickModel
```

Para iniciar Orion:

```powershell
.\scripts\windows\Start-Orion.ps1
```

El núcleo Python se ejecuta oculto y sus registros quedan en
`.orion-runtime`. La terminal desde la que se inicia Orion sigue siendo el
controlador temporal del prototipo: al detenerla, el núcleo local se cierra de
forma ordenada. El iniciador también desactiva los atajos interactivos que
podrían abrir un túnel de Cloudflare al pegar texto accidentalmente.

## Dirección del producto

El iniciador actual es transitorio. El objetivo de la siguiente etapa es
empaquetar Orion como una aplicación liviana para Windows, con acceso directo e
icono propio en la barra de tareas, sin necesitar VS Code. La aplicación deberá
iniciar y detener de forma segura el núcleo local y mantener el mismo control
de privacidad y recursos.

## Pruebas

Backend:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests
```

Interfaz:

```powershell
npm run test:windows
```

Evaluación deportiva local, con Orion iniciado:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_local_evaluation --limit 3
```

La batería completa contiene ocho casos y se ejecuta con `--limit 8`. El
prechequeo por conceptos no reemplaza la revisión humana de las respuestas.

Medición reproducible de rendimiento, con Orion iniciado:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_performance_benchmark --mode quick --runs 2
```

La primera ejecución incluye la carga del modelo; la segunda representa una
consulta con el modelo ya caliente. La interfaz muestra las mismas métricas por
respuesta para poder comparar cambios sin depender de una impresión subjetiva.

## Configuración

Los valores disponibles están documentados en `backend/.env.example`. En el
prototipo se mantienen los puertos y direcciones locales predeterminados para
reducir exposición accidental. Para un despliegue remoto, definí `ORION_API_KEY`
en el backend y la misma clave en `NEXT_PUBLIC_ORION_API_KEY` para la interfaz.
La clave protege el estado y el chat; `/api/v1/health` queda disponible para
comprobaciones de disponibilidad.
