# Orion

Agente personal de inteligencia deportiva con modelo local, control de recursos
y memoria privada por consentimiento.

## Estado actual: Módulo 1.2

Este módulo incorpora:

- interfaz inicial de chat;
- backend local en FastAPI;
- conexión con Ollama;
- modos Automático, Rápido y Profundo;
- respuesta progresiva: el texto aparece mientras Ollama lo genera;
- botón para detener una generación en curso;
- recomendación transparente del modo de respuesta;
- advertencia antes de una operación pesada cuando la PC está exigida;
- prioridad reducida continua para los procesos de Ollama;
- `qwen3:4b-instruct` para Rápido y `qwen3:8b` para Profundo;
- presupuesto configurable de 8 hilos físicos en ambos modos;
- contexto, historial y salida acotados por modo para evitar trabajo innecesario;
- servidor Python oculto, con verificación de arranque y registros de errores;
- conversación con desplazamiento independiente y panel de estado fijo;
- Markdown seguro y métricas de primer texto, carga, tiempo total, velocidad,
  tokens y pico de CPU por respuesta;
- reglas de prudencia científica y una batería reproducible de calidad deportiva;
- modelo activo en memoria durante diez minutos para acelerar preguntas seguidas;
- atajos de túnel desactivados en el iniciador local;
- cero memoria permanente y cero proveedores externos.

La memoria con Supabase se implementará en el Módulo 2. Hasta entonces, la
conversación existe únicamente en la pestaña abierta.

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
llega. Esto reduce mucho el tiempo percibido, aunque el tiempo total seguirá
dependiendo del modelo, la longitud de la respuesta y la carga del equipo.

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
reducir exposición accidental.
