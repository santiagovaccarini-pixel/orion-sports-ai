# Orion

Agente personal de inteligencia deportiva con modelo local, control de recursos
y memoria privada por consentimiento.

## Estado actual: Módulo 1.1

Este módulo incorpora:

- interfaz inicial de chat;
- backend local en FastAPI;
- conexión con Ollama;
- modos Automático, Rápido y Profundo;
- recomendación transparente del modo de respuesta;
- advertencia antes de una operación pesada cuando la PC está exigida;
- prioridad reducida continua para los procesos de Ollama;
- presupuesto configurable de 6 hilos en Rápido y 8 en Profundo;
- conversación con desplazamiento independiente y panel de estado fijo;
- Markdown seguro y métricas de tiempo, velocidad y pico de CPU por respuesta;
- reglas de prudencia científica y una batería reproducible de calidad deportiva;
- descarga del modelo tras dos minutos de inactividad;
- cero memoria permanente y cero proveedores externos.

La memoria con Supabase se implementará en el Módulo 2. Hasta entonces, la
conversación existe únicamente en la pestaña abierta.

## Arquitectura

```text
Interfaz React/Vinext -> Núcleo local FastAPI -> Ollama local
                              |
                              +-> Monitor de CPU y RAM
```

Todo el núcleo escucha únicamente en `127.0.0.1`. No queda expuesto a otros
equipos de la red durante este módulo.

## Uso de recursos

Generar texto localmente requiere cálculo. Un aumento temporal de CPU mientras
Orion responde es normal; no significa por sí solo que exista un problema. El
modo Rápido usa por defecto seis hilos y los procesos de Ollama se mantienen con
prioridad reducida para que Windows y las demás aplicaciones tengan preferencia.

El presupuesto de hilos reduce el impacto, pero no constituye un límite rígido
de porcentaje: controladores, GPU integrada y tareas auxiliares también pueden
usar CPU. Puede ajustarse con `ORION_QUICK_THREADS` y `ORION_DEEP_THREADS`.

## Preparación en Windows

Desde PowerShell, ubicado en la carpeta del proyecto:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Setup-Orion.ps1 -InstallOllama -DownloadQuickModel
```

El script crea un entorno Python aislado, instala las dependencias de Orion,
prepara la interfaz y, solamente cuando se incluyen esos parámetros, instala
Ollama y descarga `qwen3:8b`.

Para iniciar Orion:

```powershell
.\scripts\windows\Start-Orion.ps1
```

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

## Configuración

Los valores disponibles están documentados en `backend/.env.example`. En el
prototipo se mantienen los puertos y direcciones locales predeterminados para
reducir exposición accidental.
