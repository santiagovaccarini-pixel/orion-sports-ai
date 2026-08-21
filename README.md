# Orion

Agente personal de inteligencia deportiva con modelo local, control de recursos
y memoria privada por consentimiento.

## Estado actual: Módulo 1

Este módulo incorpora:

- interfaz inicial de chat;
- backend local en FastAPI;
- conexión con Ollama;
- modos Automático, Rápido y Profundo;
- recomendación transparente del modo de respuesta;
- advertencia antes de una operación pesada cuando la PC está exigida;
- prioridad reducida del proceso y descarga del modelo tras dos minutos;
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
npm test
```

## Configuración

Los valores disponibles están documentados en `backend/.env.example`. En el
prototipo se mantienen los puertos y direcciones locales predeterminados para
reducir exposición accidental.
