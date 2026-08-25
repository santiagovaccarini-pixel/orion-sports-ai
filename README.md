# Orion

Agente personal de inteligencia deportiva con modelos intercambiables, control de recursos y memoria privada por consentimiento.

## Estado actual: prototipo cloud

Orion conserva el modo local con Ollama y suma un prototipo cloud con FastAPI en Render y Cloudflare Workers AI. La interfaz React/Vinext puede conectarse a cualquiera de los dos núcleos sin cambiar su contrato de chat.

El prototipo incluye:

- interfaz inicial de chat;
- backend FastAPI local o cloud;
- proveedor intercambiable entre Ollama y Cloudflare Workers AI;
- modos Automático, Rápido y Profundo;
- respuesta progresiva por NDJSON;
- botón unificado para enviar o detener;
- selector de contexto para General, Fútbol, Básquet, Vóley, Rugby, Tenis, Atletismo, Natación y Ciclismo;
- protección de CPU/RAM y prioridad reducida únicamente cuando la inferencia es local;
- métricas de respuesta y evaluación reproducible;
- base de conocimiento local opcional para TXT, Markdown, CSV y JSON;
- orquestador que separa conocimiento general, datos locales, cálculo, gráficos y búsqueda web;
- investigación web controlada para consultas que dependen de información actual;
- extracción web orientada al fragmento relevante de la página, evitando usar solamente encabezados o navegación;
- despliegue cloud de prototipo configurado explícitamente en plan gratuito.

## Investigación web

Las preguntas que dependen de información cambiante —por ejemplo goles actuales, resultados, tablas o alineaciones— deben consultar fuentes actuales en lugar de confiar en el conocimiento interno del modelo.

En el prototipo cloud la búsqueda web está habilitada y exige al menos tres dominios independientes antes de presentar una investigación como verificada. Si no alcanza ese umbral, Orion debe mostrar la evidencia disponible como preliminar y evitar completar el dato con una suposición.

La búsqueda usa una allowlist de fuentes y prioriza dominios futbolísticos para consultas estadísticas. Los extractos se seleccionan alrededor de los términos relevantes y números de la consulta, porque muchas páginas colocan menús, cookies y navegación antes del dato real.

El enrutamiento determinístico de consultas actuales funciona como una red de seguridad para decidir cuándo buscar; no debe utilizarse como mecanismo para generar la respuesta. La evolución prevista es un intérprete estructurado de intención, entidades, período y alcance antes de seleccionar herramientas.

## Arquitectura

```text
Interfaz React/Vinext
        |
        +--> Orion local (FastAPI -> Ollama)
        |
        +--> Orion Cloud (FastAPI/Render -> Cloudflare Workers AI)
```

El modo cloud validado actualmente usa `@cf/openai/gpt-oss-20b`. El modelo no se considera definitivo: debe compararse contra alternativas mediante el mismo benchmark de Orion.

## Inicio en Windows

Preparación local inicial:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Setup-Orion.ps1 -InstallOllama -DownloadQuickModel -DownloadDeepModel
```

Modo local:

```powershell
.\scripts\windows\Start-Orion.ps1
```

Modo Cloud de prueba local:

```powershell
$env:ORION_API_KEY="TU_CLAVE_LOCAL"
.\scripts\windows\Start-Orion.ps1 -Cloud
```

En modo Cloud el iniciador no arranca el backend Python local ni Ollama. La interfaz apunta a `https://orion-core-prototype.onrender.com/api/v1` y usa la clave disponible únicamente en esa sesión de PowerShell.

Este modo con `NEXT_PUBLIC_ORION_API_KEY` es exclusivamente para pruebas locales. No debe publicarse una compilación web con esa variable porque los valores `NEXT_PUBLIC_*` quedan accesibles en el navegador. Antes de publicar la interfaz se implementará autenticación server-side.

## Persistencia

La memoria permanente continúa desactivada. El filesystem del hosting gratuito no debe utilizarse como memoria porque es efímero. La memoria v1 tendrá almacenamiento separado y gobernado cuando se implemente su etapa.

La base de conocimiento local no es memoria conversacional: guarda únicamente documentos importados explícitamente y recupera fragmentos relacionados con la consulta.

## Pruebas

Backend:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests
```

Interfaz:

```powershell
npm run test:windows
```

Smoke test cloud:

```powershell
$env:ORION_CLOUD_URL="https://orion-core-prototype.onrender.com"
$env:ORION_API_KEY="TU_CLAVE_LOCAL"
python backend\scripts\smoke_cloud.py
```

Evaluación deportiva local:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_local_evaluation --limit 3
```

Benchmark de rendimiento local:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_performance_benchmark --mode quick --runs 2
```

## Seguridad del prototipo

- Las credenciales de Cloudflare y Orion se configuran como secretos del hosting y no se versionan.
- `main` permanece estable mientras `cloud-prototype` continúa como rama de validación.
- Las operaciones pesadas locales respetan la política de recursos; las consultas cloud no quedan bloqueadas por la CPU de la notebook.
- La interfaz pública no se desplegará con una API key embebida.
- La memoria y el acceso local de Mini Orion tendrán permisos y almacenamiento separados en etapas posteriores.
