# Orion

Agente personal de inteligencia deportiva con modelos intercambiables, razonamiento semántico, herramientas determinísticas y observabilidad de desarrollo.

## Estado actual: prototipo cloud en estabilización

Orion conserva el modo local con Ollama y mantiene un prototipo cloud con FastAPI en Render y Cloudflare Workers AI. `main` continúa estable; el desarrollo cloud vive en `cloud-prototype` y el PR permanece en borrador hasta completar validación end-to-end y generalización.

El prototipo incluye:

- interfaz de chat con respuesta progresiva;
- backend FastAPI local o cloud;
- proveedor intercambiable entre Ollama y Cloudflare Workers AI;
- modos Automático, Rápido y Profundo;
- gpt-oss mediante Responses API con `reasoning effort` observable;
- presupuestos de salida independientes entre Ollama y Cloudflare;
- respuesta progresiva por NDJSON;
- botón unificado para enviar o detener;
- selector de contexto para General, Fútbol, Básquet, Vóley, Rugby, Tenis, Atletismo, Natación y Ciclismo;
- protección de CPU/RAM y prioridad reducida únicamente cuando la inferencia es local;
- base de conocimiento local opcional para TXT, Markdown, CSV y JSON;
- orquestación semántica que interpreta objetivo, entidades, restricciones, evidencia necesaria y herramientas;
- política de evidencia diferenciada entre conocimiento estable, información externa actual y datos locales;
- retrieval local por fragmentos relevantes en lugar de recortar siempre el comienzo del documento;
- calculadora segura y operaciones CSV estructuradas validadas antes de ejecutarse;
- gráficos locales construidos únicamente a partir de datos verificados;
- búsqueda web primaria con Tavily y fallback cuando `provider=auto`;
- bloqueo de búsquedas repetidas;
- lectura directa y acotada de páginas web seleccionadas por el reviewer (`web_read`);
- diagnóstico end-to-end en RAM con planificación, búsquedas, revisiones, herramientas, `finish_reason`, tokens, reasoning tokens, effort y tiempos;
- evaluación local y runner end-to-end contra el deployment cloud real;
- despliegue de prototipo configurado explícitamente en plan gratuito.

## Motor cloud actual

El benchmark cloud actual utiliza `@cf/openai/gpt-oss-120b` tanto para Quick como para Deep mientras se estabiliza la arquitectura. El modelo no se considera definitivo.

La integración cloud está diseñada específicamente para un modelo razonador:

- gpt-oss utiliza la Responses API;
- Quick usa `reasoning effort=low` por defecto;
- Deep usa `reasoning effort=medium` por defecto;
- las etapas internas pueden ajustar el esfuerzo según su función;
- se registran `finish_reason`, output tokens y reasoning tokens;
- una generación incompleta no debe presentarse silenciosamente como completa;
- el historial enviado al proveedor está acotado;
- el sampling de gpt-oss conserva `temperature=1.0` y `top_p=1.0` o sus defaults equivalentes.

Los presupuestos cloud del prototipo son independientes de Ollama:

```text
Quick cloud: 1536 output tokens
Deep cloud:  3072 output tokens
Quick local:  768 output tokens
Deep local:  1536 output tokens
```

## Razonamiento semántico

El camino cloud activo no debe interpretar preguntas mediante una lista de palabras clave. El ciclo esperado es:

```text
conversación
  -> plan semántico
  -> herramientas necesarias
  -> evidencia
  -> revisión
  -> respuesta
```

El planner puede distinguir entre:

- conocimiento estable que el modelo puede explicar sin exigir evidencia externa;
- hechos actuales o verificables que requieren web;
- datos privados/locales que requieren documentos;
- cálculos o gráficos que deben ser ejecutados de manera determinística.

La ruta legacy con heurísticas se conserva únicamente como rollback/local compatibility y no representa el diseño objetivo de Orion.

## Investigación web

Para información actual Orion utiliza un ciclo acotado:

```text
Search
  -> Review snippets
  -> Read fuentes relevantes
  -> Review contenido leído
  -> Search adicional solo si todavía falta evidencia
```

Tavily actúa principalmente como descubridor de páginas. Cuando un resultado parece relevante pero el snippet no alcanza, el reviewer puede marcar esa fuente y `web_read` abre directamente la URL.

El lector directo:

- admite solo HTTP/HTTPS;
- rechaza localhost, loopback, link-local, IP privadas/reservadas y credenciales embebidas;
- valida cada redirección;
- limita tamaño, cantidad de páginas, concurrencia y tiempo;
- elimina navegación/scripts y conserva contenido visible relevante;
- intenta detectar fecha publicada/actualizada;
- usa una caché temporal en RAM;
- nunca convierte automáticamente una página en evidencia válida: el reviewer vuelve a evaluarla.

El prototipo permite hasta tres búsquedas distintas, pero no obliga a realizar tres. Las queries duplicadas se bloquean y la lectura directa se intenta antes de consumir otra búsqueda cuando ya existen fuentes relevantes.

## Diagnóstico

La caja negra de desarrollo registra estados observables del pipeline, no cadena de pensamiento privada.

Puede incluir:

- pregunta y modo solicitado;
- modelo activo;
- plan estructurado;
- política de evidencia;
- queries exactas;
- fuentes recuperadas;
- fuentes aceptadas/descartadas;
- lecturas directas realizadas;
- resultados de herramientas locales;
- revisiones semánticas;
- llamadas al modelo por etapa;
- endpoint y reasoning effort;
- input/output/reasoning tokens;
- `finish_reason`;
- tiempos por etapa;
- respuesta final o error.

Las trazas viven únicamente en RAM del proceso y se pierden al reiniciar Render. No almacenan credenciales ni cadena de pensamiento oculta.

## Arquitectura

```text
Interfaz React/Vinext
        |
        +--> Orion local (FastAPI -> Ollama)
        |
        +--> Orion Cloud
               |
               +--> FastAPI / Render
               +--> Cloudflare Workers AI / gpt-oss-120b
               +--> Tavily Search
               +--> Web Read seguro
               +--> herramientas determinísticas
```

## Identidad de Orion

Orion tiene una identidad institucional explícita separada de la lógica de razonamiento deportivo. Santiago Vaccarini es el creador de Orion. Esa atribución no implica autoría sobre gpt-oss, Cloudflare Workers AI, Ollama ni otros motores o servicios externos utilizados por Orion.

El perfil profesional público del creador se mantiene centralizado y debe modificarse únicamente con información validada por su creador.

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

En modo Cloud el iniciador no arranca el backend Python local ni Ollama. La interfaz apunta al Orion Core desplegado y usa la clave disponible únicamente en esa sesión de PowerShell.

`NEXT_PUBLIC_ORION_API_KEY` es exclusivamente una solución temporal para pruebas locales. No debe publicarse una compilación web con esa variable porque los valores `NEXT_PUBLIC_*` quedan accesibles en el navegador. Antes de publicar la interfaz se implementará autenticación server-side.

## Persistencia

La memoria permanente continúa desactivada. El filesystem del hosting gratuito no debe utilizarse como memoria porque es efímero.

La base de conocimiento local no es memoria conversacional: guarda únicamente documentos importados explícitamente. El retrieval semántico selecciona fragmentos relevantes y las herramientas CSV validan columnas, filtros y operaciones antes de ejecutar cálculos.

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

Evaluación end-to-end contra Orion Cloud:

```powershell
$env:ORION_CLOUD_URL="https://orion-core-prototype.onrender.com"
$env:ORION_API_KEY="TU_CLAVE_LOCAL"
.\.venv\Scripts\python.exe -m backend.evals.run_cloud_evaluation --limit-per-dataset 3
```

El runner cloud verifica que `/status` reporte `cloudflare` antes de ejecutar casos y guarda un reporte JSON con latencia, respuesta, prechequeos y diagnóstico disponible. No debe ejecutarse automáticamente en cada commit porque consume la cuota gratuita de Workers AI/Tavily.

Benchmark de rendimiento local:

```powershell
.\.venv\Scripts\python.exe -m backend.evals.run_performance_benchmark --mode quick --runs 2
```

## Seguridad del prototipo

- Las credenciales de Cloudflare, Tavily y Orion se configuran como secretos del hosting y no se versionan.
- `main` permanece estable mientras `cloud-prototype` continúa como rama de validación.
- Las operaciones pesadas locales respetan la política de recursos; las consultas cloud no quedan bloqueadas por la CPU de la notebook.
- `web_read` restringe destinos y redirecciones para evitar acceso a redes privadas o locales.
- La interfaz pública no se desplegará con una API key embebida.
- La memoria y el acceso local de Mini Orion tendrán permisos y almacenamiento separados en etapas posteriores.
