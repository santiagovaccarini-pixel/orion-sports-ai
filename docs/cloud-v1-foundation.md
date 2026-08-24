# Orion Cloud v1 — Fundación

Fecha de decisión: 2026-08-24

## Objetivo

Separar el cerebro de Orion del hardware local sin romper el Orion actual. El backend debe poder usar un proveedor de modelos remoto y, más adelante, comunicarse con un agente local de Windows.

## Restricción de costo del prototipo

Durante la fase de prototipo, toda dependencia estructural debe tener un nivel gratuito no basado en créditos promocionales ni pruebas con vencimiento. Si una cuota gratuita se agota, Orion debe fallar de forma controlada o usar un proveedor alternativo; nunca debe generar cargos automáticamente.

No se aceptan como dependencia obligatoria del prototipo:

- pruebas gratuitas con fecha de vencimiento;
- GPUs alquiladas 24/7;
- servicios que obliguen a cargar una tarjeta para que Orion funcione;
- almacenamiento cuya versión gratuita expire y elimine los datos;
- una integración que impida cambiar de proveedor más adelante.

## Arquitectura objetivo de esta etapa

```text
Interfaz Orion
      |
      v
FastAPI / Orion Cloud Core
      |
      +--> Model Provider
      |       +--> Cloud provider
      |       +--> Ollama local (fallback / privacidad)
      |
      +--> conversación y razonamiento (siguiente etapa)
      |
      +--> memoria (etapa posterior)
      |
      +--> Mini Orion Windows (etapa posterior)
```

## Decisiones actuales

### Backend

El código de negocio se mantiene en FastAPI y Python. No se lo acopla a un proveedor de hosting concreto.

Para el primer despliegue se prioriza un hosting serverless/gratuito compatible con FastAPI. Vercel Hobby es apto para el prototipo personal, pero no se considera una dependencia comercial permanente: el backend debe seguir siendo portable.

### Cerebro cloud

Primera opción a evaluar: Cloudflare Workers AI mediante API compatible con OpenAI.

Motivos:

- asignación gratuita diaria real, no crédito promocional;
- modelos alojados fuera de la PC;
- streaming;
- modelos con razonamiento y function calling;
- API estándar que facilita cambiar de proveedor.

Candidatos iniciales, sujetos al benchmark de Orion:

- Rápido: `@cf/zai-org/glm-4.7-flash`
- Profundo: `@cf/nvidia/nemotron-3-120b-a12b`

No se declara ganador ningún modelo hasta ejecutar las evaluaciones de Orion.

### Memoria

No se migra todavía. La memoria debe diseñarse después de estabilizar el núcleo cloud y la conversación.

Supabase Free sigue siendo candidato, pero su pausa automática por inactividad impide tratarlo como una decisión cerrada. También se evaluarán alternativas gratuitas persistentes antes de fijar la arquitectura de memoria.

## Reglas de seguridad

1. Nunca subir claves al repositorio.
2. Mantener Ollama disponible como fallback local.
3. El proveedor cloud se habilita únicamente por configuración.
4. Si faltan credenciales cloud, Orion debe seguir pudiendo funcionar localmente.
5. Las operaciones locales futuras se ejecutarán en Mini Orion, no en el servidor cloud.
6. Datos sensibles no se enviarán a un proveedor externo sin la política de consentimiento correspondiente.

## Hitos

### C0 — Baseline

Rama protegida lógica: `baseline/local-v0-2026-08-24`.

### C1 — Provider abstraction

Agregar configuración y cliente cloud sin modificar el comportamiento local por defecto.

### C2 — Cloud chat

Permitir `ORION_MODEL_PROVIDER=cloudflare` y responder desde un modelo remoto manteniendo el mismo contrato NDJSON del frontend.

### C3 — Deploy

Desplegar FastAPI y frontend en infraestructura gratuita y comprobar que Orion conversa con Ollama apagado.

### C4 — Benchmark

Ejecutar los datasets de Orion contra local y cloud. Registrar calidad, errores, tiempo y consumo de cuota.

## Criterio para dar por terminada la etapa

La etapa Cloud v1 termina cuando:

- Orion puede abrirse desde Internet;
- el backend no depende de la CPU/RAM de la notebook para generar la respuesta;
- Ollama puede estar apagado y el chat sigue funcionando;
- el frontend conserva streaming y cancelación;
- las pruebas actuales siguen pasando;
- existe fallback local configurable;
- el gasto obligatorio mensual es $0;
- no existe ninguna clave secreta versionada en GitHub.
