# Cómo mejorar Orion

Orion no aprende automáticamente de cada conversación. Para mejorar sus
respuestas de forma confiable, cada nuevo conocimiento debe entrar por una de
estas vías:

## 1. Datos y documentos

- Usá archivos CSV con una fila de encabezados clara.
- Usá una columna estable para identificar al jugador o entidad.
- Usá nombres de columnas descriptivos y unidades en el nombre o en la documentación.
- Separá sesión, período, fecha y jugador en columnas distintas.
- Evitá mezclar totales agregados con períodos detallados sin indicarlo.
- Cargá primero una copia pequeña y verificá una respuesta antes de usar el archivo completo.

## 2. Cómo pedir un análisis

Una consulta reproducible indica:

`jugador + métrica + período + operación + unidad`

Ejemplo: `Graficá Total Distance de Ruan por Period Name en metros y mostrá la fuente.`

Si la consulta no define esos elementos, Orion debe pedir una aclaración.

## 3. Cómo convertir un error en mejora

Cuando una respuesta sea incorrecta, guardá un caso con:

- pregunta original;
- archivo o estructura relevante;
- respuesta esperada;
- respuesta incorrecta;
- regla que debería haber aplicado.

Ese caso debe convertirse en un test o una evaluación antes de cambiar el prompt.

## 4. Qué sí y qué no es aprendizaje

- Importar un documento agrega conocimiento recuperable, no modifica los pesos del modelo.
- Las evaluaciones mejoran el comportamiento del sistema, no entrenan por sí solas al modelo.
- El fine-tuning debe venir después de reunir muchos ejemplos corregidos y medir una línea base.
- Los datos médicos o personales deben permanecer locales y cargarse solo con consentimiento.

## Orden recomendado

1. CSV estructurado y cálculos deterministas.
2. Pruebas de exactitud y casos de error.
3. Herramientas de filtros, promedios, comparaciones y gráficos.
4. Embeddings locales para recuperar conceptos relacionados.
5. Memoria editable y opcional.
6. Búsqueda web con citas, fecha y fuente verificable.

## Búsqueda web controlada

La búsqueda web se activa explícitamente y solo consulta dominios de la
allowlist. Mantené `ORION_WEB_MINIMUM_SOURCES=4` o un valor mayor. Si hay menos
de cuatro fuentes permitidas o si discrepan, Orion debe informar la insuficiencia
en lugar de presentar una conclusión como consenso. Toda afirmación actual debe
conservar sus fuentes, URLs y fecha.