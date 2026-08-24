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

Desde el Módulo 1.4, un planificador semántico interpreta el objetivo real antes de
responder: puede resolver una paráfrasis, una referencia conversacional ("eso", "como
antes") o una pregunta en otro idioma sin que el usuario repita literalmente esos cinco
elementos. Eso reduce la fricción, pero no reemplaza la regla anterior: si falta una
variable indispensable (por ejemplo, qué métrica comparar), el plan semántico marca
`requires_clarification` y Orion sigue debiendo preguntar en lugar de adivinar.

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

1. ✅ CSV estructurado y cálculos deterministas.
2. ✅ Pruebas de exactitud y casos de error.
3. ✅ Herramientas de filtros, promedios, comparaciones y gráficos.
4. ✅ Interpretación semántica de la intención (Módulo 1.4): planificador previo a la
   respuesta, retrieval expandido por conceptos y fallback determinista si el modelo
   local no responde a tiempo.
5. ⏳ Embeddings locales para recuperar conceptos relacionados. `semantic_retriever.py`
   ya pondera por objetivo del usuario y conceptos inferidos, pero todavía busca por
   superposición de términos, no por similitud vectorial: es el siguiente paso natural.
6. ⏳ Memoria editable y opcional. El plan semántico ya distingue `needs_private_memory`
   de `needs_global_knowledge`, pero ningún dato persiste todavía entre sesiones.
7. ✅ Búsqueda web con citas, fecha y fuente verificable.

Los CSV ya no están limitados a la planilla de seguimiento GPS. `knowledge_base.py`
detecta el esquema de cualquier planilla tabular: columna identificadora, columnas
métricas numéricas y columna de período/fecha opcional, primero por nombre de columna
conocido y, si no encuentra ninguno, por estructura (fila de encabezado seguida de una
fila con valores numéricos). El caso GPS original sigue funcionando igual que antes; se
suman básquet, planillas de negocio o cualquier CSV con encabezados propios. Cuando hay
más de una columna numérica y la consulta no nombra cuál usar, Orion no adivina: lo trata
como ambiguo, igual que antes hacía con el jugador o la entidad.

## Búsqueda web controlada

La búsqueda web se activa explícitamente y solo consulta dominios de la
allowlist. Mantené `ORION_WEB_MINIMUM_SOURCES=4` o un valor mayor. Si hay menos
de cuatro fuentes permitidas o si discrepan, Orion debe informar la insuficiencia
en lugar de presentar una conclusión como consenso. Toda afirmación actual debe
conservar sus fuentes, URLs y fecha.