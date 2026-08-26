"use client";

import { DiagnosticTrace } from "../lib/orion-api";
import styles from "./diagnostic.module.css";

function ms(value?: number | null) {
  if (value === undefined || value === null) return "—";
  if (value < 1_000) return `${value.toFixed(0)} ms`;
  return `${(value / 1_000).toFixed(1)} s`;
}

function join(values: string[]) {
  return values.length ? values.join(", ") : "—";
}

export function DiagnosticPanel({ trace }: { trace: DiagnosticTrace }) {
  const latestReview = trace.reviews.at(-1);
  return (
    <details className={styles.panel} open>
      <summary>
        <span>Diagnóstico · {trace.status}</span>
        <code>{trace.trace_id}</code>
      </summary>
      <div className={styles.grid}>
        <section>
          <h4>Comprensión</h4>
          <p><strong>Objetivo:</strong> {trace.plan?.objective ?? "—"}</p>
          <p><strong>Entidades:</strong> {join(trace.plan?.entities ?? [])}</p>
          <p><strong>Restricciones:</strong> {join(trace.plan?.constraints ?? [])}</p>
          <p><strong>Ambigüedades:</strong> {join(trace.plan?.ambiguities ?? [])}</p>
          <p><strong>Política de evidencia:</strong> {trace.plan?.evidence_policy ?? "—"}</p>
          <p><strong>Fallback:</strong> {trace.plan_fallback ? "sí" : "no"}</p>
          {trace.plan_error ? <p><strong>Error de plan:</strong> {trace.plan_error}</p> : null}
        </section>

        <section>
          <h4>Tiempos</h4>
          <p><strong>Plan:</strong> {ms(trace.timings_ms.planning)}</p>
          <p><strong>Búsqueda web:</strong> {ms(trace.timings_ms.web_search_total)}</p>
          <p><strong>Revisión:</strong> {ms(trace.timings_ms.review_total)}</p>
          <p><strong>Pipeline previo:</strong> {ms(trace.timings_ms.reasoning_bundle_total)}</p>
        </section>
      </div>

      {trace.model_calls.length ? (
        <section className={styles.section}>
          <h4>Llamadas al modelo</h4>
          <div className={styles.sources}>
            {trace.model_calls.map((call, index) => (
              <details key={`${call.stage}-${index}`}>
                <summary>{call.stage} · {call.model ?? "modelo desconocido"}</summary>
                <p><strong>Endpoint:</strong> {call.endpoint ?? "—"}</p>
                <p><strong>Reasoning effort:</strong> {call.reasoning_effort ?? "—"}</p>
                <p><strong>Finish reason:</strong> {call.finish_reason ?? "—"}</p>
                <p><strong>Input tokens:</strong> {call.prompt_tokens ?? "—"}</p>
                <p><strong>Output tokens:</strong> {call.completion_tokens ?? "—"}</p>
                <p><strong>Reasoning tokens:</strong> {call.reasoning_tokens ?? "—"}</p>
                <p><strong>Duración registrada:</strong> {ms(call.duration_ms)}</p>
              </details>
            ))}
          </div>
        </section>
      ) : null}

      {trace.local_evidence.length ? (
        <section className={styles.section}>
          <h4>Evidencia local y herramientas</h4>
          <div className={styles.sources}>
            {trace.local_evidence.map((item) => (
              <details key={`${item.source_id}-${item.document_name}`}>
                <summary>{item.source_id} · {item.document_name}</summary>
                <p>{item.excerpt || "Sin extracto."}</p>
                <p><strong>Fragmento:</strong> {item.chunk_index === null ? "herramienta/dato estructurado" : item.chunk_index + 1}</p>
              </details>
            ))}
          </div>
        </section>
      ) : null}

      {trace.searches.map((search) => (
        <section key={`search-${search.round}`} className={styles.section}>
          <h4>Búsqueda {search.round}</h4>
          <p><strong>Consulta:</strong> {search.query}</p>
          <p><strong>Duración:</strong> {ms(search.duration_ms)}</p>
          <div className={styles.sources}>
            {search.raw_results.map((source) => (
              <details key={source.source_id}>
                <summary>{source.source_id} · {source.domain || source.title}</summary>
                <p><strong>{source.title}</strong></p>
                <p>{source.excerpt || "Sin extracto."}</p>
                {source.url ? (
                  <a href={source.url} target="_blank" rel="noreferrer">Abrir fuente</a>
                ) : null}
              </details>
            ))}
          </div>
        </section>
      ))}

      {trace.reviews.map((review) => (
        <section key={`review-${review.round}`} className={styles.section}>
          <h4>Revisión {review.round}</h4>
          <p><strong>Evidencia suficiente:</strong> {review.sufficient ? "sí" : "no"}</p>
          <p><strong>Fuentes aceptadas:</strong> {join(review.relevant_source_ids)}</p>
          <p><strong>Fuentes descartadas:</strong> {join(review.discarded_source_ids)}</p>
          <p><strong>Alcance resuelto:</strong> {review.resolved_scope ?? "—"}</p>
          <p><strong>Falta:</strong> {join(review.missing_information)}</p>
          <p><strong>Siguiente búsqueda:</strong> {review.follow_up_web_query ?? "—"}</p>
          <p><strong>Fallback:</strong> {review.fallback ? "sí" : "no"}</p>
          {review.error ? <p><strong>Error:</strong> {review.error}</p> : null}
        </section>
      ))}

      {trace.guard_events.length ? (
        <section className={styles.section}>
          <h4>Protecciones activadas</h4>
          {trace.guard_events.map((guard, index) => (
            <p key={`${guard.event}-${index}`}><strong>{guard.event}:</strong> {guard.detail}</p>
          ))}
        </section>
      ) : null}

      <section className={styles.section}>
        <h4>Resumen técnico</h4>
        <p><strong>Estado:</strong> {trace.status}</p>
        <p><strong>Modelo:</strong> {trace.model ?? "—"}</p>
        <p><strong>Fuentes recuperadas:</strong> {trace.searches.reduce((sum, item) => sum + item.raw_results.length, 0)}</p>
        <p><strong>Resultado del revisor:</strong> {latestReview ? (latestReview.sufficient ? "suficiente" : "insuficiente/no confirmado") : "sin revisión externa"}</p>
        <p><strong>Respuesta final registrada:</strong> {trace.final_answer ? "sí" : "no"}</p>
        {trace.error ? <p><strong>Error final:</strong> {trace.error}</p> : null}
        <p className={styles.privacy}>La traza vive solo en RAM del proceso. Registra decisiones, herramientas y metadatos de completado, pero no credenciales ni cadena de pensamiento oculta.</p>
      </section>
    </details>
  );
}
