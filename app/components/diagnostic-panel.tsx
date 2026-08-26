"use client";

import { DiagnosticTrace } from "../lib/orion-api";

function ms(value?: number) {
  if (value === undefined) return "—";
  if (value < 1_000) return `${value.toFixed(0)} ms`;
  return `${(value / 1_000).toFixed(1)} s`;
}

function join(values: string[]) {
  return values.length ? values.join(", ") : "—";
}

export function DiagnosticPanel({ trace }: { trace: DiagnosticTrace }) {
  const latestReview = trace.reviews.at(-1);
  return (
    <details className="diagnostic-panel">
      <summary>
        <span>Diagnóstico</span>
        <code>{trace.trace_id}</code>
      </summary>
      <div className="diagnostic-grid">
        <section>
          <h4>Comprensión</h4>
          <p><strong>Objetivo:</strong> {trace.plan?.objective ?? "—"}</p>
          <p><strong>Entidades:</strong> {join(trace.plan?.entities ?? [])}</p>
          <p><strong>Restricciones:</strong> {join(trace.plan?.constraints ?? [])}</p>
          <p><strong>Ambigüedades:</strong> {join(trace.plan?.ambiguities ?? [])}</p>
          <p><strong>Fallback:</strong> {trace.plan_fallback ? "sí" : "no"}</p>
        </section>

        <section>
          <h4>Tiempos</h4>
          <p><strong>Plan:</strong> {ms(trace.timings_ms.planning)}</p>
          <p><strong>Búsqueda web:</strong> {ms(trace.timings_ms.web_search_total)}</p>
          <p><strong>Revisión:</strong> {ms(trace.timings_ms.review_total)}</p>
          <p><strong>Pipeline previo:</strong> {ms(trace.timings_ms.reasoning_bundle_total)}</p>
        </section>
      </div>

      {trace.searches.map((search) => (
        <section key={`search-${search.round}`} className="diagnostic-section">
          <h4>Búsqueda {search.round}</h4>
          <p><strong>Consulta:</strong> {search.query}</p>
          <p><strong>Duración:</strong> {ms(search.duration_ms)}</p>
          <div className="diagnostic-sources">
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
        <section key={`review-${review.round}`} className="diagnostic-section">
          <h4>Revisión {review.round}</h4>
          <p><strong>Evidencia suficiente:</strong> {review.sufficient ? "sí" : "no"}</p>
          <p><strong>Fuentes aceptadas:</strong> {join(review.relevant_source_ids)}</p>
          <p><strong>Fuentes descartadas:</strong> {join(review.discarded_source_ids)}</p>
          <p><strong>Alcance resuelto:</strong> {review.resolved_scope ?? "—"}</p>
          <p><strong>Falta:</strong> {join(review.missing_information)}</p>
          <p><strong>Siguiente búsqueda:</strong> {review.follow_up_web_query ?? "—"}</p>
          <p><strong>Fallback:</strong> {review.fallback ? "sí" : "no"}</p>
        </section>
      ))}

      <section className="diagnostic-section diagnostic-summary">
        <h4>Resumen técnico</h4>
        <p><strong>Modelo:</strong> {trace.model ?? "—"}</p>
        <p><strong>Fuentes recuperadas:</strong> {trace.searches.reduce((sum, item) => sum + item.raw_results.length, 0)}</p>
        <p><strong>Resultado del revisor:</strong> {latestReview?.sufficient ? "suficiente" : "insuficiente/no confirmado"}</p>
        <p className="diagnostic-privacy">La traza vive solo en RAM del proceso. No contiene credenciales ni cadena de pensamiento oculta.</p>
      </section>
    </details>
  );
}
