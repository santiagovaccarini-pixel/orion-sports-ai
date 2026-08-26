"use client";

import { useState } from "react";

import {
  DiagnosticTrace,
  OrionApiError,
  getLatestDiagnosticTrace,
} from "../lib/orion-api";
import { DiagnosticPanel } from "./diagnostic-panel";

export function DiagnosticDock() {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [trace, setTrace] = useState<DiagnosticTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await getLatestDiagnosticTrace();
      setTrace(next);
      setOpen(true);
    } catch (caught) {
      if (caught instanceof OrionApiError && caught.status === 404) {
        setError("Todavía no hay una traza disponible. Hacé una consulta a Orion primero.");
      } else {
        setError(
          caught instanceof Error
            ? caught.message
            : "No se pudo cargar el diagnóstico.",
        );
      }
      setOpen(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`diagnostic-dock ${open ? "open" : ""}`}>
      <div className="diagnostic-dock-actions">
        <button
          type="button"
          className="diagnostic-dock-button"
          onClick={() => (open ? setOpen(false) : void refresh())}
        >
          {open ? "Cerrar diagnóstico" : "Diagnóstico"}
        </button>
        {open ? (
          <button
            type="button"
            className="diagnostic-refresh-button"
            onClick={() => void refresh()}
            disabled={loading}
          >
            {loading ? "Actualizando…" : "Actualizar"}
          </button>
        ) : null}
      </div>

      {open ? (
        <div className="diagnostic-dock-body">
          {error ? <div className="diagnostic-error">{error}</div> : null}
          {trace ? <DiagnosticPanel trace={trace} /> : null}
        </div>
      ) : null}
    </div>
  );
}
