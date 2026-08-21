"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ChatMessage,
  getOrionStatus,
  OrionApiError,
  OrionStatus,
  RequestedMode,
  ResourceWarning,
  sendChat,
} from "../lib/orion-api";

type UiMessage = ChatMessage & {
  id: string;
  model?: string;
  mode?: "quick" | "deep";
  recommendation?: string;
};

type PendingWarning = {
  messages: ChatMessage[];
  requestedMode: RequestedMode;
  detail: ResourceWarning;
};

const MODE_LABELS: Record<RequestedMode, string> = {
  auto: "Orion decide",
  quick: "Rápido",
  deep: "Profundo",
};

const SUGGESTIONS = [
  "Explicame qué diferencia hay entre carga externa e interna.",
  "¿Cómo validarías una base GPS antes de analizarla?",
  "Compará dos formas de calcular un promedio ponderado.",
  "Ayudame a estructurar un informe de rendimiento individual.",
];

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function OrionConsole() {
  const [status, setStatus] = useState<OrionStatus | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [mode, setMode] = useState<RequestedMode>("auto");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<PendingWarning | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();

    const refresh = async () => {
      try {
        const next = await getOrionStatus(controller.signal);
        setStatus(next);
        setStatusError(false);
      } catch {
        if (!controller.signal.aborted) setStatusError(true);
      }
    };

    void refresh();
    const timer = setInterval(refresh, 20_000);

    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, warning]);

  const runRequest = async (
    requestMessages: ChatMessage[],
    requestedMode: RequestedMode,
    allowBusy = false,
  ) => {
    setLoading(true);
    setError(null);
    setWarning(null);

    try {
      const result = await sendChat({
        messages: requestMessages,
        mode: requestedMode,
        allowBusy,
      });
      setMessages((current) => [
        ...current,
        {
          id: makeId("assistant"),
          role: "assistant",
          content: result.content,
          model: result.model,
          mode: result.selected_mode,
          recommendation: result.recommendation_reason,
        },
      ]);
    } catch (caught) {
      if (
        caught instanceof OrionApiError &&
        caught.status === 409 &&
        typeof caught.detail === "object" &&
        caught.detail.code === "resource_confirmation_required"
      ) {
        setWarning({
          messages: requestMessages,
          requestedMode,
          detail: caught.detail as ResourceWarning,
        });
      } else {
        setError(
          caught instanceof Error
            ? caught.message
            : "Orion no pudo completar la solicitud.",
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const submitPrompt = async (prompt: string) => {
    const clean = prompt.trim();
    if (!clean || loading) return;

    const nextUiMessage: UiMessage = {
      id: makeId("user"),
      role: "user",
      content: clean,
    };
    const nextMessages = [...messages, nextUiMessage];
    setMessages(nextMessages);
    setDraft("");

    await runRequest(
      nextMessages
        .slice(-20)
        .map(({ role, content }) => ({ role, content })),
      mode,
    );
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitPrompt(draft);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitPrompt(draft);
    }
  };

  const online = Boolean(status && !statusError);
  const ollamaOnline = Boolean(status?.ollama_online);
  const quickInstalled = Boolean(
    status?.installed_models.some((name) => name.startsWith("qwen3:8b")),
  );

  return (
    <main className="orion-shell">
      <aside className="orion-sidebar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">OR</div>
          <div>
            <p className="brand-name">ORION</p>
            <p className="brand-caption">Inteligencia deportiva personal</p>
          </div>
        </div>

        <div className="side-stack">
          <p className="side-label">Estado local</p>
          <div className="status-line">
            <span className="engine-status">
              <span className={`dot ${online ? "online" : "offline"}`} />
              Núcleo de Orion
            </span>
            <span className="status-value">{online ? "Activo" : "Desconectado"}</span>
          </div>
          <div className="status-line">
            <span>Ollama</span>
            <span className="status-value">{ollamaOnline ? "Activo" : "Pendiente"}</span>
          </div>
          <div className="status-line">
            <span>Qwen3 8B</span>
            <span className="status-value">{quickInstalled ? "Instalado" : "Pendiente"}</span>
          </div>
          <div className="status-line">
            <span>RAM disponible</span>
            <span className="status-value">
              {status ? `${status.snapshot.memory_available_gb.toFixed(1)} GB` : "—"}
            </span>
          </div>
          <div className="status-line">
            <span>CPU</span>
            <span className="status-value">
              {status ? `${status.snapshot.cpu_percent.toFixed(0)}%` : "—"}
            </span>
          </div>
        </div>

        <div className="side-stack">
          <p className="side-label">Privacidad</p>
          <div className="privacy-line">
            <span>Memoria permanente</span>
            <span className="status-value">Desactivada</span>
          </div>
          <div className="privacy-line">
            <span>Motor externo</span>
            <span className="status-value">Desactivado</span>
          </div>
        </div>

        <p className="side-note">
          Módulo 1 · Las conversaciones permanecen únicamente en esta sesión y
          se pierden al recargar la página.
        </p>
      </aside>

      <section className="orion-main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Conversación local</p>
            <h1>¿Qué querés analizar?</h1>
          </div>
          <div className="mode-switcher" aria-label="Modo de respuesta">
            {(Object.keys(MODE_LABELS) as RequestedMode[]).map((value) => (
              <button
                key={value}
                type="button"
                className={`mode-button ${mode === value ? "active" : ""}`}
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
              >
                {MODE_LABELS[value]}
              </button>
            ))}
          </div>
        </header>

        <div className="chat-stage">
          {messages.length === 0 ? (
            <section className="empty-state">
              <p className="eyebrow">Núcleo deportivo · Módulo 1</p>
              <h2>Tu criterio, amplificado.</h2>
              <p>
                Orion ya tiene la base para conversar con un modelo local,
                recomendar el nivel de análisis y proteger el rendimiento de tu PC.
              </p>
              <div className="suggestion-grid">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion"
                    onClick={() => setDraft(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="messages" aria-live="polite">
              {messages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  {message.content}
                  {message.role === "assistant" && message.model ? (
                    <div className="message-meta">
                      {message.mode === "deep" ? "Profundo" : "Rápido"} · {message.model}
                      {message.recommendation ? ` · ${message.recommendation}` : ""}
                    </div>
                  ) : null}
                </article>
              ))}
              {loading ? (
                <div className="thinking" role="status">
                  Orion está procesando <span /><span /><span />
                </div>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}

          <div className="composer-wrap">
            {warning ? (
              <div className="resource-warning" role="alert">
                <strong>La computadora está exigida.</strong> {warning.detail.message}
                <div>{warning.detail.reasons.join(" · ")}</div>
                <div className="warning-actions">
                  <button
                    type="button"
                    className="small-button primary"
                    onClick={() => void runRequest(warning.messages, "quick", false)}
                  >
                    Usar Rápido
                  </button>
                  <button
                    type="button"
                    className="small-button"
                    onClick={() => setWarning(null)}
                  >
                    Esperar
                  </button>
                  <button
                    type="button"
                    className="small-button"
                    onClick={() =>
                      void runRequest(
                        warning.messages,
                        warning.requestedMode,
                        true,
                      )
                    }
                  >
                    Continuar igualmente
                  </button>
                </div>
              </div>
            ) : null}

            {error ? <div className="error-banner" role="alert">{error}</div> : null}

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Escribile a Orion…"
                aria-label="Mensaje para Orion"
                maxLength={20_000}
              />
              <button
                type="submit"
                className="send-button"
                disabled={!draft.trim() || loading}
                aria-label="Enviar mensaje"
              >
                ↑
              </button>
            </form>
            <p className="composer-note">
              Enter para enviar · Shift + Enter para una nueva línea · Sin memoria permanente
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
