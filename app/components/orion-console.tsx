"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChatMessage,
  getOrionStatus,
  OrionApiError,
  OrionStatus,
  RequestedMode,
  ResourceWarning,
  sendChatStream,
} from "../lib/orion-api";

type UiMessage = ChatMessage & {
  id: string;
  model?: string;
  mode?: "quick" | "deep";
  recommendation?: string;
  totalDurationMs?: number | null;
  firstTokenMs?: number | null;
  loadDurationMs?: number | null;
  promptTokens?: number | null;
  completionTokens?: number | null;
  tokensPerSecond?: number | null;
  peakCpuPercent?: number | null;
  threadLimit?: number | null;
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

function formatDuration(milliseconds?: number | null) {
  if (milliseconds === null || milliseconds === undefined) return null;
  if (milliseconds < 1_000) return `${milliseconds.toFixed(0)} ms`;
  return `${(milliseconds / 1_000).toFixed(1)} s`;
}

export function OrionConsole() {
  const [status, setStatus] = useState<OrionStatus | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [mode, setMode] = useState<RequestedMode>("auto");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<PendingWarning | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const peakCpuRef = useRef(0);

  useEffect(() => {
    const controller = new AbortController();

    const refresh = async () => {
      try {
        const next = await getOrionStatus(controller.signal);
        setStatus(next);
        if (loading) {
          peakCpuRef.current = Math.max(
            peakCpuRef.current,
            next.snapshot.cpu_percent,
          );
        }
        setStatusError(false);
      } catch {
        if (!controller.signal.aborted) setStatusError(true);
      }
    };

    void refresh();
    const timer = setInterval(refresh, loading ? 2_500 : 15_000);

    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [loading]);

  useEffect(() => {
    if (!loading) return;

    const startedAt = Date.now();
    const timer = setInterval(() => {
      setElapsedSeconds(Math.max(1, Math.round((Date.now() - startedAt) / 1_000)));
    }, 1_000);

    return () => clearInterval(timer);
  }, [loading]);

  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: loading ? "auto" : "smooth",
    });
  }, [messages, loading, warning]);

  useEffect(
    () => () => {
      activeRequestRef.current?.abort();
    },
    [],
  );

  const runRequest = async (
    requestMessages: ChatMessage[],
    requestedMode: RequestedMode,
    allowBusy = false,
  ) => {
    const controller = new AbortController();
    activeRequestRef.current?.abort();
    activeRequestRef.current = controller;
    peakCpuRef.current = status?.snapshot.cpu_percent ?? 0;
    setElapsedSeconds(0);
    setLoading(true);
    setError(null);
    setWarning(null);

    const assistantId = makeId("assistant");
    const startedAt = performance.now();
    let firstTokenMs: number | null = null;
    let assistantStarted = false;

    try {
      const result = await sendChatStream(
        {
          messages: requestMessages,
          mode: requestedMode,
          allowBusy,
        },
        {
          onMeta: (meta) => {
            assistantStarted = true;
            setMessages((current) => [
              ...current,
              {
                id: assistantId,
                role: "assistant",
                content: "",
                model: meta.model,
                mode: meta.selected_mode,
                recommendation: meta.recommendation_reason,
              },
            ]);
          },
          onContent: (content) => {
            if (firstTokenMs === null) {
              firstTokenMs = performance.now() - startedAt;
            }
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + content }
                  : message,
              ),
            );
          },
        },
        controller.signal,
      );
      let measuredPeakCpu = peakCpuRef.current;
      try {
        const latestStatus = await getOrionStatus();
        setStatus(latestStatus);
        setStatusError(false);
        measuredPeakCpu = Math.max(
          measuredPeakCpu,
          latestStatus.snapshot.cpu_percent,
        );
      } catch {
        // The chat result remains valid even if this optional measurement fails.
      }

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                model: result.meta.model,
                mode: result.meta.selected_mode,
                recommendation: result.meta.recommendation_reason,
                totalDurationMs: result.done.total_duration_ms,
                firstTokenMs,
                loadDurationMs: result.done.load_duration_ms,
                promptTokens: result.done.prompt_tokens,
                completionTokens: result.done.completion_tokens,
                tokensPerSecond: result.done.tokens_per_second,
                peakCpuPercent: measuredPeakCpu,
                threadLimit: result.done.thread_limit,
              }
            : message,
        ),
      );
    } catch (caught) {
      if (controller.signal.aborted) {
        if (assistantStarted) {
          setMessages((current) =>
            current
              .map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content || "Respuesta detenida.",
                      recommendation: "Respuesta detenida por el usuario.",
                    }
                  : message,
              ),
          );
        }
        return;
      }
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
        if (assistantStarted) {
          setMessages((current) =>
            current.flatMap((message) => {
              if (message.id !== assistantId) return [message];
              if (!message.content) return [];
              return [
                {
                  ...message,
                  recommendation: "La respuesta se interrumpió antes de terminar.",
                },
              ];
            }),
          );
        }
        setError(
          caught instanceof Error
            ? caught.message
            : "Orion no pudo completar la solicitud.",
        );
      }
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null;
      }
      setLoading(false);
    }
  };

  const stopRequest = () => {
    activeRequestRef.current?.abort();
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
    status?.installed_models.some(
      (name) => name === status.quick_model || name.startsWith(status.quick_model),
    ),
  );
  const cpuHigh = Boolean(status && status.snapshot.cpu_percent >= 50);

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
            <span>Modelo Rápido</span>
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
            <span className={`status-value ${cpuHigh ? "resource-hot" : ""}`}>
              {status ? `${status.snapshot.cpu_percent.toFixed(0)}%` : "—"}
            </span>
          </div>
          <div className="status-line">
            <span>Protección CPU</span>
            <span className="status-value">
              {status ? `Activa · ${status.quick_threads ?? 8} hilos` : "Activa"}
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
          Módulo 1.2 · Las conversaciones permanecen únicamente en esta sesión y
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
              <p className="eyebrow">Núcleo deportivo · Módulo 1.2</p>
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
            <div ref={messagesRef} className="messages" aria-live="polite">
              {messages.map((message) => {
                const metrics = [
                  message.mode === "deep" ? "Profundo" : "Rápido",
                  message.model,
                  message.firstTokenMs !== null &&
                  message.firstTokenMs !== undefined
                    ? `primer texto ${formatDuration(message.firstTokenMs)}`
                    : null,
                  message.totalDurationMs !== null &&
                  message.totalDurationMs !== undefined
                    ? `total ${formatDuration(message.totalDurationMs)}`
                    : null,
                  message.loadDurationMs !== null &&
                  message.loadDurationMs !== undefined
                    ? `carga ${formatDuration(message.loadDurationMs)}`
                    : null,
                  message.tokensPerSecond
                    ? `${message.tokensPerSecond.toFixed(1)} tok/s`
                    : null,
                  message.promptTokens
                    ? `${message.promptTokens} tokens de entrada`
                    : null,
                  message.completionTokens
                    ? `${message.completionTokens} tokens de salida`
                    : null,
                  message.peakCpuPercent !== null &&
                  message.peakCpuPercent !== undefined
                    ? `pico CPU ${message.peakCpuPercent.toFixed(0)}%`
                    : null,
                  message.threadLimit ? `${message.threadLimit} hilos máx.` : null,
                ].filter((value): value is string => Boolean(value));

                return (
                  <article key={message.id} className={`message ${message.role}`}>
                    {message.role === "assistant" ? (
                      <div className="message-content markdown-content">
                        {message.content ? (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              a: ({ href, children }) => (
                                <a href={href} target="_blank" rel="noreferrer">
                                  {children}
                                </a>
                              ),
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        ) : (
                          <span className="stream-placeholder">Preparando respuesta…</span>
                        )}
                      </div>
                    ) : (
                      <div className="message-content plain-content">
                        {message.content}
                      </div>
                    )}
                    {message.role === "assistant" && message.model ? (
                      <div className="message-meta">
                        <div className="metric-row">
                          {metrics.map((metric) => (
                            <span key={metric}>{metric}</span>
                          ))}
                        </div>
                        {message.recommendation ? (
                          <p>{message.recommendation}</p>
                        ) : null}
                      </div>
                    ) : null}
                  </article>
                );
              })}
              {loading ? (
                <div className="thinking" role="status">
                  <span className="thinking-label">Orion está respondiendo</span>
                  <span className="thinking-clock">{elapsedSeconds}s</span>
                  <span className="thinking-dots" aria-hidden="true">
                    <i /><i /><i />
                  </span>
                  <button
                    type="button"
                    className="cancel-button"
                    onClick={stopRequest}
                  >
                    Detener
                  </button>
                  <small>La respuesta aparece progresivamente · prioridad reducida activa.</small>
                </div>
              ) : null}
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
