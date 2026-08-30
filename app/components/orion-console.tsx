"use client";

import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import { OrionMark } from "./orion-mark";
import remarkGfm from "remark-gfm";
import {
  deleteKnowledgeDocument,
  ChatMessage,
  OrionChart,
  MemoryEntry,
  deleteMemoryEntry,
  getOrionStatus,
  listMemoryEntries,
  OrionApiError,
  OrionStatus,
  RequestedMode,
  ResourceWarning,
  saveMemoryEntry,
  sendChatStream,
  uploadKnowledgeDocument,
  Sport,
} from "../lib/orion-api";

type UiMessage = ChatMessage & {
  id: string;
  attachmentName?: string;
  chart?: OrionChart;
  sport?: Sport;
  streaming?: boolean;
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
  sport: Sport;
  detail: ResourceWarning;
};

const MODE_LABELS: Record<RequestedMode, string> = {
  auto: "Orion decide",
  quick: "Rápido",
  deep: "Profundo",
};

const SPORT_OPTIONS: ReadonlyArray<{
  value: Sport;
  label: string;
  icon: string;
}> = [
  { value: "general", label: "General", icon: "◎" },
  { value: "football", label: "Fútbol", icon: "⚽" },
  { value: "basketball", label: "Básquet", icon: "🏀" },
  { value: "volleyball", label: "Vóley", icon: "🏐" },
  { value: "rugby", label: "Rugby", icon: "🏉" },
  { value: "tennis", label: "Tenis", icon: "🎾" },
  { value: "athletics", label: "Atletismo", icon: "🏃" },
  { value: "swimming", label: "Natación", icon: "🏊" },
  { value: "cycling", label: "Ciclismo", icon: "🚴" },
];

const SPORT_LABELS = Object.fromEntries(
  SPORT_OPTIONS.map(({ value, label }) => [value, label]),
) as Record<Sport, string>;

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

function simplifyFormula(expression: string) {
  return expression
    .trim()
    .replace(/\\bar\{([^{}]+)\}/g, "$1̄")
    .replace(/\\cdot|\\times/g, "×")
    .replace(/\\sum/g, "Σ")
    .replace(/\\left|\\right/g, "")
    .replace(/\s+/g, " ");
}

function normalizeCompletedMarkdown(content: string) {
  const normalized = content
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) =>
      `\n\n\`\`\`text\n${simplifyFormula(expression)}\n\`\`\`\n\n`,
    )
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) =>
      `\`${simplifyFormula(expression)}\``,
    );

  return normalized.replace(/```(?:text|txt)?\n([\s\S]*?)```/g, (match, block: string) => {
    const rows = block
      .split("\n")
      .map((line: string) => line.trim())
      .filter((line: string) => line.includes("|"))
      .filter((line: string) => !/^[|+\-=: ]+$/.test(line));
    if (rows.length < 2) return match;

    const cells = rows.map((row: string) =>
      row
        .replace(/^[|+]/, "")
        .replace(/[|+]$/, "")
        .split(/[|+]/)
        .map((cell: string) => cell.trim())
        .filter(Boolean),
    );
    const columnCount = cells[0]?.length ?? 0;
    if (columnCount < 2 || cells.some((row: string[]) => row.length !== columnCount)) {
      return match;
    }
    return [
      `| ${cells[0].join(" | ")} |`,
      `| ${cells[0].map(() => "---").join(" | ")} |`,
      ...cells.slice(1).map((row: string[]) => `| ${row.join(" | ")} |`),
    ].join("\n");
  });
}

function renderMarkdown(content: string) {
  return (
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
      {normalizeCompletedMarkdown(content)}
    </ReactMarkdown>
  );
}

function ChartPreview({ chart }: { chart: OrionChart }) {
  const width = 640;
  const height = 260;
  const padding = { top: 42, right: 24, bottom: 48, left: 52 };
  const max = Math.max(...chart.points.map((point) => point.value), 1);
  const chartHeight = height - padding.top - padding.bottom;
  const barWidth = (width - padding.left - padding.right) / chart.points.length * 0.62;
  return (
    <figure className="chart-preview">
      <figcaption>{chart.title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${chart.title}. Fuente: ${chart.source}`}>
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="chart-axis" />
        {chart.points.map((point, index) => {
          const slot = (width - padding.left - padding.right) / chart.points.length;
          const barHeight = (point.value / max) * chartHeight;
          const x = padding.left + slot * index + (slot - barWidth) / 2;
          const y = height - padding.bottom - barHeight;
          return (
            <g key={`${point.label}-${index}`}>
              <rect x={x} y={y} width={barWidth} height={barHeight} rx="4" className="chart-bar" />
              <text x={x + barWidth / 2} y={Math.max(y - 8, 18)} textAnchor="middle" className="chart-value">{point.value.toLocaleString("es-AR")}</text>
              <text x={x + barWidth / 2} y={height - 20} textAnchor="middle" className="chart-label">{point.label}</text>
            </g>
          );
        })}
      </svg>
      <small>Unidad: {chart.unit} · Fuente local: {chart.source}</small>
    </figure>
  );
}

export function OrionConsole() {
  const [status, setStatus] = useState<OrionStatus | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [mode, setMode] = useState<RequestedMode>("auto");
  const [sport, setSport] = useState<Sport>("football");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [knowledgeMessage, setKnowledgeMessage] = useState<string | null>(null);
  const [attachment, setAttachment] = useState<{ id: string; name: string } | null>(
    null,
  );
  const [attachmentBusy, setAttachmentBusy] = useState(false);
  const [memoryEntries, setMemoryEntries] = useState<MemoryEntry[]>([]);
  const [memoryDraft, setMemoryDraft] = useState("");
  const [memoryBusy, setMemoryBusy] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [warning, setWarning] = useState<PendingWarning | null>(null);
  const [showScrollToLatest, setShowScrollToLatest] = useState(false);
  const messagesRef = useRef<HTMLDivElement>(null);
  const sportPickerRef = useRef<HTMLDetailsElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const pendingContentRef = useRef("");
  const contentFrameRef = useRef<number | null>(null);
  const peakCpuRef = useRef(0);
  const shouldFollowRef = useRef(true);
  const previousScrollTopRef = useRef(0);
  const autoScrollFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listMemoryEntries(controller.signal)
      .then(setMemoryEntries)
      .catch(() => {
        // Memory is supplementary: a failure here must not block the console.
      });
    return () => controller.abort();
  }, []);

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
    if (!container || !shouldFollowRef.current) return;
    if (autoScrollFrameRef.current !== null) {
      cancelAnimationFrame(autoScrollFrameRef.current);
    }
    autoScrollFrameRef.current = requestAnimationFrame(() => {
      if (!shouldFollowRef.current) {
        autoScrollFrameRef.current = null;
        return;
      }
      container.scrollTop = container.scrollHeight;
      previousScrollTopRef.current = container.scrollTop;
      autoScrollFrameRef.current = null;
    });
    return () => {
      if (autoScrollFrameRef.current !== null) {
        cancelAnimationFrame(autoScrollFrameRef.current);
        autoScrollFrameRef.current = null;
      }
    };
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
    selectedSport: Sport,
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
    const flushPendingContent = () => {
      const pendingContent = pendingContentRef.current;
      if (!pendingContent) return;
      pendingContentRef.current = "";
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: message.content + pendingContent }
            : message,
        ),
      );
    };

    try {
      const result = await sendChatStream(
        {
          messages: requestMessages,
          mode: requestedMode,
          sport: selectedSport,
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
                sport: meta.sport,
                streaming: true,
                model: meta.model,
                mode: meta.selected_mode,
                recommendation: meta.recommendation_reason,
              },
            ]);
          },
          onChart: (chart) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, chart } : message,
              ),
            );
          },
          onContent: (content) => {
            if (firstTokenMs === null) {
              firstTokenMs = performance.now() - startedAt;
            }
            pendingContentRef.current += content;
            if (contentFrameRef.current !== null) return;
            contentFrameRef.current = requestAnimationFrame(() => {
              const pendingContent = pendingContentRef.current;
              pendingContentRef.current = "";
              contentFrameRef.current = null;
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + pendingContent }
                    : message,
                ),
              );
            });
          },
        },
        controller.signal,
      );
      flushPendingContent();
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
                sport: result.meta.sport,
                streaming: false,
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
                      streaming: false,
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
          sport: selectedSport,
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
                  streaming: false,
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
      if (contentFrameRef.current !== null) {
        cancelAnimationFrame(contentFrameRef.current);
        contentFrameRef.current = null;
      }
      pendingContentRef.current = "";
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null;
      }
      setLoading(false);
    }
  };

  const stopRequest = () => {
    activeRequestRef.current?.abort();
  };

  const handleConversationScroll = () => {
    const container = messagesRef.current;
    if (!container) return;
    const movedUp = container.scrollTop < previousScrollTopRef.current - 1;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const isNearBottom = distanceFromBottom <= 96;
    previousScrollTopRef.current = container.scrollTop;
    if (movedUp) {
      shouldFollowRef.current = false;
      setShowScrollToLatest(true);
      return;
    }
    shouldFollowRef.current = isNearBottom;
    setShowScrollToLatest(!isNearBottom);
  };

  const handleConversationWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (event.deltaY < 0) {
      shouldFollowRef.current = false;
      setShowScrollToLatest(true);
    }
  };

  const scrollToLatest = () => {
    const container = messagesRef.current;
    shouldFollowRef.current = true;
    setShowScrollToLatest(false);
    if (container) previousScrollTopRef.current = container.scrollTop;
    container?.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  };

  // La × borraba solo el chip del compositor: el documento seguía guardado y
  // alimentando cada respuesta posterior. Ahora lo retira de verdad, que es lo
  // que la cruz siempre pareció prometer.
  const removeAttachment = async () => {
    if (!attachment || attachmentBusy) return;
    const container = messagesRef.current;
    const scrollTop = container?.scrollTop ?? 0;
    setAttachmentBusy(true);
    setKnowledgeMessage(null);
    try {
      await deleteKnowledgeDocument(attachment.id);
      setAttachment(null);
    } catch (caught) {
      setKnowledgeMessage(
        caught instanceof Error
          ? caught.message
          : "No se pudo quitar el documento.",
      );
    } finally {
      setAttachmentBusy(false);
      requestAnimationFrame(() => {
        if (container) {
          container.scrollTop = scrollTop;
          previousScrollTopRef.current = scrollTop;
        }
      });
    }
  };

  const selectSport = (nextSport: Sport) => {
    setSport(nextSport);
    sportPickerRef.current?.removeAttribute("open");
  };

  const handleKnowledgeUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setKnowledgeMessage("Importando documento…");
    try {
      const document = await uploadKnowledgeDocument(file);
      setAttachment({ id: document.id, name: document.name });
      setKnowledgeMessage(null);
    } catch (caught) {
      setKnowledgeMessage(
        caught instanceof Error ? caught.message : "No se pudo importar el documento.",
      );
    }
  };

  const handleMemorySave = async () => {
    const content = memoryDraft.trim();
    if (!content || memoryBusy) return;
    setMemoryBusy(true);
    setMemoryError(null);
    try {
      const entry = await saveMemoryEntry({ content });
      setMemoryEntries((current) => [...current, entry]);
      setMemoryDraft("");
    } catch (caught) {
      setMemoryError(
        caught instanceof Error ? caught.message : "No se pudo guardar el dato.",
      );
    } finally {
      setMemoryBusy(false);
    }
  };

  const handleMemoryDelete = async (entryId: string) => {
    setMemoryError(null);
    // Optimistic removal keeps the panel responsive; on failure the entry is
    // restored by reloading, so the UI never claims a deletion that didn't happen.
    const previous = memoryEntries;
    setMemoryEntries((current) => current.filter((item) => item.id !== entryId));
    try {
      await deleteMemoryEntry(entryId);
    } catch (caught) {
      setMemoryEntries(previous);
      setMemoryError(
        caught instanceof Error ? caught.message : "No se pudo borrar el dato.",
      );
    }
  };

  const submitPrompt = async (prompt: string) => {
    const clean = prompt.trim();
    if (!clean || loading) return;

    const nextUiMessage: UiMessage = {
      id: makeId("user"),
      role: "user",
      content: clean,
      attachmentName: attachment?.name,
    };
    const nextMessages = [...messages, nextUiMessage];
    shouldFollowRef.current = true;
    setShowScrollToLatest(false);
    setMessages(nextMessages);
    setDraft("");
    setAttachment(null);

    await runRequest(
      nextMessages
        .slice(-20)
        .map(({ role, content }) => ({ role, content })),
      mode,
      sport,
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
  // Only the local (Ollama) deployment runs inference on this machine, so its
  // CPU is only worth showing there. The cloud core reports zero threads.
  const localResources = Boolean(status && status.quick_threads > 0);
  const cpuHigh = Boolean(status && status.snapshot.cpu_percent >= 50);

  return (
    <main className="orion-shell">
      <aside className="orion-sidebar">
        <div className="brand-lockup">
          <OrionMark className="brand-mark" />
          <div>
            <p className="brand-name">ORION</p>
            <p className="brand-caption">Inteligencia deportiva</p>
          </div>
        </div>

        <div className="side-stack">
          <p className="side-label">Estado</p>
          <div className="status-line">
            <span className="engine-status">
              <span className={`dot ${online ? "online" : "offline"}`} />
              Orion
            </span>
            <span className="status-value">
              {online ? "Activo" : "Desconectado"}
            </span>
          </div>
          {/* Local machine stats (RAM, CPU, Ollama) belonged to the version
              that ran inference on the user's PC. Orion runs in the cloud, so
              they described a product this no longer is. */}
          {localResources ? (
            <div className="status-line">
              <span>CPU de tu equipo</span>
              <span className={`status-value ${cpuHigh ? "resource-hot" : ""}`}>
                {status ? `${status.snapshot.cpu_percent.toFixed(0)}%` : "—"}
              </span>
            </div>
          ) : null}
        </div>

        <div className="side-stack">
          <p className="side-label">Memoria</p>
          <p className="side-hint">
            Solo lo que guardes acá. Orion no guarda nada por su cuenta.
          </p>
          <div className="memory-compose">
            <textarea
              value={memoryDraft}
              onChange={(event) => setMemoryDraft(event.target.value)}
              placeholder="Ej.: Dirijo el plantel sub-20 de Atlético Mineiro"
              rows={2}
              maxLength={1000}
              aria-label="Dato para que Orion recuerde"
            />
            <button
              type="button"
              className="memory-save"
              onClick={handleMemorySave}
              disabled={memoryBusy || !memoryDraft.trim()}
            >
              {memoryBusy ? "Guardando…" : "Recordar"}
            </button>
          </div>
          {memoryError ? (
            <p className="memory-error" role="alert">
              {memoryError}
            </p>
          ) : null}
          {memoryEntries.length === 0 ? (
            <p className="memory-empty">Todavía no guardaste nada.</p>
          ) : (
            <ul className="memory-list">
              {memoryEntries.map((entry) => (
                <li key={entry.id} className="memory-item">
                  <span className="memory-text">{entry.content}</span>
                  <button
                    type="button"
                    className="memory-remove"
                    onClick={() => handleMemoryDelete(entry.id)}
                    aria-label={`Olvidar: ${entry.content}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <p className="side-note">
          Las conversaciones se pierden al recargar la página. Lo que guardes
          en memoria y los documentos que subas sí quedan guardados.
        </p>
      </aside>

      <section className="orion-main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Conversación</p>
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
              <p className="eyebrow">Inteligencia deportiva</p>
              <h2>Tu criterio, amplificado.</h2>
              <p>
                Preguntá con tus palabras. Orion busca lo que haga falta,
                revisa las fuentes antes de responder y te dice qué no puede
                confirmar.
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
            <div
              ref={messagesRef}
              className="messages"
              aria-live="polite"
              onScroll={handleConversationScroll}
              onWheel={handleConversationWheel}
            >
              {messages.map((message) => {
                const metrics = [
                  message.mode === "deep" ? "Profundo" : "Rápido",
                  message.sport ? SPORT_LABELS[message.sport] : null,
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
                        {message.chart ? <ChartPreview chart={message.chart} /> : null}
                        {message.content ? (
                          message.streaming ? (
                            <div className="streaming-content">
                              {renderMarkdown(message.content)}
                            </div>
                          ) : (
                            renderMarkdown(message.content)
                          )
                        ) : (
                          <span className="stream-placeholder">Preparando respuesta…</span>
                        )}
                      </div>
                    ) : (
                      <div className="message-content plain-content">
                        {message.attachmentName ? (
                          <div className="message-attachment">
                            <span className="knowledge-file-icon" aria-hidden="true">▤</span>
                            <span>{message.attachmentName}</span>
                          </div>
                        ) : null}
                        <div>{message.content}</div>
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
                  <small>La respuesta aparece progresivamente · prioridad reducida activa.</small>
                </div>
              ) : null}
            </div>
          )}

          <div className="composer-wrap">
            {showScrollToLatest ? (
              <button
                type="button"
                className="scroll-to-latest"
                onClick={scrollToLatest}
              >
                ↓ Volver al final
              </button>
            ) : null}

            {warning ? (
              <div className="resource-warning" role="alert">
                <strong>La computadora está exigida.</strong> {warning.detail.message}
                <div>{warning.detail.reasons.join(" · ")}</div>
                <div className="warning-actions">
                  <button
                    type="button"
                    className="small-button primary"
                    onClick={() =>
                      void runRequest(
                        warning.messages,
                        "quick",
                        warning.sport,
                        false,
                      )
                    }
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
                        warning.sport,
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
              {attachment ? (
                <div className="knowledge-attachment">
                  <span className="knowledge-file-icon" aria-hidden="true">▤</span>
                  <span className="knowledge-file-name" title={attachment.name}>
                    {attachment.name}
                  </span>
                  <button
                    type="button"
                    className="knowledge-remove"
                    aria-label={`Quitar ${attachment.name} de Orion`}
                    title="Quitar el documento de Orion"
                    onClick={removeAttachment}
                    disabled={attachmentBusy}
                  >
                    {attachmentBusy ? "…" : "×"}
                  </button>
                </div>
              ) : null}
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Escribile a Orion…"
                aria-label="Mensaje para Orion"
                maxLength={20_000}
              />
              <div className="composer-controls">
                <label className="knowledge-upload" title="Importar documento local">
                  <span aria-hidden="true">+</span>
                  <span>Documento</span>
                  <input
                    type="file"
                    accept=".txt,.md,.csv,.json,text/plain,text/csv,application/json"
                    onChange={handleKnowledgeUpload}
                  />
                </label>
                <details ref={sportPickerRef} className="sport-picker">
                  <summary aria-label={`Deporte seleccionado: ${SPORT_LABELS[sport]}`}>
                    <span aria-hidden="true">
                      {SPORT_OPTIONS.find((option) => option.value === sport)?.icon}
                    </span>
                    {SPORT_LABELS[sport]}
                    <span className="sport-picker-chevron" aria-hidden="true">⌃</span>
                  </summary>
                  <div className="sport-menu" role="listbox" aria-label="Elegir deporte">
                    {SPORT_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        role="option"
                        aria-selected={sport === option.value}
                        className={sport === option.value ? "active" : ""}
                        onClick={() => selectSport(option.value)}
                      >
                        <span aria-hidden="true">{option.icon}</span>
                        {option.label}
                      </button>
                    ))}
                  </div>
                </details>

                <button
                  type={loading ? "button" : "submit"}
                  className={`send-button ${loading ? "stop" : ""}`}
                  disabled={!loading && !draft.trim()}
                  aria-label={loading ? "Detener respuesta" : "Enviar mensaje"}
                  title={loading ? "Detener respuesta" : "Enviar mensaje"}
                  onClick={loading ? stopRequest : undefined}
                >
                  {loading ? (
                    <span className="stop-square" aria-hidden="true" />
                  ) : (
                    <span aria-hidden="true">↑</span>
                  )}
                </button>
              </div>
            </form>
            {knowledgeMessage ? <p className="knowledge-message" role="status">{knowledgeMessage}</p> : null}
            <p className="composer-note">
              Contexto: {SPORT_LABELS[sport]} · Enter para enviar · Shift + Enter para una nueva línea
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
