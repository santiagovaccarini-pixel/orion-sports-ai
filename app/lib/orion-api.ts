export type RequestedMode = "auto" | "quick" | "deep";
export type SelectedMode = "quick" | "deep";
export type Sport =
  | "general"
  | "football"
  | "basketball"
  | "volleyball"
  | "rugby"
  | "tennis"
  | "athletics"
  | "swimming"
  | "cycling";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type SystemSnapshot = {
  cpu_percent: number;
  memory_available_gb: number;
  memory_total_gb: number;
  battery_percent: number | null;
  plugged_in: boolean | null;
};

export type OrionStatus = {
  service: "online";
  version: string;
  ollama_online: boolean;
  installed_models: string[];
  quick_model: string;
  deep_model: string;
  quick_threads: number;
  deep_threads: number;
  loaded_models: string[];
  snapshot: SystemSnapshot;
  memory_enabled: false;
};

export type KnowledgeDocument = {
  id: string;
  name: string;
  characters: number;
};

export type ChartPoint = { label: string; value: number };
export type OrionChart = {
  type: "bar";
  title: string;
  unit: string;
  source: string;
  metric: string;
  points: ChartPoint[];
};

export type ChatResult = {
  content: string;
  sport: Sport;
  selected_mode: SelectedMode;
  recommended_mode: SelectedMode;
  recommendation_reason: string;
  model: string;
  total_duration_ms: number | null;
  load_duration_ms: number | null;
  prompt_eval_duration_ms: number | null;
  eval_duration_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  tokens_per_second: number | null;
  thread_limit: number;
};

export type ChatStreamMeta = {
  type: "meta";
  sport: Sport;
  selected_mode: SelectedMode;
  recommended_mode: SelectedMode;
  recommendation_reason: string;
  model: string;
};

export type ChatStreamDone = {
  type: "done";
  total_duration_ms: number | null;
  load_duration_ms: number | null;
  prompt_eval_duration_ms: number | null;
  eval_duration_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  tokens_per_second: number | null;
  thread_limit: number;
};

type ChatStreamContent = {
  type: "content";
  content: string;
};

type ChatStreamError = {
  type: "error";
  code?: string;
  message: string;
};

type ChatStreamChart = { type: "chart"; chart: OrionChart };

type ChatStreamEvent =
  | ChatStreamMeta
  | ChatStreamContent
  | ChatStreamDone
  | ChatStreamChart
  | ChatStreamError;

export type ChatStreamResult = {
  meta: ChatStreamMeta;
  done: ChatStreamDone;
};

export type ResourceWarning = {
  code: "resource_confirmation_required";
  message: string;
  reasons: string[];
  selected_mode: SelectedMode;
  recommended_mode: SelectedMode;
  snapshot: SystemSnapshot;
};

type ApiErrorDetail =
  | ResourceWarning
  | { code?: string; message?: string }
  | string;

export class OrionApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: ApiErrorDetail,
  ) {
    super(
      typeof detail === "string"
        ? detail
        : detail.message ?? "Orion no pudo completar la solicitud.",
    );
  }
}

const API_BASE = (
  process.env.NEXT_PUBLIC_ORION_API_URL ??
  "http://127.0.0.1:8765/api/v1"
).replace(/\/$/, "");

const API_HEADERS = {
  "Content-Type": "application/json",
  ...(process.env.NEXT_PUBLIC_ORION_API_KEY
    ? { "X-Orion-Api-Key": process.env.NEXT_PUBLIC_ORION_API_KEY }
    : {}),
};

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new OrionApiError(
      response.status,
      payload?.detail ?? "Respuesta inválida del núcleo local.",
    );
  }
  return payload as T;
}

export async function getOrionStatus(signal?: AbortSignal): Promise<OrionStatus> {
  const response = await fetch(`${API_BASE}/status`, {
    cache: "no-store",
    signal,
    headers: API_HEADERS,
  });
  return parseResponse<OrionStatus>(response);
}

export async function sendChat(input: {
  messages: ChatMessage[];
  mode: RequestedMode;
  sport: Sport;
  allowBusy?: boolean;
}): Promise<ChatResult> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify({
      messages: input.messages,
      mode: input.mode,
      sport: input.sport,
      allow_busy: input.allowBusy ?? false,
    }),
  });
  return parseResponse<ChatResult>(response);
}

export async function uploadKnowledgeDocument(file: File): Promise<KnowledgeDocument> {
  const content = await file.text();
  const response = await fetch(`${API_BASE}/knowledge/documents`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify({ name: file.name, content }),
  });
  return parseResponse<KnowledgeDocument>(response);
}

export async function sendChatStream(
  input: {
    messages: ChatMessage[];
    mode: RequestedMode;
    sport: Sport;
    allowBusy?: boolean;
  },
  handlers: {
    onMeta: (event: ChatStreamMeta) => void;
    onContent: (content: string) => void;
    onChart: (chart: OrionChart) => void;
  },
  signal?: AbortSignal,
): Promise<ChatStreamResult> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify({
      messages: input.messages,
      mode: input.mode,
      sport: input.sport,
      allow_busy: input.allowBusy ?? false,
    }),
    signal,
  });

  if (!response.ok) {
    await parseResponse<never>(response);
  }
  if (!response.body) {
    throw new Error("El navegador no permitió recibir la respuesta progresiva.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let meta: ChatStreamMeta | null = null;
  let done: ChatStreamDone | null = null;

  const handleLine = (line: string) => {
    if (!line.trim()) return;
    let event: ChatStreamEvent;
    try {
      event = JSON.parse(line) as ChatStreamEvent;
    } catch {
      throw new Error("Orion recibió un fragmento inválido del núcleo local.");
    }

    if (event.type === "meta") {
      meta = event;
      handlers.onMeta(event);
    } else if (event.type === "content") {
      handlers.onContent(event.content);
    } else if (event.type === "done") {
      done = event;
    } else if (event.type === "chart") {
      handlers.onChart(event.chart);
    } else if (event.type === "error") {
      throw new OrionApiError(503, event);
    }
  };

  while (true) {
    const { value, done: readerDone } = await reader.read();
    buffer += decoder.decode(value, { stream: !readerDone });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) handleLine(line);
    if (readerDone) break;
  }
  if (buffer.trim()) handleLine(buffer);

  if (!meta || !done) {
    throw new Error("Orion cerró la respuesta antes de terminar.");
  }
  return { meta, done };
}
