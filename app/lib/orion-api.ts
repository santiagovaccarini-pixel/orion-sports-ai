export type RequestedMode = "auto" | "quick" | "deep";
export type SelectedMode = "quick" | "deep";
export type ModelProvider = "ollama" | "cloudflare" | "cerebras" | "groq";
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
  model_provider: ModelProvider;
  model_provider_online: boolean;
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

export type MemoryEntry = {
  id: string;
  content: string;
  category: string;
  created_at: string;
  updated_at: string;
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

export type DiagnosticSource = {
  source_id: string;
  title: string;
  url: string;
  domain: string;
  excerpt: string;
};

export type DiagnosticSearch = {
  round: number;
  query: string;
  duration_ms: number;
  raw_results: DiagnosticSource[];
};

export type DiagnosticReview = {
  round: number;
  fallback: boolean;
  duration_ms: number;
  error: string | null;
  sufficient: boolean;
  relevant_source_ids: string[];
  discarded_source_ids: string[];
  missing_information: string[];
  follow_up_web_query: string | null;
  needs_clarification: boolean;
  clarifying_question: string | null;
  resolved_scope: string | null;
  source_catalog: DiagnosticSource[];
};

export type DiagnosticPlan = {
  objective: string;
  entities: string[];
  constraints: string[];
  references: string[];
  information_needed: string[];
  ambiguities: string[];
  evidence_policy: "model_knowledge" | "external" | "local" | "mixed" | null;
  use_web: boolean;
  use_local_data: boolean;
  use_calculator: boolean;
  use_chart: boolean;
  needs_clarification: boolean;
  clarifying_question: string | null;
  web_query: string | null;
  local_document_names: string[];
  recommended_mode: SelectedMode | null;
};

export type DiagnosticModelCall = {
  stage: string;
  model: string | null;
  endpoint: string | null;
  reasoning_effort: string | null;
  finish_reason: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  reasoning_tokens: number | null;
  duration_ms: number | null;
};

export type DiagnosticTrace = {
  trace_id: string;
  created_at: string;
  completed_at: string | null;
  status: string;
  question: string;
  sport: string;
  requested_mode: string;
  model: string | null;
  plan: DiagnosticPlan | null;
  plan_fallback: boolean;
  plan_error: string | null;
  local_evidence: Array<{
    source_id: string;
    document_name: string;
    chunk_index: number | null;
    truncated: boolean;
    excerpt: string;
  }>;
  searches: DiagnosticSearch[];
  reviews: DiagnosticReview[];
  model_calls: DiagnosticModelCall[];
  guard_events: Array<{ event: string; detail: string }>;
  prompt_metadata: {
    template_version: string;
    system_prompt_sha256: string;
    system_prompt_characters: number;
    message_count: number;
    message_characters: number;
    full_prompt_recorded: boolean;
  } | null;
  timings_ms: Record<string, number>;
  final_answer: string | null;
  error: string | null;
  privacy: {
    persistence: string;
    max_traces: number;
    hidden_chain_of_thought_recorded: boolean;
    credentials_recorded: boolean;
    full_prompt_recorded?: boolean;
  };
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
  trace_id?: string | null;
};

export type ChatStreamDone = {
  type: "done";
  total_duration_ms: number | null;
  load_duration_ms: number | null;
  prompt_eval_duration_ms: number | null;
  eval_duration_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  reasoning_tokens?: number | null;
  finish_reason?: string | null;
  reasoning_effort?: string | null;
  endpoint?: string | null;
  tokens_per_second: number | null;
  thread_limit: number;
};

type ChatStreamContent = {
  type: "content";
  content: string;
};

/** Progress marker from the reasoning pipeline (planning, searching, reading,
 * reviewing). The core has always sent these; the interface used to drop them
 * on the floor, leaving the reader a spinner for 30 seconds while the answer
 * was already narrating exactly what it was doing. */
type ChatStreamStage = {
  type: "stage";
  stage: string;
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
  | ChatStreamStage
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

// Same-origin proxy (app/api/orion/[...path]/route.ts). The API key is attached
// server-side there and never reaches the browser: anything sent from here is
// readable by any visitor, so no credential may live in this file.
const API_BASE = "/api/orion";

const API_HEADERS = {
  "Content-Type": "application/json",
};

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new OrionApiError(
      response.status,
      payload?.detail ?? "Respuesta inválida del núcleo de Orion.",
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

export async function getLatestDiagnosticTrace(
  signal?: AbortSignal,
): Promise<DiagnosticTrace> {
  const response = await fetch(`${API_BASE}/diagnostics/traces/latest`, {
    cache: "no-store",
    signal,
    headers: API_HEADERS,
  });
  return parseResponse<DiagnosticTrace>(response);
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

export async function listMemoryEntries(
  signal?: AbortSignal,
): Promise<MemoryEntry[]> {
  const response = await fetch(`${API_BASE}/memory/entries`, {
    cache: "no-store",
    signal,
    headers: API_HEADERS,
  });
  return parseResponse<MemoryEntry[]>(response);
}

export async function saveMemoryEntry(input: {
  content: string;
  category?: string;
}): Promise<MemoryEntry> {
  const response = await fetch(`${API_BASE}/memory/entries`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify({
      content: input.content,
      category: input.category?.trim() || "general",
    }),
  });
  return parseResponse<MemoryEntry>(response);
}

export async function deleteMemoryEntry(entryId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/memory/entries/${encodeURIComponent(entryId)}`,
    { method: "DELETE", headers: API_HEADERS },
  );
  await parseResponse<{ status: string }>(response);
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

export type ConversationSummary = {
  id: string;
  title: string;
  sport: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ConversationDetail = {
  id: string;
  title: string;
  sport: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
};

/** Threads stored by the core so a reload gets the chat back.
 *
 * Every call here is a convenience around the conversation: a failure means
 * "not saving right now", never a broken chat, so callers swallow errors. */
export async function listConversations(
  signal?: AbortSignal,
): Promise<ConversationSummary[]> {
  const response = await fetch(`${API_BASE}/conversations`, {
    cache: "no-store",
    signal,
    headers: API_HEADERS,
  });
  return parseResponse<ConversationSummary[]>(response);
}

export async function createConversation(input: {
  title: string;
  sport: Sport;
}): Promise<ConversationSummary> {
  const response = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify(input),
  });
  return parseResponse<ConversationSummary>(response);
}

export async function getConversation(
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationDetail> {
  const response = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}`,
    { cache: "no-store", signal, headers: API_HEADERS },
  );
  return parseResponse<ConversationDetail>(response);
}

export async function appendConversationMessages(
  conversationId: string,
  messages: ChatMessage[],
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/messages`,
    { method: "POST", headers: API_HEADERS, body: JSON.stringify({ messages }) },
  );
  await parseResponse<{ status: string }>(response);
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE", headers: API_HEADERS },
  );
  await parseResponse<{ status: string }>(response);
}

export type MemorySuggestion = { content: string; reason: string };

/** What Orion would remember, before anything is written.
 *
 * Nothing is saved by this call: the user reads the exact sentence, edits it if
 * the wording is off, and only then does it become memory. */
export async function suggestMemories(input: {
  messages: ChatMessage[];
  answer: string;
}): Promise<MemorySuggestion[]> {
  const response = await fetch(`${API_BASE}/memory/suggestions`, {
    method: "POST",
    headers: API_HEADERS,
    body: JSON.stringify({ messages: input.messages, answer: input.answer }),
  });
  if (!response.ok) return [];
  return parseResponse<MemorySuggestion[]>(response);
}

export async function deleteKnowledgeDocument(documentId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/knowledge/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE", headers: API_HEADERS },
  );
  await parseResponse<{ status: string }>(response);
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
    onStage?: (stage: string) => void;
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
      throw new Error("Orion recibió un fragmento inválido del núcleo.");
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
    } else if (event.type === "stage") {
      handlers.onStage?.(event.stage);
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
