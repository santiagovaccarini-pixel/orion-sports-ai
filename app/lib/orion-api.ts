export type RequestedMode = "auto" | "quick" | "deep";
export type SelectedMode = "quick" | "deep";

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
  loaded_models: string[];
  snapshot: SystemSnapshot;
  memory_enabled: false;
};

export type ChatResult = {
  content: string;
  selected_mode: SelectedMode;
  recommended_mode: SelectedMode;
  recommendation_reason: string;
  model: string;
  total_duration_ms: number | null;
  tokens_per_second: number | null;
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
  });
  return parseResponse<OrionStatus>(response);
}

export async function sendChat(input: {
  messages: ChatMessage[];
  mode: RequestedMode;
  allowBusy?: boolean;
}): Promise<ChatResult> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: input.messages,
      mode: input.mode,
      allow_busy: input.allowBusy ?? false,
    }),
  });
  return parseResponse<ChatResult>(response);
}
