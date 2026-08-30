/**
 * Server-side proxy to the Orion Core backend.
 *
 * The browser talks only to this same-origin route; the API key is read here,
 * on the server, from a variable WITHOUT the NEXT_PUBLIC_ prefix so it is never
 * inlined into the client bundle. Sending the key to the browser (as
 * NEXT_PUBLIC_ORION_API_KEY did) made it readable by any visitor via devtools,
 * which in turn exposed every key-gated endpoint - including the diagnostics
 * trace log containing other people's conversations.
 */

const DEFAULT_BACKEND = "http://127.0.0.1:8765/api/v1";

// Only these backend endpoints are reachable through the proxy. Without an
// allowlist this route would forward anything the browser asked for, turning a
// key-protected backend into an open one under a different hostname.
const PROXYABLE_PATHS = new Set([
  "status",
  "chat",
  "chat/stream",
  "knowledge/documents",
  "memory/entries",
  "memory/suggestions",
  "diagnostics/traces/latest",
]);

function backendBase(): string {
  const configured = process.env.ORION_API_URL ?? DEFAULT_BACKEND;
  return configured.replace(/\/$/, "");
}

function jsonError(status: number, code: string, message: string): Response {
  return Response.json({ detail: { code, message } }, { status });
}

function isProxyable(path: string): boolean {
  if (PROXYABLE_PATHS.has(path)) return true;
  // Ids are dynamic, so match their shape rather than an exact value.
  if (/^diagnostics\/traces\/[A-Za-z0-9_-]{1,128}$/.test(path)) return true;
  if (/^knowledge\/documents\/[A-Za-z0-9_-]{1,128}$/.test(path)) return true;
  return /^memory\/entries\/[A-Za-z0-9_-]{1,128}$/.test(path);
}

async function proxy(request: Request, path: string[]): Promise<Response> {
  const target = path.join("/");
  if (!isProxyable(target)) {
    return jsonError(404, "not_found", "Ruta no disponible.");
  }

  const apiKey = process.env.ORION_API_KEY;
  if (!apiKey) {
    return jsonError(
      503,
      "server_misconfigured",
      "La interfaz de Orion no tiene configurada la clave del núcleo.",
    );
  }

  const headers = new Headers({ "X-Orion-Api-Key": apiKey });
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  let upstream: Response;
  try {
    upstream = await fetch(`${backendBase()}/${target}`, {
      method: request.method,
      headers,
      body:
        request.method === "GET" || request.method === "DELETE"
          ? undefined
          : await request.text(),
      signal: request.signal,
    });
  } catch {
    return jsonError(
      503,
      "core_unreachable",
      "No se pudo contactar al núcleo de Orion.",
    );
  }

  // Streamed straight through so /chat/stream still arrives progressively
  // rather than being buffered here and delivered all at once.
  const responseHeaders = new Headers();
  const upstreamType = upstream.headers.get("content-type");
  if (upstreamType) responseHeaders.set("content-type", upstreamType);
  const retryAfter = upstream.headers.get("retry-after");
  if (retryAfter) responseHeaders.set("retry-after", retryAfter);
  responseHeaders.set("cache-control", "no-store");

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function POST(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await context.params;
  return proxy(request, path);
}

export async function DELETE(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await context.params;
  return proxy(request, path);
}
