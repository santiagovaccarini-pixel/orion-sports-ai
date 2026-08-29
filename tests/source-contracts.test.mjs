import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("keeps the shell fixed and scrolls only the conversation", async () => {
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  assert.match(css, /\.orion-shell\s*\{[^}]*height:\s*100dvh/s);
  assert.match(css, /\.orion-shell\s*\{[^}]*overflow:\s*hidden/s);
  assert.match(css, /\.messages\s*\{[^}]*overflow-y:\s*auto/s);
  assert.match(css, /\.orion-sidebar\s*\{[^}]*overflow:\s*hidden/s);
  assert.match(css, /\.composer-wrap\s*\{[^}]*flex:\s*0 0 auto/s);
  assert.match(css, /\.composer textarea\s*\{[^}]*min-height:\s*3\.75rem/s);
  assert.match(css, /\.chat-stage\s*\{[^}]*width:\s*min\(100%,\s*88rem\)/s);
  assert.match(css, /\.message\s*\{[^}]*max-width:\s*100%/s);
  assert.match(css, /\.composer:focus-within\s*\{/s);
});

test("renders safe Markdown and visible performance metrics", async () => {
  const component = await readFile(
    new URL("app/components/orion-console.tsx", root),
    "utf8",
  );
  assert.match(component, /ReactMarkdown/);
  assert.match(component, /remarkGfm/);
  assert.match(component, /normalizeCompletedMarkdown/);
  assert.match(component, /renderMarkdown/);
  assert.match(component, /replace\(\/```\(\?:text\|txt\)\?/);
  assert.match(component, /cells\[0\]\.map/);
  assert.match(component, /ChartPreview/);
  assert.match(component, /onChart/);
  assert.match(component, /message\.streaming/);
  assert.match(component, /pico CPU/);
  assert.match(component, /tokensPerSecond/);
  assert.match(component, /threadLimit/);
  assert.match(component, /primer texto/);
  assert.match(component, /Detener respuesta/);
  assert.match(component, /stop-square/);
  assert.match(component, /AbortController/);
});

test("lets the reader leave auto-scroll and return to the latest message", async () => {
  const component = await readFile(
    new URL("app/components/orion-console.tsx", root),
    "utf8",
  );
  assert.match(component, /shouldFollowRef/);
  assert.match(component, /distanceFromBottom <= 96/);
  assert.match(component, /movedUp = container\.scrollTop < previousScrollTopRef\.current/);
  assert.match(component, /container\.scrollTop = container\.scrollHeight/);
  assert.match(component, /if \(!shouldFollowRef\.current\)/);
  assert.match(component, /autoScrollFrameRef/);
  assert.match(component, /onScroll=\{handleConversationScroll\}/);
  assert.match(component, /onWheel=\{handleConversationWheel\}/);
  assert.match(component, /Volver al final/);
});

test("sends the selected local sport context to the core", async () => {
  const component = await readFile(
    new URL("app/components/orion-console.tsx", root),
    "utf8",
  );
  const client = await readFile(new URL("app/lib/orion-api.ts", root), "utf8");
  for (const sport of [
    "General",
    "Fútbol",
    "Básquet",
    "Vóley",
    "Rugby",
    "Tenis",
    "Atletismo",
    "Natación",
    "Ciclismo",
  ]) {
    assert.match(component, new RegExp(sport));
  }
  assert.match(component, /sport-picker/);
  assert.match(client, /sport:\s*input\.sport/);
  assert.match(component, /uploadKnowledgeDocument/);
  assert.match(component, /accept="\.txt,\.md,\.csv,\.json/);
  assert.match(component, /knowledge-attachment/);
  assert.match(component, /knowledge-remove/);
  assert.match(component, /attachmentName/);
  assert.match(component, /message-attachment/);
});

test("streams chat fragments instead of waiting for one large JSON response", async () => {
  const client = await readFile(new URL("app/lib/orion-api.ts", root), "utf8");
  assert.match(client, /\/chat\/stream/);
  assert.match(client, /response\.body\.getReader\(\)/);
  assert.match(client, /TextDecoder/);
  assert.match(client, /onContent/);
});

test("never ships the core API key to the browser", async () => {
  // NEXT_PUBLIC_* values are inlined into the client bundle at build time, so
  // reading a key under that prefix anywhere in app/ would hand it to every
  // visitor. The key belongs only in the server-side proxy route.
  const client = await readFile(new URL("app/lib/orion-api.ts", root), "utf8");
  const console_ = await readFile(
    new URL("app/components/orion-console.tsx", root),
    "utf8",
  );
  const proxy = await readFile(
    new URL("app/api/orion/[...path]/route.ts", root),
    "utf8",
  );

  assert.doesNotMatch(client, /NEXT_PUBLIC_[A-Z_]*KEY/);
  assert.doesNotMatch(console_, /NEXT_PUBLIC_[A-Z_]*KEY/);
  assert.doesNotMatch(client, /X-Orion-Api-Key/);
  // The browser must call the same-origin proxy, not the core directly.
  assert.match(client, /const API_BASE = "\/api\/orion"/);
  // ...and the proxy is what attaches the credential, server-side.
  assert.match(proxy, /process\.env\.ORION_API_KEY/);
  assert.match(proxy, /X-Orion-Api-Key/);
});

test("the core proxy only forwards an explicit allowlist of routes", async () => {
  const proxy = await readFile(
    new URL("app/api/orion/[...path]/route.ts", root),
    "utf8",
  );
  assert.match(proxy, /PROXYABLE_PATHS/);
  assert.match(proxy, /isProxyable/);
  for (const path of ["status", "chat", "chat\\/stream", "knowledge\\/documents"]) {
    assert.match(proxy, new RegExp(`"${path}"`));
  }
});

test("keeps browser scroll anchoring disabled while messages grow", async () => {
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  assert.match(css, /overflow-anchor:\s*none/);
});

test("wraps response content and table cells inside the message width", async () => {
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  assert.match(css, /\.message-content,[\s\S]*overflow-wrap:\s*anywhere/);
  assert.match(css, /\.markdown-content th,[\s\S]*white-space:\s*normal/);
});
