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
});

test("renders safe Markdown and visible performance metrics", async () => {
  const component = await readFile(
    new URL("app/components/orion-console.tsx", root),
    "utf8",
  );
  assert.match(component, /ReactMarkdown/);
  assert.match(component, /remarkGfm/);
  assert.match(component, /pico CPU/);
  assert.match(component, /tokensPerSecond/);
  assert.match(component, /threadLimit/);
  assert.match(component, /primer texto/);
  assert.match(component, /Detener/);
  assert.match(component, /AbortController/);
});

test("streams chat fragments instead of waiting for one large JSON response", async () => {
  const client = await readFile(new URL("app/lib/orion-api.ts", root), "utf8");
  assert.match(client, /\/chat\/stream/);
  assert.match(client, /response\.body\.getReader\(\)/);
  assert.match(client, /TextDecoder/);
  assert.match(client, /onContent/);
});
