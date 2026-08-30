import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const launcher = new URL("../scripts/windows/Start-Orion.ps1", import.meta.url);

test("local backend remains hidden and logged only in explicit legacy mode", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /\[switch\]\$LocalLegacy/);
  assert.match(source, /-WindowStyle Hidden/);
  assert.doesNotMatch(source, /-WindowStyle Minimized/);
  assert.match(source, /-RedirectStandardOutput \$BackendOutputLog/);
  assert.match(source, /-RedirectStandardError \$BackendErrorLog/);
  assert.match(source, /api\/v1\/health/);
  assert.match(source, /Wait-OrionBackend -Process \$BackendProcess/);
  assert.match(source, /ORION_MODEL_PROVIDER\s*=\s*"ollama"/);
});

test("always stops the hidden legacy backend when that mode finishes", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /finally\s*\{[\s\S]*Stop-Process -Id \$BackendProcess\.Id/);
  assert.match(source, /\.orion-runtime/);
});

test("disables interactive tunnel shortcuts in the launcher", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /\$env:CI\s*=\s*"1"/);
  assert.match(source, /npm exec vite -- --host 127\.0\.0\.1/);
});

test("Orion Cloud is the default launch path without requiring -Cloud", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /if \(-not \$LocalLegacy\)/);
  assert.match(source, /https:\/\/orion-core-prototype\.onrender\.com/);
  assert.match(source, /\$env:ORION_API_URL/);
  // The launcher must not name the engine company: the provider is chosen at
  // deploy time, and this line announced Cloudflare while Orion ran on Cerebras.
  assert.match(source, /Motor activo: el del nucleo cloud/);
  assert.doesNotMatch(source, /Motor activo:.*Cloudflare/);
  assert.match(source, /if \(-not \$LocalLegacy\)[\s\S]*Start-OrionFrontend[\s\S]*exit 0/);
});

test("the launcher never exposes the API key to the browser bundle", async () => {
  const source = await readFile(launcher, "utf8");

  // NEXT_PUBLIC_* values are inlined into the client bundle, so the key must
  // never be set under that prefix; the server-side proxy reads it instead.
  assert.doesNotMatch(source, /NEXT_PUBLIC_[A-Z_]*KEY/);
});

test("cloud default refuses to start without the core API key", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /if \(-not \$env:ORION_API_KEY\)/);
  assert.match(source, /Falta ORION_API_KEY/);
});

test("old -Cloud parameter remains compatible but cannot be combined with legacy", async () => {
  const source = await readFile(launcher, "utf8");

  assert.match(source, /\[switch\]\$Cloud/);
  assert.match(source, /if \(\$Cloud -and \$LocalLegacy\)/);
});
