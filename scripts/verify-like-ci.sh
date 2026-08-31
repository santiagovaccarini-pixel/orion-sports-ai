#!/usr/bin/env bash
# Run every check that gates a deploy, exactly as GitHub Actions runs it.
#
# Render waits on checksPass, so any red mark on any workflow silently holds
# back every commit - the service keeps serving the old build and the endpoints
# the new one adds answer 404. Two separate checks did that for a full day:
#
#   * the backend suite, because it was run here with pytest and whatever
#     ORION_* variables the shell carried, while CI runs unittest with
#     ORION_MODEL_PROVIDER=ollama;
#   * the frontend lint, because it was simply never run locally at all.
#
# So this runs both, the same way, in one command.
#
# Usage: bash scripts/verify-like-ci.sh [--backend-only]
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON="python"

echo "== backend: compile =="
"$PYTHON" -m compileall -q backend/app backend/tests

echo "== backend: unit tests =="
# The variables come from .github/workflows/backend-tests.yml. Keep them in step
# with that file: their whole purpose is that this command and CI agree.
ORION_MODEL_PROVIDER=ollama \
ORION_WEB_ENABLED=false \
ORION_SEMANTIC_ORCHESTRATION=false \
  "$PYTHON" -m unittest discover -s backend/tests -p "test_*.py"

[ "${1:-}" = "--backend-only" ] && exit 0

echo "== frontend: audit (lo que se publica) =="
# Blocking. A high advisory in something that ships to a browser is a real hole
# in Orion; there is no reading of that which can wait.
npm audit --omit=dev --audit-level=high

echo "== frontend: audit (herramientas de desarrollo) =="
# Reported, not enforced - the same choice the workflow makes, and it has to
# stay the same choice or this script starts refusing work CI would accept.
# These packages build and serve the app on one developer's machine and never
# reach a user, so a new advisory here is a queue item, not a stop order.
npm audit --audit-level=high || true

echo "== frontend: lint =="
npm run lint

echo "== frontend: build and tests =="
npm test
