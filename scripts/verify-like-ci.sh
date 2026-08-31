#!/usr/bin/env bash
# Run the backend suite exactly as GitHub Actions runs it.
#
# A green local run used to mean nothing: this project's tests were usually run
# with pytest and whatever ORION_* variables the shell happened to carry, while
# CI runs unittest with ORION_MODEL_PROVIDER=ollama. One assertion that read the
# ambient provider passed here and failed there - and because Render waits on
# checksPass, that red mark silently held back every deploy for a day while the
# endpoints it added answered 404 in production.
#
# Usage: bash scripts/verify-like-ci.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON="python"

"$PYTHON" -m compileall -q backend/app backend/tests

# The variables come from .github/workflows/backend-tests.yml. Keep them in step
# with that file: their whole purpose is that this command and CI agree.
ORION_MODEL_PROVIDER=ollama \
ORION_WEB_ENABLED=false \
ORION_SEMANTIC_ORCHESTRATION=false \
  "$PYTHON" -m unittest discover -s backend/tests -p "test_*.py"
