"""Grade Orion's code by running it, not by looking for words in it.

Every other battery greps the answer for expected phrases. That works for an
invented club or a wrong year and fails completely on code: a function can
contain every keyword the rubric asks for and still return the wrong number, and
the same defect written differently slips past a forbidden-phrase list. Here the
answer's code block is extracted and executed against inputs with known outputs,
so the verdict is what the function actually does.

The generated code runs in a subprocess with a timeout. It is Orion's own output
rather than a stranger's, but it is still unreviewed text being executed, so it
never runs in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


DEFAULT_CLOUD_URL = "https://orion-core-prototype.onrender.com/api/v1"
CASES_PATH = Path(__file__).with_name("coding_cases.json")
EXECUTION_TIMEOUT_SECONDS = 10

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer: str) -> str:
    """The largest fenced block, or the whole answer when it has no fence.

    Orion often explains before and after the code; the block is the artefact.
    Picking the largest avoids grabbing a one-line usage example shown first.
    """

    blocks = _FENCE_RE.findall(answer or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (answer or "").strip()


def _harness(code: str, entrypoint: str, checks: list[dict]) -> str:
    return (
        code
        + "\n\nimport json as _json\n"
        + f"_checks = _json.loads({json.dumps(json.dumps(checks))})\n"
        + "_results = []\n"
        + "for _c in _checks:\n"
        + "    _args = _c['input']\n"
        + "    _repeat = _c.get('repeat_input_to')\n"
        + "    if _repeat:\n"
        + "        _args = [(_args[0] * _repeat)[:_repeat]] + list(_args[1:])\n"
        + "    try:\n"
        + f"        _got = {entrypoint}(*_args)\n"
        + "        _results.append({'ok': _got == _c['expected'], 'got': repr(_got)})\n"
        + "    except Exception as _exc:\n"
        + "        _results.append({'ok': False, 'got': f'{type(_exc).__name__}: {_exc}'})\n"
        + "print('__RESULT__' + _json.dumps(_results))\n"
    )


def run_checks(code: str, entrypoint: str, checks: list[dict]) -> tuple[bool, list[dict]]:
    with tempfile.TemporaryDirectory() as folder:
        script = Path(folder) / "candidate.py"
        script.write_text(_harness(code, entrypoint, checks), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT_SECONDS,
                cwd=folder,
            )
        except subprocess.TimeoutExpired:
            return False, [{"ok": False, "got": "tiempo agotado"}]
    marker = "__RESULT__"
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            results = json.loads(line[len(marker) :])
            return all(item["ok"] for item in results), results
    detail = (completed.stderr or completed.stdout or "sin salida").strip()
    return False, [{"ok": False, "got": detail[:400]}]


def _ask(client: httpx.Client, base_url: str, case: dict) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/chat",
        json={
            "messages": [{"role": "user", "content": case["prompt"]}],
            "mode": case.get("mode", "deep"),
            "sport": "general",
            "allow_busy": False,
        },
    )
    response.raise_for_status()
    return response.json().get("content", ""), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta el código que Orion escribe y verifica qué devuelve."
    )
    parser.add_argument("--base-url", default=os.getenv("ORION_CLOUD_URL", DEFAULT_CLOUD_URL))
    parser.add_argument("--output", type=Path, default=Path("orion-coding-eval.json"))
    args = parser.parse_args()

    api_key = os.getenv("ORION_API_KEY")
    if not api_key:
        raise SystemExit("Falta ORION_API_KEY.")

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/api/v1"):
        base_url += "/api/v1"

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    report: list[dict] = []
    passed = 0

    with httpx.Client(
        timeout=300.0,
        headers={"X-Orion-Api-Key": api_key, "Content-Type": "application/json"},
    ) as client:
        for case in cases:
            answer, seconds = _ask(client, base_url, case)
            code = extract_code(answer)
            ok, results = run_checks(code, case["entrypoint"], case["checks"])
            passed += int(ok)
            failures = [item for item in results if not item["ok"]]
            print(f"{'OK ' if ok else 'MAL'} {case['id']:26s} {seconds:5.1f}s")
            for item in failures[:3]:
                print(f"      devolvió: {item['got'][:160]}")
            report.append(
                {
                    "id": case["id"],
                    "seconds": round(seconds, 2),
                    "passed": ok,
                    "results": results,
                    "code": code,
                    "answer": answer,
                }
            )

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{passed}/{len(cases)} casos con el código funcionando. Reporte: {args.output}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
