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

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


# Injected into every harness. Compares shape, not container type: a function
# asked for "una lista de tuplas" that returns tuples is correct, and grading it
# against literal JSON lists would fail it for obeying the spec.
_NORM_SOURCE = (
    "def _norm(_v):\n"
    "    if isinstance(_v, (list, tuple)):\n"
    "        return [_norm(_i) for _i in _v]\n"
    "    if isinstance(_v, dict):\n"
    "        return {_k: _norm(_i) for _k, _i in _v.items()}\n"
    "    return _v\n"
)

# Orion's own rate limiter is a single shared bucket for the whole deployment, so
# a battery fired back to back trips it and grades its own requests as failures.
PAUSE_BETWEEN_CASES_SECONDS = 8.0


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
        + _NORM_SOURCE
        + "_results = []\n"
        + "for _c in _checks:\n"
        + "    _args = _c['input']\n"
        + "    _repeat = _c.get('repeat_input_to')\n"
        + "    if _repeat:\n"
        + "        _args = [(_args[0] * _repeat)[:_repeat]] + list(_args[1:])\n"
        + "    try:\n"
        + f"        _got = {entrypoint}(*_args)\n"
        + "        _results.append({'ok': _norm(_got) == _norm(_c['expected']), 'got': repr(_got)})\n"
        + "    except Exception as _exc:\n"
        + "        _results.append({'ok': False, 'got': f'{type(_exc).__name__}: {_exc}'})\n"
        + "print('__RESULT__' + _json.dumps(_results))\n"
    )


def _class_harness(code: str, class_name: str, checks: list[dict]) -> str:
    """Drive a class through a sequence of calls, keeping one instance alive.

    A function is judged by one return value; an object is judged by whether its
    state survives the calls in order, which is where designs actually break.
    """

    return (
        code
        + "\n\nimport json as _json\n"
        + f"_checks = _json.loads({json.dumps(json.dumps(checks))})\n"
        + _NORM_SOURCE
        + "_results = []\n"
        + "for _c in _checks:\n"
        + "    try:\n"
        + f"        _obj = {class_name}()\n"
        + "        _got = None\n"
        + "        for _call in _c['calls']:\n"
        + "            _got = getattr(_obj, _call[0])(*_call[1:])\n"
        + "        _results.append({'ok': _norm(_got) == _norm(_c['expected']), 'got': repr(_got)})\n"
        + "    except Exception as _exc:\n"
        + "        _results.append({'ok': False, 'got': f'{type(_exc).__name__}: {_exc}'})\n"
        + "print('__RESULT__' + _json.dumps(_results))\n"
    )


def _sql_harness(query: str, checks: list[dict]) -> str:
    """Create the table, load the rows, run the query, compare the result set."""

    return (
        "import json as _json, sqlite3 as _sq\n"
        + f"_query = _json.loads({json.dumps(json.dumps(query))})\n"
        + f"_checks = _json.loads({json.dumps(json.dumps(checks))})\n"
        + _NORM_SOURCE
        + "_results = []\n"
        + "for _c in _checks:\n"
        + "    try:\n"
        + "        _cx = _sq.connect(':memory:')\n"
        + "        _cx.execute(_c['schema'])\n"
        + "        _cols = ','.join('?' * len(_c['rows'][0]))\n"
        + "        _table = _c['schema'].split()[2]\n"
        + "        _cx.executemany(f'INSERT INTO {_table} VALUES ({_cols})', _c['rows'])\n"
        + "        _got = [list(_r) for _r in _cx.execute(_query).fetchall()]\n"
        + "        _results.append({'ok': _norm(_got) == _norm(_c['expected']), 'got': repr(_got)})\n"
        + "    except Exception as _exc:\n"
        + "        _results.append({'ok': False, 'got': f'{type(_exc).__name__}: {_exc}'})\n"
        + "print('__RESULT__' + _json.dumps(_results))\n"
    )


_JS_NORM = (
    "const _norm=(v)=>Array.isArray(v)?v.map(_norm)"
    ":(v&&typeof v==='object')?Object.fromEntries(Object.keys(v).sort()"
    ".map(k=>[k,_norm(v[k])])):v;\n"
    "const _same=(a,b)=>JSON.stringify(_norm(a))===JSON.stringify(_norm(b));\n"
)


def _js_harness(code: str, entrypoint: str, checks: list[dict]) -> str:
    """Run JavaScript through node, with the same contract as the Python harness.

    Async cases carry a `script` body instead of plain arguments, because what is
    being judged there is a sequence of awaited calls, not a single return value.
    """

    is_async = entrypoint.startswith("__JS_ASYNC__")
    name = entrypoint.removeprefix("__JS_ASYNC__")
    if is_async:
        call = (
            f"      const _fn = new Function('{name}', "
            f"'return (async () => {{' + c.script + '}})()');\n"
            f"      const r = await _fn({name});\n"
        )
    else:
        call = f"      const r = {name}(...c.input);\n"
    return (
        code
        + "\n\n"
        + _JS_NORM
        + f"const _checks = {json.dumps(checks)};\n"
        + "(async () => {\n"
        + "  const out = [];\n"
        + "  for (const c of _checks) {\n"
        + "    try {\n"
        + call
        + "      out.push({ok: _same(r, c.expected), got: JSON.stringify(r)});\n"
        + "    } catch (e) {\n"
        + "      out.push({ok: false, got: String(e)});\n"
        + "    }\n"
        + "  }\n"
        + "  console.log('__RESULT__' + JSON.stringify(out));\n"
        + "})();\n"
    )


def _build_harness(code: str, entrypoint: str, checks: list[dict]) -> str:
    if entrypoint == "__SQL__":
        return _sql_harness(code, checks)
    if entrypoint.startswith("__CLASS__"):
        return _class_harness(code, entrypoint.removeprefix("__CLASS__"), checks)
    return _harness(code, entrypoint, checks)



def run_checks(
    code: str, entrypoint: str, checks: list[dict], *, language: str = "python"
) -> tuple[bool, list[dict]]:
    javascript = language == "javascript"
    with tempfile.TemporaryDirectory() as folder:
        name = "candidate.js" if javascript else "candidate.py"
        script = Path(folder) / name
        source = (
            _js_harness(code, entrypoint, checks)
            if javascript
            else _build_harness(code, entrypoint, checks)
        )
        script.write_text(source, encoding="utf-8")
        command = ["node", str(script)] if javascript else [sys.executable, str(script)]
        try:
            completed = subprocess.run(
                command,
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
        for position, case in enumerate(cases):
            if position:
                time.sleep(PAUSE_BETWEEN_CASES_SECONDS)
            answer, seconds = _ask(client, base_url, case)
            code = extract_code(answer)
            ok, results = run_checks(
                code,
                case["entrypoint"],
                case["checks"],
                language=case.get("language", "python"),
            )
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
