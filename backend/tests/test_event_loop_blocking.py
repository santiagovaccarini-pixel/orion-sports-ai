"""No request handler may talk to the database on the event loop.

Storage lives in Postgres, and psycopg is synchronous, so every read is a
network round-trip to another machine. Doing one directly inside an `async def`
does not slow that request down - it stops the entire server. Nothing else is
served while it waits: not another person's answer, not the status panel, not
the health check the platform uses to decide the service is alive.

The codebase already knew this; `_memory_context` carries a comment explaining
exactly why it hands the work to a thread. The knowledge-document paths simply
never got the same treatment, and nothing noticed, because the failure only
appears under real latency with more than one request in flight - which is to
say, never on a developer's machine and always in production.

So the rule is checked by reading the code rather than by remembering it. The
check follows sync helpers too: the danger is rarely a bare `list_documents()`
in a handler, it is a small innocent-looking helper three calls away.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDLERS = ROOT / "backend" / "app" / "api"

# Methods that reach storage. Named rather than inferred, so adding a storage
# method is a deliberate act that includes adding it here.
STORAGE_METHODS = frozenset(
    {
        "list_documents",
        "add_document",
        "delete_document",
        "list_entries",
        "add_entry",
        "delete_entry",
        "delete_all",
        "list_conversations",
        "get_conversation",
        "create_conversation",
        "append_messages",
        "delete_conversation",
    }
)


def _called_names(node: ast.AST, *, enter_nested: bool = True) -> set[str]:
    """Every function or method name called inside this node.

    With `enter_nested=False` the bodies of functions defined inside are not
    read. That is the honest reading for a handler: a nested helper is the usual
    way to hand blocking work to `asyncio.to_thread`, so its body runs on a
    thread, not here. Calling that helper directly still shows up, because the
    call itself is in the handler.
    """

    names: set[str] = set()
    stack: list[ast.AST] = [node]
    seen_root = False
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if seen_root and not enter_nested:
                continue
            seen_root = True
        if isinstance(current, ast.Call):
            target = current.func
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
            elif isinstance(target, ast.Name):
                names.add(target.id)
        stack.extend(ast.iter_child_nodes(current))
    return names


def _blocking_helpers(tree: ast.AST) -> set[str]:
    """Sync functions that reach storage, directly or through another one.

    Iterated to a fixed point so a chain of helpers is caught, not just the
    last link in it.
    """

    bodies: dict[str, set[str]] = {
        node.name: _called_names(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    tainted = {name for name, calls in bodies.items() if calls & STORAGE_METHODS}
    while True:
        grown = {
            name
            for name, calls in bodies.items()
            if calls & tainted
        } | tainted
        if grown == tainted:
            return tainted
        tainted = grown


class EventLoopBlockingTests(unittest.TestCase):
    def test_no_async_handler_reaches_storage_without_a_thread(self) -> None:
        offenders: list[str] = []
        for path in sorted(HANDLERS.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forbidden = STORAGE_METHODS | _blocking_helpers(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.AsyncFunctionDef):
                    continue
                # A name handed to asyncio.to_thread is passed, not called, so
                # it is not a Call node and never reaches this set.
                called = _called_names(node, enter_nested=False)
                for name in sorted(called & forbidden):
                    offenders.append(f"{path.name}:{node.name} calls {name}()")
        self.assertEqual(
            sorted(set(offenders)),
            [],
            "these run a blocking database call on the event loop, freezing "
            "every other request until it returns; wrap them in "
            "asyncio.to_thread(...)",
        )

    def test_the_check_can_actually_see_a_violation(self) -> None:
        """A detector that finds nothing must be proven able to find something.

        Without this, deleting the rule's body would read as perfect safety.
        """

        source = """
def _helper():
    return store.list_documents()

def _wrapper():
    return _helper()

async def handler():
    return _wrapper()
"""
        tree = ast.parse(source)
        tainted = _blocking_helpers(tree)
        self.assertEqual(tainted, {"_helper", "_wrapper"})
        handler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        )
        self.assertTrue(_called_names(handler) & (STORAGE_METHODS | tainted))


if __name__ == "__main__":
    unittest.main()
