"""Every route is private unless it is listed here as deliberately public.

Authentication that has to be remembered per route is authentication that will
eventually be forgotten: adding an endpoint is one decorator, and leaving off
`dependencies=[Depends(require_api_key)]` produces a working, useful, entirely
open endpoint that no test complains about. That is how the diagnostics trace -
which returns a whole conversation, question, answer and sources - could quietly
become readable by anyone.

So the check is inverted. The suite walks the real application and demands a
reason for each public route, instead of trusting each new route to bring its
own guard.
"""

from __future__ import annotations

import unittest

from fastapi.routing import APIRoute

from backend.app.api.routes import require_api_key
from backend.app.main import app

# Public on purpose:
#   /                    - service banner, no stored content.
#   /api/v1/health       - Render's healthCheckPath. Gating it would make the
#                          platform declare the service dead and stop routing
#                          traffic to it.
# FastAPI mounts /docs, /redoc and /openapi.json itself; they describe the shape
# of the API, never its contents, and they are not APIRoute instances anyway.
PUBLIC_PATHS = frozenset({"/", "/api/v1/health"})


def _collect(app_) -> list[tuple[str, frozenset[str], bool]]:
    """Every declared endpoint as (path, methods, is-guarded).

    Routers are included rather than flattened, so this walks into them and
    carries the prefix down. A guard counts whether it sits on the route itself
    or on the `include_router(...)` call, since both really do run.
    """

    found: list[tuple[str, frozenset[str], bool]] = []

    def walk(routes, prefix: str, inherited_guard: bool) -> None:
        for route in routes:
            context = getattr(route, "include_context", None)
            if context is not None:
                guarded = inherited_guard or any(
                    dependency.dependency is require_api_key
                    for dependency in getattr(context, "dependencies", ()) or ()
                )
                walk(
                    route.original_router.routes,
                    prefix + (getattr(context, "prefix", "") or ""),
                    guarded,
                )
                continue
            if not isinstance(route, APIRoute):
                continue
            guarded = inherited_guard or any(
                dependency.call is require_api_key
                for dependency in route.dependant.dependencies
            )
            found.append((prefix + route.path, frozenset(route.methods or ()), guarded))

    walk(app_.routes, "", False)
    return found


class RouteProtectionTests(unittest.TestCase):
    def test_the_walk_actually_reaches_every_declared_endpoint(self) -> None:
        """Without this, a walker that finds nothing would report perfect safety.

        The OpenAPI schema is built by FastAPI itself, so it is the one listing
        that cannot drift from what the server really serves.
        """

        walked = {path for path, _methods, _guarded in _collect(app)}
        published = set(app.openapi()["paths"])
        self.assertEqual(published - walked, set())
        self.assertGreater(len(walked), 10)

    def test_every_route_requires_the_api_key_unless_listed_as_public(self) -> None:
        unguarded = sorted(
            f"{sorted(methods)} {path}"
            for path, methods, guarded in _collect(app)
            if not guarded and path not in PUBLIC_PATHS
        )
        self.assertEqual(
            unguarded,
            [],
            "these routes answer without the API key; add "
            "dependencies=[Depends(require_api_key)] or justify them in "
            "PUBLIC_PATHS",
        )

    def test_the_public_list_names_routes_that_exist(self) -> None:
        """A stale allowlist re-opens a route the moment one is renamed."""

        declared = {path for path, _methods, _guarded in _collect(app)}
        for path in PUBLIC_PATHS:
            self.assertIn(path, declared)

    def test_no_public_route_returns_stored_content(self) -> None:
        """What an open route returns is, in effect, published.

        Checked against the handlers' own type: both answer with a flat mapping
        of strings they build on the spot. The day one of them starts returning
        a document, an entry or a trace, it has to move behind the key first,
        and this fails until it does.
        """

        by_path = dict(_routes_with_paths(app))
        for path in PUBLIC_PATHS:
            with self.subTest(path=path):
                self.assertEqual(by_path[path].response_model, dict[str, str])


def _routes_with_paths(app_) -> list[tuple[str, APIRoute]]:
    pairs: list[tuple[str, APIRoute]] = []

    def walk(routes, prefix: str) -> None:
        for route in routes:
            context = getattr(route, "include_context", None)
            if context is not None:
                walk(
                    route.original_router.routes,
                    prefix + (getattr(context, "prefix", "") or ""),
                )
            elif isinstance(route, APIRoute):
                pairs.append((prefix + route.path, route))

    walk(app_.routes, "")
    return pairs


if __name__ == "__main__":
    unittest.main()
