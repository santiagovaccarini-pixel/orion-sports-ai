from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from html import unescape
from typing import Sequence
from urllib.parse import urljoin, urlparse

import httpx

from backend.app.services.page_text import visible_text
from backend.app.services.safe_http import create_ssrf_safe_transport
from backend.app.services.web_research import WebSource, published_age_days


MAX_PAGE_BYTES = 750_000
MAX_REDIRECTS = 3
MAX_READ_PAGES = 4
# Sized for the engine Orion runs on now. The old 6.000 dates from when every
# token was scarce and the daily quota ran out after eight questions; measured on
# a reference article it discarded 80% of the page before any model saw it, which
# is the largest single cause of Orion reporting data as unavailable from pages
# that contain it.
MAX_EXCERPT_CHARACTERS = 24_000
# A page shorter than roughly this per window is sent whole rather than sampled.
WINDOW_TARGET_CHARACTERS = 4_000
CACHE_TTL_SECONDS = 300.0
CACHE_MAX_PAGES = 64
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)


@dataclass(frozen=True, slots=True)
class CachedPage:
    title: str
    url: str
    domain: str
    visible_text: str
    published_date: str | None


@dataclass(frozen=True, slots=True)
class PageRead:
    source_id: str
    title: str
    url: str
    domain: str
    excerpt: str
    published_date: str | None


_page_cache: OrderedDict[str, tuple[float, CachedPage]] = OrderedDict()
_cache_lock = asyncio.Lock()


def _safe_url_syntax(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.casefold().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return None
    if parsed.username or parsed.password:
        return None
    return parsed.scheme, host


def _title(html: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.IGNORECASE)
    if not match:
        return fallback
    value = re.sub(r"<[^>]+>", " ", match.group(1))
    return re.sub(r"\s+", " ", unescape(value)).strip()[:500] or fallback


def _published_date(html: str) -> str | None:
    patterns = (
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|date|datePublished)["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:article:published_time|date|datePublished)["\']',
        r'"datePublished"\s*:\s*"([^"]+)"',
        r'"dateModified"\s*:\s*"([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            value = unescape(match.group(1)).strip()
            if value:
                return value[:100]
    return None


def _relevant_excerpt(text: str, query: str, limit: int = MAX_EXCERPT_CHARACTERS) -> str:
    """Fit a page inside the size budget without deciding what it is about.

    This used to score windows by how many words of the user's question appeared
    literally in them. That is lexical classification, which this project forbids,
    and measuring it showed exactly why the rule exists: on a manager's Wikipedia
    article the highest-scoring text was the citation markup, because every
    reference title repeated his name, while the career table — which names clubs
    and years without repeating the question's wording — lost and was dropped.
    Orion then reported the career as unavailable, from a page that held it in
    full.

    Python's only job here is to fit the page in the budget. Pages under the limit
    pass through whole. Longer ones are sampled at evenly spaced offsets so the
    entire document is represented end to end, instead of only the parts that echo
    the question. Judging what matters is the model's job, and it can only do that
    with the text in front of it.
    """

    _ = query  # kept for call-site compatibility; relevance is not decided here
    if len(text) <= limit:
        return text

    separator = "\n\n[...]\n\n"
    window_count = max(2, min(8, len(text) // WINDOW_TARGET_CHARACTERS))
    # The separators come out of the budget, otherwise the final trim to `limit`
    # eats the tail of the last window — the end of the page it exists to reach.
    window_size = (limit - len(separator) * (window_count - 1)) // window_count
    span = max(0, len(text) - window_size)
    pieces: list[str] = []
    for index in range(window_count):
        # Spread the windows so the first starts at the top and the last ends at
        # the very bottom: a career table or a results list often sits at the end
        # of an article, and a step that falls short would clip it off.
        start = (span * index) // (window_count - 1)
        piece = text[start : start + window_size].strip()
        if piece:
            pieces.append(piece)
    joined = "\n\n[...]\n\n".join(pieces)
    return joined[:limit] or text[:limit]


async def _cache_get(url: str) -> CachedPage | None:
    now = time.monotonic()
    async with _cache_lock:
        item = _page_cache.get(url)
        if item is None:
            return None
        timestamp, page = item
        if now - timestamp > CACHE_TTL_SECONDS:
            _page_cache.pop(url, None)
            return None
        _page_cache.move_to_end(url)
        return page


async def _cache_put(page: CachedPage) -> None:
    async with _cache_lock:
        _page_cache[page.url] = (time.monotonic(), page)
        _page_cache.move_to_end(page.url)
        while len(_page_cache) > CACHE_MAX_PAGES:
            _page_cache.popitem(last=False)


async def _download_page(
    url: str,
    *,
    client: httpx.AsyncClient,
) -> CachedPage | None:
    # Private/internal IP addresses are rejected inside the client's transport
    # (create_ssrf_safe_transport, in safe_http.py) at the moment it actually
    # opens a socket - for this request and for every redirect hop below. That
    # is the only point immune to DNS-rebinding TOCTOU, so there is no separate
    # host-resolution pre-check here anymore; _safe_url_syntax only rejects
    # obviously-bad URLs (wrong scheme, embedded credentials, literal
    # "localhost") without touching the network.
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        safe = _safe_url_syntax(current)
        if safe is None:
            return None
        _, host = safe

        cached = await _cache_get(current)
        if cached is not None:
            return cached

        try:
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(str(response.url), location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").casefold()
                if content_type and not any(item in content_type for item in ALLOWED_CONTENT_TYPES):
                    return None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_PAGE_BYTES:
                        return None
                encoding = response.encoding or "utf-8"
                html = bytes(body).decode(encoding, errors="replace")
                final_url = str(response.url)
        except (httpx.HTTPError, LookupError, UnicodeError):
            return None

        final_safe = _safe_url_syntax(final_url)
        if final_safe is None:
            return None
        visible = visible_text(html)
        if len(visible) < 40:
            return None
        domain = final_safe[1].removeprefix("www.")
        page = CachedPage(
            title=_title(html, final_url),
            url=final_url,
            domain=domain,
            visible_text=visible,
            published_date=_published_date(html),
        )
        await _cache_put(page)
        return page
    return None


async def read_source_pages(
    sources: Sequence[WebSource],
    *,
    source_ids: Sequence[str],
    query: str,
    max_pages: int = MAX_READ_PAGES,
    client: httpx.AsyncClient | None = None,
) -> tuple[PageRead, ...]:
    """Open selected search results directly and extract bounded source text.

    `source_ids` use the reviewer's W1/W2/... identifiers. The caller/model chooses
    which result deserves deeper inspection; this function only validates networking,
    downloads the page and deterministically extracts relevant text.
    """

    requested: list[tuple[str, WebSource]] = []
    seen: set[str] = set()
    for source_id in source_ids:
        match = re.fullmatch(r"W([1-9]\d*)", source_id.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(sources):
            continue
        canonical_id = f"W{index + 1}"
        source = sources[index]
        if source.url in seen:
            continue
        seen.add(source.url)
        requested.append((canonical_id, source))
        if len(requested) >= max(1, min(max_pages, MAX_READ_PAGES)):
            break
    if not requested:
        return ()

    owns_client = client is None
    web_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": "Orion-WebRead/0.1"},
        follow_redirects=False,
        transport=create_ssrf_safe_transport(),
    )
    semaphore = asyncio.Semaphore(3)

    async def read_one(source_id: str, source: WebSource) -> PageRead | None:
        async with semaphore:
            page = await _download_page(source.url, client=web_client)
        if page is None:
            return None
        excerpt = _relevant_excerpt(page.visible_text, query)
        prefix = (
            f"Fecha publicada/actualizada detectada: {page.published_date}\n"
            if page.published_date
            else ""
        )
        return PageRead(
            source_id=source_id,
            title=page.title or source.title,
            url=page.url,
            domain=page.domain,
            excerpt=(prefix + excerpt)[:MAX_EXCERPT_CHARACTERS + 150],
            published_date=page.published_date,
        )

    try:
        values = await asyncio.gather(
            *(read_one(source_id, source) for source_id, source in requested)
        )
    finally:
        if owns_client:
            await web_client.aclose()
    return tuple(value for value in values if value is not None)


def apply_page_reads(
    sources: Sequence[WebSource],
    reads: Sequence[PageRead],
) -> tuple[WebSource, ...]:
    """Replace snippets with richer direct-page excerpts without changing W ordering."""

    enriched = list(sources)
    for read in reads:
        match = re.fullmatch(r"W([1-9]\d*)", read.source_id, flags=re.IGNORECASE)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if not (0 <= index < len(enriched)):
            continue
        current = enriched[index]
        if len(read.excerpt) < len(current.excerpt):
            # Direct-page fetch yielded less text than the search snippet already
            # had (common on login-gated social platforms). Keep the richer
            # snippet instead of overwriting it with a worse "deepened" read.
            continue
        published = read.published_date or current.published_date
        enriched[index] = WebSource(
            title=read.title,
            url=read.url,
            excerpt=read.excerpt,
            domain=read.domain,
            published_date=published,
            published_age_days=published_age_days(published),
            deepened=True,
        )
    return tuple(enriched)
