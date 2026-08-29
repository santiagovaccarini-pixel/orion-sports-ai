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

from backend.app.services.safe_http import create_ssrf_safe_transport
from backend.app.services.web_research import WebSource, published_age_days


MAX_PAGE_BYTES = 750_000
MAX_REDIRECTS = 3
MAX_READ_PAGES = 4
MAX_EXCERPT_CHARACTERS = 6_000
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


def _visible_text(html: str) -> str:
    article = re.search(
        r"<article\b[^>]*>([\s\S]*?)</article>",
        html,
        flags=re.IGNORECASE,
    )
    value = article.group(1) if article else html
    value = re.sub(
        r"<(script|style|noscript|svg|nav|footer|header|form)[^>]*>[\s\S]*?</\1>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<br\s*/?>|</p>|</li>|</h[1-6]>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = unescape(value).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if len(line) >= 2).strip()


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


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            term.casefold()
            for term in re.findall(r"[\wáéíóúüñ]{3,}", query, flags=re.IGNORECASE)
        )
    )


def _relevant_excerpt(text: str, query: str, limit: int = MAX_EXCERPT_CHARACTERS) -> str:
    if len(text) <= limit:
        return text
    terms = _query_terms(query)
    if not terms:
        return text[:limit]

    folded = text.casefold()
    anchors: list[int] = []
    for term in terms:
        start = 0
        while len(anchors) < 80:
            index = folded.find(term, start)
            if index < 0:
                break
            anchors.append(index)
            start = index + len(term)
    if not anchors:
        return text[:limit]

    window_size = min(1_800, limit)
    candidates: list[tuple[int, int, str]] = []
    for anchor in anchors[:40]:
        start = max(0, anchor - window_size // 3)
        window = text[start : start + window_size]
        folded_window = window.casefold()
        score = sum(1 for term in terms if term in folded_window)
        candidates.append((score, start, window.strip()))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    pieces: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    remaining = limit
    for _, start, window in candidates:
        end = start + len(window)
        if any(not (end <= old_start or start >= old_end) for old_start, old_end in used_ranges):
            continue
        if not window:
            continue
        piece = window[:remaining].strip()
        if not piece:
            continue
        pieces.append(piece)
        used_ranges.append((start, end))
        remaining -= len(piece) + 2
        if remaining <= 250:
            break
    return "\n\n…\n\n".join(pieces)[:limit] or text[:limit]


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
        visible = _visible_text(html)
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
