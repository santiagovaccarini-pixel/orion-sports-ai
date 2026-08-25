from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from html import unescape
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx


DEFAULT_ALLOWED_DOMAINS = (
    "fifa.com",
    "theifab.com",
    "uefa.com",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "acsm.org",
    "bocajuniors.com.ar",
    "afa.com.ar",
    "ligaprofesional.ar",
    "espn.com.ar",
    "transfermarkt.es",
    "worldfootball.net",
    "fbref.com",
    "soccerway.com",
    "as.com",
    "tycsports.com",
    "elobservador.com.uy",
    "365scores.com",
)

FOOTBALL_STATS_DOMAINS = (
    "bocajuniors.com.ar",
    "afa.com.ar",
    "ligaprofesional.ar",
    "espn.com.ar",
    "transfermarkt.es",
    "worldfootball.net",
    "fbref.com",
    "soccerway.com",
    "365scores.com",
    "tycsports.com",
)


@dataclass(frozen=True, slots=True)
class WebSource:
    title: str
    url: str
    excerpt: str
    domain: str


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def _query_terms(query: str) -> tuple[str, ...]:
    stopwords = {
        "cuantos",
        "cuantas",
        "tiene",
        "hizo",
        "hacer",
        "esta",
        "este",
        "para",
        "como",
        "cual",
        "cuales",
        "donde",
        "cuando",
        "desde",
        "hasta",
        "total",
    }
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", _fold(query))
        if len(term) >= 3 and term not in stopwords
    }
    return tuple(sorted(terms, key=len, reverse=True))


def is_web_request(query: str) -> bool:
    lowered = query.lower()
    if any(
        marker in lowered
        for marker in (
            "buscá en internet",
            "busca en internet",
            "web",
            "fuentes actuales",
            "información actual",
            "informacion actual",
            "último partido",
            "ultimo partido",
            "próximo partido",
            "proximo partido",
            "cuántos goles",
            "cuantos goles",
            "goles de",
            "alineación",
            "alineacion",
            "resultado de",
            "tabla de posiciones",
            "estadísticas actuales",
            "estadisticas actuales",
            "hoy",
            "ayer",
            "esta temporada",
            "actualmente",
        )
    ):
        return True

    folded = _fold(query)
    asks_amount = bool(re.search(r"\b(cuantos?|cantidad|numero)\b", folded))
    asks_live_stat = bool(
        re.search(
            r"\b(goles?|asistencias?|partidos?|puntos?|victorias?|derrotas?)\b",
            folded,
        )
    )
    return asks_amount and asks_live_stat


def _allowed(url: str, domains: tuple[str, ...]) -> bool:
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in domains
    )


def _source_key(hostname: str, domains: tuple[str, ...]) -> str:
    clean = hostname.lower().removeprefix("www.")
    matches = [
        domain
        for domain in domains
        if clean == domain or clean.endswith(f".{domain}")
    ]
    return max(matches, key=len) if matches else clean


def _clean_text(value: str, limit: int = 900) -> str:
    return _visible_text(value)[:limit]


def _visible_text(value: str) -> str:
    without_code = re.sub(
        r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return unescape(
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_code))
    ).strip()


def _relevant_excerpt(value: str, query: str, limit: int = 1400) -> str:
    visible = _visible_text(value)
    if len(visible) <= limit:
        return visible

    folded = _fold(visible)
    terms = _query_terms(query)
    if not terms:
        return visible[:limit]

    candidate_starts: set[int] = {0}
    for term in terms:
        offset = 0
        while len(candidate_starts) < 80:
            index = folded.find(term, offset)
            if index < 0:
                break
            candidate_starts.add(max(0, index - limit // 3))
            offset = index + len(term)

    best_start = 0
    best_score = -1.0
    for start in candidate_starts:
        window = visible[start : start + limit]
        folded_window = _fold(window)
        term_hits = sum(3 for term in terms if term in folded_window)
        repeated_hits = sum(min(folded_window.count(term), 3) for term in terms)
        numeric_bonus = 2 if re.search(r"\b\d{1,4}\b", window) else 0
        score = term_hits + repeated_hits + numeric_bonus
        if score > best_score:
            best_score = score
            best_start = start

    excerpt = visible[best_start : best_start + limit].strip()
    if best_start > 0:
        excerpt = f"…{excerpt}"
    if best_start + limit < len(visible):
        excerpt = f"{excerpt}…"
    return excerpt


def _search_domains(
    query: str, allowed_domains: tuple[str, ...]
) -> tuple[str, ...]:
    folded = _fold(query)
    football_stat = bool(
        re.search(r"\b(gol|goles|asistencia|asistencias|partido|partidos)\b", folded)
    )
    if not football_stat:
        return allowed_domains
    prioritized = [
        domain for domain in FOOTBALL_STATS_DOMAINS if domain in allowed_domains
    ]
    remaining = [
        domain for domain in allowed_domains if domain not in prioritized
    ]
    return tuple(prioritized + remaining)


def _decode_search_target(candidate: str) -> str:
    candidate = unescape(candidate)
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    parsed = urlparse(candidate)
    redirected = parse_qs(parsed.query).get("uddg", [])
    return unescape(redirected[0]) if redirected else candidate


def _extract_search_urls(
    html: str,
    allowed_domains: tuple[str, ...],
) -> tuple[str, ...]:
    candidates = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    urls: list[str] = []
    for candidate in candidates:
        target = _decode_search_target(candidate)
        if target.startswith("http") and _allowed(target, allowed_domains):
            if target not in urls:
                urls.append(target)
    return tuple(urls)


def _tavily_sources(
    payload: object,
    *,
    allowed_domains: tuple[str, ...],
    limit: int,
) -> tuple[WebSource, ...]:
    if not isinstance(payload, dict):
        return ()
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return ()

    sources: list[WebSource] = []
    source_domains: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or not _allowed(url, allowed_domains):
            continue
        domain = _source_key(urlparse(url).hostname or "", allowed_domains)
        if domain in source_domains:
            continue
        title = str(item.get("title") or url).strip()
        excerpt = str(item.get("content") or "").strip()
        if len(excerpt) < 40:
            continue
        source_domains.add(domain)
        sources.append(WebSource(title, url, excerpt[:1800], domain))
        if len(sources) >= limit:
            break
    return tuple(sources)


async def _research_tavily(
    client: httpx.AsyncClient,
    query: str,
    *,
    api_key: str,
    allowed_domains: tuple[str, ...],
    minimum_sources: int,
) -> tuple[WebSource, ...]:
    try:
        response = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max(8, minimum_sources * 3),
                "include_answer": False,
                "include_domains": list(_search_domains(query, allowed_domains)),
            },
        )
        response.raise_for_status()
        return _tavily_sources(
            response.json(),
            allowed_domains=allowed_domains,
            limit=minimum_sources,
        )
    except (httpx.HTTPError, ValueError):
        return ()


async def _discover_duckduckgo_urls(
    client: httpx.AsyncClient,
    query: str,
    *,
    allowed_domains: tuple[str, ...],
    minimum_sources: int,
) -> tuple[str, ...]:
    domains = _search_domains(query, allowed_domains)
    search_queries = [
        query,
        *(f"{query} site:{domain}" for domain in domains[:10]),
    ]
    endpoints = (
        "https://html.duckduckgo.com/html/?q={query}",
        "https://lite.duckduckgo.com/lite/?q={query}",
    )
    urls: list[str] = []

    for search_query in search_queries:
        encoded = quote_plus(search_query)
        for template in endpoints:
            try:
                search = await client.get(template.format(query=encoded))
                search.raise_for_status()
            except httpx.HTTPError:
                continue
            for target in _extract_search_urls(search.text, allowed_domains):
                if target not in urls:
                    urls.append(target)
            if len(urls) >= minimum_sources * 3:
                return tuple(urls)
    return tuple(urls)


async def _research_duckduckgo(
    client: httpx.AsyncClient,
    query: str,
    *,
    allowed_domains: tuple[str, ...],
    minimum_sources: int,
) -> tuple[WebSource, ...]:
    urls = await _discover_duckduckgo_urls(
        client,
        query,
        allowed_domains=allowed_domains,
        minimum_sources=minimum_sources,
    )
    sources: list[WebSource] = []
    source_domains: set[str] = set()
    query_terms = _query_terms(query)
    for url in urls:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            response.text,
            re.IGNORECASE | re.DOTALL,
        )
        title = (
            _clean_text(re.sub(r"<[^>]+>", " ", title_match.group(1)))
            if title_match
            else url
        )
        body = _relevant_excerpt(response.text, query)
        if len(body) < 80:
            continue
        folded_body = _fold(body)
        if query_terms and not any(term in folded_body for term in query_terms):
            continue
        domain = _source_key(
            urlparse(str(response.url)).hostname or "", allowed_domains
        )
        if domain in source_domains:
            continue
        source_domains.add(domain)
        sources.append(WebSource(title, str(response.url), body, domain))
        if len(sources) >= minimum_sources:
            break
    return tuple(sources)


async def research(
    query: str,
    *,
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
    minimum_sources: int = 4,
    provider: str = "auto",
    tavily_api_key: str | None = None,
) -> tuple[WebSource, ...]:
    configured_provider = os.getenv("ORION_WEB_PROVIDER", provider).strip().lower()
    if configured_provider not in {"auto", "tavily", "duckduckgo"}:
        raise ValueError("Proveedor web no válido.")
    configured_tavily_key = tavily_api_key or os.getenv("ORION_TAVILY_API_KEY") or None

    headers = {"User-Agent": "Orion-Research/0.3"}
    async with httpx.AsyncClient(
        timeout=12.0,
        headers=headers,
        follow_redirects=True,
    ) as client:
        if configured_provider in {"auto", "tavily"} and configured_tavily_key:
            # When Tavily is configured, keep the request bounded. Returning one or
            # two sources as insufficient evidence is safer than starting a long
            # sequential HTML-search fallback before the chat stream has opened.
            return await _research_tavily(
                client,
                query,
                api_key=configured_tavily_key,
                allowed_domains=allowed_domains,
                minimum_sources=minimum_sources,
            )

        if configured_provider == "tavily":
            return ()

        return await _research_duckduckgo(
            client,
            query,
            allowed_domains=allowed_domains,
            minimum_sources=minimum_sources,
        )


def format_sources(
    sources: tuple[WebSource, ...], *, minimum_sources: int = 4
) -> str:
    independent_domains = {
        source.domain.lower().removeprefix("www.") for source in sources
    }
    if len(sources) < minimum_sources or len(independent_domains) < minimum_sources:
        warning = (
            f"INVESTIGACIÓN WEB INSUFICIENTE: solo se obtuvieron {len(sources)} fuentes "
            f"permitidas y se requieren al menos {minimum_sources}. No presentes una "
            "conclusión como confirmada. Podés dar un dato provisional solo si aparece "
            "en las fuentes encontradas, marcándolo claramente como no confirmado."
        )
        entries = [
            f"[{index}] {source.title}\nURL: {source.url}\nExtracto: {source.excerpt}"
            for index, source in enumerate(sources, start=1)
        ]
        return warning + (
            "\n\nFuentes encontradas:\n" + "\n\n".join(entries)
            if entries
            else ""
        )
    entries = [
        f"[{index}] {source.title}\nURL: {source.url}\nExtracto: {source.excerpt}"
        for index, source in enumerate(sources, start=1)
    ]
    return (
        f"INVESTIGACIÓN WEB VERIFICADA ({len(sources)} fuentes permitidas, fecha {date.today().isoformat()}):\n"
        "Usá solo estas fuentes y citá los números entre corchetes. Si discrepan, "
        "explicá la discrepancia en lugar de ocultarla.\n\n"
        + "\n\n".join(entries)
    )
