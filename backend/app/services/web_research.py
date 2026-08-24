from __future__ import annotations

import re
from html import unescape
from dataclasses import dataclass
from datetime import date
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


@dataclass(frozen=True, slots=True)
class WebSource:
    title: str
    url: str
    excerpt: str
    domain: str


def is_web_request(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "buscá en internet", "busca en internet", "web", "fuentes actuales",
            "información actual", "informacion actual", "último partido",
            "ultimo partido", "próximo partido", "proximo partido", "cuántos goles",
            "cuantos goles", "goles de", "alineación", "alineacion", "resultado de",
            "tabla de posiciones", "estadísticas actuales", "estadisticas actuales",
            "hoy", "ayer", "esta temporada", "actualmente",
        )
    )


def _allowed(url: str, domains: tuple[str, ...]) -> bool:
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def _source_key(hostname: str, domains: tuple[str, ...]) -> str:
    clean = hostname.lower().removeprefix("www.")
    matches = [domain for domain in domains if clean == domain or clean.endswith(f".{domain}")]
    return max(matches, key=len) if matches else clean


def _clean_text(value: str, limit: int = 900) -> str:
    return _visible_text(value)[:limit]


def _visible_text(value: str) -> str:
    without_code = re.sub(r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>", " ", value, flags=re.IGNORECASE)
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_code))).strip()


async def research(
    query: str,
    *,
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
    minimum_sources: int = 4,
) -> tuple[WebSource, ...]:
    headers = {"User-Agent": "Orion-Research/0.1"}
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
            urls: list[str] = []
            search_queries = [query, *(f"{query} site:{domain}" for domain in allowed_domains)]
            for search_query in search_queries:
                search = await client.get(
                    f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
                )
                search.raise_for_status()
                candidates = re.findall(r'class="result__a"[^>]*href="([^"]+)"', search.text)
                for candidate in candidates:
                    target = unescape(parse_qs(urlparse(candidate).query).get("uddg", [candidate])[0])
                    if target.startswith("//"):
                        target = f"https:{target}"
                    if _allowed(target, allowed_domains) and target not in urls:
                        urls.append(target)
                if len(urls) >= minimum_sources * 3:
                    break
            sources: list[WebSource] = []
            source_domains: set[str] = set()
            for url in urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError:
                    continue
                title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.IGNORECASE | re.DOTALL)
                title = _clean_text(re.sub(r"<[^>]+>", " ", title_match.group(1))) if title_match else url
                body = _clean_text(response.text)
                if len(body) < 80:
                    continue
                query_terms = {term for term in re.findall(r"[a-záéíóúüñ0-9]+", _visible_text(query).lower()) if len(term) > 3}
                if not any(term in body.lower() for term in query_terms):
                    continue
                domain = _source_key(urlparse(str(response.url)).hostname or "", allowed_domains)
                if domain in source_domains:
                    continue
                source_domains.add(domain)
                sources.append(WebSource(title, str(response.url), body, domain))
                if len(sources) >= minimum_sources:
                    break
            return tuple(sources)
    except (httpx.HTTPError, ValueError):
        return ()


def format_sources(sources: tuple[WebSource, ...], *, minimum_sources: int = 4) -> str:
    independent_domains = {source.domain.lower().removeprefix("www.") for source in sources}
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
        return warning + ("\n\nFuentes encontradas:\n" + "\n\n".join(entries) if entries else "")
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