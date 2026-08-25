from __future__ import annotations

import re
import unicodedata
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

# Domains that are especially useful for current football facts. They are tried
# first so a simple statistics question does not fan out over the whole allowlist.
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
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _query_terms(query: str) -> tuple[str, ...]:
    stopwords = {
        "cuantos", "cuantas", "tiene", "hizo", "hacer", "esta", "este", "para",
        "como", "cual", "cuales", "donde", "cuando", "desde", "hasta", "total",
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
            "buscá en internet", "busca en internet", "web", "fuentes actuales",
            "información actual", "informacion actual", "último partido",
            "ultimo partido", "próximo partido", "proximo partido", "cuántos goles",
            "cuantos goles", "goles de", "alineación", "alineacion", "resultado de",
            "tabla de posiciones", "estadísticas actuales", "estadisticas actuales",
            "hoy", "ayer", "esta temporada", "actualmente",
        )
    ):
        return True

    # Safety net for natural variants such as "qué cantidad de goles lleva X".
    folded = _fold(query)
    asks_amount = bool(re.search(r"\b(cuantos?|cantidad|numero)\b", folded))
    asks_live_stat = bool(
        re.search(r"\b(goles?|asistencias?|partidos?|puntos?|victorias?|derrotas?)\b", folded)
    )
    return asks_amount and asks_live_stat


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
    without_code = re.sub(
        r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_code))).strip()


def _relevant_excerpt(value: str, query: str, limit: int = 1400) -> str:
    """Return the densest query-related window instead of the page header.

    Sports pages often put navigation, cookies and metadata before the actual
    statistic. Selecting a scored window preserves nearby numbers and context,
    which is much safer than feeding the model the first N characters.
    """

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


def _search_domains(query: str, allowed_domains: tuple[str, ...]) -> tuple[str, ...]:
    folded = _fold(query)
    football_stat = bool(
        re.search(r"\b(gol|goles|asistencia|asistencias|partido|partidos)\b", folded)
    )
    if not football_stat:
        return allowed_domains
    prioritized = [domain for domain in FOOTBALL_STATS_DOMAINS if domain in allowed_domains]
    remaining = [domain for domain in allowed_domains if domain not in prioritized]
    return tuple(prioritized + remaining)


async def research(
    query: str,
    *,
    allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
    minimum_sources: int = 4,
) -> tuple[WebSource, ...]:
    headers = {"User-Agent": "Orion-Research/0.2"}
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
            urls: list[str] = []
            domains = _search_domains(query, allowed_domains)
            # General search first, then a bounded set of high-value domains. This
            # prevents a small factual question from issuing dozens of searches.
            search_queries = [query, *(f"{query} site:{domain}" for domain in domains[:10])]
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
                title = _clean_text(re.sub(r"<[^>]+>", " ", title_match.group(1))) if title_match else url
                body = _relevant_excerpt(response.text, query)
                if len(body) < 80:
                    continue
                folded_body = _fold(body)
                if query_terms and not any(term in folded_body for term in query_terms):
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