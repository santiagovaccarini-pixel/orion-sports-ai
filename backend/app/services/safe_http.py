from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpcore
import httpx


def is_public_ip(value: str) -> bool:
    """True if `value` is a public IP literal (not private/loopback/link-local/etc.)."""

    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


class _SSRFSafeNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolves the target host once and connects straight to the validated IP.

    Checking a hostname's resolved address and then letting the HTTP client
    resolve it again to actually connect is vulnerable to DNS rebinding: a
    malicious/compromised DNS server can hand back a public IP for the check
    and a private/internal one moments later for the real connection.
    `connect_tcp` is the one place httpcore actually opens a socket, so
    resolving and validating right here - then connecting to that literal IP,
    not the hostname - leaves no second, independent resolution to race.
    TLS SNI/certificate verification are unaffected: httpcore performs those
    against the original hostname in a separate step after this connects.
    """

    def __init__(self) -> None:
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        resolved = await self._resolve_public_ip(host)
        if resolved is None:
            raise httpcore.ConnectError(
                f"Orion bloqueó una conexión a una dirección no pública: {host}"
            )
        return await self._inner.connect_tcp(
            resolved,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(  # pragma: no cover - never used by this app
        self, path: str, timeout: float | None = None, socket_options=None
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("Orion no permite conexiones por socket Unix.")

    @staticmethod
    async def _resolve_public_ip(host: str) -> str | None:
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            return str(literal) if is_public_ip(str(literal)) else None

        try:
            records = await asyncio.to_thread(
                socket.getaddrinfo, host, None, type=socket.SOCK_STREAM
            )
        except OSError:
            return None
        addresses = [
            str(record[4][0])
            for record in records
            if record and len(record) >= 5 and record[4]
        ]
        if not addresses or not all(is_public_ip(address) for address in addresses):
            return None
        return addresses[0]


def create_ssrf_safe_transport() -> httpx.AsyncHTTPTransport:
    """An httpx transport that only ever connects to public IP addresses.

    httpx.AsyncHTTPTransport doesn't take a network_backend in its public
    constructor, so this builds a normal transport (letting it handle SSL
    context setup) and swaps its internal connection pool for one backed by
    _SSRFSafeNetworkBackend before any request is made.
    """

    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=transport._pool._ssl_context,
        network_backend=_SSRFSafeNetworkBackend(),
    )
    return transport
