from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpcore
import httpx

from backend.app.services.safe_http import (
    _SSRFSafeNetworkBackend,
    create_ssrf_safe_transport,
    is_public_ip,
)


class SafeHttpTests(unittest.TestCase):
    def test_is_public_ip_rejects_private_loopback_and_link_local(self) -> None:
        self.assertFalse(is_public_ip("127.0.0.1"))
        self.assertFalse(is_public_ip("10.0.0.5"))
        self.assertFalse(is_public_ip("169.254.169.254"))  # cloud metadata address
        self.assertFalse(is_public_ip("::1"))
        self.assertTrue(is_public_ip("8.8.8.8"))

    def test_connect_tcp_blocks_a_literal_private_ip_without_any_dns_lookup(
        self,
    ) -> None:
        # IP literals never touch socket.getaddrinfo, so this proves the block
        # is unconditional - not dependent on a resolver that a rebinding
        # attack could still influence.
        backend = _SSRFSafeNetworkBackend()
        with patch("socket.getaddrinfo", side_effect=AssertionError("must not resolve")):
            with self.assertRaises(httpcore.ConnectError):
                asyncio.run(backend.connect_tcp("127.0.0.1", 443))

    def test_connect_tcp_blocks_a_hostname_that_resolves_to_a_private_address(
        self,
    ) -> None:
        # Simulates what a DNS-rebinding attacker would try to serve: a
        # hostname whose resolved address is internal/private.
        backend = _SSRFSafeNetworkBackend()
        fake_records = [
            (None, None, None, "", ("169.254.169.254", 443)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_records):
            with self.assertRaises(httpcore.ConnectError):
                asyncio.run(backend.connect_tcp("attacker.example", 443))

    def test_connect_tcp_connects_to_the_single_resolved_ip_when_public(
        self,
    ) -> None:
        # The critical property: connect_tcp must be called with the exact IP
        # that was validated, never with the original hostname again (that
        # second, independent resolution is the TOCTOU gap this closes).
        backend = _SSRFSafeNetworkBackend()
        fake_records = [(None, None, None, "", ("93.184.216.34", 443))]
        inner_connect = AsyncMock(return_value="fake-stream")
        with (
            patch("socket.getaddrinfo", return_value=fake_records),
            patch.object(backend._inner, "connect_tcp", new=inner_connect),
        ):
            asyncio.run(backend.connect_tcp("example.com", 443))
        inner_connect.assert_called_once()
        called_host = inner_connect.call_args.args[0]
        self.assertEqual(called_host, "93.184.216.34")

    def test_create_ssrf_safe_transport_returns_a_working_httpx_transport(
        self,
    ) -> None:
        transport = create_ssrf_safe_transport()
        self.assertIsInstance(transport, httpx.AsyncHTTPTransport)
        self.assertIsInstance(
            transport._pool._network_backend, _SSRFSafeNetworkBackend
        )


if __name__ == "__main__":
    unittest.main()
