"""SSRF protection for user-supplied ``url`` inputs.

Before fetching a remote URL we resolve its host and reject anything that maps
to a private, loopback, link-local, or otherwise non-public address. We also
restrict schemes to http/https and cap redirects/size/time in the fetcher
(``images.fetch_url``). This blocks the classic "fetch http://169.254.169.254/"
metadata-service exfiltration and internal port scans.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from app.common.errors import bad_request, not_found

_ALLOWED_SCHEMES = {"http", "https"}


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_url(url: str) -> str:
    """Validate scheme + host, ensuring every resolved IP is public.

    Returns the URL unchanged if safe. Raises APIError otherwise.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise bad_request("Only http(s) URLs are allowed")
    host = parsed.hostname
    if not host:
        raise bad_request("URL has no host")

    # Reject raw private IPs given directly as host.
    try:
        literal = ipaddress.ip_address(host)
        if not _is_public_ip(str(literal)):
            raise bad_request("URL host resolves to a non-public address")
        return url
    except ValueError:
        pass  # hostname, resolve below

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise not_found("Could not resolve URL host")

    if not infos:
        raise not_found("Could not resolve URL host")

    for info in infos:
        ip = info[4][0]
        if not _is_public_ip(ip):
            raise bad_request("URL host resolves to a non-public address")
    return url
