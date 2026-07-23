from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import ssl
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .models import FetchReceipt, FetchResult


class FetchError(RuntimeError):
    """A typed, fail-closed retrieval error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# A crawler that touches government sites must say what it is.  The polite
# requirement is a self-identifying User-Agent that names the software and
# points at its documentation; a monitored mailbox is a stronger courtesy that
# an operator can add through CRAWLER_CONTACT.  Shipping with no identity at
# all -- or with a literal "unset" placeholder -- is worse for the origin than
# shipping this default, so the default identifies the project.
DEFAULT_CRAWLER_CONTACT = (
    "+https://github.com/odisha-public-health-intelligence-hub; public-health-evidence-index"
)
PLACEHOLDER_CONTACT_MARKERS = ("example.invalid", "replace-with", "change-me", "unset")


def crawler_contact() -> str:
    """Return the configured contact, or the project-identifying default.

    A placeholder value left over from `.env.example` is treated as unset so an
    origin never receives a User-Agent whose contact says "replace me".
    """

    contact = os.getenv("CRAWLER_CONTACT", "").strip()
    lowered = contact.casefold()
    if not contact or any(marker in lowered for marker in PLACEHOLDER_CONTACT_MARKERS):
        return DEFAULT_CRAWLER_CONTACT
    return contact


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    user_agent: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    maximum_redirects: int
    maximum_response_bytes: int
    allowed_schemes: tuple[str, ...]
    allowed_content_types: tuple[str, ...]
    allow_http_hosts: tuple[str, ...]
    accept_encoding: str = "identity"

    @classmethod
    def load(cls, path: str | Path = "config/fetch_policy.json") -> FetchPolicy:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        user_agent = f"{value['user_agent']} ({crawler_contact()})"
        return cls(
            user_agent=user_agent,
            connect_timeout_seconds=float(value["connect_timeout_seconds"]),
            read_timeout_seconds=float(value["read_timeout_seconds"]),
            maximum_redirects=int(value["maximum_redirects"]),
            maximum_response_bytes=int(value["maximum_response_bytes"]),
            allowed_schemes=tuple(value["allowed_schemes"]),
            allowed_content_types=tuple(value["allowed_content_types"]),
            allow_http_hosts=tuple(host.lower() for host in value["allow_http_hosts"]),
            accept_encoding=str(value.get("accept_encoding", "identity")),
        )


Resolver = Callable[..., list[tuple]]


@dataclass(frozen=True, slots=True)
class _ValidatedDestination:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PinnedResponse:
    status: int
    headers: dict[str, str]
    body: bytes


def _normalise_host(hostname: str) -> str:
    try:
        return hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise FetchError("invalid_hostname", "hostname is not valid IDNA") from exc


def _is_permitted_ip(raw: str) -> bool:
    address = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return bool(address.is_global)


def _validated_destination(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    policy: FetchPolicy,
    resolver: Resolver = socket.getaddrinfo,
) -> _ValidatedDestination:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in policy.allowed_schemes:
        raise FetchError("scheme_not_allowed", f"scheme {parsed.scheme!r} is not allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise FetchError("invalid_authority", "URL must have a host and no userinfo")
    host = _normalise_host(parsed.hostname)
    allowed = {_normalise_host(item) for item in allowed_hosts}
    if host not in allowed:
        raise FetchError("host_not_allowlisted", f"host {host!r} is not allowlisted")
    if parsed.scheme.lower() == "http" and host not in policy.allow_http_hosts:
        raise FetchError("cleartext_not_allowed", f"HTTP is not allowed for host {host!r}")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise FetchError("invalid_port", "URL contains an invalid port") from exc
    if port not in {80, 443}:
        raise FetchError("port_not_allowed", f"port {port} is not allowed")
    try:
        addresses = resolver(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise FetchError("dns_failed", f"DNS resolution failed for {host}") from exc
    resolved = {item[4][0] for item in addresses}
    if not resolved or any(not _is_permitted_ip(address) for address in resolved):
        raise FetchError("non_public_address", f"host {host!r} resolved to a non-public address")
    normalised = urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )
    return _ValidatedDestination(
        url=normalised,
        hostname=host,
        port=port,
        addresses=tuple(sorted(resolved)),
    )


def validate_url(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    policy: FetchPolicy,
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    """Return the canonical URL after exact-host and public-address validation."""

    return _validated_destination(
        url,
        allowed_hosts=allowed_hosts,
        policy=policy,
        resolver=resolver,
    ).url


def _pinned_https_get(
    destination: _ValidatedDestination,
    *,
    headers: dict[str, str],
    policy: FetchPolicy,
) -> _PinnedResponse:
    """Connect only to an IP from the validated DNS answer.

    The URL host is replaced with that approved address for TCP routing while
    the original registered hostname remains both the HTTP Host header and the
    TLS SNI/certificate-verification name.  No second DNS lookup occurs.
    """

    parsed = urllib.parse.urlsplit(destination.url)
    if parsed.scheme != "https":
        raise FetchError("cleartext_not_allowed", "production fetch transport requires HTTPS")
    request_target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    host_header = destination.hostname
    if destination.port != 443:
        host_header = f"{host_header}:{destination.port}"
    request_headers = {**headers, "Host": host_header}
    timeout = httpx.Timeout(
        connect=policy.connect_timeout_seconds,
        read=policy.read_timeout_seconds,
        write=policy.read_timeout_seconds,
        pool=policy.connect_timeout_seconds,
    )
    last_error: Exception | None = None
    for address in destination.addresses:
        authority = f"[{address}]" if ":" in address else address
        pinned_url = f"https://{authority}:{destination.port}{request_target}"
        try:
            with httpx.Client(
                verify=ssl.create_default_context(),
                trust_env=False,
                follow_redirects=False,
                timeout=timeout,
            ) as client:
                with client.stream(
                    "GET",
                    pinned_url,
                    headers=request_headers,
                    extensions={"sni_hostname": destination.hostname},
                ) as response:
                    response_headers = {
                        str(key).lower(): str(value) for key, value in response.headers.items()
                    }
                    declared = response_headers.get("content-length")
                    if declared:
                        try:
                            declared_bytes = int(declared)
                        except ValueError as exc:
                            raise FetchError(
                                "invalid_content_length",
                                "upstream returned an invalid Content-Length",
                            ) from exc
                        if declared_bytes > policy.maximum_response_bytes:
                            raise FetchError(
                                "response_too_large",
                                "declared response exceeds byte limit",
                            )
                    body = bytearray()
                    if response.status_code == 200:
                        for chunk in response.iter_bytes():
                            body.extend(chunk)
                            if len(body) > policy.maximum_response_bytes:
                                raise FetchError(
                                    "response_too_large",
                                    "response exceeds byte limit",
                                )
                    return _PinnedResponse(
                        status=response.status_code,
                        headers=response_headers,
                        body=bytes(body),
                    )
        except FetchError:
            raise
        except httpx.TransportError as exc:
            last_error = exc
            continue
    raise FetchError(
        "network_error",
        f"network retrieval failed for {destination.url}",
    ) from last_error


def _media_type(headers) -> str:  # noqa: ANN001
    # Persisted receipt headers are normalised to lowercase.  Accept both
    # representations so JSON/XML/text responses do not fall through to
    # application/octet-stream after a successful fetch.
    value = headers.get("content-type")
    if value is None:
        value = headers.get("Content-Type", "application/octet-stream")
    return (
        str(value)
        .split(";", 1)[0]
        .strip()
        .lower()
    )


SCHEME_DOWNGRADE_ACCESS_PATH_SUFFIX = "+origin_scheme_downgrade_refetched_over_https"


def _same_host_scheme_downgrade(current: str, target: str) -> bool:
    """Whether `target` is `current`'s origin answering over cleartext.

    Several Odisha district portals answer an HTTPS request with `301
    http://<same host>/...` while publishing nothing at all on port 80, so the
    literal redirect is unfollowable and the page was being dropped.  The hop
    is only treated as a downgrade when the host is unchanged: a redirect to a
    *different* host stays an ordinary hop and is re-validated as one.
    """

    source = urllib.parse.urlsplit(current)
    destination = urllib.parse.urlsplit(target)
    if source.scheme.lower() != "https" or destination.scheme.lower() != "http":
        return False
    if destination.username or destination.password:
        return False
    if (source.hostname or "").lower() != (destination.hostname or "").lower():
        return False
    return (source.port or 443) == 443 and (destination.port or 80) == 80


def _upgraded_to_https(target: str) -> str:
    parsed = urllib.parse.urlsplit(target)
    authority = parsed.hostname or ""
    if ":" in authority:
        authority = f"[{authority}]"
    return urllib.parse.urlunsplit(
        ("https", authority, parsed.path, parsed.query, parsed.fragment)
    )


def _sniff_content_type(body: bytes, declared: str) -> str:
    prefix = body[:512].lstrip().lower()
    if body.startswith(b"%PDF-"):
        return "application/pdf"
    if prefix.startswith((b"<!doctype html", b"<html")):
        return "text/html"
    return declared


def fetch_url(
    url: str,
    *,
    source_id: str,
    allowed_hosts: Iterable[str],
    policy: FetchPolicy | None = None,
    resolver: Resolver = socket.getaddrinfo,
    access_path: str = "live_origin",
    archive_timestamp: str | None = None,
    archive_digest: str | None = None,
    fallback_reason: str | None = None,
) -> FetchResult:
    policy = policy or FetchPolicy.load()
    allowed_hostnames = tuple(allowed_hosts)
    destination = _validated_destination(
        url, allowed_hosts=allowed_hostnames, policy=policy, resolver=resolver
    )
    current = destination.url
    redirects: list[str] = []
    scheme_downgrades: list[str] = []
    for redirect_number in range(policy.maximum_redirects + 1):
        # `destination` carries the exact public IP set approved for this hop.
        # Production connects to one of those addresses without resolving the
        # hostname again. Redirects build a new independently validated hop.
        headers = {
            "User-Agent": policy.user_agent,
            "Accept": (
                "text/html,application/pdf,application/json,application/xml,text/plain;q=0.8"
            ),
            "Accept-Encoding": policy.accept_encoding,
        }
        pinned_response = _pinned_https_get(
            destination,
            headers=headers,
            policy=policy,
        )
        status = pinned_response.status
        response_headers = pinned_response.headers
        body = pinned_response.body

        if status in {301, 302, 303, 307, 308}:
            location = response_headers.get("location")
            if not location:
                raise FetchError("redirect_without_location", "redirect response omitted Location")
            if redirect_number >= policy.maximum_redirects:
                raise FetchError("too_many_redirects", "maximum redirect count exceeded")
            next_url = urllib.parse.urljoin(current, location)
            downgraded = _same_host_scheme_downgrade(current, next_url)
            if downgraded:
                # Follow the resource the origin named, over TLS.  The host is
                # re-validated below exactly as any other hop, so the
                # allow-list, port, public-address and IP-pinning guards all
                # still apply; only the origin's scheme choice is refused.
                scheme_downgrades.append(next_url)
                next_url = _upgraded_to_https(next_url)
            destination = _validated_destination(
                next_url,
                allowed_hosts=allowed_hostnames,
                policy=policy,
                resolver=resolver,
            )
            if downgraded and destination.url == current:
                raise FetchError(
                    "scheme_downgrade_loop",
                    "origin redirects this HTTPS URL to its own cleartext form",
                )
            redirects.append(destination.url)
            current = destination.url
            continue
        if status != 200:
            raise FetchError(f"http_{status}", f"upstream returned HTTP {status}")
        content_type = _sniff_content_type(body, _media_type(response_headers))
        if content_type not in policy.allowed_content_types:
            raise FetchError(
                "content_type_not_allowed", f"content type {content_type!r} is not allowed"
            )
        receipt = FetchReceipt(
            source_id=source_id,
            requested_url=url,
            final_url=current,
            retrieved_at=datetime.now(UTC),
            status_code=status,
            content_type=content_type,
            byte_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            access_path=(
                f"{access_path}{SCHEME_DOWNGRADE_ACCESS_PATH_SUFFIX}"
                if scheme_downgrades
                else access_path
            ),
            redirect_chain=tuple(redirects),
            archive_timestamp=archive_timestamp,
            archive_digest=archive_digest,
            fallback_reason=fallback_reason,
            response_headers=response_headers,
            scheme_downgrades=tuple(scheme_downgrades),
        )
        return FetchResult(receipt=receipt, body=body)
    raise FetchError("too_many_redirects", "maximum redirect count exceeded")
