"""Stable URL identities shared by discovery and durable queue storage."""

from __future__ import annotations

import posixpath
import urllib.parse

_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
    }
)


def canonicalize_discovered_url(value: str) -> str:
    """Return a conservative canonical form without changing article identity.

    Host case, fragments, redundant slashes and analytics parameters cannot make
    a second crawl target.  Other query parameters are retained and sorted: many
    government CMS endpoints use them as the document identifier.
    """

    parsed = urllib.parse.urlsplit(value.strip())
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    if not scheme or not host:
        return value.strip()
    port = parsed.port
    netloc = host
    if port is not None and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    raw_path = parsed.path or "/"
    path = posixpath.normpath("/" + raw_path.lstrip("/"))
    if raw_path.endswith("/") and path != "/":
        path = path.rstrip("/")
    query = [
        (key, item)
        for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in _TRACKING_QUERY_KEYS
    ]
    return urllib.parse.urlunsplit(
        (scheme, netloc, path, urllib.parse.urlencode(sorted(query)), "")
    )
