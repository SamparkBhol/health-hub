"""Runtime robots.txt evaluation and per-host politeness pacing.

The registry records a human-checked `robots_state` string per source.  That is
a review artefact, not an enforcement mechanism: it is written once and can go
stale the moment an origin edits its file.  This module performs the actual
check at crawl time, on the exact path being requested, and it paces requests so
that a wide registry never becomes a burst against one origin.

Verdicts are typed rather than boolean so the collector can persist *why* a URL
was skipped:

``allowed``
    robots.txt was retrieved and permits this path for our User-Agent.
``disallowed``
    robots.txt was retrieved and forbids this path.  Fail closed.
``robots_unavailable``
    robots.txt could not be retrieved (404, 403, TLS or network failure).  The
    REP treats an unreachable robots.txt as "no restrictions published", which
    is also how the reviewed registry states describe these origins, so the
    fetch proceeds and the reason is recorded.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass

from .safe_fetch import FetchError, FetchPolicy, fetch_url

ROBOTS_CACHE_SECONDS = 3600.0
# Never let a published Crawl-delay stall the whole queue behind one origin.
MAXIMUM_HONOURED_CRAWL_DELAY = 30.0


@dataclass(frozen=True, slots=True)
class RobotsVerdict:
    allowed: bool
    state: str
    crawl_delay: float | None = None


@dataclass(slots=True)
class _CachedRobots:
    parser: urllib.robotparser.RobotFileParser | None
    state: str
    fetched_at: float


def host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").casefold()


class RobotsPolicy:
    """Cache and evaluate robots.txt per host, using the project fetch stack."""

    def __init__(
        self,
        *,
        policy: FetchPolicy | None = None,
        cache_seconds: float = ROBOTS_CACHE_SECONDS,
    ) -> None:
        self._policy = policy
        self._cache_seconds = cache_seconds
        self._cache: dict[str, _CachedRobots] = {}
        self._lock = threading.Lock()

    def _fetch_policy(self) -> FetchPolicy:
        if self._policy is None:
            self._policy = FetchPolicy.load()
        return self._policy

    @property
    def user_agent_token(self) -> str:
        """The product token robots.txt rules are matched against."""

        return self._fetch_policy().user_agent.split("/", 1)[0]

    def _load(self, host: str) -> _CachedRobots:
        with self._lock:
            cached = self._cache.get(host)
            if cached is not None and (time.monotonic() - cached.fetched_at) < self._cache_seconds:
                return cached
        entry = self._retrieve(host)
        with self._lock:
            self._cache[host] = entry
        return entry

    def _retrieve(self, host: str) -> _CachedRobots:
        url = f"https://{host}/robots.txt"
        try:
            result = fetch_url(
                url,
                source_id="robots",
                allowed_hosts=[host],
                policy=self._fetch_policy(),
            )
        except FetchError as error:
            return _CachedRobots(None, f"robots_unavailable_{error.code}", time.monotonic())
        body = result.body.decode("utf-8", errors="replace")
        # Several Indian government hosts answer /robots.txt with a themed HTML
        # error page and HTTP 200.  That is not a robots file; treating it as
        # one would parse zero rules and silently claim "allowed by robots".
        if "<html" in body[:600].casefold() or "<!doctype" in body[:600].casefold():
            return _CachedRobots(None, "robots_unavailable_html_error_page", time.monotonic())
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(body.splitlines())
        return _CachedRobots(parser, "robots_retrieved", time.monotonic())

    def evaluate(self, url: str) -> RobotsVerdict:
        host = host_of(url)
        if not host:
            return RobotsVerdict(False, "invalid_host")
        entry = self._load(host)
        if entry.parser is None:
            return RobotsVerdict(True, entry.state)
        token = self.user_agent_token
        allowed = bool(entry.parser.can_fetch(token, url))
        delay_value = entry.parser.crawl_delay(token)
        delay = None
        if delay_value is not None:
            delay = min(float(delay_value), MAXIMUM_HONOURED_CRAWL_DELAY)
        return RobotsVerdict(
            allowed=allowed,
            state="allowed" if allowed else "disallowed",
            crawl_delay=delay,
        )


class HostRateLimiter:
    """Serialise requests per origin host and hold a minimum gap between them.

    Parallelism in this collector is *across* hosts only.  Two jobs that target
    the same origin queue behind the same lock, so widening the worker pool
    never increases the request rate any single site sees.
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._guard = threading.Lock()

    def _lock_for(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def acquire(self, host: str, minimum_interval_seconds: float) -> None:
        lock = self._lock_for(host)
        lock.acquire()
        wait = 0.0
        with self._guard:
            previous = self._last_request.get(host)
            if previous is not None:
                wait = max(0.0, minimum_interval_seconds - (time.monotonic() - previous))
        if wait > 0:
            time.sleep(wait)

    def release(self, host: str) -> None:
        with self._guard:
            self._last_request[host] = time.monotonic()
        lock = self._lock_for(host)
        if lock.locked():
            lock.release()
