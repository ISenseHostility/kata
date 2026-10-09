"""Polite async HTTP client: per-host rate limits, concurrency caps, backoff."""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from kata.config import USER_AGENT

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class HostPolicy:
    rate: float  # requests per second
    concurrency: int


DEFAULT_POLICIES = {
    "api.beatsaver.com": HostPolicy(rate=4.0, concurrency=4),
    "r2cdn.beatsaver.com": HostPolicy(rate=8.0, concurrency=4),
    "api.beatleader.com": HostPolicy(rate=3.0, concurrency=3),
    "cdn.replays.beatleader.com": HostPolicy(rate=8.0, concurrency=4),
}
FALLBACK_POLICY = HostPolicy(rate=2.0, concurrency=2)


class TokenBucket:
    """Spaces requests at least 1/rate seconds apart (no bursts)."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self._next = 0.0
        self._lock = asyncio.Lock()
        self.pause_until = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next, self.pause_until)
            self._next = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


class HttpError(Exception):
    def __init__(self, url: str, status: int | None, message: str):
        super().__init__(f"{status} {url}: {message}")
        self.url = url
        self.status = status


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


class PoliteClient:
    def __init__(self, policies: dict[str, HostPolicy] | None = None, max_attempts: int = 6,
                 backoff_base: float = 1.0, transport: httpx.AsyncBaseTransport | None = None):
        self.policies = {**DEFAULT_POLICIES, **(policies or {})}
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._buckets: dict[str, TokenBucket] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True, transport=transport, http2=False)
        self.request_count = 0

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    def _host_state(self, url: str) -> tuple[TokenBucket, asyncio.Semaphore]:
        host = urlsplit(url).hostname or ""
        if host not in self._buckets:
            policy = self.policies.get(host, FALLBACK_POLICY)
            self._buckets[host] = TokenBucket(policy.rate)
            self._sems[host] = asyncio.Semaphore(policy.concurrency)
        return self._buckets[host], self._sems[host]

    def _observe_rate_headers(self, bucket: TokenBucket, resp: httpx.Response) -> None:
        # BeatLeader advertises its window; back off when nearly exhausted.
        remaining = resp.headers.get("x-rate-limit-remaining")
        reset = resp.headers.get("x-rate-limit-reset")
        if remaining is not None and reset is not None and int(remaining) <= 3:
            try:
                reset_at = datetime.fromisoformat(reset.replace("Z", "+00:00"))
                wait = (reset_at - datetime.now(timezone.utc)).total_seconds()
                bucket.pause_until = max(bucket.pause_until, time.monotonic() + max(wait, 0) + 0.5)
            except ValueError:
                pass

    async def _send(self, method: str, url: str, *, stream_to: Path | None = None,
                    params: dict | None = None) -> httpx.Response | str:
        bucket, sem = self._host_state(url)
        last_error = ""
        for attempt in range(self.max_attempts):
            await bucket.acquire()
            try:
                async with sem:
                    self.request_count += 1
                    if stream_to is None:
                        resp = await self._client.request(method, url, params=params)
                    else:
                        async with self._client.stream(method, url, params=params) as resp:
                            if resp.status_code == 200:
                                return await _stream_file(resp, stream_to)
                            await resp.aread()
                self._observe_rate_headers(bucket, resp)
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in RETRY_STATUS:
                    raise HttpError(url, resp.status_code, resp.text[:200])
                last_error = f"HTTP {resp.status_code}"
                wait = _retry_after_seconds(resp)
            except (httpx.TransportError, httpx.StreamError) as e:
                last_error = f"{type(e).__name__}: {e}"
                wait = None
            if wait is None:
                wait = self.backoff_base * (2 ** attempt) * (0.5 + random.random())
            bucket.pause_until = max(bucket.pause_until, time.monotonic() + wait)
        raise HttpError(url, None, f"gave up after {self.max_attempts} attempts ({last_error})")

    async def get_json(self, url: str, params: dict | None = None):
        resp = await self._send("GET", url, params=params)
        return resp.json()

    async def download(self, url: str, dest: Path) -> str:
        """Stream `url` to `dest` atomically; returns the sha256 of the file."""
        return await self._send("GET", url, stream_to=dest)


async def _stream_file(resp: httpx.Response, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    sha = hashlib.sha256()
    with open(tmp, "wb") as f:
        async for chunk in resp.aiter_bytes(1 << 16):
            sha.update(chunk)
            f.write(chunk)
    tmp.replace(dest)
    return sha.hexdigest()
