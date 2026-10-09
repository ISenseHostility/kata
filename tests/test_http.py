import asyncio
import hashlib
import time

import httpx
import pytest
import respx

from kata.data.http import HostPolicy, HttpError, PoliteClient, TokenBucket

FAST = {"example.com": HostPolicy(rate=1000, concurrency=4)}


@respx.mock
async def test_retries_429_then_succeeds():
    route = respx.get("https://example.com/x").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json={"ok": True})])
    async with PoliteClient(FAST, backoff_base=0.01) as c:
        assert await c.get_json("https://example.com/x") == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_404_is_not_retried():
    route = respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
    async with PoliteClient(FAST, backoff_base=0.01) as c:
        with pytest.raises(HttpError) as e:
            await c.get_json("https://example.com/missing")
    assert e.value.status == 404 and route.call_count == 1


@respx.mock
async def test_gives_up_after_max_attempts():
    respx.get("https://example.com/down").mock(return_value=httpx.Response(503))
    async with PoliteClient(FAST, max_attempts=3, backoff_base=0.001) as c:
        with pytest.raises(HttpError, match="gave up"):
            await c.get_json("https://example.com/down")


@respx.mock
async def test_download_is_atomic_and_hashed(tmp_path):
    body = b"x" * 100_000
    respx.get("https://example.com/f.zip").mock(return_value=httpx.Response(200, content=body))
    dest = tmp_path / "sub" / "f.zip"
    async with PoliteClient(FAST) as c:
        sha = await c.download("https://example.com/f.zip", dest)
    assert dest.read_bytes() == body and sha == hashlib.sha256(body).hexdigest()
    assert not list(tmp_path.rglob("*.part"))


async def test_token_bucket_spaces_requests():
    bucket = TokenBucket(rate=20)
    start = time.monotonic()
    await asyncio.gather(*(bucket.acquire() for _ in range(5)))
    assert time.monotonic() - start >= 0.19
