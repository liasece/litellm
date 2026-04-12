"""
Regression tests for the duplicate ``message_start`` SSE event bug fixed across:

  * litellm/llms/anthropic/experimental_pass_through/responses_adapters/handler.py
  * litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py
  * litellm/proxy/common_request_processing.py

The bug: Anthropic ``/v1/messages`` streaming responses contained 2-3 ``message_start``
events for a single client request, because:

  1. ``AnthropicResponsesStreamWrapper.__anext__``'s fallback branch could emit a
     ``message_start`` before the first upstream event was consumed, and then
     ``_process_event("response.created")`` unconditionally emitted a second one.
  2. ``async_sse_data_generator_with_immediate_start``'s "skip the first upstream
     ``message_start``" guard only matched ``str`` chunks, but the responses_adapters
     wrapper / Anthropic passthrough handler both emit ``bytes``, so the skip never
     fired.

These tests pin both behaviours so the fix can not silently regress.
"""

import asyncio
import os
import sys
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)
from litellm.proxy import common_request_processing as crp
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


class _FakeEvent:
    """Minimal stand-in for a Responses API streaming event."""

    def __init__(self, event_type: str) -> None:
        self.type = event_type


async def _async_iter(items: List[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _collect_message_starts(payloads: List[bytes]) -> List[bytes]:
    return [p for p in payloads if p.startswith(b"event: message_start")]


def test_wrapper_emits_single_message_start_when_response_created_arrives() -> None:
    """
    When ``__anext__`` triggers its fallback branch first (queue empty,
    ``_sent_message_start`` False), the subsequent ``response.created`` event
    must NOT emit a second ``message_start``. The fallback already set the
    flag, and ``_process_event`` honours it.
    """
    upstream = _async_iter(
        [
            _FakeEvent("response.created"),
            _FakeEvent("response.completed"),
        ]
    )
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=upstream, model="gpt-test"
    )

    async def _drain() -> List[bytes]:
        return [chunk async for chunk in wrapper.async_anthropic_sse_wrapper()]

    chunks = asyncio.run(_drain())
    assert len(_collect_message_starts(chunks)) == 1, (
        "Wrapper must emit message_start exactly once even when both the "
        "fallback branch and response.created would otherwise produce one."
    )


def test_wrapper_response_created_alone_emits_single_message_start() -> None:
    """
    When ``response.created`` arrives before the fallback branch runs (i.e.
    upstream events are consumed during the very first ``__anext__`` call),
    exactly one ``message_start`` should still be emitted.
    """
    upstream = _async_iter(
        [
            _FakeEvent("response.created"),
            _FakeEvent("response.completed"),
        ]
    )
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=upstream, model="gpt-test"
    )
    # Simulate the case where the fallback branch is bypassed because the
    # caller has already advanced past it (mirroring the post-fix behaviour
    # of the proxy/handler chain).
    wrapper._sent_message_start = False

    async def _drain() -> List[bytes]:
        return [chunk async for chunk in wrapper.async_anthropic_sse_wrapper()]

    chunks = asyncio.run(_drain())
    assert len(_collect_message_starts(chunks)) == 1


def test_proxy_skip_handles_bytes_message_start_chunk() -> None:
    """
    ``async_sse_data_generator_with_immediate_start`` injects its own
    ``message_start`` and then skips the FIRST upstream chunk if it is a
    ``message_start``. The upstream chunk for both the responses_adapters
    wrapper and the Anthropic passthrough handler is ``bytes`` (encoded SSE
    frame), so the skip guard must accept ``bytes`` in addition to ``str``.
    """
    upstream_message_start = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_up"}}\n\n'
    )
    upstream_delta = (
        b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
    )

    async def _fake_upstream() -> AsyncIterator[bytes]:
        yield upstream_message_start
        yield upstream_delta

    proxy_logging = MagicMock()
    # ``async_post_call_streaming_iterator_hook`` is consumed via ``async for``;
    # return the raw upstream iterator unchanged so we exercise the real skip
    # path inside ``async_streaming_data_generator``.
    proxy_logging.async_post_call_streaming_iterator_hook = MagicMock(
        return_value=_fake_upstream()
    )
    # Per-chunk hook is awaited and expected to return the (possibly
    # modified) chunk; identity is sufficient for this test.
    proxy_logging.async_post_call_streaming_hook = AsyncMock(
        side_effect=lambda response, **_kw: response
    )

    user_api_key_dict = MagicMock()
    request_data = {"model": "gpt-test"}

    async def _drain() -> List[Any]:
        gen = ProxyBaseLLMRequestProcessing.async_sse_data_generator_with_immediate_start(
            response=_fake_upstream(),  # unused; iterator hook supplies chunks
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            proxy_logging_obj=proxy_logging,
        )
        return [chunk async for chunk in gen]

    chunks = asyncio.run(_drain())

    # The proxy injects exactly one (str) message_start.
    proxy_injected = [
        c
        for c in chunks
        if isinstance(c, str) and c.startswith("event: message_start")
    ]
    assert len(proxy_injected) == 1, (
        "Proxy must inject its own message_start once."
    )

    # The bytes ``message_start`` from upstream must be skipped, but the
    # subsequent ``content_block_delta`` bytes chunk must pass through.
    upstream_passed_through = [
        c
        for c in chunks
        if isinstance(c, (bytes, bytearray)) and c.startswith(b"event: message_start")
    ]
    assert upstream_passed_through == [], (
        "Upstream bytes message_start must be skipped to avoid duplicates."
    )

    delta_passed_through = [
        c
        for c in chunks
        if isinstance(c, (bytes, bytearray))
        and c.startswith(b"event: content_block_delta")
    ]
    assert len(delta_passed_through) == 1, (
        "Skip must only consume the first upstream chunk; subsequent chunks "
        "(content_block_delta) must still flow through."
    )


def test_proxy_emits_keepalive_ping_during_upstream_silence() -> None:
    """
    When upstream stalls between chunks for longer than
    ``_KEEPALIVE_INTERVAL_SECONDS``, the proxy must yield an Anthropic
    ``ping`` SSE frame to keep the client connection alive (Claude Code's
    SDK retries the entire request after ~6s of silence).

    This regression covers the case where ``upstream_aiter.__anext__`` is
    blocked deep inside ``litellm.aresponses`` while the upstream model is
    still processing a large prompt and has not produced any tokens yet.
    """
    # Compress the keep-alive interval so the test stays fast. The fix uses
    # a module-level constant so we monkey-patch it on the imported module.
    original_interval = crp._KEEPALIVE_INTERVAL_SECONDS
    crp._KEEPALIVE_INTERVAL_SECONDS = 0.1  # type: ignore[assignment]

    try:
        upstream_delta = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
        )

        async def _slow_upstream() -> AsyncIterator[bytes]:
            # Sleep longer than the (compressed) keep-alive interval to force
            # at least two ping emissions before the first real chunk.
            await asyncio.sleep(0.35)
            yield upstream_delta

        proxy_logging = MagicMock()
        proxy_logging.async_post_call_streaming_iterator_hook = MagicMock(
            return_value=_slow_upstream()
        )
        proxy_logging.async_post_call_streaming_hook = AsyncMock(
            side_effect=lambda response, **_kw: response
        )

        user_api_key_dict = MagicMock()
        request_data = {"model": "gpt-test"}

        async def _drain() -> List[Any]:
            gen = ProxyBaseLLMRequestProcessing.async_sse_data_generator_with_immediate_start(
                response=_slow_upstream(),
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
                proxy_logging_obj=proxy_logging,
            )
            return [chunk async for chunk in gen]

        chunks = asyncio.run(_drain())
    finally:
        crp._KEEPALIVE_INTERVAL_SECONDS = original_interval  # type: ignore[assignment]

    pings = [
        c
        for c in chunks
        if (isinstance(c, str) and c.startswith("event: ping"))
        or (isinstance(c, (bytes, bytearray)) and c.startswith(b"event: ping"))
    ]
    assert len(pings) >= 2, (
        f"Expected at least 2 keep-alive pings during a 0.35s upstream stall "
        f"with a 0.1s interval, got {len(pings)}. Chunks: {chunks!r}"
    )

    # The real upstream chunk must still arrive after the pings.
    deltas = [
        c
        for c in chunks
        if isinstance(c, (bytes, bytearray))
        and c.startswith(b"event: content_block_delta")
    ]
    assert len(deltas) == 1, (
        "Real upstream chunks must still flow through after keep-alive pings."
    )

    # The injected message_start must come BEFORE any ping.
    proxy_message_start_idx = next(
        i
        for i, c in enumerate(chunks)
        if isinstance(c, str) and c.startswith("event: message_start")
    )
    first_ping_idx = next(
        i
        for i, c in enumerate(chunks)
        if (isinstance(c, str) and c.startswith("event: ping"))
        or (isinstance(c, (bytes, bytearray)) and c.startswith(b"event: ping"))
    )
    assert proxy_message_start_idx < first_ping_idx, (
        "Keep-alive pings must never appear before the immediate message_start."
    )
