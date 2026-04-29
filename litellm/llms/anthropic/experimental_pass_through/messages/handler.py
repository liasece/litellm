"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here
- Ensure requests are logged in the DB - stream + non-stream

"""

import asyncio
import contextvars
from functools import partial
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union, cast

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_messages.anthropic_request import AnthropicMetadata
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

from ..adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
from ..responses_adapters.handler import LiteLLMMessagesToResponsesAPIHandler
from .utils import AnthropicMessagesRequestUtils, mock_response

# Providers that are routed directly to the OpenAI Responses API instead of
# going through chat/completions.
_RESPONSES_API_PROVIDERS = frozenset({"openai"})


def _should_route_to_responses_api(custom_llm_provider: Optional[str]) -> bool:
    """Return True when the provider should use the Responses API path.

    Set ``litellm.use_chat_completions_url_for_anthropic_messages = True`` to
    opt out and route OpenAI/Azure requests through chat/completions instead.
    """
    if litellm.use_chat_completions_url_for_anthropic_messages:
        return False
    return custom_llm_provider in _RESPONSES_API_PROVIDERS


####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


async def _execute_pre_request_hooks(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: Optional[bool],
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Dict:
    """
    Execute pre-request hooks from CustomLogger callbacks.

    Allows CustomLoggers to modify request parameters before the API call.
    Used for WebSearch tool conversion, stream modification, etc.

    Args:
        model: Model name
        messages: List of messages
        tools: Optional tools list
        stream: Optional stream flag
        custom_llm_provider: Provider name (if not set, will be extracted from model)
        **kwargs: Additional request parameters

    Returns:
        Dict containing all (potentially modified) request parameters including tools, stream
    """
    # If custom_llm_provider not provided, extract from model
    if not custom_llm_provider:
        try:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
        except Exception:
            # If extraction fails, continue without provider
            pass

    # Build complete request kwargs dict
    request_kwargs = {
        "tools": tools,
        "stream": stream,
        "litellm_params": {
            "custom_llm_provider": custom_llm_provider,
        },
        **kwargs,
    }

    if not litellm.callbacks:
        return request_kwargs

    from litellm.integrations.custom_logger import CustomLogger as _CustomLogger

    for callback in litellm.callbacks:
        if not isinstance(callback, _CustomLogger):
            continue

        # Call the pre-request hook
        modified_kwargs = await callback.async_pre_request_hook(
            model, messages, request_kwargs
        )

        # If hook returned modified kwargs, use them
        if modified_kwargs is not None:
            request_kwargs = modified_kwargs

    return request_kwargs


async def _try_websearch_short_circuit(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    custom_llm_provider: Optional[str],
    stream: Optional[bool],
) -> Optional[Union[AnthropicMessagesResponse, AsyncIterator]]:
    """
    Attempt to short-circuit a web-search-only request.

    Claude Code sends web search as a separate, standalone /v1/messages
    request. For providers that don't natively support web search (e.g.
    github_copilot), we detect this pattern, execute the search via
    Tavily/Perplexity, and return a synthetic Anthropic response — bypassing
    the backend LLM entirely.

    Returns the synthetic response if short-circuited, or None to continue
    normal processing.
    """
    if not litellm.callbacks:
        return None

    from litellm.integrations.websearch_interception.handler import (
        WebSearchInterceptionLogger,
    )

    for callback in litellm.callbacks:
        if not isinstance(callback, WebSearchInterceptionLogger):
            continue

        response = await callback.try_short_circuit_search(
            model=model,
            messages=messages,
            tools=tools,
            custom_llm_provider=custom_llm_provider,
        )
        if response is not None:
            anthropic_response = cast(AnthropicMessagesResponse, response)
            if stream:
                from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
                    FakeAnthropicMessagesStreamIterator,
                )

                return FakeAnthropicMessagesStreamIterator(anthropic_response)
            return anthropic_response

    return None


@client
async def anthropic_messages(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    """
    Async: Make llm api request in Anthropic /messages API spec
    """
    # Save original stream flag before pre-request hooks can convert it.
    # The websearch interception hook converts stream=True → stream=False
    # for the agentic loop, but the short-circuit path needs to know
    # whether the caller originally requested streaming.
    original_stream = stream

    # Execute pre-request hooks to allow CustomLoggers to modify request
    request_kwargs = await _execute_pre_request_hooks(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )

    # Extract modified parameters
    tools = request_kwargs.pop("tools", tools)
    stream = request_kwargs.pop("stream", stream)
    # Propagate the provider derived inside pre-request hooks, if not already set.
    # The litellm_params dict may have been overwritten by **kwargs in
    # _execute_pre_request_hooks, so fall back to get_llm_provider() if needed.
    if not custom_llm_provider:
        custom_llm_provider = request_kwargs.get("litellm_params", {}).get(
            "custom_llm_provider"
        )
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
            except Exception:
                pass
    # Remove litellm_params from kwargs (only needed for hooks)
    request_kwargs.pop("litellm_params", None)
    # Merge back any other modifications
    kwargs.update(request_kwargs)

    # Short-circuit web-search-only requests: detect the pattern, execute
    # search directly via Tavily/Perplexity, and return a synthetic response
    # without ever touching the backend LLM or the adapter path.
    # Use original_stream (not the hook-converted stream) so streaming
    # callers get SSE events instead of a plain dict.
    short_circuit_response = await _try_websearch_short_circuit(
        model=model,
        messages=messages,
        tools=tools,
        custom_llm_provider=custom_llm_provider,
        stream=original_stream,
    )
    if short_circuit_response is not None:
        return short_circuit_response

    loop = asyncio.get_event_loop()
    kwargs["is_async"] = True

    func = partial(
        anthropic_messages_handler,
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=thinking,
        tool_choice=tool_choice,
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        api_key=api_key,
        api_base=api_base,
        client=client,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)
    init_response = await loop.run_in_executor(None, func_with_context)

    if asyncio.iscoroutine(init_response):
        response = await init_response
    else:
        response = init_response
    return response


def validate_anthropic_api_metadata(metadata: Optional[Dict] = None) -> Optional[Dict]:
    """
    Validate Anthropic API metadata - This is done to ensure only allowed `metadata` fields are passed to Anthropic API

    If there are any litellm specific metadata fields, use `litellm_metadata` key to pass them.
    """
    if metadata is None:
        return None
    anthropic_metadata_obj = AnthropicMetadata(**metadata)
    return anthropic_metadata_obj.model_dump(exclude_none=True)


def _normalize_messages_for_deepseek(messages: List[Dict]) -> None:
    """Normalize messages for DeepSeek's Anthropic-compatible endpoint.

    DeepSeek requires:
    1. No ``server_tool_use`` → convert to ``tool_use``
    2. No ``tool_result`` inside assistant messages
    3. All ``tool_use`` blocks must be at the END of assistant content —
       no text blocks after any tool_use (DeepSeek rejects this with
       "tool_use ids were found without tool_result blocks immediately after")

    Strategy: for each assistant message with embedded tool_results or
    server_tool_use, rebuild content so that:
    - server_tool_use → tool_use
    - tool_result blocks are extracted (moved to next user message)
    - All tool_use blocks are moved to the END of the content array
    - tool_results are ordered to match tool_use order
    """
    needs_normalize = False
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") in (
                "server_tool_use", "tool_result"
            ):
                needs_normalize = True
                break
        if needs_normalize:
            break

    if not needs_normalize:
        return

    normalized: List[Dict] = []
    pending_results: List[Dict] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")

        # Inject pending tool_results into the next user message
        if role == "user" and pending_results:
            content = msg.get("content")
            if isinstance(content, list):
                merged = list(pending_results) + list(content)
                normalized.append({"role": "user", "content": merged})
            else:
                normalized.append({"role": "user", "content": list(pending_results)})
                normalized.append(msg)
            pending_results = []
            continue

        if role == "assistant" and isinstance(msg.get("content"), list):
            content = msg["content"]
            has_server_or_result = any(
                isinstance(b, dict) and b.get("type") in ("server_tool_use", "tool_result")
                for b in content
            )
            if not has_server_or_result:
                normalized.append(msg)
                continue

            # Rebuild: text blocks first, then tool_use blocks (at end)
            text_blocks: List[Dict] = []
            tool_use_blocks: List[Dict] = []
            result_blocks: List[Dict] = []
            tool_use_ids: List[str] = []

            for block in content:
                if not isinstance(block, dict):
                    text_blocks.append(block)
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    result_blocks.append(block)
                elif btype == "server_tool_use":
                    new_block = dict(block)
                    new_block["type"] = "tool_use"
                    tool_use_blocks.append(new_block)
                    tool_use_ids.append(block.get("id", ""))
                elif btype == "tool_use":
                    tool_use_blocks.append(block)
                    tool_use_ids.append(block.get("id", ""))
                else:
                    text_blocks.append(block)

            # Reorder result_blocks to match tool_use_ids order
            result_by_id = {b.get("tool_use_id", ""): b for b in result_blocks}
            ordered_results = []
            for uid in tool_use_ids:
                if uid in result_by_id:
                    ordered_results.append(result_by_id.pop(uid))
            ordered_results.extend(result_by_id.values())

            # Build new content: text first, tool_use at end
            new_content = text_blocks + tool_use_blocks
            normalized.append({"role": "assistant", "content": new_content})
            pending_results = ordered_results
            continue

        normalized.append(msg)

    # If there are still pending results at the end, append a user message
    if pending_results:
        normalized.append({"role": "user", "content": pending_results})

    messages[:] = normalized


def anthropic_messages_handler(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    container: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    AnthropicMessagesResponse,
    AsyncIterator[Any],
    Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec

    Args:
        container: Container config with skills for code execution
    """
    from litellm.types.utils import LlmProviders

    metadata = validate_anthropic_api_metadata(metadata)

    local_vars = locals()
    is_async = kwargs.pop("is_async", False)
    # Use provided client or create a new one
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore

    # Store original model name before get_llm_provider strips the provider prefix
    # This is needed by agentic hooks (e.g., websearch_interception) to make follow-up requests
    original_model = model

    litellm_params = GenericLiteLLMParams(
        **kwargs,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
    )
    (
        model,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )

    # DeepSeek does not support Anthropic's interleaved mode where
    # server_tool_use + tool_result blocks coexist inside assistant
    # messages. Other providers (Claude, GLM) produce this pattern
    # when using server-side tools (web search, MCP tools, etc.).
    # We must extract those blocks into separate user messages and
    # convert server_tool_use → tool_use so DeepSeek can handle them.
    if model.startswith("deepseek-"):
        _normalize_messages_for_deepseek(messages)

    # DeepSeek requires every assistant message to carry a thinking
    # block. History from other providers (Claude, GLM, etc.) may
    # include assistant messages without one — fill in empty blocks
    # so DeepSeek doesn't reject the request.
    if model.startswith("deepseek-"):
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                if not any(
                    b.get("type") in ("thinking", "redacted_thinking")
                    for b in content
                ):
                    content.append(
                        {"type": "thinking", "thinking": "", "signature": ""}
                    )
            elif isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content},
                    {"type": "thinking", "thinking": "", "signature": ""},
                ]

    # Store agentic loop params in logging object for agentic hooks
    # This provides original request context needed for follow-up calls
    if litellm_logging_obj is not None:
        litellm_logging_obj.model_call_details["agentic_loop_params"] = {
            "model": original_model,
            "custom_llm_provider": custom_llm_provider,
        }

        # Check if stream was converted for WebSearch interception
        # This is set in the async wrapper above when stream=True is converted to stream=False
        if kwargs.get("_websearch_interception_converted_stream", False):
            litellm_logging_obj.model_call_details[
                "websearch_interception_converted_stream"
            ] = True

    if litellm_params.mock_response and isinstance(litellm_params.mock_response, str):
        return mock_response(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            mock_response=litellm_params.mock_response,
        )

    anthropic_messages_provider_config: Optional[BaseAnthropicMessagesConfig] = None

    if custom_llm_provider is not None and custom_llm_provider in [
        provider.value for provider in LlmProviders
    ]:
        anthropic_messages_provider_config = (
            ProviderConfigManager.get_provider_anthropic_messages_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )
    if anthropic_messages_provider_config is None:
        # Route to Responses API for OpenAI / Azure, chat/completions for everything else.
        _shared_kwargs = dict(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            _is_async=is_async,
            api_key=api_key,
            api_base=api_base,
            client=client,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        if _should_route_to_responses_api(custom_llm_provider):
            return LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler(
                **_shared_kwargs
            )
        return (
            LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                **_shared_kwargs
            )
        )

    if custom_llm_provider is None:
        raise ValueError(
            f"custom_llm_provider is required for Anthropic messages, passed in model={model}, custom_llm_provider={custom_llm_provider}"
        )

    local_vars.update(kwargs)
    anthropic_messages_optional_request_params = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params=local_vars
        )
    )
    return base_llm_http_handler.anthropic_messages_handler(
        model=model,
        messages=messages,
        anthropic_messages_provider_config=anthropic_messages_provider_config,
        anthropic_messages_optional_request_params=dict(
            anthropic_messages_optional_request_params
        ),
        _is_async=is_async,
        client=client,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=litellm_logging_obj,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        kwargs=kwargs,
    )
