"""Usage tracking + pricing.

Combines the eval/usage.py callback handler and eval/pricing.py rate
table from agent_fin into one module.

Bedrock rates: https://aws.amazon.com/bedrock/pricing/
DeepSeek rates: https://platform.deepseek.com/api-docs/pricing
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write": 1.25},
    # DeepSeek v4 rates are placeholders at v3-era levels — confirm from the
    # DeepSeek pricing page. DeepSeek's cache is server-side automatic (no
    # separate cache-write SKU), so cache_write is set equal to input here
    # and will only be read if usage metadata reports cache_creation tokens.
    "deepseek-v4-pro":   {"input": 0.27, "output": 1.10, "cache_read": 0.07, "cache_write": 0.27},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.55, "cache_read": 0.035, "cache_write": 0.14},
}

_VERSION_SUFFIX = re.compile(r"-\d{8}(-v\d+:\d+)?$")
_GEO_PREFIXES = ("us.", "eu.", "au.", "apac.", "global.")


def normalize_model_id(model_id: str) -> str:
    """us.anthropic.claude-sonnet-4-5-20250929-v1:0 -> claude-sonnet-4-5"""
    s = model_id
    for prefix in _GEO_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("anthropic."):
        s = s[len("anthropic."):]
    return _VERSION_SUFFIX.sub("", s)


def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """USD cost. cache_read_tokens and cache_creation_tokens are assumed to be
    subsets of input_tokens (LangChain's UsageMetadata reports them that way)."""
    rates = _PRICING.get(normalize_model_id(model_id))
    if not rates:
        return 0.0
    plain_input = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)
    return (
        plain_input * rates["input"]
        + cache_read_tokens * rates["cache_read"]
        + cache_creation_tokens * rates["cache_write"]
        + output_tokens * rates["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Usage collector (LangChain callback)
# ---------------------------------------------------------------------------


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0


@dataclass
class ToolCall:
    tool_name: str
    result_tokens_est: int
    result_chars: int


class UsageCollector(BaseCallbackHandler):
    """LangChain callback that accumulates token usage by model_id.

    Sync handler so it works for both `llm.invoke(...)` and
    `graph.astream_events(...)` call paths.
    """

    def __init__(self) -> None:
        self.by_model: dict[str, ModelUsage] = defaultdict(ModelUsage)
        self._run_models: dict[UUID, str] = {}
        self.tool_calls: list[ToolCall] = []
        self._tool_starts: dict[UUID, str] = {}

    def on_chat_model_start(
        self, serialized: dict, messages, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._run_models[run_id] = _extract_model_id(serialized, kwargs)

    def on_llm_start(
        self, serialized: dict, prompts, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._run_models[run_id] = _extract_model_id(serialized, kwargs)

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        model_id = self._run_models.pop(run_id, "unknown")
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    self._record(model_id, msg)

    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or "unknown"
        self._tool_starts[run_id] = name

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._tool_starts.pop(run_id, "unknown")
        s = output if isinstance(output, str) else str(output)
        # ~4 chars per token approximation (no tiktoken dep).
        self.tool_calls.append(
            ToolCall(
                tool_name=name,
                result_tokens_est=len(s) // 4,
                result_chars=len(s),
            )
        )

    def _record(self, model_id: str, msg: BaseMessage) -> None:
        meta = getattr(msg, "usage_metadata", None) or {}
        if not meta:
            return
        details = meta.get("input_token_details") or {}
        u = self.by_model[model_id]
        u.input_tokens += meta.get("input_tokens", 0)
        u.output_tokens += meta.get("output_tokens", 0)
        u.cache_read_tokens += details.get("cache_read", 0)
        u.cache_creation_tokens += details.get("cache_creation", 0)
        u.calls += 1


def _extract_model_id(serialized: dict | None, kwargs: dict) -> str:
    if serialized:
        kw = serialized.get("kwargs") or {}
        for key in ("model_id", "model"):
            if kw.get(key):
                return kw[key]
    invocation = kwargs.get("invocation_params") or {}
    for key in ("model_id", "model"):
        if invocation.get(key):
            return invocation[key]
    metadata = kwargs.get("metadata") or {}
    return metadata.get("ls_model_name") or metadata.get("model_id") or "unknown"
