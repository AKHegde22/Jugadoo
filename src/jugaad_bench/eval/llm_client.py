"""
Unified LLM client using individual provider SDKs.

Provides a common interface for OpenAI, Anthropic, and Google Gemini models,
with rate limiting (via aiolimiter + tenacity), token counting, and cost estimation.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any

import tiktoken

from jugaad_bench.models import CompletionResult, ModelConfig
from jugaad_bench.utils.config import get_api_key
from jugaad_bench.utils.rate_limiter import rate_limited_call

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pricing tables (USD per 1K tokens, as of mid-2025)
# ─────────────────────────────────────────────────────────────────────────────

_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.060),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
}

_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}

_GOOGLE_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-preview-05-20": (0.00015, 0.0006),
    "gemini-2.5-pro-preview-05-06": (0.00125, 0.01),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
}

# Fallback for unknown / open-weight models served via OpenAI-compatible APIs
_DEFAULT_PRICING: tuple[float, float] = (0.001, 0.002)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class BaseLLMProvider(abc.ABC):
    """Abstract base class for all LLM provider implementations."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.model_id = model_config.model_id
        self.provider_name = model_config.provider

    @abc.abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Send a completion request and return a structured result."""

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*."""

    @abc.abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated cost in USD."""

    def _lookup_pricing(
        self, table: dict[str, tuple[float, float]]
    ) -> tuple[float, float]:
        """Look up pricing for the current model, falling back gracefully."""
        # Try exact match first
        if self.model_id in table:
            return table[self.model_id]
        # Try prefix match (handles dated model slugs)
        for key, val in table.items():
            if self.model_id.startswith(key) or key.startswith(self.model_id):
                return val
        return _DEFAULT_PRICING


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI (+ OpenAI-compatible: Together, DeepSeek, Mistral, Krutrim)
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIProvider(BaseLLMProvider):
    """Provider backed by the ``openai`` SDK (supports custom api_base)."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        from openai import AsyncOpenAI

        api_key = get_api_key(model_config.provider, model_config.api_key_env)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if model_config.api_base:
            kwargs["base_url"] = model_config.api_base

        self._client = AsyncOpenAI(**kwargs)

        # Tiktoken encoding – fall back to cl100k_base for unknown models
        try:
            self._encoding = tiktoken.encoding_for_model(self.model_id)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    # ── public API ──────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        # o1/o3 and gpt-5.5 models don't support max_tokens, they use max_completion_tokens
        # They also don't support temperature=0.0 (must be 1 or omitted)
        if any(m in self.model_id for m in ("o1", "o3", "gpt-5.5")):
            kwargs["max_completion_tokens"] = max_tokens
            kwargs.pop("temperature", None)
        else:
            kwargs["max_tokens"] = max_tokens
        
        t0 = time.perf_counter()
        response = await rate_limited_call(
            self.provider_name,
            self._client.chat.completions.create,
            **kwargs
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        raw_output = choice.message.content or ""
        usage = response.usage

        input_tok = usage.prompt_tokens if usage else self.count_tokens(prompt)
        output_tok = usage.completion_tokens if usage else self.count_tokens(raw_output)

        return CompletionResult(
            problem_id="",  # caller sets this
            model_name=self.model_config.name,
            prompt_sent=prompt,
            raw_output=raw_output,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency_ms,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        inp_price, out_price = self._lookup_pricing(_OPENAI_PRICING)
        return (input_tokens / 1000 * inp_price) + (output_tokens / 1000 * out_price)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────────────────────


class AnthropicProvider(BaseLLMProvider):
    """Provider backed by the ``anthropic`` SDK."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        from anthropic import AsyncAnthropic

        api_key = get_api_key("anthropic", model_config.api_key_env)
        self._client = AsyncAnthropic(api_key=api_key)

        # Use tiktoken cl100k as a rough approximation for token counting
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        response = await rate_limited_call(
            "anthropic", self._client.messages.create, **kwargs
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_output = ""
        for block in response.content:
            if block.type == "text":
                raw_output += block.text

        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens

        return CompletionResult(
            problem_id="",
            model_name=self.model_config.name,
            prompt_sent=prompt,
            raw_output=raw_output,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency_ms,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        inp_price, out_price = self._lookup_pricing(_ANTHROPIC_PRICING)
        return (input_tokens / 1000 * inp_price) + (output_tokens / 1000 * out_price)


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini (via google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────


class GoogleProvider(BaseLLMProvider):
    """Provider backed by the ``google-genai`` SDK."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        from google import genai

        api_key = get_api_key("google", model_config.api_key_env)
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_config.model_id

        # Fallback tiktoken for fast local estimates
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        t0 = time.perf_counter()
        response = await rate_limited_call(
            "google",
            self._async_generate,
            prompt=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_output = response.text or ""

        # Extract token counts from usage_metadata if available
        input_tok = 0
        output_tok = 0
        if response.usage_metadata:
            input_tok = response.usage_metadata.prompt_token_count or 0
            output_tok = response.usage_metadata.candidates_token_count or 0

        if input_tok == 0:
            input_tok = self.count_tokens(prompt)
        if output_tok == 0:
            output_tok = self.count_tokens(raw_output)

        return CompletionResult(
            problem_id="",
            model_name=self.model_config.name,
            prompt_sent=prompt,
            raw_output=raw_output,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency_ms,
        )

    async def _async_generate(self, prompt: str, config: Any) -> Any:
        """Wrap the sync generate_content call for async rate-limited execution."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            ),
        )

    def count_tokens(self, text: str) -> int:
        try:
            resp = self._client.models.count_tokens(
                model=self._model_id, contents=text
            )
            return resp.total_tokens
        except Exception:
            # Fall back to tiktoken estimate
            return len(self._encoding.encode(text))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        inp_price, out_price = self._lookup_pricing(_GOOGLE_PRICING)
        return (input_tokens / 1000 * inp_price) + (output_tokens / 1000 * out_price)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

# Providers whose OpenAI-compatible endpoints use the OpenAIProvider
_OPENAI_COMPAT_PROVIDERS = frozenset(
    {"openai", "together", "deepseek", "fireworks", "mistral", "krutrim", "vllm"}
)


class LLMClientFactory:
    """Factory that creates the right provider from a ``ModelConfig``."""

    @staticmethod
    def create(model_config: ModelConfig) -> BaseLLMProvider:
        """
        Instantiate the appropriate LLM provider.

        Args:
            model_config: Configuration for the model.

        Returns:
            A provider instance ready for ``complete()`` calls.

        Raises:
            ValueError: If the provider is not recognised.
        """
        provider = model_config.provider.lower()

        if provider in _OPENAI_COMPAT_PROVIDERS:
            logger.info(
                "Creating OpenAI-compatible provider for %s (%s)",
                model_config.name,
                model_config.model_id,
            )
            return OpenAIProvider(model_config)

        if provider == "anthropic":
            logger.info(
                "Creating Anthropic provider for %s (%s)",
                model_config.name,
                model_config.model_id,
            )
            return AnthropicProvider(model_config)

        if provider == "google":
            logger.info(
                "Creating Google provider for %s (%s)",
                model_config.name,
                model_config.model_id,
            )
            return GoogleProvider(model_config)

        raise ValueError(
            f"Unknown provider '{provider}' for model '{model_config.name}'. "
            f"Supported: openai, anthropic, google, together, deepseek, "
            f"fireworks, mistral, krutrim, vllm."
        )
