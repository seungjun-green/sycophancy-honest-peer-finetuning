"""Bounded Anthropic concurrency with retries, validation, and durable response caching."""

import asyncio
import os
import random

from .artifacts import Cache

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeRefusal(RuntimeError):
    """The model (and any configured fallback) declined; retrying the same request will not help."""

    def __init__(self, identity, category, model):
        super().__init__(f"Anthropic model declined the request (category={category}); identity={identity}")
        self.identity = identity
        self.category = category
        self.model = model


def _field(value, name):
    # SDK 0.51 does not know newer block/field types; they may arrive as dicts or extra attributes.
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def request_options(settings):
    """Optional request fields; None omits the field so the API default applies.

    Claude Sonnet 5.5 rejects non-default sampling values and `thinking: disabled`, so the
    defaults omit temperature and send `between_tools`. Fields the pinned SDK predates
    travel through extra_body/extra_headers unchanged.
    """
    options, extra_body = {}, {}
    if settings.get("temperature") is not None:
        options["temperature"] = settings["temperature"]
    if settings.get("thinking"):
        options["thinking"] = settings["thinking"]
    if settings.get("effort"):
        extra_body["output_config"] = {"effort": settings["effort"]}
    if settings.get("fallbacks"):
        extra_body["fallbacks"] = settings["fallbacks"]
        options["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
    if extra_body:
        options["extra_body"] = extra_body
    return options


class Claude:
    def __init__(self, settings, cache_path):
        self.settings = settings
        self.cache = Cache(cache_path)
        self.semaphore = asyncio.Semaphore(settings["concurrency"])
        self.client = None

    async def ask(self, prompt, identity, validate=None):
        settings = self.settings
        payload = {
            "identity": identity,
            "prompt": prompt,
            "model": settings["model"],
            "temperature": settings.get("temperature"),
            "thinking": settings.get("thinking"),
            "effort": settings.get("effort"),
            "fallbacks": settings.get("fallbacks"),
            "max_tokens": settings["max_tokens"],
        }
        cached = self.cache.get(payload)
        if cached is not None:
            if cached.get("refused"):
                # Declines are cached so reruns do not pay for the same refusal again.
                raise ClaudeRefusal(identity, cached["refusal_category"], cached["model"])
            if validate:
                validate(cached["text"])
            return cached
        if self.client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("Set ANTHROPIC_API_KEY in the environment before API stages.")
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(max_retries=0, timeout=settings["timeout_seconds"])
        from anthropic import APIConnectionError, APIStatusError, APITimeoutError

        for attempt in range(settings["retries"] + 1):
            try:
                async with self.semaphore:
                    response = await self.client.messages.create(
                        model=settings["model"],
                        max_tokens=settings["max_tokens"],
                        messages=[{"role": "user", "content": prompt}],
                        **request_options(settings),
                    )
                if response.stop_reason == "refusal":
                    category = _field(getattr(response, "stop_details", None), "category")
                    self.cache.put(
                        payload,
                        {
                            "refused": True,
                            "refusal_category": category,
                            "model": response.model,
                            "request_id": response.id,
                        },
                    )
                    raise ClaudeRefusal(identity, category, response.model)
                if response.stop_reason == "max_tokens":
                    raise RuntimeError(
                        f"Anthropic response hit max_tokens={settings['max_tokens']}; increase it. identity={identity}"
                    )
                text = "".join(
                    _field(block, "text") or ""
                    for block in response.content
                    if _field(block, "type") == "text"
                ).strip()
                if not text or response.stop_reason != "end_turn":
                    raise ValueError(f"Incomplete API response ({response.stop_reason}).")
                if validate:
                    validate(text)
                result = {
                    "text": text,
                    "model": response.model,
                    "request_id": response.id,
                    "stop_reason": response.stop_reason,
                    "usage": response.usage.model_dump(),
                }
                self.cache.put(payload, result)
                return result
            except (APIConnectionError, APITimeoutError, APIStatusError, ValueError) as exc:
                retryable = (
                    not isinstance(exc, APIStatusError)
                    or exc.status_code in (408, 409, 429)
                    or exc.status_code >= 500
                )
                if not retryable or attempt == settings["retries"]:
                    # The API's error body (e.g. an unsupported parameter) is safe to show; never
                    # log request bodies/headers or exception strings that could contain secrets.
                    status = (
                        f" HTTP {exc.status_code}: {_field(_field(exc.body, 'error'), 'message')}"
                        if isinstance(exc, APIStatusError)
                        else ""
                    )
                    raise RuntimeError(
                        f"Anthropic request failed ({type(exc).__name__}{status}); identity={identity}"
                    ) from None
                delay = min(settings["backoff_max_seconds"], settings["backoff_seconds"] * 2**attempt)
                if isinstance(exc, APIStatusError):
                    try:
                        delay = min(
                            settings["backoff_max_seconds"],
                            max(delay, float(exc.response.headers.get("retry-after", 0))),
                        )
                    except ValueError:
                        pass
                await asyncio.sleep(delay * (1 + random.random() * 0.1))

    async def close(self):
        if self.client is not None:
            await self.client.close()
        self.cache.close()


RATE_LIMIT_HEADERS = {
    "requests_per_minute": "anthropic-ratelimit-requests-limit",
    "input_tokens_per_minute": "anthropic-ratelimit-input-tokens-limit",
    "output_tokens_per_minute": "anthropic-ratelimit-output-tokens-limit",
}


def plan_concurrency(limits, input_tokens, output_tokens, latency_seconds, safety=0.7, cap=32):
    """Little's law: in-flight requests = sustainable request rate x request latency.

    The sustainable rate is the tightest of the request, input-token, and output-token
    per-minute limits for the expected tokens per call. Returns (concurrency, binding limit),
    or (None, None) when the API reported no limits.
    """
    per_second = {}
    if limits.get("requests_per_minute"):
        per_second["requests_per_minute"] = limits["requests_per_minute"] / 60
    if limits.get("input_tokens_per_minute"):
        per_second["input_tokens_per_minute"] = limits["input_tokens_per_minute"] / 60 / max(1, input_tokens)
    if limits.get("output_tokens_per_minute"):
        per_second["output_tokens_per_minute"] = (
            limits["output_tokens_per_minute"] / 60 / max(1, output_tokens)
        )
    if not per_second:
        return None, None
    binding = min(per_second, key=per_second.get)
    return max(1, min(cap, int(safety * per_second[binding] * latency_seconds))), binding


async def probe_rate_limits(settings):
    """One tiny request (8 output tokens) to read this key's rate-limit headers for the model."""
    import time

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Set ANTHROPIC_API_KEY in the environment before probing rate limits.")
    from anthropic import APIStatusError, AsyncAnthropic

    client = AsyncAnthropic(max_retries=2, timeout=settings["timeout_seconds"])
    try:
        started = time.monotonic()
        raw = await client.messages.with_raw_response.create(
            model=settings["model"],
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with OK."}],
            **request_options(settings),
        )
        latency = time.monotonic() - started
    except APIStatusError as exc:
        message = _field(_field(exc.body, "error"), "message")
        raise RuntimeError(f"Rate-limit probe failed (HTTP {exc.status_code}: {message})") from None
    finally:
        await client.close()
    limits = {}
    for name, header in RATE_LIMIT_HEADERS.items():
        try:
            limits[name] = int(raw.headers.get(header)) if raw.headers.get(header) else None
        except ValueError:
            limits[name] = None
    return {"limits": limits, "probe_latency_seconds": round(latency, 2)}
