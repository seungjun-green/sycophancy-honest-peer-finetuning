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
