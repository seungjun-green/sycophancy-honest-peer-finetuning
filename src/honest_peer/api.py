"""Bounded Anthropic concurrency with retries, validation, and durable response caching."""

import asyncio
import os
import random

from .artifacts import Cache


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
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
        }
        cached = self.cache.get(payload)
        if cached is not None:
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
                        temperature=settings["temperature"],
                        messages=[{"role": "user", "content": prompt}],
                    )
                text = "".join(block.text for block in response.content if block.type == "text").strip()
                if not text or response.stop_reason != "end_turn":
                    raise ValueError(
                        f"Incomplete API response ({response.stop_reason}); increase max_tokens if truncated."
                    )
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
                    # Do not log API request bodies/headers or exception strings containing secrets.
                    raise RuntimeError(
                        f"Anthropic request failed ({type(exc).__name__}); identity={identity}"
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
