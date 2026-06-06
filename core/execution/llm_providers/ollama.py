"""
core/execution/llm_providers/ollama.py — Ollama LLM provider.

Communicates with a local Ollama server via HTTP.
No API keys. Models run locally.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import httpx

from config import settings
from core.exceptions import OllamaUnavailableError, OllamaModelMissingError, AgentTimeoutError


class OllamaProvider:
    """
    LLM provider backed by a local Ollama server.

    Satisfies the LLMProvider Protocol:
        async def generate(prompt, model, **kwargs) -> str
    """

    def __init__(self):
        self._base_url = settings.OLLAMA_BASE_URL
        self._timeout  = settings.OLLAMA_TIMEOUT_SECONDS

    async def health_check(self):
        """
        Returns True if Ollama is running AND the configured model is available.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check if primary model is pulled (partial match — "qwen2.5:14b" matches "qwen2.5:14b")
                return any(
                    settings.OLLAMA_LLM_MODEL in m or m in settings.OLLAMA_LLM_MODEL
                    for m in models
                )
        except Exception:
            return False

    async def list_models(self):
        """Return names of all locally available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def generate(
        self,
        prompt: str,
        model:Optional[str] = None,
        system:Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Generate a response from Ollama.

        Args:
            prompt:      The user prompt
            model:       Model name (defaults to settings.OLLAMA_LLM_MODEL)
            system:      Optional system prompt
            temperature: Sampling temperature
        """
        model = model or settings.OLLAMA_LLM_MODEL
        payload: dict = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0)
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")

        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Try: ollama serve"
            ) from e
        except httpx.TimeoutException as e:
            raise AgentTimeoutError("ollama_generate", self._timeout) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelMissingError(model) from e
            raise OllamaUnavailableError(f"Ollama HTTP error: {e}") from e

    async def pull_model(self, model: str):
        """Pull a model if not already available. Returns True on success."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=10.0)
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/pull",
                    json={"name": model},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("status") == "success":
                                    return True
                            except json.JSONDecodeError:
                                pass
            return True
        except Exception:
            return False


if __name__ == "__main__":
    """Quick test: python -m core.execution.llm_providers.ollama"""
    import asyncio

    async def demo():
        provider = OllamaProvider()
        print("Health check:", await provider.health_check())
        models = await provider.list_models()
        print("Available models:", models)
        if models:
            model = models[0]
            print(f"\nGenerating with {model}…")
            resp = await provider.generate("Say hello in one sentence.", model=model)
            print("Response:", resp)

    asyncio.run(demo())
