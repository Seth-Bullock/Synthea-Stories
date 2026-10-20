"""
A minimal client for a local LLM served by Ollama (https://ollama.com).

Ollama runs open-weight models on a laptop or a Colab GPU and exposes a small
HTTP API. We call its native /api/chat endpoint with the standard library, so
nothing extra needs installing. The native endpoint (rather than Ollama's
OpenAI-compatible one) lets us set the context window size, which otherwise
defaults to a few thousand tokens and would silently truncate long prompts.

DryRunClient has the same interface but never calls a model; it is used to
check the pipeline and inspect prompts without a GPU.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, model: str = "gemma3:12b", host: str = "http://localhost:11434",
                 num_ctx: int = 8192, temperature: float = 0.8, timeout: int = 900, retries: int = 3):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_ctx": self.num_ctx,
                        "temperature": self.temperature if temperature is None else temperature},
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())["message"]["content"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                if attempt == self.retries:
                    raise RuntimeError(f"Ollama request failed after {attempt} attempts: {err}") from err
                time.sleep(5 * attempt)
        raise AssertionError("unreachable")


class DryRunClient:
    """Stands in for a model: returns placeholder text so the whole pipeline can be checked."""
    model = "dry-run"

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float | None = None) -> str:
        if json_mode:
            return json.dumps({"preferred_name": "(dry run)", "gp_history": [
                {"name": "Dr Dry Run", "from_year": 1900, "to_year": 2100}]})
        return (f"[DRY RUN: a real model would write the document here. "
                f"Prompt was {len(system) + len(user)} characters.]\n"
                f"### SUMMARY\nDry-run placeholder summary.")
