"""Provider adapters. Each is ~20 lines: build the request, read text + usage.

Two wire formats cover almost every model on the market:
  * OpenAI-compatible /chat/completions: OpenAI, Groq, OpenRouter, Together,
    DeepSeek, Mistral, Ollama, vLLM, LM Studio, any in-house gateway
  * Google Gemini generateContent
plus `offline`, a no-model heuristic baseline.
"""
from __future__ import annotations

import os

from .base import LLM, LLMError, LLMResponse, RetryableError, post_json

# No URLs are hard-coded in the adapters: each one gets its endpoint from .env
# (<PROVIDER>_BASE_URL), passed in by the registry in llm/__init__.py.


class OpenAICompatLLM(LLM):
    provider = "openai-compatible"

    def __init__(self, model: str, *, base_url: str, api_key: str | None,
                 token_param: str = "max_tokens", provider_name: str | None = None, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token_param = token_param
        if provider_name:
            self.provider = provider_name
        self._json_mode_supported = True

    def _complete(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.token_param: self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if json_mode and self._json_mode_supported:
            body["response_format"] = {"type": "json_object"}
        body.update(self.options.get("extra_body", {}))
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self.options.get("headers", {}))
        try:
            d = post_json(f"{self.base_url}/chat/completions", headers, body, self.timeout)
        except LLMError as e:
            # Some servers/models reject JSON mode; fall back to prompt-only JSON once.
            if (not isinstance(e, RetryableError) and "response_format" in body
                    and ("response_format" in str(e) or "json" in str(e).lower())):
                self._json_mode_supported = False
                return self._complete(system, user, json_mode)
            raise
        try:
            choice = d["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.spec}: unexpected response shape: {str(d)[:400]}") from e
        if choice.get("finish_reason") == "length":
            raise LLMError(f"{self.spec}: output truncated at max_tokens={self.max_tokens}")
        u = d.get("usage") or {}
        return LLMResponse(text=text, model=d.get("model", self.model),
                           input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"))


class GeminiLLM(LLM):
    provider = "gemini"

    def __init__(self, model: str, *, api_key: str | None, base_url: str, **kw):
        super().__init__(model, **kw)
        if not api_key:
            raise LLMError("Gemini needs GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _complete(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        gen: dict = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        d = post_json(f"{self.base_url}/models/{self.model}:generateContent",
                      {"x-goog-api-key": self.api_key}, body, self.timeout)
        cands = d.get("candidates") or []
        if not cands:
            raise LLMError(f"{self.spec}: no candidates (blocked?): {str(d.get('promptFeedback'))[:300]}")
        c = cands[0]
        if c.get("finishReason") == "MAX_TOKENS":
            raise LLMError(f"{self.spec}: output truncated at maxOutputTokens={self.max_tokens} "
                           "(on Gemini 2.5 this limit includes thinking; raise GEMINI_MAX_TOKENS)")
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = d.get("usageMetadata") or {}
        # Thinking tokens are billed as output, so count them, or the cost line is too low.
        thoughts = u.get("thoughtsTokenCount") or 0
        answer = u.get("candidatesTokenCount") or 0
        return LLMResponse(text=text, model=self.model, input_tokens=u.get("promptTokenCount"),
                           output_tokens=answer + thoughts,
                           meta={"thinking_tokens": thoughts, "answer_tokens": answer})


class OfflineLLM(LLM):
    """Sentinel for the heuristic, no-model baseline. The finder checks for it."""

    provider = "offline"

    def _complete(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        raise LLMError("offline mode does not call a model")


def env_key(*names: str | None) -> str | None:
    for n in names:
        if n and os.environ.get(n):
            return os.environ[n]
    return None
