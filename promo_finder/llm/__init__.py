"""Model registry: turn a spec string into an LLM.

    get_llm("gemini")                         # provider's model from GEMINI_MODEL in .env
    get_llm("groq:openai/gpt-oss-120b")       # or name the model; split on the FIRST colon only
    get_llm("custom")                         # any OpenAI-compatible endpoint: CUSTOM_BASE_URL etc.
    get_llm("offline")                        # keyword baseline, no model

WHERE each provider is and WHICH model to use come from .env:
    <PROVIDER>_BASE_URL, <PROVIDER>_MODEL, <PROVIDER>_API_KEY, <PROVIDER>_MAX_TOKENS
"""
from __future__ import annotations

from ..config import S, provider_env
from .base import LLM, LLMError, LLMResponse, RetryableError, Usage
from .providers import GeminiLLM, OfflineLLM, OpenAICompatLLM, env_key

# HOW to talk to each provider (wire format and quirks). Everything that isn't gemini or
# offline speaks the OpenAI /chat/completions format.
PRESETS: dict[str, dict] = {
    "openai": {"kind": "openai", "token_param": "max_completion_tokens", "temperature": None},
    "groq": {"kind": "openai"},
    "openrouter": {"kind": "openai"},
    "together": {"kind": "openai"},
    "deepseek": {"kind": "openai"},
    "mistral": {"kind": "openai"},
    "ollama": {"kind": "openai", "needs_key": False},
    "custom": {"kind": "openai"},            # e.g. a gateway Kuku provides
    "gemini": {"kind": "gemini", "alt_key_env": "GOOGLE_API_KEY"},
    "offline": {"kind": "offline", "default_model": "heuristic", "needs_key": False},
}
ALIASES = {"google": "gemini", "heuristic": "offline", "none": "offline"}


def load_presets() -> dict[str, dict]:
    """The table above plus each provider's URL, model and output limit from .env."""
    presets = {k: dict(v) for k, v in PRESETS.items()}
    for name, cfg in presets.items():
        cfg["key_env"] = f"{name.upper()}_API_KEY"
        if (v := provider_env(name, "BASE_URL")):
            cfg["base_url"] = v
        if (v := provider_env(name, "MODEL")):
            cfg["default_model"] = v
        if (v := provider_env(name, "MAX_TOKENS")):
            try:
                cfg["max_tokens"] = int(v)
            except ValueError as e:
                raise LLMError(f"{name.upper()}_MAX_TOKENS={v!r} is not a whole number") from e
    return presets


def parse_spec(spec: str) -> tuple[str, str | None]:
    spec = (spec or "offline").strip()
    provider, _, model = spec.partition(":")
    provider = ALIASES.get(provider.lower(), provider.lower())
    return provider, (model or None)


def get_llm(spec: str, *, base_url: str | None = None, api_key_env: str | None = None, **kw) -> LLM:
    """Build one model. kw (max_tokens, temperature, cache_dir, min_interval) override the defaults."""
    presets = load_presets()
    provider, model = parse_spec(spec)
    if provider not in presets:
        raise LLMError(f"Unknown provider '{provider}'. Known: {', '.join(sorted(presets))}. "
                       "Any other OpenAI-compatible endpoint: use 'custom' with CUSTOM_BASE_URL.")
    cfg = presets[provider]
    env_name = provider.upper()
    model = model or cfg.get("default_model")
    if not model:
        raise LLMError(f"No model for provider '{provider}': set {env_name}_MODEL in .env, "
                       f"or use {provider}:<model>")
    if cfg["kind"] == "offline":
        return OfflineLLM(model, cache_dir=None)

    # Precedence: argument (CLI flag) > <PROVIDER>_MAX_TOKENS > the fixed default.
    kw = dict(kw)
    if kw.get("max_tokens") is None:
        kw["max_tokens"] = cfg.get("max_tokens") or S.max_tokens
    if "temperature" in cfg:
        kw.setdefault("temperature", cfg["temperature"])
    key = env_key(api_key_env, cfg["key_env"], cfg.get("alt_key_env"))
    url = base_url or cfg.get("base_url")
    if not url:
        raise LLMError(f"No endpoint for provider '{provider}': set {env_name}_BASE_URL in .env "
                       "(see .env.example) or pass --base-url")

    if cfg["kind"] == "gemini":
        return GeminiLLM(model, api_key=key, base_url=url, **kw)
    is_local = "localhost" in url or "127.0.0.1" in url
    if cfg.get("needs_key", True) and not key and not is_local:
        raise LLMError(f"Missing API key: set {api_key_env or cfg['key_env']} in .env")
    return OpenAICompatLLM(model, base_url=url, api_key=key, provider_name=provider,
                           token_param=cfg.get("token_param", "max_tokens"), **kw)


__all__ = ["LLM", "LLMError", "LLMResponse", "RetryableError", "Usage", "OfflineLLM",
           "get_llm", "parse_spec", "load_presets", "PRESETS"]
