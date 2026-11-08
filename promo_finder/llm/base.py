"""The single interface every model provider implements.

The pipeline only ever calls `LLM.generate(system, user)` and gets text back.
Retries, rate limiting and the on-disk response cache live here, so a new
provider only has to implement `_complete` (one HTTP call).
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..config import S

USER_AGENT = f"promo-finder/{__version__}"


class LLMError(Exception):
    """Non-retryable failure (bad key, bad model name, refusal, malformed request)."""


class RetryableError(LLMError):
    """429 / 5xx / network failure: worth retrying with backoff."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0                  # real API calls made in this run (paid)
    cached_calls: int = 0           # answers reused from .cache (free this run)
    input_tokens: int = 0           # tokens actually spent in this run
    output_tokens: int = 0
    cached_input_tokens: int = 0    # what the reused answers cost when they were first made
    cached_output_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class LLM(ABC):
    provider: str = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = S.temperature,
        max_tokens: int = S.max_tokens,
        timeout: float = S.timeout,
        max_retries: int = S.max_retries,
        min_interval: float = 0.0,
        cache_dir: str | Path | None = S.cache_dir if S.cache else None,
        options: dict | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval  # seconds between calls (free-tier RPM limits)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.options = options or {}
        self.usage = Usage()
        self._lock = threading.Lock()
        self._last_call = 0.0

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    # ---- to implement per provider -------------------------------------
    @abstractmethod
    def _complete(self, system: str, user: str, json_mode: bool) -> LLMResponse: ...

    # ---- shared machinery ----------------------------------------------
    def generate(self, system: str, user: str, *, json_mode: bool = True) -> LLMResponse:
        key = self._cache_key(system, user, json_mode)
        cached = self._cache_get(key)
        if cached:
            with self._lock:
                self.usage.cached_calls += 1
                self.usage.cached_input_tokens += cached.input_tokens or 0
                self.usage.cached_output_tokens += cached.output_tokens or 0
            return cached

        attempt = 0
        while True:
            self._throttle()
            t0 = time.time()
            try:
                resp = self._complete(system, user, json_mode)
            except RetryableError as e:
                attempt += 1
                with self._lock:
                    self.usage.errors += 1
                if attempt > self.max_retries:
                    raise LLMError(f"{self.spec}: giving up after {attempt} attempts: {e}") from e
                # Exponential backoff, or the provider's own hint, but never longer than
                # PROMO_RETRY_MAX_WAIT: a daily-quota "retry in 1 hour" should fail over to the
                # fallback model instead of stalling the whole run.
                wait = min(S.retry_max_wait, e.retry_after or (2 ** attempt) + random.random())
                time.sleep(wait)
                continue
            with self._lock:
                self.usage.calls += 1
                self.usage.seconds += time.time() - t0
                self.usage.input_tokens += resp.input_tokens or 0
                self.usage.output_tokens += resp.output_tokens or 0
            if not resp.text.strip():
                raise LLMError(f"{self.spec}: empty response")
            self._cache_put(key, resp)
            return resp

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _cache_key(self, system: str, user: str, json_mode: bool) -> str:
        blob = json.dumps(
            [self.spec, self.temperature, self.max_tokens, json_mode, system, user, self.options],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cache_get(self, key: str) -> LLMResponse | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return LLMResponse(text=d["text"], model=d["model"], input_tokens=d.get("input_tokens"),
                               output_tokens=d.get("output_tokens"), cached=True)
        except Exception:
            return None

    def _cache_put(self, key: str, resp: LLMResponse) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"text": resp.text, "model": resp.model, "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens}),
            encoding="utf-8",
        )


def post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """Minimal JSON POST with error classification (stdlib only)."""
    data = json.dumps(body).encode("utf-8")
    # Some providers sit behind Cloudflare, which blocks Python's default "Python-urllib"
    # user-agent (HTTP 403, "error code: 1010"). Identify as this tool instead.
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        if e.code == 429 or e.code >= 500:
            ra = e.headers.get("retry-after") if e.headers else None
            try:
                ra_f = float(ra) if ra else None
            except ValueError:
                ra_f = None
            raise RetryableError(f"HTTP {e.code}: {detail}", retry_after=ra_f) from e
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RetryableError(f"network error calling {url}: {e}") from e
