"""All settings, read from `.env`. No values are written in this file.

Copy `.env.example` to `.env` (it lists every setting with its recommended value and a
comment) and add your API key. A missing or invalid setting stops the program with a message
naming it. Variables already set in the real environment win over `.env`.

Each provider's endpoint, model and output limit follow one naming rule:
    <PROVIDER>_API_KEY, <PROVIDER>_BASE_URL, <PROVIDER>_MODEL, <PROVIDER>_MAX_TOKENS
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path = ".env") -> bool:
    """Tiny .env reader (KEY=VALUE lines, # comments). Real environment variables win."""
    p = Path(path)
    if not p.is_file():
        return False
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("#"):  # KEY=   # comment  -> empty value
            v = ""
        elif len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:  # allow trailing comments: KEY=value  # note
            v = v.split(" #", 1)[0].strip()
        if k and v and not os.environ.get(k):  # empty = not set
            os.environ[k] = v
    return True


for _p in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
    load_dotenv(_p)


def env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v is not None and v.strip() else default


def provider_env(provider: str, field: str) -> str | None:
    """provider_env('gemini', 'BASE_URL') -> value of GEMINI_BASE_URL, if set."""
    return env_str(f"{provider.upper().replace('-', '_')}_{field}")


class _Reader:
    """Reads required settings and collects every problem, so one error lists them all."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def _raw(self, name: str) -> str | None:
        v = env_str(name)
        if v is None:
            self.problems.append(f"{name} is missing")
        return v

    def text(self, name: str) -> str:
        return self._raw(name) or ""

    def int(self, name: str) -> int:
        v = self._raw(name)
        try:
            return int(v) if v is not None else 0
        except ValueError:
            self.problems.append(f"{name}={v!r} is not a whole number")
            return 0

    def float(self, name: str) -> float:
        v = self._raw(name)
        try:
            return float(v) if v is not None else 0.0
        except ValueError:
            self.problems.append(f"{name}={v!r} is not a number")
            return 0.0

    def bool(self, name: str) -> bool:
        v = (self._raw(name) or "").lower()
        if v and v not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            self.problems.append(f"{name}={v!r} should be true or false")
        return v in {"true", "1", "yes", "on"}

    def floats(self, name: str, count: int) -> list[float]:
        v = self._raw(name)
        try:
            items = [float(x) for x in (v or "").split(",") if x.strip()]
        except ValueError:
            items = []
        if v is not None and len(items) != count:
            self.problems.append(f"{name}={v!r} should be {count} comma-separated numbers")
        return items or [0.0] * count


class Settings:
    def __init__(self) -> None:
        r = _Reader()
        # Run
        self.model = r.text("PROMO_MODEL")                    # provider[:model]
        self.rpm = r.float("PROMO_RPM")                       # max requests/minute, 0 = no limit
        self.workers = r.int("PROMO_WORKERS")                 # episodes processed in parallel
        self.cache = r.bool("PROMO_CACHE")
        self.cache_dir = r.text("PROMO_CACHE_DIR")
        self.input = r.text("PROMO_INPUT")
        self.out_dir = r.text("PROMO_OUT_DIR")
        self.per_episode = r.int("PROMO_PER_EPISODE")         # max promo candidates per episode
        self.price_in = r.float("PROMO_PRICE_IN")             # USD per 1M input tokens (cost line)
        self.price_out = r.float("PROMO_PRICE_OUT")

        # The brief's rules (the prompts are generated from these)
        self.min_seconds = r.float("PROMO_MIN_SECONDS")
        self.max_seconds = r.float("PROMO_MAX_SECONDS")
        self.hook_window = r.float("PROMO_HOOK_WINDOW")

        # Model calls
        self.max_tokens = r.int("PROMO_MAX_TOKENS")           # when <PROVIDER>_MAX_TOKENS isn't set
        self.temperature = r.float("PROMO_TEMPERATURE")
        self.timeout = r.float("PROMO_TIMEOUT")
        self.max_retries = r.int("PROMO_MAX_RETRIES")
        self.retry_max_wait = r.float("PROMO_RETRY_MAX_WAIT")

        # Validation details
        self.length_tolerance = r.float("PROMO_LENGTH_TOLERANCE")
        self.cliff_tail = r.float("PROMO_CLIFF_TAIL")
        self.max_segments = r.int("PROMO_MAX_SEGMENTS")
        self.min_segment_seconds = r.float("PROMO_MIN_SEGMENT_SECONDS")
        self.safety_margin = r.float("PROMO_SAFETY_MARGIN_SECONDS")
        self.reorder_warn_seconds = r.float("PROMO_REORDER_WARN_SECONDS")

        # Pipeline
        self.include_camera = r.bool("PROMO_INCLUDE_CAMERA")
        self.story_window = r.int("PROMO_STORY_WINDOW")
        self.shortlist = r.int("PROMO_SHORTLIST")
        self.top = r.int("PROMO_TOP")
        self.dedupe_overlap = r.float("PROMO_DEDUPE_OVERLAP")
        self.dedupe_same_moment = r.float("PROMO_DEDUPE_SAME_MOMENT_SECONDS")
        self.blind_check_top = r.int("PROMO_BLIND_CHECK_TOP")
        self.blind_fail_below = r.float("PROMO_BLIND_FAIL_BELOW")
        # Variety rule for the first posts: at most MAX_PER_EPISODE promos from the same
        # episode (± SPREAD episodes, i.e. one storyline) in the top CAP_N.
        self.top_cap_n = r.int("PROMO_TOP_CAP_N")
        self.top_max_per_episode = r.int("PROMO_TOP_MAX_PER_EPISODE")
        self.top_episode_spread = r.int("PROMO_TOP_EPISODE_SPREAD")

        # Scoring
        self.score_weights = r.floats("PROMO_SCORE_WEIGHTS", 3)   # hook, cliffhanger, standalone
        self.sweet_spot_min = r.float("PROMO_SWEET_SPOT_MIN_SECONDS")
        self.sweet_spot_max = r.float("PROMO_SWEET_SPOT_MAX_SECONDS")

        if not r.problems and not 0 < self.hook_window < self.min_seconds < self.max_seconds:
            r.problems.append("need 0 < PROMO_HOOK_WINDOW < PROMO_MIN_SECONDS < PROMO_MAX_SECONDS")
        if r.problems:
            raise ValueError("Problems with settings in .env (copy any missing lines from "
                             ".env.example):\n  - " + "\n  - ".join(r.problems))

    def as_dict(self) -> dict:
        return dict(self.__dict__)


S = Settings()
