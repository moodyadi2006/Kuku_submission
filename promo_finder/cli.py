"""Command line entry point. Defaults come from .env (see .env.example).

  python -m promo_finder                                       # everything from .env
  python -m promo_finder data/ --model groq --episodes 1-10
  python -m promo_finder data/ --model offline                 # no API key needed
  python -m promo_finder data/ --model custom:any-model --base-url https://gateway/v1 --api-key-env MY_KEY
"""
from __future__ import annotations

import argparse
import sys

from .config import S
from .llm import LLMError, get_llm
from .parser import load_show
from .pipeline import PromoPipeline, RunConfig
from .report import write_outputs


def parse_episodes(spec: str | None) -> list[int] | None:
    """'1-10,15' -> [1, 2, ..., 10, 15]. Raises ValueError with a readable message."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, b = sorted(int(x) for x in part.split("-", 1))  # '10-1' works too
                out.update(range(a, b + 1))
            elif part:
                out.add(int(part))
        except ValueError:
            raise ValueError(f"--episodes: can't read {part!r}; use numbers like 1-10,15") from None
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promo_finder",
                                 description="Find ranked promos in a microdrama. Defaults come from .env.")
    ap.add_argument("input", nargs="?", default=S.input,
                    help=f"folder of episode .txt files, a .zip, or one .txt (now: {S.input})")
    ap.add_argument("-m", "--model", default=S.model, help=f"provider[:model] (PROMO_MODEL, now: {S.model})")
    ap.add_argument("--base-url", help="override the model's endpoint")
    ap.add_argument("--api-key-env", help="name of the env var holding the API key")
    ap.add_argument("--episodes", help="subset, e.g. '1-10,15'")
    ap.add_argument("-o", "--out", default=S.out_dir, help=f"output folder (now: {S.out_dir})")
    ap.add_argument("--per-episode", type=int, default=S.per_episode, help="max candidates per episode")
    ap.add_argument("--workers", type=int, default=S.workers, help="episodes processed in parallel")
    ap.add_argument("--rpm", type=float, default=S.rpm, help="max requests/minute (free tiers), 0 = no limit")
    ap.add_argument("--max-tokens", type=int, default=None, help="output-token limit (else from .env)")
    ap.add_argument("--no-camera", action="store_true", help="drop CAMERA lines from the prompt (fewer tokens)")
    ap.add_argument("--no-cache", action="store_true", default=not S.cache, help="ignore the response cache")
    ap.add_argument("--price-in", type=float, default=S.price_in, help="USD per 1M input tokens (cost line)")
    ap.add_argument("--price-out", type=float, default=S.price_out, help="USD per 1M output tokens")
    a = ap.parse_args(argv)
 
    try:
        llm = get_llm(a.model, base_url=a.base_url, api_key_env=a.api_key_env, max_tokens=a.max_tokens,
                      min_interval=(60.0 / a.rpm if a.rpm else 0.0),
                      cache_dir=None if a.no_cache else S.cache_dir)
        show = load_show(a.input)
        episodes = parse_episodes(a.episodes)
    except (LLMError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if episodes:
        missing = sorted(set(episodes) - {e.number for e in show.episodes})
        if len(missing) == len(episodes):
            print(f"error: none of the requested episodes exist (show has "
                  f"{show.episodes[0].number}-{show.episodes[-1].number})", file=sys.stderr)
            return 2
        if missing:
            print(f"  note: episodes not found, skipped: {', '.join(map(str, missing))}")

    print(f"Show: {show.title} · {len(show.episodes)} episodes · model {llm.spec}")
    warns = show.warnings
    if episodes:  # only show warnings for the episodes being run
        keep = {f"E{n:02d}:" for n in episodes}
        warns = [w for w in warns if not w.startswith("E") or w.split(" ", 1)[0] in keep]
    for w in warns[:10]:
        print(f"  parse warning: {w}")

    cfg = RunConfig(max_candidates=a.per_episode, workers=a.workers, include_camera=not a.no_camera)
    res = PromoPipeline(llm, cfg).run(show, episodes)
    out = write_outputs(res, a.out, a.price_in, a.price_out)

    print(f"\n{res.stats['promos']} promos, {res.stats['rejected']} rejected. Top 5:")
    for p in res.promos[:5]:
        print(f"  {p.rank}. [{p.id}] {' + '.join(s.label() for s in p.segments)} ({p.duration:.0f}s) {p.title}")
    print(f"\nOutputs: {out}")
    if res.errors:
        print(f"{len(res.errors)} episode errors (see report.md); rerun the same command to retry "
              "them (finished episodes come from the cache)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    