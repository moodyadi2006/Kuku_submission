"""Compare two runs' picks, e.g. the LLM against the keyword baseline.

    python -m promo_finder.compare outputs/<show>/gemini-gemini-2-5-flash outputs/<show>/offline-heuristic

Two promos "match" when they're from the same episode and open on the same moment (hooks within
MATCH_SECONDS) or share at least half their screen time. Writes compare.md into the first folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import fmt_ts

MATCH_SECONDS = 5.0   # hooks this close count as the same moment
MATCH_OVERLAP = 0.5   # ...or this share of the shorter clip's screen time is shared


def _load(folder: Path) -> tuple[dict, list[dict]]:
    d = json.loads((folder / "promos.json").read_text(encoding="utf-8"))
    return d, d["promos"]


def _overlap(a: dict, b: dict) -> float:
    shared = sum(max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
                 for x in a["segments"] for y in b["segments"] if x["episode"] == y["episode"])
    return shared / (min(a["duration"], b["duration"]) or 1.0)


def _match(p: dict, others: list[dict]) -> dict | None:
    for q in others:
        same_ep = p["segments"][0]["episode"] == q["segments"][0]["episode"]
        if same_ep and (abs(p.get("hook_at", 0) - q.get("hook_at", 0)) <= MATCH_SECONDS
                        or _overlap(p, q) >= MATCH_OVERLAP):
            return q
    return None


def _where(p: dict) -> str:
    return " + ".join(f"E{s['episode']:02d} {fmt_ts(s['start'])}-{fmt_ts(s['end'])}" for s in p["segments"])


def compare(a_dir: Path, b_dir: Path, top: int = 15) -> str:
    a_meta, a = _load(a_dir)
    b_meta, b = _load(b_dir)
    matched_a = [p for p in a if _match(p, b)]
    top_a = a[:top]
    a_eps = {p["segments"][0]["episode"] for p in a}
    b_eps = {p["segments"][0]["episode"] for p in b}
    L = [f"# Picks compared: `{a_meta['model']}` vs `{b_meta['model']}`", "",
         f"Show: {a_meta['show']}", "",
         "| | " + a_meta["model"] + " | " + b_meta["model"] + " |", "|---|---|---|",
         f"| Promos | {len(a)} | {len(b)} |",
         f"| Episodes with at least one promo | {len(a_eps)} | {len(b_eps)} |",
         f"| Average length | {sum(p['duration'] for p in a) / max(1, len(a)):.0f}s | "
         f"{sum(p['duration'] for p in b) / max(1, len(b)):.0f}s |",
         "",
         f"**Overlap:** {len(matched_a)} of {len(a)} `{a_meta['model']}` promos ({len(matched_a) / max(1, len(a)):.0%}) "
         f"are on a moment the `{b_meta['model']}` run also picked (same episode, hooks within "
         f"{MATCH_SECONDS:g}s or half the footage shared).", "",
         f"## `{a_meta['model']}` top {len(top_a)}: did `{b_meta['model']}` find them too?", "",
         "| Rank | Promo | Where | Hook | Also found by the other run? |", "|---|---|---|---|---|"]
    for p in top_a:
        q = _match(p, b)
        found = f"yes: {q['id']} '{q['title']}'" if q else "**no**"
        hook = (p.get("hook_text") or p.get("hook", ""))[:70].replace("|", "/")
        L.append(f"| {p.get('rank')} | {p['id']} {p['title'][:40]} | {_where(p)} | {hook} | {found} |")
    only_b = [q for q in b if not _match(q, a)][:10]
    L += ["", f"## Moments only `{b_meta['model']}` picked (first {len(only_b)})", "",
          "| Promo | Where | Hook |", "|---|---|---|"]
    for q in only_b:
        L.append(f"| {q['id']} {q['title'][:40]} | {_where(q)} | {(q.get('hook_text') or '')[:80].replace('|', '/')} |")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promo_finder.compare", description=__doc__.splitlines()[0])
    ap.add_argument("run_a", help="output folder of the first run (e.g. the LLM run)")
    ap.add_argument("run_b", help="output folder of the second run (e.g. offline-heuristic)")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    try:
        md = compare(Path(a.run_a), Path(a.run_b), a.top)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.run_a) / "compare.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
