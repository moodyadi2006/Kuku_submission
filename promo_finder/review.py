"""Hand-review sheet for the top promos: what a human editor checks before posting.

    python -m promo_finder.review outputs/<show>/gemini-gemini-2-5-flash [--top 15]

For each promo it lists the timestamps, the full clip text (what's seen and heard), the model's
claims, the automatic checks, and an empty verdict line to fill in. Writes review_sheet.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import S
from .parser import load_show
from .pipeline import clip_text
from .promos import Promo, Segment

# Common reasons a promo is a bad pick, to tick off while reviewing.
CHECKLIST = ("visual-only hook (the script can't show it lands)", "spoiler (gives away a big twist)",
             "near-duplicate of a higher-ranked promo", "needs backstory to make sense",
             "starts or ends awkwardly", "hook isn't really in the first seconds")


def _promo(d: dict) -> Promo:
    """Rebuild just enough of a Promo from promos.json to cut its clip text."""
    segs = [Segment(s["episode"], s["start"], s["end"], s["start_block"], s["end_block"]) for s in d["segments"]]
    return Promo(segments=segs)


def build(folder: Path, top: int, input_path: str | None = None) -> str:
    meta = json.loads((folder / "promos.json").read_text(encoding="utf-8"))
    show = load_show(input_path or S.input)
    picks = meta["promos"][:top]
    L = [f"# Review sheet: top {len(picks)} promos ({meta['model']})", "",
         f"Show: {meta['show']}. For each promo, read the clip as a stranger would and fill in the verdict.",
         "", "Things to look for: " + "; ".join(CHECKLIST) + ".", ""]
    for p in picks:
        segs = " + ".join(f"E{s['episode']:02d} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        L += [f"## {p.get('rank')}. {p['id']} · {p['title']} · {p['duration']:.0f}s", "",
              f"**Where:** {segs}  ",
              f"**Model's hook:** {p['hook']}  ",
              f"**Model's cliffhanger:** {p['cliffhanger']}  ",
              f"**Model's scores:** hook {p['scores'].get('hook')} · cliffhanger {p['scores'].get('cliffhanger')} · "
              f"standalone {p['scores'].get('standalone')}"
              + (f" · **blind standalone {p['blind']['standalone']}** ({p['blind'].get('confusing', '')})"
                 if p.get("blind") else ""), ""]
        if p.get("warnings"):
            L += ["**Automatic warnings:** " + "; ".join(p["warnings"]), ""]
        L += ["```", clip_text(show, _promo(p)), "```", "",
              "**Verdict:** ✅ post / ⚠️ post with an edit / ❌ don't post  ",
              "**Why:** ", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="promo_finder.review", description=__doc__.splitlines()[0])
    ap.add_argument("run", help="output folder of a run")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--input", help="the scripts used for that run (default: PROMO_INPUT)")
    a = ap.parse_args(argv)
    try:
        md = build(Path(a.run), a.top, a.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.run) / "review_sheet.md"
    out.write_text(md, encoding="utf-8")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
