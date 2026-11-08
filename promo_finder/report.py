"""Write run outputs: promos.json / report.md / run_meta.json / raw/ (unedited model output)."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .parser import fmt_ts
from .pipeline import RunResult
from .promos import HOOK_WINDOW, Promo

HOOK = f"{HOOK_WINDOW:g}s"  # e.g. "3s", from PROMO_HOOK_WINDOW, so labels match the rule in use
TOP_TABLE = 10      # promos in the "Post first" table at the top of report.md
CELL_CHARS = 90     # table cells are shortened to this many characters
TOP_REASONS = 10    # rejection reasons listed in the quality check


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "run"


def segments_str(p: Promo) -> str:
    return " + ".join(s.label() for s in p.segments)


def write_outputs(res: RunResult, out_root: str | Path, price_in: float = 0.0, price_out: float = 0.0) -> Path:
    """Write this run's files, replacing whatever an earlier run left in the same folder, so the
    folder always matches exactly one run. (Use -o <folder> to keep runs separate.)"""
    out = Path(out_root) / slug(res.show) / slug(res.model)
    # Clear the previous run's files: otherwise a smaller rerun (e.g. --episodes 1-5) would leave
    # old raw/episode-*.json files behind, and an old compare.md / review_sheet.md would describe
    # results that no longer exist (rerun `promo_finder.compare` / `promo_finder.review` for them).
    if (out / "raw").is_dir():
        shutil.rmtree(out / "raw")
    for stale in ("compare.md", "review_sheet.md"):
        (out / stale).unlink(missing_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    u = res.stats["llm_usage"]
    priced = bool(price_in or price_out)
    cost = (u["input_tokens"] * price_in + u["output_tokens"] * price_out) / 1e6
    full = cost + (u["cached_input_tokens"] * price_in + u["cached_output_tokens"] * price_out) / 1e6
    meta = {"show": res.show, "model": res.model, "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": res.stats,
            "estimated_cost_usd": round(cost, 4) if priced else None,          # spent in THIS run
            "estimated_full_cost_usd": round(full, 4) if priced else None,     # incl. cached answers
            "errors": res.errors}

    (out / "promos.json").write_text(json.dumps(
        {**meta, "promos": [p.to_dict() for p in res.promos], "rejected": res.rejected},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # Raw, unedited model responses for every call: per episode, the ranking, the blind checks.
    for ep_no, rec in res.raw["episodes"].items():
        (out / "raw" / f"episode-{int(ep_no):03d}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, fname in (("rank", "ranking.json"), ("blind", "blind_check.json")):
        if res.raw.get(name):
            (out / "raw" / fname).write_text(json.dumps(res.raw[name], indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(res, meta), encoding="utf-8")
    return out


def render_markdown(res: RunResult, meta: dict) -> str:
    s = res.stats
    L = [f"# Promo picks: {res.show}", "",
         f"Model: `{res.model}` · generated {meta['generated_at'][:16].replace('T', ' ')} UTC · "
         f"{s['episodes']} episodes · {s['promos']} promos", ""]

    L += [f"## Post first (top {TOP_TABLE})", "",
          "| # | ID | Episode · timestamps | Len | Hook | Cliffhanger | H/C/S (S = blind) |",
          "|---|---|---|---|---|---|---|"]
    for p in res.promos[:TOP_TABLE]:
        stand = p.blind.get("standalone", p.scores.get("standalone"))
        L.append(f"| {p.rank} | {p.id} | {segments_str(p)} | {p.duration:.0f}s | {_cell(p.hook)} | "
                 f"{_cell(p.cliffhanger)} | {p.scores.get('hook')}/{p.scores.get('cliffhanger')}/"
                 f"{stand} |")
    L.append("")

    L += ["## All promos, ranked", ""]
    for p in res.promos:
        L += _card(p)

    L += ["## Quality check", "",
          f"- Accepted: **{s['promos']}** promos, lengths {s['duration_min']:.0f}-{s['duration_max']:.0f}s "
          f"(avg {s['duration_avg']}s).",
          f"- Auto-fixed by code: {s['auto_fixed']} (hook moved into first {HOOK}, tail trimmed to the "
          f"cliffhanger, or cuts snapped off mid-line). Re-proposed by the model after failing checks: "
          f"{s['repaired_by_model']}.",
          f"- Flagged for human review: {s['with_warnings']} (e.g. a hook or cliffhanger that is only a "
          f"camera direction, so the script can't show it lands).",
          f"- Rejected or dropped: {s['rejected']}."
          + (f" This counts every original failure, including ones the model then fixed: "
             f"{s['repaired_by_model']} repaired version(s) were accepted." if s.get("repaired_by_model") else "")]
    for reason, n in list(s["rejected_by_reason"].items())[:TOP_REASONS]:
        L.append(f"  - {n} × {reason}")
    if s["episodes_without_promos"]:
        L.append(f"- Episodes with no accepted promo: {', '.join(map(str, s['episodes_without_promos']))}")
    bc = s.get("blind_check")
    if bc:
        L.append(f"- Blind standalone check (a separate call that sees only the clip's lines, no show context): "
                 f"{bc['checked']} top promos checked. Average standalone score {bc['picker_standalone_avg']}/10 "
                 f"from the picking model vs {bc['blind_standalone_avg']}/10 blind. Its notes on what a stranger "
                 f"would find confusing are on each promo's card"
                 + (f"; {bc['errors']} check(s) failed" if bc["errors"] else "") + ".")
        L.append(f"  - Failed (blind score < 5): {bc.get('blind_below_5', 0)} of {bc['checked']}. "
                 f"Distribution: {bc.get('blind_distribution', {})}")
    # Lead with what the results took in total (live + cached answers), then say how this run got them.
    u = s["llm_usage"]
    priced = meta.get("estimated_full_cost_usd") is not None
    line = (f"- Model answers behind these results: {u['calls'] + u.get('cached_calls', 0)}, tokens in/out "
            f"{u['input_tokens'] + u.get('cached_input_tokens', 0):,}/"
            f"{u['output_tokens'] + u.get('cached_output_tokens', 0):,}"
            + (f", est. cost ${meta['estimated_full_cost_usd']}" if priced else "") + ".")
    if not u["calls"] and not u.get("cached_calls"):
        line = f"- No model calls (keyword baseline). The run took {s['wall_seconds']}s."
    elif not u.get("cached_calls"):
        line += f" All made live in this run, which took {s['wall_seconds']}s."
    elif not u["calls"]:
        line += (f" This run reused all of them from the cache (made live in earlier runs), so it "
                 + ("cost $0" if priced else "made no model calls") + f" and took {s['wall_seconds']}s.")
    else:
        line += (f" This run made {u['calls']} live calls ({u['input_tokens']:,}/{u['output_tokens']:,} tokens"
                 + (f", ${meta['estimated_cost_usd']}" if priced else "")
                 + f") and reused {u['cached_calls']} from the cache; it took {s['wall_seconds']}s.")
    L += [line, ""]
    low = [p for p in res.promos if p.blind.get("standalone", p.scores.get("standalone", 10)) <= 5]
    if low:
        L += ["**May not stand alone** (blind score where checked, else the picking model's, ≤ 5): "
              + ", ".join(p.id for p in low), ""]
    if res.errors:
        L += ["**Errors:**", *[f"- {e}" for e in res.errors], ""]
    return "\n".join(L)


def _cell(t: str, n: int = CELL_CHARS) -> str:
    t = (t or "").replace("|", "/").replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def _card(p: Promo) -> list[str]:
    lines = [f"### {p.rank}. {p.id} · {p.title or '(untitled)'} · {p.duration:.0f}s", "",
             "| Segment | Start | End | Length |", "|---|---|---|---|"]
    for s in p.segments:
        lines.append(f"| Episode {s.episode} | {fmt_ts(s.start)} | {fmt_ts(s.end)} | {s.duration:.1f}s |")
    lines += ["",
              f"- **Hook (first {HOOK}):** {p.hook}",
              f"  - script: {p.hook_text}",
              f"- **Cliffhanger:** {p.cliffhanger}",
              f"  - script: {p.cliffhanger_text}",
              f"- **Why it works:** {p.why}",
              f"- **Why this rank:** {p.rank_reason}",
              f"- **Standalone:** {p.standalone_note or '-'}",
              f"- **Scores** hook {p.scores.get('hook')} · cliffhanger {p.scores.get('cliffhanger')} · "
              f"standalone {p.scores.get('standalone')} · composite {p.score}"]
    if p.blind:
        lines.append(f"- **Blind check** (sees only the clip): standalone {p.blind['standalone']}/10, "
                     f"keep watching {p.blind.get('keep_watching')}/10. Understood: \"{p.blind['understood']}\" "
                     f"Confusing: \"{p.blind['confusing']}\"")
    if p.fixes:
        lines.append(f"- **Auto-fixes:** {'; '.join(p.fixes)}")
    if p.warnings:
        lines.append(f"- ⚠ **Check:** {'; '.join(p.warnings)}")
    lines.append("")
    return lines
