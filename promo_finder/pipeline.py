"""End-to-end pipeline.

  parse (code) -> per-episode candidates (LLM, or keyword heuristic offline)
  -> validate + auto-fix (code) -> one repair round for failures (LLM)
  -> remove duplicates (code) -> posting order for the show (LLM, code fallback)
  -> blind standalone check on the top promos (LLM)
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import heuristic, prompts
from .config import S
from .llm import LLM, LLMError, OfflineLLM
from .parser import Episode, Show, fmt_ts, render_blocks
from .promos import DEFAULT_SCORE, EPS, Promo, composite, overlap_ratio, validate


@dataclass
class RunConfig:
    """Defaults come from .env via config.S; the CLI overrides the ones it has flags for."""
    max_candidates: int = S.per_episode     # per episode
    workers: int = S.workers
    include_camera: bool = S.include_camera
    story_window: int = S.story_window      # previous loglines given as context
    shortlist: int = S.shortlist            # how many go to the ranking model
    top_k: int = S.top                      # how many it orders
    blind_check_top: int = S.blind_check_top


@dataclass
class RunResult:
    show: str
    model: str
    promos: list[Promo]
    rejected: list[dict]
    raw: dict
    stats: dict
    errors: list[str] = field(default_factory=list)


def parse_json(text: str) -> dict:
    """Read the model's JSON even when it adds ```json fences, chatter or trailing commas."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    attempts = [t]
    if "{" in t:
        attempts.append(t[t.find("{"): t.rfind("}") + 1])
    for a in list(attempts):
        attempts.append(re.sub(r",\s*([}\]])", r"\1", a))  # trailing commas
    for a in attempts:
        try:
            d = json.loads(a)
        except json.JSONDecodeError:
            continue
        if isinstance(d, list):
            return {"promos": d}
        if isinstance(d, dict):
            return d
        break  # valid JSON but a number/string: not something we can use
    raise ValueError(f"could not parse a JSON object from model output: {text[:200]!r}")


def call_record(kind: str, resp) -> dict:
    """What we keep in raw/ for every model call: the unedited text plus tokens."""
    rec = {"type": kind, "model": resp.model, "cached": resp.cached,
           "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}
    if resp.meta.get("thinking_tokens"):
        rec["thinking_tokens"] = resp.meta["thinking_tokens"]
    rec["text"] = resp.text
    return rec


class PromoPipeline:
    def __init__(self, llm: LLM, cfg: RunConfig | None = None, log=print):
        self.llm = llm
        self.cfg = cfg or RunConfig()
        self.log = log
        self.offline = isinstance(llm, OfflineLLM)

    # ------------------------------------------------------------ public
    def run(self, show: Show, episodes: list[int] | None = None) -> RunResult:
        t0 = time.time()
        eps = [e for e in show.episodes if not episodes or e.number in episodes]
        raw: dict = {"episodes": {}, "rank": None, "blind": None}
        accepted: list[Promo] = []
        rejected: list[dict] = []
        errors: list[str] = []

        def job(ep: Episode):
            # An unexpected error in ONE episode (odd model output, a bug) is recorded for that
            # episode; the rest of the run continues. Rerunning later reuses the cache.
            try:
                return ep, self._episode(show, ep)
            except Exception as e:  # noqa: BLE001 - deliberately broad, see above
                return ep, ([], [], {"episode": ep.number, "calls": [],
                                     "error": f"unexpected {type(e).__name__}: {e}"})

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as pool:
            futures = [pool.submit(job, ep) for ep in eps]
            for n, fut in enumerate(as_completed(futures), 1):
                ep, (acc, rej, rec) = fut.result()
                accepted += acc
                rejected += rej
                raw["episodes"][ep.number] = rec
                if rec.get("error"):
                    errors.append(f"E{ep.number:02d}: {rec['error']}")
                self.log(f"[{n}/{len(eps)}] E{ep.number:02d}: {len(acc)} accepted, {len(rej)} rejected"
                         + (f"  ERROR: {rec['error']}" if rec.get("error") else ""))

        # Episodes finish in a different order each run (they run in parallel), so put promos in a
        # fixed order first: ties then always break the same way and reruns give identical results
        # (and identical model requests, so the cache is hit).
        accepted.sort(key=lambda p: (p.episodes[0], p.segments[0].start, p.title))
        rejected.sort(key=lambda r: (r.get("episode") or 0, str(r.get("title"))))
        promos, dups = self._dedupe(accepted)
        rejected += dups
        promos.sort(key=lambda p: (-p.score, p.episodes[0], p.segments[0].start))

        # Blind check BEFORE ranking, so it counts: the blind score replaces the picking model's
        # own standalone score in the composite, and the ranking model sees it on each card.
        # It repeats until every promo in the top N has been checked: checked promos usually score
        # lower than the picking model's self-grade, so unchecked ones would otherwise overtake
        # them and reach the shortlist without ever being checked.
        raw["blind"] = None
        tried: set[int] = set()
        while not self.offline:
            todo = [p for p in promos[: self.cfg.blind_check_top] if id(p) not in tried]
            if not todo:
                break
            tried.update(id(p) for p in todo)
            rec = self._blind_check(show, todo)
            if raw["blind"] is None:
                raw["blind"] = {"calls": [], "rounds": 0}
            raw["blind"]["calls"] += rec["calls"]
            raw["blind"]["rounds"] += 1
            if rec.get("errors"):
                raw["blind"].setdefault("errors", []).extend(rec["errors"])
            for p in promos:
                p.score = composite(p)
            promos.sort(key=lambda p: (-p.score, p.episodes[0], p.segments[0].start))
        promos.sort(key=lambda p: (-p.score, p.episodes[0], p.segments[0].start))
        for i, p in enumerate(promos, 1):
            p.id = f"P{i:03d}"

        ranked, raw["rank"] = self._rank(show, promos)

        stats = self._stats(eps, ranked, rejected, time.time() - t0)
        checked = [p for p in ranked if p.blind]
        if checked:
            own = [p.scores.get("standalone", DEFAULT_SCORE) for p in checked]
            blind = [p.blind["standalone"] for p in checked]
            stats["blind_check"] = {
                "checked": len(checked),
                "picker_standalone_avg": round(sum(own) / len(own), 1),
                "blind_standalone_avg": round(sum(blind) / len(blind), 1),
                "blind_below_5": sum(1 for p in checked if p.blind.get("standalone", 10) < 5),
                "blind_distribution": dict(sorted(Counter(p.blind.get("standalone") for p in checked).items())),
                "errors": len((raw["blind"] or {}).get("errors", [])),
            }
        return RunResult(show=show.title, model=self.llm.spec, promos=ranked,
                         rejected=rejected, raw=raw, stats=stats, errors=errors)

    # ------------------------------------------------------------ per episode
    def _episode(self, show: Show, ep: Episode) -> tuple[list[Promo], list[dict], dict]:
        rec: dict = {"episode": ep.number, "calls": []}
        acc: list[Promo] = []
        rej: list[dict] = []
        if not ep.blocks:
            rec["error"] = "episode has no parsable blocks"
            return acc, rej, rec
        transcript = render_blocks(ep.blocks, self.cfg.include_camera)
        system = prompts.find_system(self.cfg.max_candidates)
        if self.offline:
            cands = heuristic.propose(ep, self.cfg.max_candidates)
            rec["calls"].append({"type": "heuristic", "output": cands})
        else:
            prior = [e.logline for e in show.episodes if e.number < ep.number][-self.cfg.story_window:]
            user = prompts.find_user(show.title, ep.number, ep.logline, prior, fmt_ts(ep.duration), transcript)
            try:
                cands = self._ask(system, user, rec, "find")
            except (LLMError, ValueError) as e:
                rec["error"] = str(e)
                return acc, rej, rec

        failed: list[tuple[dict, list[str]]] = []
        for c in cands:
            p, errs = validate(c, ep, source="heuristic" if self.offline else "llm")
            if p:
                acc.append(p)
            else:
                failed.append((c, errs))

        # Every original failure is logged (the quality report counts them), then one repair round.
        for c, errs in failed:
            rej.append({"episode": ep.number, "stage": "validation", "reasons": errs,
                        "title": c.get("title") if isinstance(c, dict) else None, "candidate": c})
        if failed and not self.offline:
            failures = "\n".join(json.dumps({"promo": c, "errors": e}, ensure_ascii=False) for c, e in failed)
            user = prompts.repair_user(failures, transcript)
            try:
                fixed = self._ask(system, user, rec, "repair")
            except (LLMError, ValueError) as e:
                fixed = []
                rec["repair_error"] = str(e)
            for c in fixed:
                p, errs = validate(c, ep, source="repair")
                if p:
                    p.fixes.insert(0, "re-proposed by the model after failing automatic checks")
                    acc.append(p)
                else:
                    rej.append({"episode": ep.number, "stage": "repair", "reasons": errs,
                                "title": c.get("title"), "candidate": c})
        rec["accepted"] = [p.title for p in acc]
        return acc, rej, rec

    def _ask(self, system: str, user: str, rec: dict, kind: str) -> list[dict]:
        resp = self.llm.generate(system, user)
        rec["calls"].append(call_record(kind, resp))
        try:
            data = parse_json(resp.text)
        except ValueError:
            resp = self.llm.generate(system, user + "\n\nYour previous reply was not valid JSON. "
                                                    "Reply with the JSON object only.")
            rec["calls"].append(call_record(kind + "-retry", resp))
            data = parse_json(resp.text)
        promos = data.get("promos", []) if isinstance(data, dict) else []
        return [p for p in promos if isinstance(p, dict)]

    # ------------------------------------------------------------ show level
    def _dedupe(self, promos: list[Promo]) -> tuple[list[Promo], list[dict]]:
        """Drop a promo only if it's the SAME ad as a better one already kept: same hook moment,
        same cliffhanger moment, or most of the same footage. Overlapping clips with different
        hooks and endings are different ads, so they stay."""
        keep: list[Promo] = []
        dropped: list[dict] = []
        for p in sorted(promos, key=lambda p: -p.score):
            reason = None
            for k in keep:
                reason = duplicate_reason(p, k)
                if reason:
                    dropped.append({"episode": p.episodes[0], "stage": "dedupe", "title": p.title,
                                    "reasons": [f"{reason} '{k.title}'"], "candidate": p.to_dict()})
                    break
            if not reason:
                keep.append(p)
        return keep, dropped

    def _rank(self, show: Show, promos: list[Promo]) -> tuple[list[Promo], dict | None]:
        """Posting order: the ranking model orders the shortlist; the rest (or everything, if the
        model is offline or its answer is unusable) follows in score order."""
        by_id = {p.id: p for p in promos}
        judged: list[Promo] = []
        rank_raw: dict | None = None
        if not self.offline and len(promos) >= 2:
            shortlist = promos[: self.cfg.shortlist]
            cards = [{"id": p.id, "episodes": p.episodes, "seconds": p.duration, "title": p.title,
                      "hook": p.hook, "cliffhanger": p.cliffhanger, "why": p.why,
                      "standalone_note": p.standalone_note, "tags": p.tags, "scores": p.scores,
                      "blind_standalone": p.blind.get("standalone")}
                     for p in shortlist]
            rank_raw = {"calls": []}
            try:
                resp = self.llm.generate(prompts.rank_system(min(self.cfg.top_k, len(shortlist))),
                                         prompts.rank_user(show.title, cards))
                rank_raw["calls"].append(call_record("rank", resp))
                for item in parse_json(resp.text).get("ranking") or []:
                    pid = str(item.get("id", "")) if isinstance(item, dict) else str(item)
                    p = by_id.get(pid)
                    if p and p not in judged:
                        p.rank_reason = item.get("reason", "") if isinstance(item, dict) else ""
                        judged.append(p)
            except (LLMError, ValueError, AttributeError, TypeError) as e:
                rank_raw["error"] = f"ranking model's answer not used ({type(e).__name__}: {e}); " \
                                    "ordered by score instead"
                judged = []
                for p in promos:
                    p.rank_reason = ""

        ordered = judged + [p for p in promos if p not in judged]  # promos are already in score order
        # Variety is a hard rule in code: the ranking model was asked for it but still clustered
        # one storyline at the top (5 of 15 from E41-E46 in the first full run).
        ordered = cap_per_episode(ordered, S.top_cap_n, S.top_max_per_episode, S.top_episode_spread)
        for i, p in enumerate(ordered, 1):
            p.rank = i
            if not p.rank_reason:
                p.rank_reason = "ordered by score (not ranked by the model)" if judged else "ordered by score"
        return ordered, rank_raw

    def _blind_check(self, show: Show, promos: list[Promo]) -> dict | None:
        """Ask a fresh model call, which sees ONLY the clip's lines, whether a stranger follows it.

        Runs before ranking: its score replaces the picking model's own standalone score in the
        composite, the ranking model sees it, and a low score adds a warning for the reviewer.
        """
        if self.offline:
            return None
        targets = promos
        record: dict = {"calls": []}

        def check(p: Promo):
            resp = self.llm.generate(prompts.BLIND_SYSTEM, prompts.blind_user(clip_text(show, p)))
            return p, resp, parse_json(resp.text)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as pool:
            for fut in as_completed([pool.submit(check, p) for p in targets]):
                try:
                    p, resp, data = fut.result()
                except Exception as e:  # noqa: BLE001 - a failed check must not lose the run
                    record.setdefault("errors", []).append(f"{type(e).__name__}: {e}")
                    continue
                rec = call_record("blind", resp)
                rec["promo"] = f"{p.title} ({' + '.join(s.label() for s in p.segments)})"  # ids come later
                record["calls"].append(rec)
                try:
                    blind_score = max(1, min(10, int(round(float(data.get("standalone"))))))
                except (TypeError, ValueError):
                    continue
                p.blind = {"standalone": blind_score, "keep_watching": data.get("keep_watching"),
                           "understood": str(data.get("understood", "")),
                           "confusing": str(data.get("confusing", ""))}
                if blind_score < S.blind_fail_below:
                    p.warnings.append(f"blind check failed ({blind_score}/10): a new viewer read it as "
                                      f"'{p.blind['understood'][:140]}' and was confused by: "
                                      f"{p.blind['confusing'][:140]}")
        return record

    def _stats(self, eps, promos, rejected, seconds) -> dict:
        reasons: dict[str, int] = {}
        for r in rejected:
            # Group similar reasons: hide the specifics (numbers, block ids, titles)
            # so "same cliffhanger moment as 'X'" and "... as 'Y'" count as one reason.
            key = r["stage"] + ": " + (r.get("reasons") or ["?"])[0]
            key = re.sub(r"'.*'", "'…'", key)                 # quoted title (may contain an apostrophe)
            key = re.sub(r"B\d{3}", "Bxxx", key)               # block ids
            key = re.sub(r"\d+(\.\d+)?%", "N%", key)           # percentages
            key = re.sub(r"\d+(\.\d+)?s\b", "Ns", key)         # seconds
            reasons[key] = reasons.get(key, 0) + 1
        durs = [p.duration for p in promos]
        return {
            "episodes": len(eps),
            "promos": len(promos),
            "rejected": len(rejected),
            "rejected_by_reason": dict(sorted(reasons.items(), key=lambda x: -x[1])),
            "auto_fixed": sum(1 for p in promos if p.fixes),
            "repaired_by_model": sum(1 for p in promos if p.source == "repair"),
            "with_warnings": sum(1 for p in promos if p.warnings),
            "episodes_without_promos": [e.number for e in eps
                                        if not any(e.number in p.episodes for p in promos)],
            "duration_avg": round(sum(durs) / len(durs), 1) if durs else 0,
            "duration_min": min(durs, default=0),
            "duration_max": max(durs, default=0),
            "llm_usage": self.llm.usage.as_dict(),
            "wall_seconds": round(seconds, 1),
        }


def duplicate_reason(p: Promo, k: Promo) -> str | None:
    """Why p is the same ad as k, or None if it isn't."""
    near = S.dedupe_same_moment
    if p.segments[0].episode == k.segments[0].episode and abs(p.hook_at - k.hook_at) <= near:
        return "same hook moment as"
    if p.segments[-1].episode == k.segments[-1].episode and abs(p.cliff_at - k.cliff_at) <= near:
        return "same cliffhanger moment as"
    ov = overlap_ratio(p, k)
    if ov > S.dedupe_overlap:
        return f"overlaps {ov:.0%} with"
    return None


def clip_text(show: Show, p: Promo) -> str:
    """What a stranger sees and hears in the clip: no block ids, timestamps, titles or loglines."""
    parts = []
    for s in p.segments:
        ep = show.episode(s.episode)
        lines = []
        for b in (ep.blocks if ep else []):
            if s.start - EPS <= b.start < s.end:
                if b.is_dialogue:
                    tone = f" ({b.emotion})" if b.emotion else ""
                    lines.append(f'{b.speaker}{tone}: "{b.text}"')
                else:
                    lines.append(f"[{b.kind}] {b.text}")
        parts.append("\n".join(lines))
    return "\n--- cut ---\n".join(parts)



def cap_per_episode(ordered: list[Promo], top_n: int, max_per_ep: int, spread: int = 0) -> list[Promo]:
    """Variety rule: at most `max_per_ep` promos from the same episode (± `spread` episodes, i.e.
    one storyline) in the top `top_n`. A promo that would break it keeps its relative order and
    slides to just below the top `top_n`."""
    head: list[Promo] = []
    bumped: list[Promo] = []
    tail: list[Promo] = []
    for p in ordered:
        if len(head) >= top_n:
            tail.append(p)
            continue
        crowded = any(sum(1 for q in head for qe in q.episodes if abs(qe - e) <= spread) >= max_per_ep
                      for e in p.episodes)
        if crowded:
            near = f"E{p.episodes[0]:02d}" + (f" ±{spread}" if spread else "")
            p.rank_reason = (f"moved below the top {top_n}: already {max_per_ep} promos from {near} "
                             f"there (variety rule in code; ranking model's reason: {p.rank_reason or '-'})")
            bumped.append(p)
        else:
            head.append(p)
    return head + bumped + tail
