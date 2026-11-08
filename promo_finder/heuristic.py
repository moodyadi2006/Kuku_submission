"""No-model baseline: keyword + emotion-tag scoring.

Purpose: (1) the tool runs end to end with no API key, (2) a floor to compare
LLM picks against in the quality check. It emits the SAME JSON the LLM does, so
it goes through the same validation.
"""
from __future__ import annotations

import re

from .config import S
from .parser import Block, Episode
from .promos import DEFAULT_SCORE, MAX_LEN, MIN_LEN, describe

# Aim a little inside the allowed range (PROMO_SAFETY_MARGIN_SECONDS), so snapping cuts to
# line edges can't push a clip out of it.
TARGET_MIN, TARGET_MAX = MIN_LEN + S.safety_margin, MAX_LEN - S.safety_margin

# Tuning for this baseline (it is a deliberately simple yardstick, so these stay in code):
MIN_HOOK_SCORE = 2.0      # lines scoring below this are never used as hooks
SCORE_OFFSET = 2          # keyword points + this = the 1-10 score shown in the report
MARKER_WINDOW = 2.5       # seconds after a line in which a "cliffhanger"/"cut to black" note counts
EPISODE_END_WINDOW = 6.0  # seconds before the episode's end that count as "the episode cliffhanger"
OVERLAP_MARGIN = 5.0      # a new hook this close before an already-chosen clip is skipped

HOOK_WORDS = {
    # physical / violent beats
    "slap": 4, "slaps": 4, "strike": 3, "strikes": 3, "struck": 3, "punch": 3, "crop": 3, "whip": 3,
    "blood": 3, "bleeding": 3, "knife": 3, "blade": 3, "sword": 3, "dagger": 3, "chains": 2.5,
    "chained": 2.5, "fire": 3, "flames": 3, "burn": 3, "burning": 3, "explosion": 4, "explodes": 4,
    "shatters": 2.5, "throws": 2, "shoves": 2.5, "grabs": 2, "seize": 2, "seized": 2, "drags": 2.5,
    "kneel": 2, "collapses": 2.5, "faints": 2.5, "poison": 3, "poisoned": 3, "arrest": 2.5,
    "kiss": 3, "kisses": 3, "iron": 2, "dungeon": 1.5, "execution": 3, "hang": 2.5, "witch": 2.5,
    # reveals / humiliation
    "reveal": 2.5, "reveals": 2.5, "duke": 2, "duchess": 1.5, "betray": 2.5, "abandon": 2,
    "disown": 3, "disowned": 3, "humiliat": 2.5, "pregnant": 3, "married": 1.5, "marrying": 2,
    "liar": 2, "thief": 2, "murder": 3, "dead": 2, "alive": 2,
}
HOOK_EMOTIONS = {
    "shocked": 2, "furious": 2, "outraged": 2, "gloating": 2, "cruel": 2, "vicious": 2, "mocking": 1.5,
    "menacing": 2, "threatening": 2, "screaming": 2, "sobbing": 1.5, "horrified": 2, "venomous": 2,
    "cold": 1, "smug": 1.5, "contemptuous": 1.5, "enraged": 2, "desperate": 1.5, "seething": 2,
}
CLIFF_WORDS = {
    "will": 0.6, "never": 1, "until": 0.8, "before": 0.6, "kill": 2, "die": 2, "pay": 1.5,
    "destroy": 2, "burn": 1.5, "hang": 1.5, "execute": 2, "who are you": 2.5, "what have you": 2,
    "or else": 2, "you'll": 1, "unless": 1.2, "no one": 1, "secret": 1.5, "truth": 1.2, "wait": 1,
    "reveals": 2, "raises": 1, "lunges": 2, "draws": 1.2, "freezes": 1.5, "stops": 0.8, "turns": 0.6,
}
CLIFF_EMOTIONS = {"menacing": 1.5, "threatening": 1.5, "ominous": 1.5, "shocked": 1.2, "stunned": 1.2,
                  "cold": 0.8, "deadly": 1.5, "chilling": 1.5, "horrified": 1.2}


def _kw(text: str, table: dict[str, float]) -> tuple[float, list[str]]:
    t = text.lower()
    score, hits = 0.0, []
    for w, v in table.items():
        if (" " in w and w in t) or re.search(rf"\b{re.escape(w)}", t):
            score += v
            hits.append(w)
    return score, hits


def hook_score(b: Block) -> tuple[float, list[str]]:
    s, hits = _kw(b.text, HOOK_WORDS)
    if b.emotion:
        es, eh = _kw(b.emotion, HOOK_EMOTIONS)
        s, hits = s + es, hits + eh
    if b.is_dialogue:
        s += 0.8 * b.text.count("!") + (0.5 if len(b.text) < 60 else 0)  # short, punchy lines
    return s, hits


def cliff_score(b: Block, ep: Episode) -> float:
    s, _ = _kw(b.text, CLIFF_WORDS)
    if b.emotion:
        s += _kw(b.emotion, CLIFF_EMOTIONS)[0]
    txt = b.text.rstrip()
    if b.is_dialogue:
        s += 2.0 if txt.endswith("?") else 0
        s += 1.5 if txt.endswith(("...", "…", "—", "-")) else 0
        s += 0.6 if txt.endswith("!") else 0
    # The scriptwriter often marks the cut: "cliffhanger" SFX or "Cut to black" right after.
    for nb in ep.blocks:
        if b.end - 0.1 <= nb.start <= b.end + MARKER_WINDOW:
            low = nb.text.lower()
            if "cliffhanger" in low or "cut to black" in low or "ominous" in low:
                s += 2.5
                break
    if ep.duration and ep.duration - b.end < EPISODE_END_WINDOW:
        s += 1.5  # episodes are written to end on a cliffhanger
    return s


def propose(ep: Episode, max_candidates: int = 3) -> list[dict]:
    blocks = ep.blocks
    hooks = sorted(
        ((hook_score(b), i) for i, b in enumerate(blocks) if b.kind in ("dialogue", "action")),
        key=lambda x: -x[0][0],
    )
    taken: list[tuple[float, float]] = []
    out: list[dict] = []
    for (hs, hits), i in hooks:
        if hs < MIN_HOOK_SCORE or len(out) >= max_candidates:
            break
        hb = blocks[i]
        if any(a - OVERLAP_MARGIN <= hb.start <= b for a, b in taken):
            continue
        best: tuple[float, int] | None = None
        for j in range(i + 1, len(blocks)):
            cb = blocks[j]
            dur = cb.end - hb.start
            if dur > TARGET_MAX:
                break
            if dur < TARGET_MIN or cb.kind == "camera":
                continue
            cs = cliff_score(cb, ep) + (0.5 if S.sweet_spot_min <= dur <= S.sweet_spot_max else 0)
            if best is None or cs > best[0]:
                best = (cs, j)
        if best is None:
            continue
        cs, j = best
        cb = blocks[j]
        taken.append((hb.start, cb.end))
        out.append({
            "title": f"E{ep.number} {', '.join(hits[:2]) or 'dramatic beat'}",
            "segments": [{"start_block": hb.id, "end_block": cb.id}],
            "hook_block": hb.id,
            "cliffhanger_block": cb.id,
            "hook": describe(hb),
            "cliffhanger": describe(cb),
            "why": f"Keyword hook ({', '.join(hits[:3])}) followed by the strongest unresolved line in range.",
            "standalone_note": "not assessed by the heuristic",
            "tags": hits[:4],
            "scores": {"hook": min(10, round(SCORE_OFFSET + hs)), "cliffhanger": min(10, round(SCORE_OFFSET + cs)),
                       "standalone": DEFAULT_SCORE},  # the heuristic can't judge this
        })
    return out
