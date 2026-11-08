"""Promo data model + deterministic validation / auto-repair (no AI).

The model proposes block ids; this module decides the actual timestamps and
enforces the brief's hard rules:
  * 20-60 s total (PROMO_MIN/MAX_SECONDS) -> reject (sent back to the model to fix)
  * hook within the first 3 s (PROMO_HOOK_WINDOW) -> auto-fix: move the start to the hook
  * clip ends on the cliffhanger       -> auto-fix: trim the tail to the cliffhanger
  * never start/end mid dialogue line  -> auto-fix: snap outward to line edges
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import S
from .parser import Block, Episode, fmt_ts, normalize_block_id

# Rules: all from .env (PROMO_*), defaults match the brief. See config.py.
MIN_LEN, MAX_LEN = S.min_seconds, S.max_seconds   # promo length bounds
HOOK_WINDOW = S.hook_window   # hook must start within this many seconds of the clip start
CLIFF_TAIL = S.cliff_tail     # allowed reaction shot after the cliffhanger line
TOL = S.length_tolerance      # tolerance on the length bounds
MAX_SEGMENTS = S.max_segments
MIN_SEGMENT = S.min_segment_seconds

# Internal constants (technical, not policy, so they stay in code):
EPS = 0.05             # seconds: timestamps closer than this count as the same moment
EPS_BLOCK = 0.01       # seconds: tolerance when finding the block at a cut point
DEFAULT_SCORE = 5      # used when the model leaves a 1-10 score out
MAX_TITLE, MAX_TAGS = 80, 6


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_block: str
    end_block: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def label(self) -> str:
        return f"E{self.episode:02d} {fmt_ts(self.start)}-{fmt_ts(self.end)}"


@dataclass
class Promo:
    segments: list[Segment]
    title: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    standalone_note: str = ""
    hook_block: str = ""
    cliffhanger_block: str = ""
    tags: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    fixes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "llm"           # llm | repair | heuristic
    kind: str = "single"          # single | multi
    id: str = ""
    score: float = 0.0
    rank: int | None = None
    rank_reason: str = ""
    hook_text: str = ""           # verbatim script line of the hook block, shown next to the
                                  # model's description so a reviewer can spot a mislabel
    cliffhanger_text: str = ""
    hook_at: float = 0.0          # episode time where the hook block starts (for duplicate checks)
    cliff_at: float = 0.0         # episode time where the cliffhanger block ends
    blind: dict = field(default_factory=dict)  # blind standalone check result (top promos only)

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        d["episodes"] = self.episodes
        for s, sd in zip(self.segments, d["segments"]):
            sd["start_ts"], sd["end_ts"], sd["duration"] = fmt_ts(s.start), fmt_ts(s.end), s.duration
        return d


# ---------------------------------------------------------------- snapping
def snap_start(ep: Episode, t: float) -> float:
    """Move t earlier until it is not inside a line of dialogue."""
    changed = True
    while changed:
        changed = False
        for b in ep.blocks:
            if b.is_dialogue and b.start < t - EPS and b.end > t + EPS:
                t, changed = b.start, True
    return round(t, 2)


def snap_end(ep: Episode, t: float) -> float:
    """Move t later until it is not inside a line of dialogue."""
    changed = True
    while changed:
        changed = False
        for b in ep.blocks:
            if b.is_dialogue and b.start < t - EPS and b.end > t + EPS:
                t, changed = b.end, True
    return round(min(t, ep.duration or t), 2)


def block_starting_at(ep: Episode, t: float) -> Block:
    after = [b for b in ep.blocks if b.start >= t - EPS_BLOCK]
    return min(after, key=lambda b: b.start) if after else ep.blocks[-1]


def block_ending_at(ep: Episode, t: float) -> Block:
    before = [b for b in ep.blocks if b.end <= t + EPS_BLOCK]
    return max(before, key=lambda b: (b.end, b.start)) if before else ep.blocks[0]


def describe(b: Block) -> str:
    if b.is_dialogue:
        return f'{b.speaker}{f" ({b.emotion})" if b.emotion else ""}: "{b.text}"'
    return f"{b.kind.upper()}: {b.text}"


def _score(v, warnings: list[str], name: str) -> int:
    try:
        return max(1, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        warnings.append(f"missing {name} score; defaulted to {DEFAULT_SCORE}")
        return DEFAULT_SCORE


# ---------------------------------------------------------------- validation
def validate(raw: dict, ep: Episode, source: str = "llm") -> tuple[Promo | None, list[str]]:
    """Return (promo, errors). errors non-empty -> promo is None and can be sent for repair."""
    errors: list[str] = []
    fixes: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, ["promo is not a JSON object"]

    segs_raw = raw.get("segments")
    if not segs_raw and raw.get("start_block"):
        segs_raw = [{"start_block": raw.get("start_block"), "end_block": raw.get("end_block")}]
    if not isinstance(segs_raw, list) or not segs_raw:
        return None, ["no segments given"]
    if len(segs_raw) > MAX_SEGMENTS:
        return None, [f"{len(segs_raw)} segments; max is {MAX_SEGMENTS}"]

    segs: list[Segment] = []
    for s in segs_raw:
        if not isinstance(s, dict):  # e.g. ["B010", "B031"] instead of [{"start_block": ...}]
            return None, [f"segment {s!r} is not an object with start_block/end_block"]
        sid, eid = str(s.get("start_block", "")), str(s.get("end_block", ""))
        si, ei = ep.index_of(sid), ep.index_of(eid)
        if si < 0 or ei < 0:
            errors.append(f"unknown block id in segment {sid}..{eid}")
            continue
        if ei < si:
            si, ei = ei, si
            fixes.append(f"swapped reversed segment {sid}..{eid}")
        span = ep.blocks[si:ei + 1]
        start = snap_start(ep, ep.blocks[si].start)
        end = snap_end(ep, max(b.end for b in span))
        if start != ep.blocks[si].start or end != max(b.end for b in span):
            fixes.append(f"snapped segment {sid}..{eid} outward to dialogue line edges")
        segs.append(Segment(ep.number, start, end, ep.blocks[si].id, ep.blocks[ei].id))
    if errors:
        return None, errors

    # chronological order + merge overlaps
    segs.sort(key=lambda s: s.start)
    merged: list[Segment] = []
    for s in segs:
        if merged and s.start <= merged[-1].end + EPS:
            merged[-1].end = max(merged[-1].end, s.end)
            merged[-1].end_block = s.end_block
            fixes.append("merged overlapping segments")
        else:
            merged.append(s)
    segs = merged

    # ---- hook: must land in the first 3 s of the first segment
    hb = ep.block(str(raw.get("hook_block", "")))
    first = segs[0]
    if hb is None:
        hb = ep.block(first.start_block)
        warnings.append("no valid hook_block; assumed first block of the clip")
    if hb.start < first.start - EPS:
        new = snap_start(ep, hb.start)
        fixes.append(f"started {first.start - new:.1f}s earlier to include the hook")
        first.start, first.start_block = new, block_starting_at(ep, new).id
    elif hb.start - first.start > HOOK_WINDOW:
        if hb.start > first.end:
            return None, [f"hook_block {hb.id} is not inside the first segment"]
        new = snap_start(ep, hb.start)
        fixes.append(f"cut {new - first.start:.1f}s of lead-in so the hook lands in the first {HOOK_WINDOW:g}s")
        first.start, first.start_block = new, block_starting_at(ep, new).id

    # ---- cliffhanger: the clip must end on it (plus a short reaction shot)
    cb = ep.block(str(raw.get("cliffhanger_block", "")))
    last = segs[-1]
    if cb is None:
        cb = ep.block(last.end_block)
        warnings.append("no valid cliffhanger_block; assumed last block of the clip")
    if cb.end > last.end + EPS:
        new = snap_end(ep, cb.end)
        fixes.append(f"extended end {new - last.end:.1f}s to include the cliffhanger")
        last.end, last.end_block = new, block_ending_at(ep, new).id
    elif last.end - cb.end > CLIFF_TAIL:
        if cb.end < last.start:
            return None, [f"cliffhanger_block {cb.id} is not inside the last segment"]
        new = snap_end(ep, cb.end)
        fixes.append(f"trimmed {last.end - new:.1f}s after the cliffhanger so the clip cuts on it")
        last.end, last.end_block = new, block_ending_at(ep, new).id

    for s in segs:
        if s.duration < MIN_SEGMENT:
            return None, [f"segment {s.start_block}..{s.end_block} is under {MIN_SEGMENT:g}s"]

    total = round(sum(s.duration for s in segs), 2)
    if total < MIN_LEN - TOL:
        return None, [f"total length {total:.1f}s is under {MIN_LEN:.0f}s (after hook/cliffhanger fixes)"]
    if total > MAX_LEN + TOL:
        return None, [f"total length {total:.1f}s is over {MAX_LEN:.0f}s (after hook/cliffhanger fixes)"]

    scores_raw = raw.get("scores") or {}
    scores = {k: _score(scores_raw.get(k), warnings, k) for k in ("hook", "cliffhanger", "standalone")}
    if hb.kind == "camera":
        warnings.append("hook is a camera direction only; its impact depends on the actual footage")
    if cb.kind in ("camera", "sfx"):
        warnings.append(f"cliffhanger is a {cb.kind} block; check the footage actually lands it")

    promo = Promo(
        segments=segs,
        title=str(raw.get("title", "")).strip()[:MAX_TITLE],
        hook=str(raw.get("hook", "")).strip(),
        cliffhanger=str(raw.get("cliffhanger", "")).strip(),
        why=str(raw.get("why", "")).strip(),
        standalone_note=str(raw.get("standalone_note", "")).strip(),
        hook_block=hb.id,
        cliffhanger_block=cb.id,
        tags=[str(t).lower() for t in (raw.get("tags") or [])][:MAX_TAGS],
        scores=scores,
        fixes=fixes,
        warnings=warnings,
        source=source,
        kind="multi" if len(segs) > 1 else "single",
        hook_text=describe(hb),
        cliffhanger_text=describe(cb),
        hook_at=hb.start,
        cliff_at=cb.end,
    )
    promo.score = composite(promo)
    return promo, []


def composite(p: Promo) -> float:
    """Pre-rank score on a 10-point scale: the weighted average of the hook, cliffhanger and
    standalone scores (PROMO_SCORE_WEIGHTS). For standalone, the blind check's score is used when
    there is one: the picking model has read the whole episode, so its own score is optimistic."""
    s = p.scores
    w_hook, w_cliff, w_stand = S.score_weights
    total = (w_hook + w_cliff + w_stand) or 1.0
    stand = p.blind.get("standalone", s.get("standalone", DEFAULT_SCORE))
    base = (w_hook * s.get("hook", DEFAULT_SCORE) + w_cliff * s.get("cliffhanger", DEFAULT_SCORE)
            + w_stand * stand) / total   # weights needn't sum to 1
    return round(base, 2)


def overlap_ratio(a: Promo, b: Promo) -> float:
    """Shared screen time as a fraction of the shorter promo."""
    shared = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                shared += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    shorter = min(a.duration, b.duration) or 1.0
    return shared / shorter


__all__ = ["Promo", "Segment", "validate", "composite", "overlap_ratio", "snap_start", "snap_end",
           "describe", "normalize_block_id", "MIN_LEN", "MAX_LEN"]
