"""Deterministic screenplay parser (no AI).

Turns the timestamped screenplay format into structured, addressable blocks:

    **LADY JULIANA (Outraged, Accusing):**
    Start: (00:46.80) | End: (00:50.00)
    "You knew this before I walked through those doors!"

Every block gets a stable id (B001, B002, ...) per episode. The LLM only ever
refers to these ids, so it can never invent a timestamp: times come from here.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import S

TS_RE = re.compile(
    r"Start:\s*\((?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\)",
    re.I,
)
LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
SCENE_HEAD_RE = re.compile(r"^scene heading:\s*(.*)$", re.I)
EP_NUM_RE = re.compile(r"^EPISODE NUMBER:\s*.*?(\d+)", re.I)
LOGLINE_RE = re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.I)
CONTINUITY_RE = re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.I)
FILE_EP_RE = re.compile(r"episode[-_ ]?0*(\d+)", re.I)
HEADER_RE = re.compile(r"^(EPISODE [A-Z ]+|CONTINUITY NOTES|SCENE [A-Z ]+|---)\b", re.I)
HEADER_LINES = 20  # the episode header (number, logline, notes) is looked for in this many lines
BOM = chr(0xFEFF)  # invisible byte-order mark some editors put at the start of a file
# Parentheticals that are delivery directions, not emotions.
NON_EMOTION = {"o.s.", "v.o.", "cont'd", "contd", "o.c.", "off screen", "voice over"}


@dataclass
class Block:
    id: str
    episode: int
    scene: int
    kind: str  # dialogue | camera | action | sfx | other
    label: str
    speaker: str | None
    emotion: str | None
    text: str
    start: float
    end: float

    @property
    def is_dialogue(self) -> bool:
        return self.kind == "dialogue"


@dataclass
class Episode:
    number: int
    show: str
    logline: str
    continuity: str
    source: str
    blocks: list[Block] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.blocks), default=0.0)

    def block(self, block_id: str) -> Block | None:
        return self._index().get(normalize_block_id(block_id))

    def index_of(self, block_id: str) -> int:
        b = self.block(block_id)
        return -1 if b is None else self.blocks.index(b)

    def _index(self) -> dict[str, Block]:
        if not hasattr(self, "_idx_cache"):
            self._idx_cache = {b.id: b for b in self.blocks}
        return self._idx_cache


@dataclass
class Show:
    title: str
    episodes: list[Episode]
    warnings: list[str] = field(default_factory=list)

    def episode(self, number: int) -> Episode | None:
        return next((e for e in self.episodes if e.number == number), None)


BLOCK_ID_RE = re.compile(r"\bB\s*(\d+)", re.I)   # 'B008', 'b8', 'B 8', inside any text
TRAILING_NUM_RE = re.compile(r"(\d+)\s*$")        # bare '8' or '...8'


def normalize_block_id(raw: str) -> str:
    """Map whatever the model wrote to our canonical id.

    'B008', 'b8', 'B 8', 'B008.', 'B008 (Matriarch)', 'E01-B008', '8'  ->  'B008'
    1. Prefer a 'B' followed by a number anywhere in the text (so trailing words or
       punctuation don't matter, and the episode number in 'E01-B008' is ignored).
    2. Otherwise fall back to a bare number at the end ('8').
    3. Otherwise return the text unchanged; the lookup then fails and the checker
       reports it as an unknown block.
    """
    text = str(raw).strip()
    m = BLOCK_ID_RE.search(text) or TRAILING_NUM_RE.search(text)
    return f"B{int(m.group(1)):03d}" if m else text.upper()


def _secs(h: str | None, m: str, s: str) -> float:
    return (int(h) * 3600 if h else 0) + int(m) * 60 + float(s)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _classify(label: str) -> tuple[str, str | None, str | None]:
    up = label.strip().upper()
    if up == "CAMERA":
        return "camera", None, None
    if up == "ACTION":
        return "action", None, None
    if up.startswith("SOUND") or up in {"SFX", "MUSIC"}:
        return "sfx", None, None
    speaker = label.split("(")[0].strip()
    parens = [p.strip() for p in re.findall(r"\(([^)]*)\)", label)]
    emotions = [p for p in parens if p.lower() not in NON_EMOTION]
    return "dialogue", speaker or None, (emotions[-1] if emotions else None)


def parse_episode(text: str, source: str = "") -> Episode:
    text = text.lstrip(BOM)  # drop a byte-order mark (some Windows editors add one)
    lines = text.splitlines()
    # A title line is optional: the input pack's files start straight with "EPISODE NUMBER:",
    # and the show name only exists as the folder name (filled in by _build_show).
    first = next((l.strip() for l in lines if l.strip()), "")
    show = "" if HEADER_RE.match(first) or LABEL_RE.match(first) or SCENE_RE.match(first) else first
    number, logline, continuity = None, "", ""
    for l in lines[:HEADER_LINES]:
        if (m := EP_NUM_RE.match(l.strip())):
            number = int(m.group(1))
        elif (m := LOGLINE_RE.match(l.strip())):
            logline = m.group(1).strip()
        elif (m := CONTINUITY_RE.match(l.strip())):
            continuity = m.group(1).strip()
    if number is None:
        m = FILE_EP_RE.search(Path(source).name)
        number = int(m.group(1)) if m else 0

    ep = Episode(number=number, show=show, logline=logline, continuity=continuity, source=source)
    scene_no = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (m := SCENE_RE.match(line)):
            scene_no = int(m.group(1))
            ep.scenes.append({"scene": scene_no, "heading": ""})
        elif (m := SCENE_HEAD_RE.match(line)) and ep.scenes:
            ep.scenes[-1]["heading"] = m.group(1).strip()
        elif (m := LABEL_RE.match(line)):
            label = m.group(1).strip()
            # Find timestamp on the next non-empty line.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            ts = TS_RE.search(lines[j]) if j < len(lines) else None
            if not ts:
                ep.warnings.append(f"line {i + 1}: label '{label}' has no timestamp; skipped")
                i += 1
                continue
            start = _secs(ts.group(1), ts.group(2), ts.group(3))
            end = _secs(ts.group(4), ts.group(5), ts.group(6))
            if end < start:
                ep.warnings.append(f"line {j + 1}: end before start for '{label}'; swapped")
                start, end = end, start
            # Text runs until the next label / scene marker / script end.
            k = j + 1
            body: list[str] = []
            while k < len(lines):
                nxt = lines[k].strip()
                if LABEL_RE.match(nxt) or SCENE_RE.match(nxt) or nxt.startswith("--- SCRIPT"):
                    break
                if nxt:
                    body.append(nxt)
                k += 1
            kind, speaker, emotion = _classify(label)
            txt = " ".join(body).strip()
            if kind == "dialogue":
                txt = txt.strip('"“” ')
            ep.blocks.append(
                Block(
                    id=f"B{len(ep.blocks) + 1:03d}",
                    episode=number,
                    scene=scene_no,
                    kind=kind,
                    label=label,
                    speaker=speaker,
                    emotion=emotion,
                    text=txt,
                    start=round(start, 2),
                    end=round(end, 2),
                )
            )
            i = k
            continue
        i += 1
    if not ep.blocks:
        ep.warnings.append("no timed blocks found")
    _sort_blocks(ep)
    return ep


def _sort_blocks(ep: Episode) -> None:
    """Put blocks in time order and renumber, so 'B010..B030' is always a time range.

    Scripts occasionally list a block slightly before the one above it (the input pack has
    8 such cases, the worst 1.0 s). The sort is stable, so blocks with the same start time
    keep their file order. Ids are assigned after sorting, so ids always follow time.
    """
    limit = S.reorder_warn_seconds
    big = sum(1 for a, b in zip(ep.blocks, ep.blocks[1:]) if b.start < a.start - limit)
    ep.blocks.sort(key=lambda b: b.start)
    for n, b in enumerate(ep.blocks, 1):
        b.id = f"B{n:03d}"
    if big:
        ep.warnings.append(f"{big} block(s) were listed more than {limit:g}s out of time order; "
                           "sorted by start time")


def load_show(path: str | Path) -> Show:
    """Load a show from a folder of .txt files, a .zip of them, or one .txt file."""
    path = Path(path)
    texts: list[tuple[str, str]] = []
    # "utf-8-sig" = UTF-8 that also strips a byte-order mark.
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                    texts.append((name, zf.read(name).decode("utf-8-sig", errors="replace")))
    elif path.is_file():
        texts.append((str(path), path.read_text(encoding="utf-8-sig", errors="replace")))
    elif path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            texts.append((str(p), p.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        raise FileNotFoundError(f"No such input: {path}")
    return _build_show(texts)


def load_show_from_bytes(name: str, data: bytes) -> Show:
    """For uploads (e.g. Streamlit): a .zip or a single .txt as raw bytes."""
    if name.lower().endswith(".zip"):
        texts = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                    texts.append((n, zf.read(n).decode("utf-8-sig", errors="replace")))
        return _build_show(texts)
    return _build_show([(name, data.decode("utf-8-sig", errors="replace"))])


def _build_show(texts: list[tuple[str, str]]) -> Show:
    if not texts:
        raise ValueError("No .txt screenplay files found in the input")
    parsed = [(parse_episode(t, src), t) for src, t in texts]
    parsed.sort(key=lambda p: (p[0].number, p[0].source))
    warnings: list[str] = []

    # Duplicate episode numbers: every episode must have a unique number, because
    # lookups and the raw/ output files are all keyed by it.
    episodes: list[Episode] = []
    by_number: dict[int, tuple[Episode, str]] = {}
    for ep, text in parsed:
        if ep.number not in by_number:
            by_number[ep.number] = (ep, text)
            episodes.append(ep)
            continue
        first, first_text = by_number[ep.number]
        if text.strip() == first_text.strip():
            warnings.append(f"skipped {ep.source}: identical copy of episode {ep.number} ({first.source})")
            continue
        m = FILE_EP_RE.search(Path(ep.source).name)
        from_name = int(m.group(1)) if m else None
        old = ep.number
        if from_name is not None and from_name not in by_number:
            new, why = from_name, "taken from the filename"
        else:
            new, why = max(by_number) + 1, "next free number"
        _renumber(ep, new)
        by_number[new] = (ep, text)
        episodes.append(ep)
        warnings.append(f"{ep.source} also says episode {old}; renumbered to {new} ({why}). Check the order.")
    episodes.sort(key=lambda e: e.number)

    for e in episodes:
        if not e.show:  # no title line in the file: use the folder it sits in
            e.show = Path(e.source).parent.name or "Untitled show"
        warnings += [f"E{e.number:02d}: {w}" for w in e.warnings]

    # Files from more than one show mixed together.
    titles: dict[str, int] = {}
    for e in episodes:
        titles[e.show] = titles.get(e.show, 0) + 1
    title = max(titles, key=titles.get) if titles else "Untitled show"
    if len(titles) > 1:
        detail = "; ".join(f"'{t}' x{n}" for t, n in sorted(titles.items(), key=lambda x: -x[1]))
        warnings.insert(0, f"input mixes {len(titles)} show titles ({detail}). Using '{title}'; "
                           "run each show separately")
    return Show(title=title, episodes=episodes, warnings=warnings)


def _renumber(ep: Episode, number: int) -> None:
    ep.number = number
    for b in ep.blocks:
        b.episode = number


def render_blocks(blocks: list[Block], include_camera: bool = True) -> str:
    """Compact, LLM-friendly transcript. One line per block, addressed by id."""
    out = []
    for b in blocks:
        if b.kind == "camera" and not include_camera:
            continue
        ts = f"[{fmt_ts(b.start)}-{fmt_ts(b.end)}]"
        if b.kind == "dialogue":
            emo = f" ({b.emotion})" if b.emotion else ""
            out.append(f'{b.id} {ts} {b.speaker}{emo}: "{b.text}"')
        else:
            out.append(f"{b.id} {ts} {b.kind.upper()}: {b.text}")
    return "\n".join(out)
