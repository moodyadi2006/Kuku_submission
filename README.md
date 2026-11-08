# Promo Finder

**Solution doc:** [docs/solution.pdf](docs/solution.pdf) · **Video (3 min):** [YouTube](https://youtu.be/GTw5nuS48Jg) · **Raw outputs:** [Google Drive](https://drive.google.com/drive/folders/121C8M7x-gaq2_NIWC1XlozhlUh6yNGA2?usp=sharing)

Reads a microdrama screenplay (one timestamped `.txt` per episode) and returns ranked
**20–60 second promo clips** for Instagram, TikTok and YouTube. Each promo opens on a hook in its
first 3 seconds, cuts on a cliffhanger, and is meant to make sense to someone who has never seen
the show. For each one you get the episode, the start and end of every segment, the hook, the
cliffhanger, one line on why it works, and its place in the posting order.

Any LLM can be plugged in: the model is one setting (`--model provider[:model]`).

## Setup

Python 3.10+ (tested on 3.12). Nothing to install: only the standard library is used.

1. Copy `.env.example` to `.env`. It lists every setting with its recommended value.
2. Add the API key for the model you'll use. The default is Gemini on its free tier
   (free key: https://aistudio.google.com/apikey).
3. Put the episode `.txt` files (or the zip) in `data/`. The input pack isn't in this repo
   because it's confidential.

## Run

```bash
python -m promo_finder data/                          # all episodes, model from .env (Gemini)
python -m promo_finder data/ --episodes 1-10          # a subset
python -m promo_finder show.zip --model openai        # another provider: one flag
python -m promo_finder data/ --model offline          # keyword baseline, no API key
python -m promo_finder data/ --model custom:their-model --base-url https://gateway/v1 --api-key-env KEY
```

The input can be a folder of `.txt` episodes, a `.zip` of them, or one `.txt`.

| Flag | What it does |
|---|---|
| `--model provider[:model]` | gemini, openai, groq, openrouter, together, deepseek, mistral, ollama, custom, offline |
| `--episodes 1-10,15` | Run on a subset of episodes |
| `--per-episode N` | Maximum candidates per episode |
| `--workers N` / `--rpm N` | Parallel episodes / stay under a free-tier rate limit |
| `--no-camera` | Leave camera lines out of the prompt (about half the input tokens) |
| `--no-cache` | Ignore saved answers and ask the model again |
| `--price-in` / `--price-out` | USD per million tokens, for the cost line in the report |
| `--base-url URL` / `--api-key-env NAME` | Point at any OpenAI-compatible endpoint, e.g. one Kuku provides |
| `-o DIR` / `--max-tokens N` | Output folder / output-token limit for the model |

## How it works

```
 scripts ─► parse into numbered, timed blocks (code)
         ─► per episode: AI proposes up to 3 promos as block IDs
         ─► check and fix every proposal (code) ─► one repair round for failures (AI)
         ─► remove duplicates (code)
         ─► blind standalone check on the top 30 (AI, sees only the clip's lines)
         ─► pick and order the best 15 from the top 30 (AI) ─► variety rule (code) ─► report
```

| Step | Done by | Why |
|---|---|---|
| Parse scripts into blocks `B001…` with timestamps | Code | Deterministic; tolerant of messy input (format drift, duplicates, out-of-order lines) |
| Propose promos (hook, cliffhanger, up to 3 pieces of one episode) | AI | Needs judgement about drama |
| The model answers with **block IDs only**; code turns them into timestamps | Code | The model can't invent or misquote a timestamp |
| Enforce the brief: 20–60 s, hook in the first 3 s, cut on the cliffhanger, no mid-line cuts | Code | Obvious fixes are applied automatically; anything needing judgement is sent back |
| Fix candidates that failed a check | AI, once | It gets the exact error, e.g. "total 61.8 s is over 60 s" |
| Remove duplicates: same hook, same cliffhanger, or >80% shared footage | Code | Overlapping clips with different hooks and endings are different ads |
| Blind standalone check on the top 30 | AI, a fresh call per promo | The picking model has read the whole episode, so it can't judge what a stranger follows. Repeats until every promo in the top 30 has been checked. |
| Picks and orders the best 15 from the top 30 | AI, one call | Needs judgement across the show; the rest follow in score order (and everything does if the call fails) |
| Variety rule: at most 2 promos from any 5-episode window in the top 10 | Code | The ranking model was asked for variety and still clustered one storyline |

The report shows the model's description of each hook and cliffhanger **next to the exact script
line**, so a reviewer can spot a mislabel at a glance.

## Settings

All settings live in `.env`; `promo_finder/config.py` contains no values. A missing or invalid
setting stops the program with a message naming it. Groups:

- **Model:** `PROMO_MODEL`, and per provider `<NAME>_API_KEY`, `<NAME>_BASE_URL`, `<NAME>_MODEL`, `<NAME>_MAX_TOKENS`
- **Run:** `PROMO_RPM`, `PROMO_WORKERS`, `PROMO_INPUT`, `PROMO_OUT_DIR`, `PROMO_CACHE`, `PROMO_PER_EPISODE`, prices
- **The brief's rules** (the prompts are generated from these): `PROMO_MIN_SECONDS=20`, `PROMO_MAX_SECONDS=60`, `PROMO_HOOK_WINDOW=3`
- **Checks, duplicates, scoring, blind check, variety rule:** see the comments in `.env.example`

Changing anything that goes into a model request (the model, the rules, candidates per episode)
re-asks the model for every episode; changing only checks, scoring or ordering reuses saved answers.

## Outputs

Each run writes to `outputs/<show>/<model>/`:

- `report.md`: top-10 table, a card per promo, and a **quality check** section (rejections by
  reason, auto-fixes, blind-check scores, tokens, cost)
- `promos.json`: everything, including rejected candidates and why
- `raw/`: the model's **unedited** response to every call
- `run_meta.json`: token usage, timings, cost estimate

Each run replaces the files in its folder; use `-o <folder>` to keep runs separate.
Answers are cached in `.cache/`, so reruns are free and a stopped run resumes where it left off.

## Checking quality

```bash
python -m promo_finder.compare outputs/<show>/openai-gpt-5-mini outputs/<show>/offline-heuristic
python -m promo_finder.review  outputs/<show>/openai-gpt-5-mini --top 15
```

- `compare` lines up two runs' picks, e.g. the AI against the free keyword baseline (`compare.md`).
- `review` writes `review_sheet.md`: the top promos with their full clip text, scores, warnings,
  and a verdict line for a human editor.

## Results on the input pack (50 episodes, gpt-5-mini)

These come from gpt-5-mini because Gemini's free daily cap ran out after about 19 episodes. The
default stays Gemini so the tool runs for free; switching models is one flag.

- 104 promos, all 20–60 s (average 37.5 s); 24 auto-fixed by code, 17 rescued by the repair round.
- 94 model calls, about 261k input / 233k output tokens: **about $0.53 per show** at assumed
  gpt-5-mini prices ($0.25 / $2 per million tokens), about 10 minutes with 4 workers.
- 74% of the AI's picks are moments the free keyword baseline also found.
- Hand review of the top 15 (AI-drafted first pass, checked and confirmed by me): 5 post,
  7 post with an edit, 3 don't post.

Known limitations: timing comes from the script, not the edited footage; hooks that are only camera
directions can't be verified from text; the variety rule counts episodes, not story beats; the
blind check's score doesn't match human judgement (its notes are useful, its number isn't a filter).

## Note on data

The input pack is Kuku's confidential content, so this repository holds **code only**: `data/`,
the zips and `outputs/` are git-ignored. The outputs quote lines of the scripts, so they're shared
separately: **[Google Drive](https://drive.google.com/drive/folders/121C8M7x-gaq2_NIWC1XlozhlUh6yNGA2?usp=sharing)**.
In it:

- `openai-gpt-5-mini/`: the submitted run (`report.md`, `promos.json`, `raw/` = unedited model responses)
- `offline-heuristic/`: the free keyword baseline, for comparison
- `human_review/`: my verdicts on the top 15
- `blind_check_history/`: the v1 and v2 blind-check runs from the solution doc
