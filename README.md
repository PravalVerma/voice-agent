# Voice Calling Agent

Real-time AI voice agent that runs locally (FastAPI + browser WebSocket). Two
interchangeable call pipelines, picked per call from the Voice Quality dropdown:

- **Standard (cascaded)**: browser microphone → Silero VAD → Whisper STT (Groq) →
  LLM (Groq `openai/gpt-oss-20b` by default) → Google Cloud TTS → browser speaker.
  Three priced synthesis tiers, shown to callers as **Core / Enhanced / Premium**.
- **Studio (Gemini Live)**: a single continuous Gemini Live API session handles
  speech in and out directly — no separate STT/LLM/TTS legs. Picked by choosing
  the **Studio** tier in the same dropdown.

No caller- or admin-facing text ever names the underlying vendor/model behind a tier
— see [Voice quality tiers](#voice-quality-tiers) below.

- **Low latency**: LLM replies stream sentence-by-sentence into TTS, so audio starts
  playing while the model is still generating.
- **Barge-in**: talk over the agent and it stops instantly and listens (tuned
  separately for each pipeline — see `[vad]` in `config.toml` and the Gemini Live
  activity-detection settings in `app/live_session.py`).
- **Pluggable providers**: swap STT/LLM/TTS backends via `config.toml`, no code changes.
- **Cost-aware**: every call is logged with an estimated ₹ cost per call.
- **Multi-user, bot-based**: admins register named "bots" (persona + per-language
  system prompt) and assign them to user accounts; each user only sees the bots
  assigned to them and only the Voice Quality tiers they're allowed to use.
- **Auditable**: every call's full transcript, barge-ins, and cost are on disk and
  viewable from an admin-only Call Logs screen — no server access needed to see
  what a caller said or how the agent replied.

## Setup (Windows, PowerShell — uses [uv](https://docs.astral.sh/uv/))

```powershell
uv sync
copy .env.example .env   # then fill in your keys
```

Required credentials in `.env` (for the default providers):

- `GROQ_API_KEY` — from https://console.groq.com/keys (STT + LLM)
- `GOOGLE_APPLICATION_CREDENTIALS` — path to a Google Cloud service-account JSON
  with the **Text-to-Speech API** enabled
- `GEMINI_API_KEY` — from https://aistudio.google.com/apikey. Required for any bot
  whose call uses the Studio (Gemini Live) tier, or a per-bot pipeline set to
  Gemini. Billed against that Google AI Studio project's own spending cap —
  independent of the Groq/Google TTS pipeline, so a cap hit there only breaks
  Studio-tier calls, not the rest of the app.
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — seeds the first login the very first time
  the app starts with an empty user database. Change the password, or just create
  a proper account from Advanced Settings → Users afterwards and stop using this one.
  Usernames are matched case-insensitively.

## Run

```powershell
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000 — you'll land on **Login**. Sign in, then:

- **Voice Assistant** (`/`) — the call UI: pick a bot, language, Voice Quality tier
  (Core/Enhanced/Premium/Studio) and gender, fill in Call Details, press
  **Start New Session**, allow the microphone, and talk. (Browsers only grant
  mic access on `localhost` or over HTTPS, so run it from `localhost`.)
- **Advanced Settings** (gear icon in the topbar, admin-only, `/settings`) — a
  dedicated page with three tabs:
  - **Users** — create/delete logins, and assign each user any number of
    (bot, language, allowed Voice Quality tiers) combinations.
  - **Bots** — register named bots, each with its own persona/behavior prompt
    per language (e.g. a separate English and Hindi script for the same bot).
    Every bot supports every pipeline — which one runs is purely the caller's
    per-call Voice Quality choice, not a bot setting.
  - **Call Logs** — every call ever placed, who placed it, duration, tier, and
    an estimated cost; click a row for the full turn-by-turn transcript,
    including which turns were interrupted (barge-in).

**Diagnostic modes** (append `?mode=echo`/`?mode=vad` to the Voice Assistant
URL, e.g. `http://localhost:8000/?mode=echo`):
- `?mode=echo` — your mic audio is echoed straight back; proves capture/playback work
  without touching any AI provider.
- `?mode=vad` — no audio comes back; the state lamp and status line show live speech
  detection so you can see/tune endpointing in isolation.
- default (`?mode=call` or no param) — the real agent.

## Voice quality tiers

Four tiers appear in the Voice Quality dropdown. The first three are Google Cloud
TTS quality levels layered on the standard (cascaded) pipeline; the fourth switches
the whole call to the Gemini Live pipeline instead:

| Caller sees | What it actually is | Rate |
|---|---|---|
| Core | Google Cloud TTS, `Standard` voices | $4 / M characters |
| Enhanced | Google Cloud TTS, `Neural2` voices | $16 / M characters |
| Premium | Google Cloud TTS, `Chirp3-HD` voices | $30 / M characters |
| Studio | Gemini Live (native audio-to-audio) | $3 / $12 per M audio tokens (in/out) |

An admin can restrict which tiers a given user is allowed to pick, per bot
assignment, from Advanced Settings → Users.

## Auth

Accounts live in `data/app.db` (SQLite, created automatically) — no external auth
service. Passwords are hashed (PBKDF2, per-user salt); usernames are matched
case-insensitively. Sessions are opaque tokens in an `HttpOnly` cookie, tracked
server-side in a `sessions` table, so deleting a user via Advanced Settings
immediately kills their active login. Two roles: `admin` (sees the Advanced
Settings gear, manages users/bots, sees all call logs) and `user` (only the bots
assigned to them).

**Known simplifications** — fine for a small admin-provisioned tool, worth revisiting
before wider rollout: no CSRF tokens, no login rate-limiting. Also, `COOKIE_SECURE`
in `.env` must be flipped to `true` if this is ever served over HTTPS — left
`false` for the local http setup.

## Configuration

- `config.toml` — enabled languages, VAD tuning, model/provider selection, cost
  rates. No secrets here. Per-bot persona/prompts live in the database instead
  (edit them via Advanced Settings → Bots), not in this file.
- `.env` — API keys only.

**Provider switching** — set `[stt]`/`[llm]`/`[tts]` `.provider` in `config.toml`.
`model` must match whichever provider you pick:

| Section | provider | model example | extra install |
|---|---|---|---|
| `[stt]` | `groq` (default) | `whisper-large-v3` | - |
| `[stt]` | `local` | `small`, `medium`, ... (faster-whisper sizes) | `uv sync --extra local-stt` |
| `[stt]` | `gemini` | (used automatically by the Studio/Gemini pipelines) | - |
| `[llm]` | `groq` (default) | `openai/gpt-oss-20b` | - |
| `[llm]` | `ollama` | `llama3.2` (needs a local Ollama server) | - |
| `[llm]` | `openai` | `gpt-4o-mini` | `uv sync --extra openai` |
| `[llm]` | `gemini` | `gemini-2.0-flash` | - |
| `[tts]` | `google` (default) | voice name, chosen per call from the live catalog | - |

Ollama and Gemini talk over plain REST (no extra installs); Ollama needs
`OLLAMA_BASE_URL` in `.env` if it isn't on `http://localhost:11434`, and Gemini
needs `GEMINI_API_KEY`. The Gemini Live (Studio) pipeline is separate from all of
the above — it's a synthetic tier chosen per call, not a `[stt]`/`[llm]`/`[tts]`
provider setting (see `app/live_session.py`).

## Stage-by-stage checks

```powershell
uv run python scripts/check_stt.py sample.wav              # transcribe a WAV
uv run python scripts/check_stt.py sample.wav --local       # ...via faster-whisper
uv run python scripts/check_llm.py "Tell me a joke"         # stream LLM tokens + TTFT
uv run python scripts/check_llm.py "hi" --provider ollama   # ...against another provider
uv run python scripts/check_tts.py "Hello there!"           # synthesize out.wav
uv run python scripts/check_vad.py sample.wav               # print detected speech segments
uv run python scripts/check_pipeline.py question.wav        # full pipeline offline -> reply.wav
```

Each script fails with a single clear line (missing key, unknown provider, missing
optional dependency) rather than a stack trace, so they're safe to use for quick
troubleshooting.

## Call logs & cost

Each call writes `logs/calls/<timestamp>_<id>/transcript.jsonl` (one JSON event per
line: call start incl. which user placed it, greeting, user utterances, agent
replies, barge-ins) and `meta.json`, written on a clean call end:

```json
{
  "session_id": "c76b1224",
  "user_id": 3,
  "username": "arcil",
  "tts_tier": "Enhanced",
  "duration_s": 187.3,
  "turn_count": 9,
  "usage": {"stt_requests": 4, "stt_billed_seconds": 40.0, "llm_input_tokens": 812, "llm_output_tokens": 210, "tts_chars": 640},
  "cost_usd": {"stt": 0.0004, "llm": 0.0002, "tts": 0.0026, "total": 0.0032},
  "cost_inr": 0.31
}
```

The cost estimate is approximate (STT bills a minimum duration per request, matching
Groq's pricing; LLM/TTS rates come from `[cost]` in `config.toml`; Gemini Live calls
are estimated from streamed audio-token rates instead) — good enough to catch a
runaway call, not a substitute for your provider's actual invoice. All of this is
browsable without SSH access from Advanced Settings → Call Logs (admin-only).

Calls also end automatically after `[call] max_call_minutes` (default 10): the agent
speaks a short goodbye and the connection closes.

## Project layout

```
app/
  main.py         FastAPI app: pages, auth routes, users/bots/call-log APIs, the /ws endpoint
  auth.py         password hashing, session tokens, require_login/require_admin, user-bot assignments
  db.py           SQLite connection + schema (users, sessions, bots, bot_prompts, user_bots) -> data/app.db
  bots_store.py   admin-registered "bots" (named, per-language prompt sets - bots/bot_prompts tables)
  tts_catalog.py  live Google voice catalog (tier/gender/quality-label per language),
                  plus the synthetic Studio/Gemini Live "tier" definition
  call_logs.py    read-only: lists/parses logs/calls/ for the admin Call Logs page
  calllog.py      per-call transcript writer + cost estimate, written to logs/calls/ as the call happens
  session.py      CallSession: state machine (listening/speaking/...), barge-in - standard pipeline
  live_session.py Gemini Live pipeline: a single bidirectional audio session (no discrete STT/LLM/TTS legs)
  pipeline.py     one agent turn on the standard pipeline: STT -> streaming LLM -> sentence chunks -> TTS
  config.py       config.toml + .env loading and validation, global prompt-suffix rules (language lock, currency phrasing)
  transport/      AudioTransport interface + the browser WebSocket implementation
                  (the seam where a future Twilio/SIP adapter would plug in)
  audio/          Silero VAD wrapper + endpointing, PCM/WAV helpers
  text/           streaming sentence chunker
  providers/      STT/LLM/TTS backends behind small ABCs, config-driven factories
web/              plain HTML/JS, no framework/build step
  login.html/js   sign-in page
  index.html/app.js  the Voice Assistant call UI
  settings.html/admin-panel.js  Advanced Settings page (Users / Bots / Call Logs tabs, admin-only)
  shared.css/js   topbar shell, auth chip, shared table/card styles
  call.css        Live Call card styles, shared between index.html and login.html's preview
scripts/          check_stt/llm/tts/vad/pipeline.py - test each stage in isolation
docs/             misc reference docs (voice tier history, architecture notes)
```

## Running it as a service (optional)

Local development just needs `uv run uvicorn app.main:app --port 8000`. To keep it
running in the background on a Linux host, the repo is set up to run as a
`voice-calling-agent` systemd service; Python code changes then need
`sudo systemctl restart voice-calling-agent.service`. Static files under `web/`
(HTML/CSS/JS) are served straight off disk and go live immediately — bump the
`?v=NN` cache-busting query param on `shared.css`/`call.css` links when editing
those two specifically, since browsers cache them aggressively.
