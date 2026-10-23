# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time AI voice agent (FastAPI + browser WebSocket), run locally at
`http://localhost:8000`. Two interchangeable call pipelines, picked per call from the
Voice Quality dropdown:

- **Standard (cascaded)**: browser mic → Silero VAD → Whisper STT (Groq) → LLM (Groq
  `openai/gpt-oss-20b` by default) → Google Cloud TTS → browser speaker. Three priced
  synthesis tiers, shown to callers as **Core / Enhanced / Premium**.
- **Studio (Gemini Live)**: a single continuous Gemini Live API session handles speech
  in and out directly — no separate STT/LLM/TTS legs. Picked via the **Studio** tier in
  the same dropdown, implemented entirely in `app/live_session.py`, not through the
  `[stt]`/`[llm]`/`[tts]` provider settings used by the standard pipeline.

No caller- or admin-facing text ever names the underlying vendor/model behind a tier —
preserve that when touching UI copy or prompts.

Multi-user, bot-based: admins register named "bots" (persona + per-language system
prompt, stored in the DB) and assign each to users along with which Voice Quality
tiers they're allowed to use. Every bot supports every pipeline — pipeline choice is a
per-call Voice Quality pick, not a bot property.

## Commands

```powershell
uv sync                                        # install deps
uv sync --extra local-stt                      # + faster-whisper, for stt.provider = "local"
uv sync --extra openai                         # + openai SDK, for llm.provider = "openai"
uv run uvicorn app.main:app --port 8000        # run the server (http://localhost:8000)
```

There is no automated test suite. Verification is done via the stage-by-stage check
scripts (each fails with one clear line — missing key, unknown provider — rather than a
stack trace) and by exercising the app in a browser:

```powershell
uv run python scripts/check_stt.py sample.wav              # transcribe a WAV
uv run python scripts/check_stt.py sample.wav --local       # ...via faster-whisper
uv run python scripts/check_llm.py "Tell me a joke"         # stream LLM tokens + TTFT
uv run python scripts/check_llm.py "hi" --provider ollama   # ...against another provider
uv run python scripts/check_tts.py "Hello there!"           # synthesize out.wav
uv run python scripts/check_vad.py sample.wav               # print detected speech segments
uv run python scripts/check_pipeline.py question.wav        # full standard pipeline offline -> reply.wav
uv run python scripts/check_regressions.py                  # every fixed defect, no network needed
uv run python scripts/check_compliance.py "sentence"        # outbound/inbound screens
uv run python scripts/check_window.py --tz Asia/Kolkata     # calling-window gate
uv run python scripts/check_mcp.py --customer 1             # tool server: schemas + execution
uv run python scripts/check_cache.py                        # measure the LLM prompt-cache floor
uv run python scripts/run_post_jobs.py --dry-run            # post-call queue
```

`check_regressions.py` is the closest thing to a test suite here. Every case in
it is a defect that once shipped silently - raising nothing, logging nothing,
and producing plausible output while doing the wrong thing. Run it after
touching compliance, the numeral parsing, the tool loop, or history handling.

Diagnostic modes in the browser UI (append to the Voice Assistant URL):
- `?mode=echo` — mic audio echoed straight back; proves capture/playback without touching any provider.
- `?mode=vad` — no audio back; state lamp + status line show live speech detection, for tuning endpointing in isolation.
- default (`?mode=call` or no param) — the real agent.

No build step for `web/` (plain HTML/JS/CSS, no framework/bundler) — edits there are
live on refresh when running locally. In production, static files are served straight
off disk and go live immediately on save, but `shared.css`/`call.css` need their
`?v=NN` query param bumped on referencing `<link>` tags since browsers cache those two
aggressively; Python changes need a service restart (see Deployment below).

## Architecture

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
  call_state.py   State enum, split out so session.py and pipeline.py can both import it without a cycle
  compliance.py   pure gates: calling window, contact eligibility, outbound/inbound screens.
                  No LLM, no network. The prompt is a hint; this is the enforcement.
  collections_flow.py  scripted stages (GREETING/VERIFY/NEGOTIATE/CLOSE). Only NEGOTIATE
                  calls the LLM; opt-out and distress are handled here, before the model.
  customers_store.py   borrower records. Money is integer paise throughout; verify_value
                  never leaves the server (see get_verify_value).
  mcp_client.py   tool-server client. Schemas cached per PROCESS, execution gets one
                  server process per CALL bound to a customer via the environment.
  post_call.py    after-call audit/summary/retention. The queue is the log directory.
  tts_cache.py    content-addressed disk cache for fixed script lines.
  session.py      CallSession: state machine (listening/speaking/...), barge-in - standard pipeline
  live_session.py Gemini Live pipeline: a single bidirectional audio session (no discrete STT/LLM/TTS legs)
  pipeline.py     one agent turn on the standard pipeline: STT -> streaming LLM -> sentence chunks -> TTS
  config.py       config.toml + .env loading/validation; global prompt-suffix rules (language lock, currency phrasing)
  recording.py    mono mixdown of each call -> recording.wav in its log dir
  s3_archive.py   archives recording.wav to S3 when S3_RECORDINGS_BUCKET is set
  transport/      AudioTransport interface + the browser WebSocket implementation
                  (the seam where a future Twilio/SIP adapter would plug in)
  audio/          Silero VAD wrapper + endpointing, PCM/WAV helpers
  text/           streaming sentence chunker
  providers/      STT/LLM/TTS backends behind small ABCs (providers/base.py), config-driven factories
web/              plain HTML/JS, no framework/build step
  login.html/js   sign-in page
  index.html/app.js  the Voice Assistant call UI
  settings.html/admin-panel.js  Advanced Settings page (Users / Bots / Call Logs tabs, admin-only)
  shared.css/js   topbar shell, auth chip, shared table/card styles
  call.css        Live Call card styles, shared between index.html and login.html's preview
mcp_server/       standalone MCP tool server over stdio; imports app.db/config/compliance,
                  never app.main. Tools take no customer_id/session_id/phone - those are
                  bound into the process at spawn, so the model cannot supply or forge them.
scripts/          check_stt/llm/tts/vad/pipeline.py - test each stage in isolation
docs/             misc reference docs (voice tier history, architecture notes)
```

### Standard pipeline flow

`main.py` `ws_call` accepts the WebSocket, resolves which bot/language/tier the caller
is allowed to use (admins get a free-choice selector; everyone else is re-validated
server-side against their DB assignment — never trust the client payload for bot,
language, or tier), then builds `stt`/`llm`/`tts` providers from `config.py` and hands
off to `CallSession.run()` (`session.py`). Each turn: VAD (`audio/vad.py`) detects an
utterance → `pipeline.py` runs STT → streams the LLM response, chunked into sentences
(`text/sentences.py`) → each sentence streams to TTS and out to the browser, so audio
starts playing before the LLM finishes generating. Barge-in (interrupting the agent
mid-reply) is handled in `session.py`'s state machine, tuned via `[vad]` in
`config.toml`.

### Gemini Live (Studio) pipeline

Entirely separate code path (`live_session.py`): one continuous bidirectional Gemini
Live API session does speech-in/speech-out directly, with its own
activity-detection/barge-in tuning (not the `[vad]` config block). Selected via the
synthetic `gemini_live` tier returned by `tts_catalog.gemini_voice_tier()`, which is
appended to the dropdown alongside the real Google TTS tiers — not a separate UI
control or bot setting.

### Config and prompts

- `config.toml` — enabled languages, VAD tuning, provider/model selection, cost rates.
  No secrets. Provider switching: set `[stt]`/`[llm]`/`[tts]` `.provider`; `model` must
  match whichever provider is picked (see table in README).
- `.env` — API keys only (see `.env.example`).
- Bot personas/system prompts are **not** in `config.toml` — they're admin-edited per
  (bot, language) pairs stored in SQLite (`bots_store.py`, Advanced Settings → Bots).
  `config.py` layers code-generated rules on top of whatever prompt text is stored:
  a language-lock instruction (reply only in the selected language/script — matters
  because Google's voices read romanized Hindi with English pronunciation, so the
  script rule must live in code rather than depend on how an admin wrote the prompt)
  and currency/number phrasing rules. `prompt.md` is a reference example of a bot
  prompt (a collections-call script), not something loaded by the app at runtime.
  `docs/archived_language_prompts.md` has the history/reasoning for currently
  disabled languages in `config.toml`.
- Adding/changing a provider: implement the matching ABC in `providers/base.py`
  (`STTProvider`/`LLMProvider`/`TTSProvider`), add a factory case, wire the
  `provider`/`model` strings through `config.py`.

### Auth

Accounts live in `data/app.db` (SQLite, auto-created) — no external auth service.
Passwords: PBKDF2 + per-user salt; usernames matched case-insensitively. Sessions are
opaque tokens in an `HttpOnly` cookie, tracked server-side in a `sessions` table (so
deleting a user immediately kills their active login). Two roles: `admin` (Advanced
Settings, all call logs, free-choice bot/language/tier selector) and `user` (only
their assigned bots, re-validated server-side on every `/ws` connection — see
`ws_call` in `main.py`). `COOKIE_SECURE` in `.env` must be `true` once behind HTTPS;
left `false` for local http. Known simplifications: no CSRF tokens, no login
rate-limiting.

### Call logs & cost

Each call writes `logs/calls/<timestamp>_<id>/transcript.jsonl` (one JSON event per
line) and `meta.json` on clean call end, including a per-call cost estimate in USD/INR
(`calllog.py`). Rates come from `[cost]` in `config.toml`; STT bills a minimum
duration per request matching Groq's pricing; Gemini Live calls are estimated from
streamed audio-token rates instead. This is approximate — good enough to catch a
runaway call, not a substitute for the provider's actual invoice. All of it is
browsable from Advanced Settings → Call Logs (admin-only), including the audio
recording when `[call] record_calls` is on.

### Collections mode

`bots.mode` is `'assistant'` (today's behaviour, unchanged) or `'collections'`.
A collections bot requires a customer selected at call time and refuses to
start without one - without an account there is no window to check, no
eligibility to test, and nothing to validate spoken figures against, so running
anyway would be the compliance layer quietly switched off.

Things worth knowing before changing any of it:

- **Groq's free tier does not cache prompt prefixes.** Measured 2026-10-23 with
  `scripts/check_cache.py`: `cached_tokens` is 0 at every prefix size and
  `prompt_tokens_details` comes back `None`. The fixed system prompt is kept
  because it is cheap and pays off on a paid tier, but raw token count is the
  only budget lever that currently works. Re-run that script if the account is
  ever upgraded.
- **Compliance patterns in `config.toml` must use TOML *literal* strings**
  (single quotes, or `''' '''` when the pattern contains an apostrophe). A
  double-quoted `"\b"` decodes to a backspace, compiles fine, and silently
  matches nothing. `ComplianceConfig` rejects control characters for this reason.
- **Devanagari needs the explicit `\u0900-\u097F` range** in any tokenizer;
  Python's `\w` excludes combining marks and will shred every Hindi word.
- **Tool messages never enter `session.history`**, which is what keeps
  `trim_history`'s slicing safe. Round two of a tool turn declares the tools
  and discards what it asks for - with `tools=None` gpt-oss emits a call
  anyway and Groq rejects the whole request.
- The outbound screen is inert unless `session.allowed_facts` is set, so
  assistant bots are completely unaffected by any of it.

## Running as a service (optional)

Normal use is local: `uv run uvicorn app.main:app --port 8000`. The repo is also set up
to run as a `voice-calling-agent` systemd service on a Linux host; there, Python changes
need `sudo systemctl restart voice-calling-agent.service` while static `web/` files are
live immediately (see cache-busting note above).

The UI carries no branding and the app is not tied to any domain — keep it that way when
touching page titles, the topbar brand, or docs.
