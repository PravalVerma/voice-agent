"""FastAPI app: serves the browser demo and the /ws call endpoint."""

import logging
import os
import sqlite3
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, bots_store, call_logs, tts_catalog
from .audio.frames import FrameBuffer, wav_wrap
from .audio.vad import SileroVAD, SpeechEnd, SpeechStart, VADGate
from .config import ROOT, apply_language, build_call_vars, get_config, missing_secrets
from .db import init_db
from .live_session import run_live_call
from .providers import build_llm, build_stt, build_tts
from .recording import CallRecorder
from .session import CallSession
from .transport.base import AudioFrame, AudioTransport, ControlMessage, Disconnected
from .transport.browser_ws import BrowserWebSocketTransport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

app = FastAPI(title="Voice Calling Agent")

WEB_DIR = ROOT / "web"

# Off on plain http (local dev); set COOKIE_SECURE=true once this runs behind HTTPS.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").strip().lower() == "true"


@app.on_event("startup")
async def startup() -> None:
    cfg = get_config()  # loads .env as a side effect - must run before seeding reads it
    init_db()
    auth.seed_admin_if_empty()
    bots_store.migrate_legacy_base_prompt_if_needed([l.code for l in cfg.languages])
    tts_catalog.init_catalog([l.locale for l in cfg.languages if l.enabled])
    log.info(
        "providers: stt=%s (%s), llm=%s (%s), tts=%s",
        cfg.stt.provider, cfg.stt.model,
        cfg.llm.provider, cfg.llm.model,
        cfg.tts.provider,
    )
    for problem in missing_secrets(cfg):
        log.warning("credentials: %s — calls will fail until this is fixed", problem)


def _serve_protected_page(request: Request, filename: str) -> Response:
    user = auth.get_current_user(request)
    if user is None:
        return RedirectResponse(f"/login?next={request.url.path}")
    return FileResponse(WEB_DIR / filename)


# ---- auth pages/API -----------------------------------------------------------

class LoginBody(BaseModel):
    username: str
    password: str
    remember: bool = True


class CreateUserBody(BaseModel):
    username: str
    password: str
    role: str = "user"


class AddUserBotBody(BaseModel):
    bot_id: int
    language_code: str
    allowed_tiers: list[str] | None = None


class SetAssignmentTiersBody(BaseModel):
    allowed_tiers: list[str] | None = None


class CreateBotBody(BaseModel):
    name: str


class RenameBotBody(BaseModel):
    name: str


class BotPromptBody(BaseModel):
    language: str
    prompt: str


@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(WEB_DIR / "login.html")


@app.post("/api/login")
async def api_login(body: LoginBody, response: Response) -> dict:
    user = auth.get_user_by_username(body.username)
    if user is None or not auth.verify_password(body.password, user["password_hash"], user["salt"]):
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = auth.create_session(user["id"])
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        # unchecked "stay logged in" -> browser-session cookie (dropped when the
        # browser closes); the server-side session still lives out its normal
        # SESSION_TTL_HOURS, so an open tab keeps working either way
        max_age=auth.SESSION_TTL_HOURS * 3600 if body.remember else None,
    )
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/logout")
async def api_logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        auth.delete_session(token)
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me")
async def api_me(user: dict = Depends(auth.require_login)) -> dict:
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


# ---- pages ------------------------------------------------------------------

@app.get("/")
async def index(request: Request) -> Response:
    return _serve_protected_page(request, "index.html")


@app.get("/call")
async def call_page_redirect() -> Response:
    return RedirectResponse("/")


@app.get("/logs")
async def logs_page_redirect() -> Response:
    return RedirectResponse("/")


@app.get("/users")
async def users_page_redirect() -> Response:
    return RedirectResponse("/")


@app.get("/settings")
async def settings_page(request: Request) -> Response:
    # admin-only is enforced client-side (admin-panel.js redirects non-admins to
    # "/"), same pattern as the rest of this app's page-level access control
    return _serve_protected_page(request, "settings.html")


@app.get("/languages")
async def languages(user: dict = Depends(auth.require_login)) -> list[dict]:
    cfg = get_config()
    return [{"code": l.code, "label": l.label} for l in cfg.languages if l.enabled]


@app.get("/tts-options")
async def tts_options(
    language: str, assignment_id: int | None = None, user: dict = Depends(auth.require_login)
) -> dict:
    cfg = get_config()
    lang = next((l for l in cfg.languages if l.code == language and l.enabled), None)
    if lang is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled language {language!r}")
    # every bot supports every pipeline - the Gemini Live pipeline is just one
    # more entry in the same Voice Quality dropdown, alongside the real
    # Standard/Neural2/Chirp3-HD quality ladder, not a separate control
    tiers = tts_catalog.tiers_for(lang.locale) + [tts_catalog.gemini_voice_tier()]
    # admins get the free-choice topbar selector (no assignment_id) and always
    # see every tier, for testing; everyone else may be restricted to a subset
    # of tiers an admin picked for their specific (bot, language) assignment
    if assignment_id is not None:
        assignment = auth.get_user_bot_assignment(assignment_id, user["id"])
        allowed = assignment["allowed_tiers"] if assignment else None
        if allowed:
            tiers = [t for t in tiers if t["name"] in allowed]
    return {"tiers": tiers}


@app.get("/health")
async def health(user: dict = Depends(auth.require_login)) -> dict:
    cfg = get_config()
    return {
        "status": "ok",
        "stt": f"{cfg.stt.provider}:{cfg.stt.model}",
        "llm": f"{cfg.llm.provider}:{cfg.llm.model}",
        "tts": cfg.tts.provider,
        "missing_secrets": missing_secrets(cfg),
    }


# ---- bots API ---------------------------------------------------------------
# Bots are named, per-language prompt sets (e.g. one collections script in
# English, another in Hindi). Everyone logged in can list bots by name to pick
# one for a call; only admins can see/edit the actual prompt text.

@app.get("/api/bots")
async def api_list_bots(user: dict = Depends(auth.require_login)) -> list[dict]:
    # A bot may still have prompt rows for languages that used to be enabled (e.g.
    # the initial migration seeded every configured language) - only surface the
    # ones actually selectable today, so the list doesn't show stale codes.
    enabled_codes = {l.code for l in get_config().languages if l.enabled}
    bots = bots_store.list_bots()
    for bot in bots:
        bot["languages"] = [c for c in bot["languages"] if c in enabled_codes]
    return bots


@app.get("/api/my-bots")
async def api_my_bots(user: dict = Depends(auth.require_login)) -> list[dict]:
    """Non-admins pick from this list in the Bot dropdown - only the specific
    (bot, language) pairs an admin has assigned them, never the full roster."""
    labels = {l.code: l.label for l in get_config().languages if l.enabled}
    assignments = auth.list_user_bots(user["id"])
    return [
        {
            "id": a["id"],
            "bot_id": a["bot_id"],
            "bot_name": a["bot_name"],
            "language_code": a["language_code"],
            "language_label": labels[a["language_code"]],
        }
        for a in assignments
        if a["language_code"] in labels
    ]


@app.get("/api/bots/{bot_id}")
async def api_get_bot(bot_id: int, admin: dict = Depends(auth.require_admin)) -> dict:
    bot = bots_store.get_bot(bot_id)
    if bot is None:
        raise HTTPException(status_code=404, detail="bot not found")
    return bot


@app.post("/api/bots")
async def api_create_bot(body: CreateBotBody, admin: dict = Depends(auth.require_admin)) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        return bots_store.create_bot(name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="a bot with that name already exists")


@app.put("/api/bots/{bot_id}")
async def api_rename_bot(bot_id: int, body: RenameBotBody, admin: dict = Depends(auth.require_admin)) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        bots_store.rename_bot(bot_id, name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="a bot with that name already exists")
    return {"ok": True}


@app.delete("/api/bots/{bot_id}")
async def api_delete_bot(bot_id: int, admin: dict = Depends(auth.require_admin)) -> dict:
    if bots_store.count_bots() <= 1:
        raise HTTPException(status_code=400, detail="can't delete the last remaining bot")
    bots_store.delete_bot(bot_id)
    return {"ok": True}


@app.put("/api/bots/{bot_id}/prompt")
async def api_set_bot_prompt(bot_id: int, body: BotPromptBody, admin: dict = Depends(auth.require_admin)) -> dict:
    cfg = get_config()
    if body.language not in {l.code for l in cfg.languages}:
        raise HTTPException(status_code=400, detail=f"unknown language code {body.language!r}")
    text = body.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt can't be empty")
    bots_store.set_prompt(bot_id, body.language, text)
    return {"ok": True}


# ---- users API (admin-only) ----------------------------------------------------

@app.get("/api/users")
async def api_list_users(admin: dict = Depends(auth.require_admin)) -> list[dict]:
    return auth.list_users()


@app.post("/api/users")
async def api_create_user(body: CreateUserBody, admin: dict = Depends(auth.require_admin)) -> dict:
    username = body.username.strip()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail="username and password are required")
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
    try:
        return auth.create_user(username, body.password, body.role)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="username already exists")


@app.delete("/api/users/{user_id}")
async def api_delete_user(user_id: int, admin: dict = Depends(auth.require_admin)) -> dict:
    auth.delete_user(user_id)
    return {"ok": True}


@app.get("/api/users/{user_id}/bots")
async def api_list_user_bots(user_id: int, admin: dict = Depends(auth.require_admin)) -> list[dict]:
    return auth.list_user_bots(user_id)


def _validate_allowed_tiers(allowed_tiers: list[str] | None) -> None:
    if allowed_tiers is None:
        return
    valid = set(tts_catalog.all_tier_names())
    unknown = set(allowed_tiers) - valid
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown voice quality tier(s): {sorted(unknown)}")
    if not allowed_tiers:
        raise HTTPException(status_code=400, detail="allowed_tiers can't be an empty list - omit it instead to allow every tier")


@app.post("/api/users/{user_id}/bots")
async def api_add_user_bot(user_id: int, body: AddUserBotBody, admin: dict = Depends(auth.require_admin)) -> dict:
    if bots_store.get_bot(body.bot_id) is None:
        raise HTTPException(status_code=400, detail="bot not found")
    enabled_codes = {l.code for l in get_config().languages if l.enabled}
    if body.language_code not in enabled_codes:
        raise HTTPException(status_code=400, detail=f"unknown or disabled language {body.language_code!r}")
    _validate_allowed_tiers(body.allowed_tiers)
    try:
        return auth.add_user_bot(user_id, body.bot_id, body.language_code, body.allowed_tiers)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="that bot is already assigned to this user in this language")


@app.put("/api/users/{user_id}/bots/{assignment_id}/tiers")
async def api_set_user_bot_tiers(
    user_id: int, assignment_id: int, body: SetAssignmentTiersBody, admin: dict = Depends(auth.require_admin)
) -> dict:
    if auth.get_user_bot_assignment(assignment_id, user_id) is None:
        raise HTTPException(status_code=404, detail="assignment not found")
    _validate_allowed_tiers(body.allowed_tiers)
    auth.set_user_bot_tiers(assignment_id, body.allowed_tiers)
    return {"ok": True}


@app.delete("/api/users/{user_id}/bots/{bot_id}/{language_code}")
async def api_remove_user_bot(
    user_id: int, bot_id: int, language_code: str, admin: dict = Depends(auth.require_admin)
) -> dict:
    auth.remove_user_bot(user_id, bot_id, language_code)
    return {"ok": True}


# ---- call logs (admin-only; read-only view of logs/calls/ on disk) ------------

@app.get("/api/call-logs")
async def api_list_call_logs(admin: dict = Depends(auth.require_admin)) -> list[dict]:
    return call_logs.list_call_logs()


@app.get("/api/call-logs/{dir_name}")
async def api_get_call_log(dir_name: str, admin: dict = Depends(auth.require_admin)) -> dict:
    log = call_logs.get_call_log(dir_name)
    if log is None:
        raise HTTPException(status_code=404, detail="call log not found")
    return log


@app.get("/api/call-logs/{dir_name}/recording")
async def api_get_call_recording(
    dir_name: str, admin: dict = Depends(auth.require_admin)
) -> FileResponse:
    path = call_logs.recording_path(dir_name)
    if path is None:
        raise HTTPException(status_code=404, detail="no recording for this call")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=call_logs.recording_download_name(dir_name),
    )


# ---- call websocket -----------------------------------------------------------

@app.websocket("/ws")
async def ws_call(websocket: WebSocket) -> None:
    user = auth.get_session_user(websocket.cookies.get(auth.SESSION_COOKIE))
    if user is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    transport = BrowserWebSocketTransport(websocket)

    # the first control message opens the session and selects the mode
    event = await transport.recv()
    if isinstance(event, Disconnected):
        return
    if not isinstance(event, ControlMessage) or event.data.get("type") != "start":
        await transport.send_control({"type": "error", "message": "expected a 'start' message"})
        await transport.close()
        return

    mode = event.data.get("mode", "call")
    if mode == "echo":
        # loopback diagnostic: your mic comes straight back at 16 kHz
        transport.out_sample_rate = 16000
        await run_echo(transport)
    elif mode == "vad":
        # endpointing diagnostic: no audio back, just live speech detection
        await run_vad_probe(transport)
    elif mode == "call":
        if user["role"] == "admin":
            # admins keep a free-choice Bot + Language selector in the UI for testing
            try:
                bot_id = int(event.data.get("bot_id"))
            except (TypeError, ValueError):
                bot_id = None
            language = event.data.get("language")
        else:
            # everyone else picks from their own assigned (bot, language) pairs -
            # enforced here server-side by re-looking-up the assignment, not just
            # hidden in the UI, so a crafted WS payload can't pick an arbitrary
            # bot or language either
            try:
                assignment_id = int(event.data.get("assignment_id"))
            except (TypeError, ValueError):
                assignment_id = None
            assignment = auth.get_user_bot_assignment(assignment_id, user["id"]) if assignment_id else None
            if assignment is None:
                await transport.send_control(
                    {"type": "error", "message": "no bot assigned - contact your admin"}
                )
                await transport.close()
                return
            bot_id = assignment["bot_id"]
            language = assignment["language_code"]
            # re-check the Voice Quality tier server-side too, same reasoning as
            # the bot/language re-lookup above - a crafted WS payload shouldn't
            # be able to pick a tier the admin didn't allow for this assignment
            allowed_tiers = assignment["allowed_tiers"]
            if allowed_tiers and event.data.get("tts_tier") not in allowed_tiers:
                await transport.send_control(
                    {"type": "error", "message": "that voice quality isn't allowed for this bot"}
                )
                await transport.close()
                return
        # only real calls are recorded - the echo/vad modes above are
        # diagnostics, and a loopback of your own mic isn't worth archiving
        if get_config().call.record_calls:
            transport.recorder = CallRecorder(
                caller_rate=transport.in_sample_rate, agent_rate=transport.out_sample_rate
            )
        company_name = event.data.get("company_name")
        if not company_name and user.get("username") and "collegevidya" in user["username"].lower():
            company_name = "College Vidya"
        await run_call(
            transport,
            language,
            event.data.get("gender"),
            event.data.get("tts_tier"),
            build_call_vars(event.data.get("customer_name"), company_name),
            bot_id,
            event.data.get("tts_voice"),
            user["id"],
            user["username"],
        )
    else:
        await transport.send_control({"type": "error", "message": f"unknown mode '{mode}'"})
        await transport.close()


async def run_echo(transport: AudioTransport) -> None:
    log.info("echo session started")
    await transport.send_control(
        {"type": "ready", "mode": "echo", "out_sample_rate": transport.out_sample_rate}
    )
    await transport.send_control({"type": "turn_start", "turn_id": 1})
    await transport.send_control({"type": "state", "state": "listening"})
    frames = 0
    try:
        while True:
            event = await transport.recv()
            if isinstance(event, Disconnected):
                break
            if isinstance(event, AudioFrame):
                frames += 1
                await transport.send_audio(1, event.pcm)
            elif isinstance(event, ControlMessage) and event.data.get("type") == "stop":
                break
    finally:
        log.info("echo session ended (%d frames echoed)", frames)
        await transport.close()


async def run_vad_probe(transport: AudioTransport) -> None:
    """Phase 3 diagnostic: run the VAD gate live and report state + utterances."""
    cfg = get_config()
    gate = VADGate(SileroVAD(), cfg.vad)
    frame_buffer = FrameBuffer()
    dump_dir = ROOT / "logs" / "utterances"
    n_utterances = 0

    log.info("vad probe session started")
    await transport.send_control(
        {"type": "ready", "mode": "vad", "out_sample_rate": transport.out_sample_rate}
    )
    await transport.send_control({"type": "state", "state": "listening"})
    try:
        while True:
            event = await transport.recv()
            if isinstance(event, Disconnected):
                break
            if isinstance(event, ControlMessage):
                if event.data.get("type") == "stop":
                    break
                continue
            for frame in frame_buffer.push(event.pcm):
                for vad_event in gate.push(frame):
                    if isinstance(vad_event, SpeechStart):
                        log.info("speech start")
                        await transport.send_control({"type": "state", "state": "user_speaking"})
                    elif isinstance(vad_event, SpeechEnd):
                        n_utterances += 1
                        log.info(
                            "utterance captured: %.2fs%s",
                            vad_event.duration_s,
                            " (forced)" if vad_event.forced else "",
                        )
                        await transport.send_control({"type": "state", "state": "listening"})
                        await transport.send_control(
                            {
                                "type": "vad_utterance",
                                "duration_s": round(vad_event.duration_s, 2),
                                "forced": vad_event.forced,
                            }
                        )
                        if cfg.call.dump_utterances:
                            dump_dir.mkdir(parents=True, exist_ok=True)
                            name = f"{time.strftime('%Y%m%d_%H%M%S')}_{n_utterances:03d}.wav"
                            (dump_dir / name).write_bytes(wav_wrap(vad_event.utterance))
                            log.info("dumped %s", name)
    finally:
        log.info("vad probe session ended (%d utterances)", n_utterances)
        await transport.close()


async def run_call(
    transport: AudioTransport,
    language_code: str | None = None,
    gender: str | None = None,
    tts_tier: str | None = None,
    call_vars: dict | None = None,
    bot_id: int | None = None,
    tts_voice: str | None = None,
    user_id: int | None = None,
    username: str | None = None,
) -> None:
    # every bot supports every pipeline - which one runs is purely a per-call
    # choice, picked via the Voice Quality dropdown's synthetic "gemini_live"
    # tier (tts_catalog.gemini_voice_tier()), not a bot property
    pipeline = "gemini_live" if tts_tier == "gemini_live" else "standard"
    cfg = apply_language(get_config(), language_code, gender, tts_tier, call_vars, bot_id, tts_voice)
    problems = missing_secrets(cfg, pipeline)
    if problems:
        await transport.send_control(
            {"type": "error", "message": "server is missing credentials: " + "; ".join(problems)}
        )
        await transport.close()
        return

    if pipeline == "gemini_live":
        await run_live_call(transport, cfg, gender, tts_voice, user_id, username)
        return

    try:
        stt = build_stt(cfg)
        llm = build_llm(cfg)
        tts = build_tts(cfg)
        session = CallSession(transport, cfg, stt, llm, tts, user_id=user_id, username=username)
    except Exception as e:  # unknown provider, bad SDK creds, VAD/log setup failure, etc.
        log.exception("failed to set up the call")
        await transport.send_control({"type": "error", "message": f"call setup failed: {e}"})
        await transport.close()
        return

    await session.run()


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
