"""FastAPI app: serves the browser demo and the /ws call endpoint."""

import asyncio
import datetime
import logging
import os
import sqlite3
import time
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import (auth, bots_store, call_logs, calllog, compliance, customers_store, post_call,
               tts_cache, tts_catalog)
from .audio.frames import FrameBuffer, wav_wrap
from .collections_flow import CollectionsFlow
from .audio.vad import SileroVAD, SpeechEnd, SpeechStart, VADGate
from .config import ROOT, apply_language, build_call_vars, get_config, missing_secrets
from .db import init_db
from .live_session import run_live_call
from .mcp_client import McpClient
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
    # Tool schemas are identical for every bot and every call, so they are
    # listed once per process. Fire-and-forget: a missing collections extra
    # must not stop the server booting for the assistant bots that don't need
    # it, and a collections call refuses cleanly later if this never succeeded.
    asyncio.create_task(_warm_tools())
    # Same fire-and-forget reasoning for the fixed script audio: a cold cache
    # puts several serial synthesis round trips in front of the opening
    # disclosure, and a provider that is down at boot must not stop the server.
    asyncio.create_task(tts_cache.prewarm(cfg, build_tts))
    if cfg.llm_post.in_process:
        # Post-call work shares this process by default. Set [llm_post]
        # in_process = false and schedule scripts/run_post_jobs.py instead if
        # it should not. Either way the queue is the log directory, so nothing
        # is lost by switching between them or by a restart mid-sweep.
        asyncio.create_task(post_call.loop(cfg))


async def _warm_tools() -> None:
    try:
        schemas = await McpClient.warm()
        if schemas:
            log.info("tool server ready: %d tools", len(schemas))
        else:
            log.info("tool server unavailable - collections bots will run without tools")
    except Exception as e:
        log.info("tool server unavailable (%s) - collections bots will run without tools", e)


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


class CustomerBody(BaseModel):
    customer_ref: str
    name: str
    phone: str
    amount_due_rupees: float
    due_date: str
    lender_name: str
    timezone: str = "Asia/Kolkata"
    language_code: str | None = None
    verify_kind: str = "dob"
    verify_value: str = ""


class CustomerFlagsBody(BaseModel):
    opt_out: bool | None = None
    dispute_open: bool | None = None
    grievance_open: bool | None = None
    hardship: bool | None = None


class BotModeBody(BaseModel):
    mode: str


class BotToolsBody(BaseModel):
    allowed_tools: list[str] | None = None


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
    tiers = [tts_catalog.cascaded_tier(lang.locale, cfg), tts_catalog.gemini_voice_tier()]
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


def _validate_allowed_tools(allowed_tools: list[str] | None) -> None:
    """Mirrors _validate_allowed_tiers, including the empty-list refusal.

    None means "every tool", so an empty list is almost always a UI slip rather
    than an intent to disable the bot entirely - and silently storing it would
    leave an agent that can talk but never act, with nothing to show why.
    """
    if allowed_tools is None:
        return
    if not allowed_tools:
        raise HTTPException(
            status_code=400,
            detail="allowed_tools can't be an empty list - omit it instead to allow every tool",
        )
    known = {s["function"]["name"] for s in McpClient.schemas()}
    if not known:
        return  # tool server never came up; nothing to validate against
    unknown = sorted(set(allowed_tools) - known)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown tool(s): {unknown}")


@app.get("/api/tools")
async def api_list_tools(admin: dict = Depends(auth.require_admin)) -> list[dict]:
    """The tool catalogue the allowlist editor offers.

    Names and descriptions only - these are what the admin is choosing between,
    and the parameter schemas would just be noise in a picker.
    """
    return [
        {"name": s["function"]["name"], "description": s["function"].get("description", "")}
        for s in McpClient.schemas()
    ]


@app.put("/api/bots/{bot_id}/mode")
async def api_set_bot_mode(bot_id: int, body: BotModeBody, admin: dict = Depends(auth.require_admin)) -> dict:
    if body.mode not in ("assistant", "collections"):
        raise HTTPException(status_code=400, detail="mode must be 'assistant' or 'collections'")
    if bots_store.get_bot(bot_id) is None:
        raise HTTPException(status_code=404, detail="no such bot")
    bots_store.set_mode(bot_id, body.mode)
    return {"ok": True}


@app.put("/api/bots/{bot_id}/tools")
async def api_set_bot_tools(bot_id: int, body: BotToolsBody, admin: dict = Depends(auth.require_admin)) -> dict:
    if bots_store.get_bot(bot_id) is None:
        raise HTTPException(status_code=404, detail="no such bot")
    _validate_allowed_tools(body.allowed_tools)
    bots_store.set_tool_allowlist(bot_id, body.allowed_tools)
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


# ---- customers API -------------------------------------------------------------
# Listing requires a login rather than admin: a user has to pick who they are
# calling. Every mutation is admin-only, and nothing here ever returns
# verify_value - customers_store strips it (see _public there).

_VERIFY_KINDS = ("dob", "phone_last4", "pin")


def _validate_customer(body: CustomerBody) -> None:
    if body.verify_kind not in _VERIFY_KINDS:
        raise HTTPException(status_code=400, detail=f"verify_kind must be one of {list(_VERIFY_KINDS)}")
    if body.amount_due_rupees < 0:
        raise HTTPException(status_code=400, detail="amount due can't be negative")
    try:
        datetime.date.fromisoformat(body.due_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="due_date must be YYYY-MM-DD")
    # Fail here rather than at call time: an unresolvable timezone makes the
    # calling-window gate fail closed, which looks like a mysteriously
    # un-callable customer rather than a typo in an import.
    try:
        ZoneInfo(body.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(status_code=400, detail=f"unknown timezone {body.timezone!r}")


def _customer_fields(body: CustomerBody) -> dict:
    return {
        "customer_ref": body.customer_ref.strip(),
        "name": body.name.strip(),
        "phone": body.phone.strip(),
        "timezone": body.timezone,
        "language_code": body.language_code,
        "amount_due_paise": int(round(body.amount_due_rupees * 100)),
        "due_date": body.due_date,
        "lender_name": body.lender_name.strip(),
        "verify_kind": body.verify_kind,
        "verify_value": body.verify_value.strip(),
    }


@app.get("/api/customers")
async def api_list_customers(user: dict = Depends(auth.require_login)) -> list[dict]:
    return customers_store.list_customers()


@app.post("/api/customers")
async def api_create_customer(body: CustomerBody, admin: dict = Depends(auth.require_admin)) -> dict:
    _validate_customer(body)
    try:
        return customers_store.create_customer(**_customer_fields(body))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="a customer with that reference already exists")


@app.put("/api/customers/{customer_id}")
async def api_update_customer(customer_id: int, body: CustomerBody,
                              admin: dict = Depends(auth.require_admin)) -> dict:
    _validate_customer(body)
    if customers_store.get_customer(customer_id) is None:
        raise HTTPException(status_code=404, detail="no such customer")
    fields = _customer_fields(body)
    if not fields["verify_value"]:
        # an empty box in the edit form means "leave it as it is", not "clear
        # the identity answer" - the UI never receives it, so it cannot resend it
        fields.pop("verify_value")
    customers_store.update_customer(customer_id, **fields)
    return {"ok": True}


@app.put("/api/customers/{customer_id}/flags")
async def api_set_customer_flags(customer_id: int, body: CustomerFlagsBody,
                                 admin: dict = Depends(auth.require_admin)) -> dict:
    if customers_store.get_customer(customer_id) is None:
        raise HTTPException(status_code=404, detail="no such customer")
    for flag, value in body.model_dump(exclude_none=True).items():
        customers_store.set_flag(customer_id, flag, value)
    return {"ok": True}


@app.delete("/api/customers/{customer_id}")
async def api_delete_customer(customer_id: int, admin: dict = Depends(auth.require_admin)) -> dict:
    customers_store.delete_customer(customer_id)
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
        # Whoever picked it, the tier name itself has to be one the app still
        # offers. The entitlement check above is skipped when an assignment
        # allows every tier, and admins skip it entirely, so without this a
        # retired name (or an invented one) reached resolve_voice and used to
        # be silently downgraded to the cheapest tier rather than refused.
        if event.data.get("tts_tier") not in tts_catalog.all_tier_names():
            await transport.send_control(
                {"type": "error", "message": "that voice quality is not available"}
            )
            await transport.close()
            return
        # Pre-call compliance gates. A customer_id is what makes this a
        # collections call: an ordinary assistant call has no borrower and no
        # window to be outside of. Both gates run server-side off the stored
        # record, never off anything the client sent - the client supplies only
        # which customer, and even that is re-read from the database here.
        customer = None
        try:
            customer_id = int(event.data.get("customer_id"))
        except (TypeError, ValueError):
            customer_id = None
        bot_mode = bots_store.get_mode(bot_id)
        if bot_mode == "collections" and customer_id is None:
            # A collections bot has nothing to check without an account: no
            # calling window, no eligibility, no figures to validate speech
            # against. Running it anyway would be the compliance layer quietly
            # switched off, so refuse instead.
            await transport.send_control(
                {"type": "error", "message": "this bot needs an account selected before a call can start"}
            )
            await transport.close()
            return
        if customer_id is not None:
            cfg_now = get_config()
            customer = customers_store.get_customer(customer_id)
            if customer is None:
                decision = compliance.Decision(
                    False, "unknown_customer", "no such customer record", str(customer_id)
                )
            else:
                decision = compliance.check_calling_window(customer["timezone"], cfg=cfg_now)
                if decision.ok:
                    decision = compliance.check_contact_eligibility(customer_id, cfg=cfg_now)
            if not decision.ok:
                # Refused before anything is allocated: no recorder, no
                # providers, no session. The attempt is still recorded, because
                # a refusal is itself a contact decision an auditor will ask
                # about, and it leaves a call-log directory for the same reason.
                compliance.record_contact_event(customer_id, "refused", reason=decision.code)
                calllog.log_refusal(
                    cfg_now, decision,
                    user_id=user["id"], username=user["username"],
                    customer_id=customer_id, language=language,
                    tts_tier=event.data.get("tts_tier"),
                )
                await transport.send_control(
                    {"type": "error", "message": "call not permitted: " + decision.detail}
                )
                await transport.close()
                return
            # The cap counts attempts, so this is written before the call
            # connects rather than after it succeeds.
            compliance.record_contact_event(customer_id, "attempt")

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
            customer,
            bot_mode,
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
    customer: dict | None = None,
    mode: str = "assistant",
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
        # The Studio pipeline gets the same tool server, with the same per-bot
        # allowlist and the same server-side customer binding - only the
        # transport differs.
        live_mcp = None
        if customer and mode == "collections":
            live_mcp = McpClient(customer["id"], uuid.uuid4().hex[:8],
                                 allowed=bots_store.get_tool_allowlist(bot_id))
            await live_mcp.connect()
        try:
            await run_live_call(transport, cfg, gender, tts_voice, user_id, username, mcp=live_mcp)
        finally:
            if live_mcp is not None:
                await live_mcp.aclose()
        return

    try:
        stt = build_stt(cfg)
        llm = build_llm(cfg)
        tts = build_tts(cfg)
        # A bound customer is what makes this a collections call. The gates in
        # ws_call have already cleared it by this point.
        # The bot's mode decides, not merely whether an account was picked: an
        # assistant bot stays an assistant bot even if one was.
        flow = CollectionsFlow(cfg, customer) if (customer and mode == "collections") else None
        session = CallSession(transport, cfg, stt, llm, tts, user_id=user_id,
                              username=username, flow=flow)
        if customer:
            # Spawned here, alongside the providers, so the ~300ms process
            # start lands inside the setup window the caller is already waiting
            # through rather than in the middle of a turn.
            session.mcp = McpClient(
                customer["id"], session.session_id,
                allowed=bots_store.get_tool_allowlist(bot_id),
            )
            await session.mcp.connect()
            if not session.mcp.available:
                log.warning("call %s: no tool server - the agent will talk but cannot act",
                            session.session_id)
    except Exception as e:  # unknown provider, bad SDK creds, VAD/log setup failure, etc.
        log.exception("failed to set up the call")
        await transport.send_control({"type": "error", "message": f"call setup failed: {e}"})
        await transport.close()
        return

    await session.run()


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
