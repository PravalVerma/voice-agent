"""Named, per-language agent prompts ("bots"), stored in SQLite (bots/bot_prompts
tables) so the admin can register multiple personas - each with its own English
prompt, Hindi prompt, etc. - and callers pick one per call, same as they pick a
language or voice gender. Every bot supports every pipeline (standard/gemini_live) -
which one runs is a separate per-call choice (see config.apply_language), not
a bot property."""

import json

from .db import get_connection

DEFAULT_BOT_PROMPT = """You are a friendly and concise voice assistant on a phone call.
If the voice speaking for you on this call is female, your name is Asha. If it is male,
your name is Rohan.
The caller does not know who you are yet - in your very first reply of the call, start with
a brief self-introduction (your name and that you're a voice assistant) before addressing what
they said. Don't reintroduce yourself in later replies.
Keep answers short - one to three sentences - and conversational.
If you don't know something, say so briefly rather than guessing."""


def list_bots() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, name, mode, created_at FROM bots ORDER BY id").fetchall()
        bots = [dict(r) for r in rows]
        for bot in bots:
            lang_rows = conn.execute(
                "SELECT language_code FROM bot_prompts WHERE bot_id = ? ORDER BY language_code",
                (bot["id"],),
            ).fetchall()
            bot["languages"] = [r["language_code"] for r in lang_rows]
        return bots
    finally:
        conn.close()


def count_bots() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM bots").fetchone()["n"]
    finally:
        conn.close()


def create_bot(name: str) -> dict:
    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO bots (name) VALUES (?)", (name,))
        conn.commit()
        return {"id": cur.lastrowid, "name": name}
    finally:
        conn.close()


def rename_bot(bot_id: int, name: str) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE bots SET name = ? WHERE id = ?", (name, bot_id))
        conn.commit()
    finally:
        conn.close()


def delete_bot(bot_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM bots WHERE id = ?", (bot_id,))
        conn.commit()
    finally:
        conn.close()


def get_bot(bot_id: int) -> dict | None:
    conn = get_connection()
    try:
        bot_row = conn.execute(
            "SELECT id, name, mode, created_at FROM bots WHERE id = ?", (bot_id,)
        ).fetchone()
        if bot_row is None:
            return None
        prompt_rows = conn.execute(
            "SELECT language_code, prompt FROM bot_prompts WHERE bot_id = ?", (bot_id,)
        ).fetchall()
        bot = dict(bot_row)
        bot["prompts"] = {r["language_code"]: r["prompt"] for r in prompt_rows}
        tools_row = conn.execute(
            "SELECT allowed FROM bot_tools WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        raw = tools_row["allowed"] if tools_row is not None else None
        bot["allowed_tools"] = json.loads(raw) if raw else None
        return bot
    finally:
        conn.close()


def get_prompt(bot_id: int, language_code: str) -> str:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT prompt FROM bot_prompts WHERE bot_id = ? AND language_code = ?",
            (bot_id, language_code),
        ).fetchone()
        return row["prompt"] if row is not None else ""
    finally:
        conn.close()


def set_prompt(bot_id: int, language_code: str, text: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO bot_prompts (bot_id, language_code, prompt) VALUES (?, ?, ?) "
            "ON CONFLICT(bot_id, language_code) DO UPDATE SET prompt = excluded.prompt",
            (bot_id, language_code, text),
        )
        conn.commit()
    finally:
        conn.close()


def _resolve_bot(bot_id: int | None) -> dict | None:
    """Falls back to the first registered bot if bot_id is missing/unknown,
    and to None if no bots exist at all - the shared lookup
    resolve_prompt_for_call() builds on."""
    bots = list_bots()
    if not bots:
        return None
    return next((b for b in bots if b["id"] == bot_id), None) or bots[0]


def resolve_prompt_for_call(bot_id: int | None, language_code: str) -> str:
    """Used by config.apply_language(). Falls back to the first registered bot
    if bot_id is missing/unknown, and to DEFAULT_BOT_PROMPT if that bot has no
    prompt saved yet for this language (or no bots exist at all)."""
    bot = _resolve_bot(bot_id)
    if bot is None:
        return DEFAULT_BOT_PROMPT
    return get_prompt(bot["id"], language_code) or DEFAULT_BOT_PROMPT


def get_tool_allowlist(bot_id: int | None) -> list[str] | None:
    """Tools this bot may execute, or None when every tool is allowed.

    None rather than the full list, matching user_bots.allowed_tiers: a tool
    added later is then allowed automatically instead of being silently
    excluded because it postdates when the row was saved.

    This gates EXECUTION only. The schemas sent to the model never vary by bot
    - see pipeline._tools_for.
    """
    if bot_id is None:
        return None
    conn = get_connection()
    try:
        row = conn.execute("SELECT allowed FROM bot_tools WHERE bot_id = ?", (bot_id,)).fetchone()
    finally:
        conn.close()
    if row is None or not row["allowed"]:
        return None
    try:
        decoded = json.loads(row["allowed"])
    except json.JSONDecodeError:
        return None
    return decoded or None


def set_tool_allowlist(bot_id: int, allowed: list[str] | None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO bot_tools (bot_id, allowed) VALUES (?, ?) "
            "ON CONFLICT(bot_id) DO UPDATE SET allowed = excluded.allowed",
            (bot_id, json.dumps(allowed) if allowed else None),
        )
        conn.commit()
    finally:
        conn.close()


def get_mode(bot_id: int | None) -> str:
    """'assistant' (the default) or 'collections'."""
    if bot_id is None:
        return "assistant"
    conn = get_connection()
    try:
        row = conn.execute("SELECT mode FROM bots WHERE id = ?", (bot_id,)).fetchone()
    finally:
        conn.close()
    return (row["mode"] if row is not None else "assistant") or "assistant"


def set_mode(bot_id: int, mode: str) -> None:
    if mode not in ("assistant", "collections"):
        raise ValueError(f"unknown bot mode {mode!r}")
    conn = get_connection()
    try:
        conn.execute("UPDATE bots SET mode = ? WHERE id = ?", (mode, bot_id))
        conn.commit()
    finally:
        conn.close()


def migrate_legacy_base_prompt_if_needed(language_codes: list[str]) -> None:
    """One-time migration: before bots existed, every language shared a single
    admin-edited `base_prompt` (settings table, key='base_prompt'). If no bots
    have been registered yet, carry that legacy text (if any) into a "Default"
    bot's prompt for every currently-configured language, so existing calls
    keep behaving exactly as before until the admin splits them apart."""
    if count_bots() > 0:
        return
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM settings WHERE key = 'base_prompt'").fetchone()
    finally:
        conn.close()
    legacy_text = row["value"] if row is not None else DEFAULT_BOT_PROMPT
    bot = create_bot("Default")
    for code in language_codes:
        set_prompt(bot["id"], code, legacy_text)
