"""User accounts and cookie sessions, backed by the SQLite tables in db.py.

Server-side sessions (not a signed cookie) so deleting a user immediately kills
their active login, without needing a separate revocation list.
"""

import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta

from fastapi import HTTPException, Request

from .db import get_connection

log = logging.getLogger("agent")

SESSION_COOKIE = "session_id"
SESSION_TTL_HOURS = 24 * 7
_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt_hex


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    candidate, _ = hash_password(password, salt_hex)
    return secrets.compare_digest(candidate, password_hash)


def create_user(username: str, password: str, role: str = "user") -> dict:
    password_hash, salt = hash_password(password)
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, role),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username, "role": role}
    finally:
        conn.close()


def list_users() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
        users = [dict(r) for r in rows]
    finally:
        conn.close()
    for u in users:
        u["bots"] = list_user_bots(u["id"])
    return users


def _decode_allowed_tiers(raw: str | None) -> list[str] | None:
    """None means "every Voice Quality tier is allowed" (the default, and the
    only possibility before this column existed) - kept as None rather than
    the full tier list so a tier added later is automatically allowed too,
    instead of silently excluded because it postdates when this was saved."""
    return json.loads(raw) if raw else None


def _encode_allowed_tiers(allowed_tiers: list[str] | None) -> str | None:
    return json.dumps(allowed_tiers) if allowed_tiers else None


def list_user_bots(user_id: int) -> list[dict]:
    """A user's assigned (bot, language) pairs - each is an independently
    clickable option in their Bot dropdown, since the same bot can be assigned
    twice with two different languages."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT user_bots.id, user_bots.bot_id, bots.name AS bot_name,
                      user_bots.language_code, user_bots.allowed_tiers
               FROM user_bots JOIN bots ON bots.id = user_bots.bot_id
               WHERE user_bots.user_id = ?
               ORDER BY bots.name, user_bots.language_code""",
            (user_id,),
        ).fetchall()
        assignments = [dict(r) for r in rows]
        for a in assignments:
            a["allowed_tiers"] = _decode_allowed_tiers(a["allowed_tiers"])
        return assignments
    finally:
        conn.close()


def add_user_bot(
    user_id: int, bot_id: int, language_code: str, allowed_tiers: list[str] | None = None
) -> dict:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO user_bots (user_id, bot_id, language_code, allowed_tiers) VALUES (?, ?, ?, ?)",
            (user_id, bot_id, language_code, _encode_allowed_tiers(allowed_tiers)),
        )
        conn.commit()
        return {
            "id": cur.lastrowid,
            "user_id": user_id,
            "bot_id": bot_id,
            "language_code": language_code,
            "allowed_tiers": allowed_tiers,
        }
    finally:
        conn.close()


def set_user_bot_tiers(assignment_id: int, allowed_tiers: list[str] | None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE user_bots SET allowed_tiers = ? WHERE id = ?",
            (_encode_allowed_tiers(allowed_tiers), assignment_id),
        )
        conn.commit()
    finally:
        conn.close()


def remove_user_bot(user_id: int, bot_id: int, language_code: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM user_bots WHERE user_id = ? AND bot_id = ? AND language_code = ?",
            (user_id, bot_id, language_code),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_bot_assignment(assignment_id: int, user_id: int) -> dict | None:
    """Looks up one of *this* user's assignments by id - used to authorize a
    call: a client-supplied assignment_id that doesn't belong to the caller
    resolves to None, same as an unknown id."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT bot_id, language_code, allowed_tiers FROM user_bots WHERE id = ? AND user_id = ?",
            (assignment_id, user_id),
        ).fetchone()
        if row is None:
            return None
        assignment = dict(row)
        assignment["allowed_tiers"] = _decode_allowed_tiers(assignment["allowed_tiers"])
        return assignment
    finally:
        conn.close()


def count_users() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    finally:
        conn.close()


def delete_user(user_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    conn = get_connection()
    try:
        # case-insensitive - "Alex@example.com" and "alex@example.com" are the
        # same login, matching how every real email/username system behaves
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def delete_session(token: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def get_session_user(token: str | None) -> dict | None:
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT users.id, users.username, users.role, sessions.expires_at
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token = ?""",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return {"id": row["id"], "username": row["username"], "role": row["role"]}
    finally:
        conn.close()


def seed_admin_if_empty() -> None:
    if count_users() > 0:
        return
    username = os.environ.get("ADMIN_USERNAME", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not username or not password:
        log.warning(
            "no users exist yet and ADMIN_USERNAME/ADMIN_PASSWORD are not set in .env "
            "- nobody will be able to log in until an admin user is created"
        )
        return
    create_user(username, password, role="admin")
    log.info("seeded initial admin user %r", username)


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    return get_session_user(token)


def require_login(request: Request) -> dict:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


def require_admin(request: Request) -> dict:
    user = require_login(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin access required")
    return user
