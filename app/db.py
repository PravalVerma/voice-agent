"""SQLite storage for users and sessions. No ORM - the schema is two small tables."""

import sqlite3
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    pipeline TEXT NOT NULL DEFAULT 'standard' CHECK(pipeline IN ('standard', 'gemini', 'gemini_live')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bot_prompts (
    bot_id INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    language_code TEXT NOT NULL,
    prompt TEXT NOT NULL,
    PRIMARY KEY (bot_id, language_code)
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_bots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    bot_id INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    language_code TEXT NOT NULL,
    allowed_tiers TEXT,
    UNIQUE (user_id, bot_id, language_code)
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA)
        # users.bot_id (one bot per user) predates the many-to-many user_bots table -
        # carry any existing single assignments over as a row per user, then drop the
        # column. Language wasn't tracked per-assignment yet at that point, so fall
        # back to "en" (this app's own default_language) for the migrated row.
        if _has_column(conn, "users", "bot_id"):
            conn.execute(
                """INSERT OR IGNORE INTO user_bots (user_id, bot_id, language_code)
                   SELECT id, bot_id, 'en' FROM users WHERE bot_id IS NOT NULL"""
            )
            conn.execute("ALTER TABLE users DROP COLUMN bot_id")
        # pipeline predates this column - existing bots default to 'standard'
        # (today's Whisper+gpt-oss+Google TTS stack), unaffected.
        if not _has_column(conn, "bots", "pipeline"):
            conn.execute("ALTER TABLE bots ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'standard'")
        # allowed_tiers predates this column - existing assignments get NULL,
        # which auth.py treats as "every Voice Quality tier is allowed" (today's
        # behavior), unaffected until an admin explicitly restricts one.
        if not _has_column(conn, "user_bots", "allowed_tiers"):
            conn.execute("ALTER TABLE user_bots ADD COLUMN allowed_tiers TEXT")
        # username lookups are case-insensitive (auth.get_user_by_username) so
        # uniqueness has to be too, or "Alex@example.com" and "alex@example.com"
        # could both be created and login would be ambiguous between them - the
        # column's own UNIQUE constraint is case-sensitive (SQLite's default
        # BINARY collation), so this adds a second, case-insensitive one
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_nocase "
            "ON users (username COLLATE NOCASE)"
        )
        conn.commit()
    finally:
        conn.close()
