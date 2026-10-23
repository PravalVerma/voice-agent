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
    mode TEXT NOT NULL DEFAULT 'assistant' CHECK(mode IN ('assistant', 'collections')),
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

-- Borrowers a collections-mode bot may be pointed at. Nothing here is supplied
-- by the caller or the model: the compliance gates read it before a call is
-- placed, and the MCP tools read it to resolve who they are acting on.
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_ref TEXT UNIQUE NOT NULL,          -- the lender's own id; the CSV upsert key
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',  -- IANA name; the calling window is
                                                    -- evaluated here, not on server time
    language_code TEXT,
    -- integer paise, never a float: these are compared for equality against
    -- figures parsed out of speech, and a representation error there would
    -- block a correct sentence in a system whose whole job is checking figures
    amount_due_paise INTEGER NOT NULL,
    due_date TEXT NOT NULL,                     -- 'YYYY-MM-DD'
    lender_name TEXT NOT NULL,
    verify_kind TEXT NOT NULL DEFAULT 'dob'
        CHECK(verify_kind IN ('dob', 'phone_last4', 'pin')),
    verify_value TEXT NOT NULL,                 -- never spoken, never sent to the model
    opt_out INTEGER NOT NULL DEFAULT 0,
    dispute_open INTEGER NOT NULL DEFAULT 0,
    grievance_open INTEGER NOT NULL DEFAULT 0,
    hardship INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per dialling attempt, which is what the regulatory cap counts - a
-- call that rings out still spends one, so this is written before the call
-- connects rather than after.
CREATE TABLE IF NOT EXISTS contact_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT,
    kind TEXT NOT NULL CHECK(kind IN ('attempt', 'connected', 'refused')),
    reason TEXT,                                -- compliance Decision.code when refused
    at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_contact_events_customer_at
    ON contact_events (customer_id, at);

-- What the in-call tools write. Each row names the session that produced it,
-- so an outcome can always be traced back to the call and its transcript.
CREATE TABLE IF NOT EXISTS promises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    promise_date TEXT NOT NULL,
    amount_paise INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS callbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    callback_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per customer: opting out twice is not a different fact. The
-- authoritative flag the eligibility gate reads is customers.opt_out; this is
-- the audit record of when and in which call it happened.
CREATE TABLE IF NOT EXISTS optouts (
    customer_id INTEGER PRIMARY KEY REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'voice',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Intent only: no message is dispatched. phone is snapshotted from the account
-- at the time, so a later change to the record cannot rewrite what was sent.
CREATE TABLE IF NOT EXISTS payment_link_intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    phone TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Which tools a bot may actually execute. NULL or empty means every tool is
-- allowed, following user_bots.allowed_tiers. The declared schema list sent to
-- the model never varies - only execution is gated - so that restricting a bot
-- cannot change the prompt.
CREATE TABLE IF NOT EXISTS bot_tools (
    bot_id INTEGER PRIMARY KEY REFERENCES bots(id) ON DELETE CASCADE,
    allowed TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The MCP tool server runs as its own process against this same file, so
    # writes genuinely overlap. The default rollback journal takes a database-
    # wide write lock and the loser gets SQLITE_BUSY immediately; WAL lets a
    # reader and a writer coexist, and busy_timeout makes two writers queue
    # instead of failing. journal_mode is a persistent property of the file and
    # only has to take once; busy_timeout is per-connection and must be set on
    # every open.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
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
        # mode predates this column - every existing bot stays an ordinary
        # assistant, behaving exactly as it did, until an admin switches it
        # over. SQLite cannot add a CHECK via ALTER, so the constraint is in
        # _SCHEMA for new databases and validated in main.py for migrated ones
        # - the same split bots.pipeline already has.
        if not _has_column(conn, "bots", "mode"):
            conn.execute("ALTER TABLE bots ADD COLUMN mode TEXT NOT NULL DEFAULT 'assistant'")
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
