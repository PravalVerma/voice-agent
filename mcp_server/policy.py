"""Input validation and persistence for the tools.

Every tool validates its own arguments here. Model output is never trusted:
a date can arrive in the past, an amount can arrive below the settlement floor
or above the balance, and a callback can land outside the permitted calling
window - each of which is a compliance breach if written through unchecked.

Rejections are terse and never quote the limit they enforce. Told the exact
bound, the model re-proposes precisely that bound and presents it to the caller
as the lender's offer, which is how a negotiating floor becomes an opening bid.
"""

import logging
from datetime import date, datetime, timedelta

from app.compliance import check_calling_window
from app.config import get_config
from app.db import get_connection

log = logging.getLogger("mcp_server")

_MAX_REASON_CHARS = 500


def _fail(error: str) -> dict:
    return {"ok": False, "error": error}


def record_promise(customer_id: int, session_id: str, when: str, amount_rupees: float) -> dict:
    cfg = get_config()
    try:
        promise_date = date.fromisoformat(when)
    except (ValueError, TypeError):
        return _fail("date not understood")

    today = date.today()
    if promise_date < today:
        return _fail("date is in the past")
    if promise_date > today + timedelta(days=cfg.compliance.max_promise_days):
        return _fail("date is too far out")

    conn = get_connection()
    try:
        row = conn.execute("SELECT amount_due_paise FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if row is None:
            return _fail("account not found")
        due = row["amount_due_paise"]
        try:
            paise = int(round(float(amount_rupees) * 100))
        except (TypeError, ValueError):
            return _fail("amount not understood")
        if paise <= 0:
            return _fail("amount must be positive")
        floor = int(round(due * cfg.compliance.settlement_floor_pct / 100))
        if paise < floor:
            return _fail("amount is below what can be accepted")
        if paise > due:
            return _fail("amount is more than the balance")

        conn.execute(
            "INSERT INTO promises (customer_id, session_id, promise_date, amount_paise) VALUES (?, ?, ?, ?)",
            (customer_id, session_id, promise_date.isoformat(), paise),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "date": promise_date.isoformat(), "amount_rupees": round(paise / 100, 2)}


def schedule_callback(customer_id: int, session_id: str, when: str) -> dict:
    cfg = get_config()
    try:
        moment = datetime.fromisoformat(when)
    except (ValueError, TypeError):
        return _fail("time not understood")

    conn = get_connection()
    try:
        row = conn.execute("SELECT timezone FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if row is None:
            return _fail("account not found")
        tz = row["timezone"]
    finally:
        conn.close()

    if moment.tzinfo is None:
        from zoneinfo import ZoneInfo
        try:
            moment = moment.replace(tzinfo=ZoneInfo(tz))
        except Exception:
            return _fail("time not understood")
    if moment < datetime.now(moment.tzinfo):
        return _fail("time is in the past")

    # The same gate that governs placing a call governs promising one. A
    # callback booked for 21:00 would be refused at dial time, so accepting it
    # here would mean telling the caller something untrue.
    decision = check_calling_window(tz, moment, cfg=cfg)
    if not decision.ok:
        return _fail("that time is outside when we are able to call")

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO callbacks (customer_id, session_id, callback_at) VALUES (?, ?, ?)",
            (customer_id, session_id, moment.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "at": moment.isoformat()}


def register_opt_out(customer_id: int, session_id: str) -> dict:
    """Record the opt-out and set it on the account, in one transaction.

    Both writes or neither: the optouts row is the audit record and
    customers.opt_out is what check_contact_eligibility actually reads before
    the next call. Writing only the first would leave a documented opt-out that
    the system keeps calling through.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO optouts (customer_id, session_id, channel) VALUES (?, ?, 'voice')",
            (customer_id, session_id),
        )
        conn.execute(
            "UPDATE customers SET opt_out = 1, updated_at = datetime('now') WHERE id = ?",
            (customer_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


def log_dispute(customer_id: int, session_id: str, reason: str) -> dict:
    reason = (reason or "").strip()[:_MAX_REASON_CHARS]
    if not reason:
        return _fail("reason is required")
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO disputes (customer_id, session_id, reason) VALUES (?, ?, ?)",
            (customer_id, session_id, reason),
        )
        conn.execute(
            "UPDATE customers SET dispute_open = 1, updated_at = datetime('now') WHERE id = ?",
            (customer_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


def send_payment_link(customer_id: int, session_id: str) -> dict:
    """Record the intent to send a link to the number already on file.

    No message is dispatched - there is no SMS vendor wired in. The phone is
    read from the account here rather than taken as an argument, which is the
    entire security property: the model has no way to redirect it.
    """
    conn = get_connection()
    try:
        row = conn.execute("SELECT phone FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if row is None:
            return _fail("account not found")
        conn.execute(
            "INSERT INTO payment_link_intents (customer_id, session_id, phone) VALUES (?, ?, ?)",
            (customer_id, session_id, row["phone"]),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}
