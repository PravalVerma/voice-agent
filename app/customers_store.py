"""Borrower records for collections-mode calls (customers table).

Separate from bots and users: a bot is a persona, a user is a login, and a
customer is the person being called. The compliance gates read these before a
call is placed, and the MCP tools read them to resolve who an action applies
to - which is why neither the caller nor the model ever supplies a customer id.

Money is stored as integer paise throughout. These figures are compared for
equality against amounts parsed out of speech, and a float representation error
would block a correct sentence in a system whose entire job is checking
figures. Rupees exist only at the UI boundary, in `rupees` on the way out.
"""

from .db import get_connection

#: Columns a CSV re-import is allowed to overwrite on an existing row.
#: Deliberately excludes every status flag - see upsert_customer().
_CSV_UPDATABLE = (
    "name", "phone", "timezone", "language_code",
    "amount_due_paise", "due_date", "lender_name",
)

#: Never leaves the server: the answer VERIFY checks against.
_SECRET = "verify_value"


def _public(row) -> dict:
    """A customer as the API and UI may see it - without the verify answer."""
    d = {k: row[k] for k in row.keys() if k != _SECRET}
    d["rupees"] = round(d["amount_due_paise"] / 100, 2)
    d["has_verify_value"] = bool(row[_SECRET])
    return d


def list_customers() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM customers ORDER BY name, id").fetchall()
        return [_public(r) for r in rows]
    finally:
        conn.close()


def get_customer(customer_id: int) -> dict | None:
    """Public view: safe to serialise. Use get_verify_value() for the secret."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return _public(row) if row is not None else None
    finally:
        conn.close()


def get_verify_value(customer_id: int) -> tuple[str, str] | None:
    """(verify_kind, verify_value) for the identity check.

    Isolated in its own function so the secret is read deliberately, at the one
    place that compares against it, rather than riding along in every customer
    dict and ending up in a log line or an API response by accident.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT verify_kind, verify_value FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        return (row["verify_kind"], row["verify_value"]) if row is not None else None
    finally:
        conn.close()


def create_customer(**fields) -> dict:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    conn = get_connection()
    try:
        cur = conn.execute(f"INSERT INTO customers ({cols}) VALUES ({marks})", tuple(fields.values()))
        conn.commit()
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _public(row)
    finally:
        conn.close()


def update_customer(customer_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE customers SET {sets}, updated_at = datetime('now') WHERE id = ?",
            (*fields.values(), customer_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_customer(customer_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
        conn.commit()
    finally:
        conn.close()


def set_flag(customer_id: int, flag: str, value: bool = True) -> None:
    """Set one status flag. Used by the MCP tools (opt-out, dispute)."""
    if flag not in ("opt_out", "dispute_open", "grievance_open", "hardship"):
        raise ValueError(f"unknown customer flag {flag!r}")
    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE customers SET {flag} = ?, updated_at = datetime('now') WHERE id = ?",
            (1 if value else 0, customer_id),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_customer(customer_ref: str, **fields) -> str:
    """Insert, or update an existing row matched on the lender's own id.

    Returns "inserted" | "updated". Only _CSV_UPDATABLE columns are written on
    an update, and that restriction is the whole point of this function.

    A naive INSERT OR REPLACE would carry the file's values over every column,
    so a nightly re-import of an unchanged extract would silently clear
    opt_out, dispute_open, grievance_open and hardship for everyone who had
    them set - un-opting-out people who asked not to be called, invisibly, with
    no error and nothing in a log to notice. Those flags are set by what
    happened on a call, never by an upstream file, so they are never updated
    here.
    """
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM customers WHERE customer_ref = ?", (customer_ref,)
        ).fetchone()
        if existing is None:
            cols = ["customer_ref", *fields]
            marks = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO customers ({', '.join(cols)}) VALUES ({marks})",
                (customer_ref, *fields.values()),
            )
            conn.commit()
            return "inserted"

        safe = {k: v for k, v in fields.items() if k in _CSV_UPDATABLE}
        if safe:
            sets = ", ".join(f"{k} = ?" for k in safe)
            conn.execute(
                f"UPDATE customers SET {sets}, updated_at = datetime('now') WHERE customer_ref = ?",
                (*safe.values(), customer_ref),
            )
            conn.commit()
        return "updated"
    finally:
        conn.close()
