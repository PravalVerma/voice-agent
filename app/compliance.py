"""Compliance gates for collections calls: pure checks, enforced in code.

The system prompt is a hint. This module is the enforcement. Nothing here calls
an LLM or the network, and nothing here depends on how an admin worded a bot
prompt - which is the point, because prompt text cannot be audited, cannot be
checked in isolation, and is exactly what a model stops honouring once a caller
pushes back on it.

Four gates, two of them before a call is placed:

  check_calling_window      permitted hours, in the BORROWER's timezone
  check_contact_eligibility opt-out, disputes, grievances, hardship, attempt caps
  screen_outbound           every sentence, before it reaches TTS
  screen_inbound            every caller utterance, before the model sees it

What screen_outbound cannot catch, stated plainly so nobody mistakes it for a
guarantee: figures split across turns (not merely across sentences); romanised
Hindi spellings missing from the configured lexicon; implied amounts ("double
what you paid last month"); relative dates ("next Tuesday"); and anything the
voice mispronounces, since this reads text while the caller hears audio.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import AppConfig
from .db import get_connection
from .text.numerals import EN_LEXICON, find_word_numbers, has_numeric_token, normalize

log = logging.getLogger("agent")


@dataclass(frozen=True, slots=True)
class Decision:
    """The result of a gate. `code` is machine-stable and safe to branch on;
    `detail` is admin-facing English and must never name a vendor or model;
    `evidence` is the offending span verbatim, for the audit log."""

    ok: bool
    code: str = ""
    detail: str = ""
    evidence: str = ""


#: Shared singleton - screen_outbound runs on every sentence of every call, so
#: the overwhelmingly common "nothing wrong" answer should not allocate.
PASS = Decision(ok=True)


@dataclass(frozen=True, slots=True)
class Flag:
    code: str
    severity: str  # "high" | "medium" | "low"
    evidence: str


# Precedence is fixed HERE, in code, and deliberately not in config: a
# compliance staffer editing the pattern lists must not be able to reorder
# self-harm below opt-out. Callers act on flags[0].
_FLAG_ORDER = (
    "self_harm", "distress", "opt_out", "dispute",
    "third_party", "wrong_number", "human_request", "injection",
)
_FLAG_SEVERITY = {
    "self_harm": "high", "distress": "high", "opt_out": "high",
    "dispute": "medium", "third_party": "medium", "wrong_number": "medium",
    "human_request": "low", "injection": "low",
}


@dataclass(frozen=True, slots=True)
class AllowedFacts:
    """Everything the agent may put a number to on this call.

    Amounts are integer paise, pre-expanded to include each policy percentage
    of the principal: the script negotiates in percentages, and a caller may
    restate one as a rupee figure for the agent to confirm back.

    Dates are a RANGE, not a set. A promise-to-pay date is negotiated live and
    cannot be enumerated in advance.
    """

    amounts_paise: frozenset[int]
    percents: frozenset[int]
    counts: frozenset[int]
    date_lo: date
    date_hi: date
    tolerance_paise: int = 100


@dataclass(frozen=True, slots=True)
class Figure:
    kind: str  # "amount" | "percent" | "count" | "date"
    value: object  # int paise / int percent / int count / date
    text: str


# ---- patterns -----------------------------------------------------------------

# Word boundaries are load-bearing on the Latin alternatives: without them
# "rs" matches inside "understand" and every innocent sentence containing it
# trips the money marker. Devanagari forms are listed whole for the same
# reason - a bare "रु" occurs inside unrelated words.
_MONEY_WORDS = (
    r"(?:₹|\b(?:rupees?|rupaye|rupaiya|rs|inr)\b\.?"
    r"|रुपये|रुपए|रुपया)"
)
_PERCENT_WORDS = (
    r"(?:%|\b(?:percent|per ?cent)\b"
    r"|प्रतिशत|परसेंट|फ़ीसदी|फीसदी)"
)

_MONEY_MARKER_RE = re.compile(_MONEY_WORDS, re.IGNORECASE)
_PERCENT_MARKER_RE = re.compile(_PERCENT_WORDS, re.IGNORECASE)

# Rs 2,65,000 / Rs. 265000.50 / 265000 rupees
_CURRENCY_BEFORE_RE = re.compile(r"(?:₹|rs\.?|inr|रु\.?)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)
_CURRENCY_AFTER_RE = re.compile(r"([\d,]+(?:\.\d{1,2})?)\s*" + _MONEY_WORDS, re.IGNORECASE)
_PERCENT_DIGIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*" + _PERCENT_WORDS, re.IGNORECASE)
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_DMY_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_BARE_INT_RE = re.compile(r"\b\d+\b")


def _compiled(cfg: AppConfig, group: str, key: str) -> list[re.Pattern]:
    block = getattr(cfg.compliance, group, {}) or {}
    return [re.compile(p) for p in block.get(key, [])]


def _lexicon(cfg: AppConfig) -> dict[str, int]:
    """English words stay in scope for every language: a Hindi call still says
    "lakh", and an English call still uses the Indian scale words."""
    hi = (cfg.compliance.numerals or {}).get("hi", {}) or {}
    # Keys are normalised the same way the scanned text is, so a config entry
    # typed in one Devanagari form still matches speech transcribed in another.
    return {**EN_LEXICON, **{normalize(k.lower()): int(v) for k, v in hi.items()}}


# ---- pre-call gates -----------------------------------------------------------

def check_calling_window(borrower_tz: str, now: datetime | None = None, *, cfg: AppConfig) -> Decision:
    """Permitted hours, in the borrower's own local time.

    Fails CLOSED on an unknown timezone. A typo in an imported CSV must not
    quietly become an unrestricted calling window.
    """
    c = cfg.compliance
    now = now or datetime.now(timezone.utc)
    try:
        local = now.astimezone(ZoneInfo(borrower_tz))
    except (ZoneInfoNotFoundError, ValueError):
        return Decision(
            False, "unknown_timezone",
            "no calling window could be established for timezone " + repr(borrower_tz),
            borrower_tz,
        )

    stamp = local.strftime("%Y-%m-%d %H:%M %Z")
    if local.weekday() not in c.allowed_weekdays:
        return Decision(False, "outside_window",
                        local.strftime("%A") + " is outside the permitted calling days", stamp)
    if local.strftime("%Y-%m-%d") in set(c.holidays):
        return Decision(False, "holiday",
                        local.strftime("%Y-%m-%d") + " is a listed holiday", stamp)

    start_h, start_m = (int(x) for x in c.call_window_start.split(":"))
    end_h, end_m = (int(x) for x in c.call_window_end.split(":"))
    minutes = local.hour * 60 + local.minute
    if not (start_h * 60 + start_m <= minutes < end_h * 60 + end_m):
        return Decision(
            False, "outside_window",
            local.strftime("%H:%M") + " borrower-local is outside the permitted window ("
            + c.call_window_start + "-" + c.call_window_end + ")",
            stamp,
        )
    return PASS


def check_contact_eligibility(customer_id: int, now: datetime | None = None, *, cfg: AppConfig) -> Decision:
    """Opt-out, open matters, hardship, and the attempt caps.

    Evaluated in order of legal significance, first failure wins, so the reason
    that gets logged is the most consequential one rather than whichever check
    happened to run first.
    """
    c = cfg.compliance
    now = now or datetime.now(timezone.utc)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT opt_out, dispute_open, grievance_open, hardship FROM customers WHERE id = ?",
            (customer_id,),
        ).fetchone()
        if row is None:
            return Decision(False, "unknown_customer", "no such customer record", str(customer_id))

        for field, code, detail in (
            ("opt_out", "opt_out", "this contact has asked not to be called"),
            ("grievance_open", "grievance_open", "an open grievance is recorded against this account"),
            ("dispute_open", "dispute_open", "this account is under dispute"),
            ("hardship", "hardship", "this contact is flagged as being in hardship"),
        ):
            if row[field]:
                return Decision(False, code, detail, field)

        # Attempts, not connections: the cap is on contacting someone, so a
        # call that rings out unanswered still spends one.
        for days, limit, code in (
            (1, c.max_attempts_per_day, "daily_cap"),
            (7, c.max_attempts_per_week, "weekly_cap"),
        ):
            since = (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM contact_events "
                "WHERE customer_id = ? AND kind = 'attempt' AND at >= ?",
                (customer_id, since),
            ).fetchone()["n"]
            if n >= limit:
                window = "today" if days == 1 else "this week"
                return Decision(False, code,
                                "already attempted " + str(n) + " time(s) " + window
                                + " (limit " + str(limit) + ")", str(n))
        return PASS
    finally:
        conn.close()


def record_contact_event(customer_id: int, kind: str, session_id: str | None = None,
                         reason: str | None = None) -> None:
    """Write the attempt/refusal that the caps above count. Best-effort: a
    bookkeeping failure must never be the thing that breaks a call."""
    try:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO contact_events (customer_id, session_id, kind, reason) VALUES (?, ?, ?, ?)",
                (customer_id, session_id, kind, reason),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.exception("failed to record contact event for customer %s", customer_id)


# ---- figure extraction --------------------------------------------------------

def _marker_follows(text: str, end: int, pattern: re.Pattern, window: int = 14) -> bool:
    """Is there a currency/percent marker just after this span?

    Direction matters. Spoken amounts put the unit last - "two lakh sixty-five
    thousand rupees", "ninety percent" - so looking only forward is both more
    accurate and much safer than a symmetric window. A symmetric one reads the
    "one" in "Rs 2,38,500 as a one-time settlement" as a second rupee figure,
    and then blocks a sentence whose only real figure was correct.
    """
    tail = text[end:min(len(text), end + window)]
    m = pattern.search(tail)
    # must be adjacent - whitespace or punctuation only between them
    return bool(m and not tail[:m.start()].strip(" \t.,:-"))


def _to_paise(rupees: str) -> int | None:
    try:
        return int(round(float(rupees.replace(",", "")) * 100))
    except ValueError:
        return None


def extract_figures(sentence: str, lexicon: dict[str, int]) -> tuple[list[Figure], list[str]]:
    """Every figure the sentence states, plus spans that look numeric but did
    not resolve. Residue is what the caller must treat as unverifiable rather
    than as "no figure was said"."""
    figures: list[Figure] = []
    consumed: list[tuple[int, int]] = []

    def taken(pos: int) -> bool:
        return any(s <= pos < e for s, e in consumed)

    for m in _DATE_ISO_RE.finditer(sentence):
        try:
            figures.append(Figure("date", date(int(m[1]), int(m[2]), int(m[3])), m[0]))
        except ValueError:
            pass
        consumed.append(m.span())
    for m in _DATE_DMY_RE.finditer(sentence):
        year = int(m[3])
        try:
            figures.append(Figure("date", date(year + 2000 if year < 100 else year, int(m[2]), int(m[1])), m[0]))
        except ValueError:
            pass
        consumed.append(m.span())

    for m in _PERCENT_DIGIT_RE.finditer(sentence):
        if not taken(m.start()):
            figures.append(Figure("percent", int(float(m[1])), m[0]))
            consumed.append(m.span())

    for rx in (_CURRENCY_BEFORE_RE, _CURRENCY_AFTER_RE):
        for m in rx.finditer(sentence):
            if taken(m.start()):
                continue
            paise = _to_paise(m[1])
            if paise is not None:
                figures.append(Figure("amount", paise, m[0]))
            consumed.append(m.span())

    # Written-out numbers, classified by whichever marker sits beside them.
    for value, span_text, start, end in find_word_numbers(sentence, lexicon):
        if taken(start):
            continue
        if _marker_follows(sentence, end, _MONEY_MARKER_RE):
            figures.append(Figure("amount", value * 100, span_text))
        elif _marker_follows(sentence, end, _PERCENT_MARKER_RE):
            figures.append(Figure("percent", value, span_text))
        else:
            figures.append(Figure("count", value, span_text))
        consumed.append((start, end))

    for m in _BARE_INT_RE.finditer(sentence):
        if not taken(m.start()):
            figures.append(Figure("count", int(m[0]), m[0]))
            consumed.append(m.span())

    # Residue: a currency or percent marker with nothing resolved beside it.
    # That is the shape of an unparsed numeral - most often a romanisation
    # missing from the lexicon - and it must not read as "no figure stated".
    residue: list[str] = []
    for marker_rx, kind in ((_MONEY_MARKER_RE, "amount"), (_PERCENT_MARKER_RE, "percent")):
        if any(f.kind == kind for f in figures):
            continue
        m = marker_rx.search(sentence)
        if m:
            residue.append(sentence[max(0, m.start() - 30):m.end() + 10].strip())
    return figures, residue


def build_allowed_facts(amount_due_paise: int, *, cfg: AppConfig, today: date | None = None,
                        extra_counts: frozenset[int] = frozenset()) -> AllowedFacts:
    """The facts a call is permitted to state, derived from the account.

    Percentages run from the settlement floor to 100 in steps of 5, matching
    how the script actually negotiates, and each one is also expanded into the
    rupee amount it represents so the agent may confirm either form.
    """
    c = cfg.compliance
    today = today or date.today()
    percents = set(range(c.settlement_floor_pct, 101, 5)) | {100}
    amounts = {amount_due_paise}
    for p in percents:
        amounts.add(int(round(amount_due_paise * p / 100)))
    return AllowedFacts(
        amounts_paise=frozenset(amounts),
        percents=frozenset(percents),
        counts=frozenset(extra_counts),
        date_lo=today,
        date_hi=today + timedelta(days=c.max_promise_days),
        tolerance_paise=c.amount_tolerance_paise,
    )


# ---- in-call screens ----------------------------------------------------------

def _hedged(sentence: str, start: int, end: int, markers: list[str], window: int) -> bool:
    chunk = sentence[max(0, start - window):min(len(sentence), end + window)].lower()
    return any(mark.lower() in chunk for mark in markers)


def screen_outbound(sentence: str, allowed: AllowedFacts, language: str, *, cfg: AppConfig) -> Decision:
    """Gate one sentence on its way to TTS. A non-ok Decision means do not speak it."""
    threats = cfg.compliance.threats or {}
    window = int(threats.get("hedge_window_chars", 60))

    for lang in {language, "en"}:
        markers = threats.get("hedge_markers_" + lang, [])
        for rx in _compiled(cfg, "threats", lang):
            m = rx.search(sentence)
            if m and not _hedged(sentence, m.start(), m.end(), markers, window):
                return Decision(False, "threat",
                                "the reply asserted a consequence the agent may not state",
                                m.group(0))

    lexicon = _lexicon(cfg)
    # Fast path: most sentences carry no number at all and must not pay for a parse.
    if not has_numeric_token(sentence, lexicon) and not _MONEY_MARKER_RE.search(sentence):
        return PASS

    figures, residue = extract_figures(sentence, lexicon)
    for f in figures:
        if f.kind == "amount":
            if not any(abs(f.value - a) <= allowed.tolerance_paise for a in allowed.amounts_paise):
                return Decision(False, "unverified_amount",
                                "the reply stated a rupee figure this call was not given", f.text)
        elif f.kind == "percent":
            if f.value not in allowed.percents:
                return Decision(False, "unverified_percent",
                                "the reply stated a percentage outside the approved range", f.text)
        elif f.kind == "date":
            if not (allowed.date_lo <= f.value <= allowed.date_hi):
                return Decision(False, "unverified_date",
                                "the reply stated a date outside the permitted range", f.text)

    if residue:
        return Decision(False, "unresolved_figure",
                        "the reply stated a figure that could not be checked", residue[0])
    return PASS


def screen_inbound(transcript: str, language: str, *, cfg: AppConfig) -> list[Flag]:
    """Flags on what the caller said, highest precedence first.

    English patterns always run alongside the call's language: callers
    code-switch constantly, and a Hindi call still gets "stop calling me".
    """
    found: dict[str, str] = {}
    for lang in {language, "en"}:
        for code in _FLAG_ORDER:
            if code in found:
                continue
            for rx in _compiled(cfg, "inbound", code + "_" + lang):
                m = rx.search(transcript)
                if m:
                    found[code] = m.group(0)
                    break
    return [Flag(code, _FLAG_SEVERITY[code], found[code]) for code in _FLAG_ORDER if code in found]
