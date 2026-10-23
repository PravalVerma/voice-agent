"""The scripted stages of a collections call.

Four stages, and only one of them talks to a model:

  GREETING   fixed. States that this is an automated assistant, who it calls
             on behalf of, and that the call is recorded. Never skipped and
             never model-generated - the disclosure is a regulatory
             requirement, not an opening line.
  VERIFY     fixed. Asks one identifying question. The correct answer is never
             spoken and never enters the model's context; only a boolean comes
             back. After the configured number of strikes the call offers a
             callback rather than hanging up.
  NEGOTIATE  the only stage that calls the LLM.
  CLOSE      fixed. Confirms the outcome and ends.

Several caller intents are handled here, in code, before the model is ever
consulted: an opt-out registers and ends the call, distress gets a fixed
empathetic script, and a third party gets told nothing whatsoever about the
debt. These are the cases where consulting a model is both unnecessary and the
thing most likely to go wrong.
"""

import enum
import logging

from . import compliance, customers_store
from .call_state import State
from .pipeline import run_agent_turn, speak_script
from .text.numerals import EN_LEXICON, digit_sequence, spell_to_digits, spoken_values

log = logging.getLogger("agent")


class Stage(str, enum.Enum):
    """str-mixin so a stage serialises straight into transcript.jsonl."""

    GREETING = "greeting"
    VERIFY = "verify"
    NEGOTIATE = "negotiate"
    CLOSE = "close"


def _normalize_answer(text: str) -> str:
    """Reduce a spoken answer to comparable characters.

    Identity answers are dates, digit runs and PINs, and the caller may say
    them as words or digits while STT transcribes whichever they said. Word
    numbers are mapped to digits, then everything that is not alphanumeric is
    dropped, so "fifteen oh three" and "15/03" converge.
    """
    return "".join(ch for ch in spell_to_digits(text.lower(), EN_LEXICON) if ch.isalnum())


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def verify_answer(stored_kind: str, stored_value: str, spoken: str) -> bool:
    """Compare a spoken answer with the stored one. Never reveals the stored value.

    Deliberately lenient about form and strict about content. People say a date
    of birth as "the fifteenth of March, nineteen eighty-eight" at least as
    often as "15-03-1988", and Indic STT transcribes whichever they said; a
    comparison that only accepted the stored spelling would fail real customers
    constantly and burn their three attempts on nothing. So the day, month and
    year are matched independently, in any order and in either form. What it
    never does is accept a near miss on the values themselves.
    """
    want = _normalize_answer(stored_value)
    got = _normalize_answer(spoken)
    if not want or not got:
        return False

    # Digits may be read out one by one, which the ordinary number parse sums
    # instead of concatenating - check that reading too.
    digits = digit_sequence(spoken, EN_LEXICON)

    if stored_kind == "phone_last4":
        # last four accepted anywhere in the utterance: callers volunteer the
        # whole number, or a prefix, or "ending in ...".
        last4 = want[-4:]
        return last4 in got or last4 in digits

    if stored_kind == "pin":
        return want in got or want in digits

    if stored_kind == "dob":
        if want in got:
            return True
        parts = [p for p in stored_value.replace("/", "-").split("-") if p]
        if len(parts) < 3:
            return False
        year, month, day = (int(parts[0]), int(parts[1]), int(parts[2]))
        values = spoken_values(spoken, EN_LEXICON)
        words = (w.strip(",.;:!?'\"") for w in spoken.lower().split())
        said_months = {_MONTHS[w] for w in words if w in _MONTHS}
        return year in values and day in values and (month in values or month in said_months)

    return want in got


class CollectionsFlow:
    """Per-call state. One instance lives on the CallSession for the call."""

    def __init__(self, cfg, customer: dict):
        self.cfg = cfg
        self.customer = customer
        self.stage = Stage.GREETING
        self.strikes = 0
        self.verified = False
        self.disposition = "incomplete"
        self.ending = False

    # ---- helpers --------------------------------------------------------------

    @property
    def language(self) -> str:
        return self.cfg.stt.language

    def _line(self, key: str, **fields) -> str:
        return self.cfg.collections.script(self.language, key, **fields)

    def _to(self, session, stage: Stage, reason: str) -> None:
        if stage is self.stage:
            return
        session.call_logger.log_event({
            "type": "stage_transition", "from": self.stage.value,
            "to": stage.value, "reason": reason,
        })
        log.info("call %s: stage %s -> %s (%s)", session.session_id, self.stage.value, stage.value, reason)
        self.stage = stage

    async def _say(self, session, key: str, stage: str, **fields) -> None:
        session.turn_id += 1
        await speak_script(session, session.turn_id, self._line(key, **fields), stage)

    async def _end(self, session, disposition: str) -> None:
        self.disposition = disposition
        self.ending = True
        self._to(session, Stage.CLOSE, disposition)
        await session.transport.send_control({"type": "call_ending", "reason": disposition})

    # ---- stages ---------------------------------------------------------------

    async def start(self, session) -> None:
        """Open the call. Runs before the caller has said anything, because the
        disclosure has to come first - the standard pipeline otherwise waits
        for the caller to speak and introduces itself in its first reply."""
        session.allowed_facts = compliance.build_allowed_facts(
            self.customer["amount_due_paise"], cfg=self.cfg
        )
        await self._say(session, "greeting", Stage.GREETING.value,
                        customer=self.customer["name"], lender=self.customer["lender_name"])
        self._to(session, Stage.VERIFY, "disclosure given")
        await self._ask_verify(session)

    async def _ask_verify(self, session) -> None:
        kind = self.customer.get("verify_kind", "dob")
        await self._say(session, f"verify_{kind}", Stage.VERIFY.value)

    async def handle_turn(self, session, turn_id: int, utterance_pcm: bytes) -> None:
        """One caller turn. Replaces pipeline.run_agent_turn for collections bots."""
        if self.ending:
            return

        await session.set_state(State.PROCESSING)
        from .audio.frames import PIPELINE_SAMPLE_RATE, pcm_duration_s
        duration_s = pcm_duration_s(utterance_pcm, PIPELINE_SAMPLE_RATE)
        text = (await session.stt.transcribe(utterance_pcm, PIPELINE_SAMPLE_RATE)).strip()
        session.call_logger.record_stt(duration_s)
        if not text:
            await session.set_state(State.LISTENING)
            return
        session.call_logger.log_event({"type": "user_utterance", "turn_id": turn_id, "text": text})
        await session.transport.send_control({"type": "transcript", "role": "user", "text": text})

        flags = compliance.screen_inbound(text, self.language, cfg=self.cfg)
        for f in flags:
            session.call_logger.log_event({
                "type": "compliance_flag", "turn_id": turn_id, "code": f.code,
                "severity": f.severity, "evidence": f.evidence, "stage": self.stage.value,
            })
        if flags and await self._handle_flag(session, flags[0], text):
            return

        if self.stage is Stage.VERIFY:
            await self._handle_verify(session, text)
            return

        # NEGOTIATE: the only path that reaches a model. The utterance is
        # replayed into the standard turn, which already carries the outbound
        # screen via session.allowed_facts.
        await run_agent_turn(session, turn_id, utterance_pcm, transcript=text)

    async def _handle_flag(self, session, flag, text: str) -> bool:
        """Code-driven responses that bypass the model entirely.

        Returns True if the flag was handled and the turn is over. These are
        exactly the situations where asking a model what to do is both
        unnecessary and the most dangerous thing available.
        """
        if flag.code in ("self_harm", "distress"):
            await self._say(session, "distress", self.stage.value)
            session.call_logger.log_event({
                "type": "compliance_flag", "code": "welfare_concern", "severity": "high",
                "evidence": flag.evidence, "stage": self.stage.value,
                "detail": "fixed welfare script given; no handoff is available on this line",
            })
            await self._end(session, "welfare_concern")
            return True

        if flag.code == "opt_out":
            customers_store.set_flag(self.customer["id"], "opt_out", True)
            compliance.record_contact_event(
                self.customer["id"], "refused", session.session_id, "opt_out"
            )
            await self._say(session, "optout_ack", self.stage.value)
            await self._end(session, "opt_out")
            return True

        if flag.code in ("third_party", "wrong_number") and not self.verified:
            # Say nothing about the debt - not the amount, not that it is
            # overdue, not that this concerns an account at all.
            await self._say(session, "third_party", self.stage.value)
            await self._end(session, flag.code)
            return True

        if flag.code == "dispute":
            customers_store.set_flag(self.customer["id"], "dispute_open", True)
            # Not terminal: the model is told to stop pursuing payment, which
            # it does through the normal NEGOTIATE path.
            return False

        return False

    async def _handle_verify(self, session, text: str) -> None:
        secret = customers_store.get_verify_value(self.customer["id"])
        ok = bool(secret) and verify_answer(secret[0], secret[1], text)
        session.call_logger.log_event({
            "type": "tool_call", "name": "verify_identity", "stage": Stage.VERIFY.value,
            "args": {"answer": "***"},  # never logged: it is the identity secret
        })
        session.call_logger.log_event({
            "type": "tool_result", "name": "verify_identity",
            "ok": True, "result": {"verified": ok,
                                   "strikes_left": self.cfg.collections.verify_strikes - self.strikes - (0 if ok else 1)},
        })
        if ok:
            self.verified = True
            self._to(session, Stage.NEGOTIATE, "identity confirmed")
            # The account facts enter the model's context here and nowhere
            # else - a tool round-trip for data already in hand would cost a
            # request and a second of dead air.
            session.history.append({"role": "user", "content": self._account_line()})
            await session.set_state(State.LISTENING)
            return

        self.strikes += 1
        if self.strikes >= self.cfg.collections.verify_strikes:
            await self._say(session, "verify_failed", Stage.VERIFY.value)
            await self._end(session, "verification_failed")
            return
        await self._say(session, "verify_retry", Stage.VERIFY.value)

    def _account_line(self) -> str:
        """The compact account block, injected once after verification."""
        c = self.customer
        return (
            f"[ACCOUNT] borrower={c['name']} lender={c['lender_name']} "
            f"outstanding_rupees={c['amount_due_paise'] / 100:.2f} due_date={c['due_date']}"
        )
