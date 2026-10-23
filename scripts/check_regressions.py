"""Re-check the defects found while building the compliance and tool layers.

Usage: uv run python scripts/check_regressions.py [-v]

Every case here is a bug that shipped silently once: it raised nothing, logged
nothing, and produced plausible-looking output while doing the wrong thing.
That is the only reason they are worth a standing check - a loud failure would
not need one.

Runs entirely on fakes. No network, no API keys, no credentials.
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile

from pathlib import Path

from _bootstrap import load_config_or_die

from app import tts_cache

from app.providers.base import Done, TextDelta, ToolCall
from app.transport.base import AudioTransport

VERBOSE = False
_RESULTS: list[tuple[bool, str, str]] = []

#: a short run of silence, long enough to clear the minimum-utterance check
SILENCE = bytes(16000)


def check(name: str, passed: bool, detail: str = "") -> None:
    _RESULTS.append((passed, name, detail))


class _Transport(AudioTransport):
    def __init__(self):
        self.in_sample_rate = 16000
        self.out_sample_rate = 24000
        self.spoken: list[str] = []
        self.audio: list[bytes] = []

    async def recv(self):
        raise NotImplementedError("not used in this offline check")

    async def send_audio(self, turn_id, pcm):
        self.audio.append(pcm)

    async def send_control(self, data):
        if data["type"] == "agent_text":
            self.spoken.append(data["text"])

    async def close(self):
        pass


class _TTS:
    sample_rate = 24000

    # Mirrors TTSProvider: returns the whole utterance at once, so these
    # checks exercise the non-streaming branch of _speak_text. The
    # streaming branch is covered by _TTSStreaming below.
    def supports_streaming(self):
        return False

    async def synthesize(self, text):
        return b"\x00\x00" * 600


class _TTSStreaming(_TTS):
    """A streaming provider whose pieces do not line up with the 500 ms send
    frame - the case where _speak_text's re-chunking buffer has to carry a
    partial frame across iterations."""

    def supports_streaming(self):
        return True

    async def synthesize_stream(self, text):
        for _ in range(30):
            yield bytes(1700)


def _session(cfg, llm, **kw):
    from app.session import CallSession

    base = cfg.model_copy(update={"stt": cfg.stt.model_copy(update={"language": "en"})})
    return CallSession(_Transport(), base, None, llm, _TTS(), username="regress", **kw)


# ---- the compliance-layer defects --------------------------------------------

def check_toml_control_chars(cfg) -> None:
    """A regex written with TOML double quotes decodes "\\b" to a backspace.

    It compiles, matches nothing, and silently disables that screen. Four
    inbound screens shipped this way, the opt-out detector among them.
    """
    patterns = [p for v in cfg.compliance.inbound.values() for p in v]
    patterns += cfg.compliance.threats.get("en", []) + cfg.compliance.threats.get("hi", [])
    bad = [p for p in patterns if any(ch in p for ch in "\x08\x0b\x0c")]
    check("config patterns carry no control characters", not bad, f"{len(bad)} bad" if bad else "")

    from app.compliance import screen_inbound

    flags = {f.code for f in screen_inbound("Please stop calling me", "en", cfg=cfg)}
    check("opt-out screen actually matches", "opt_out" in flags, str(sorted(flags)))

    # the validator must reject it rather than accept a dead pattern
    from app.config import ComplianceConfig
    from pydantic import ValidationError

    try:
        ComplianceConfig(inbound={"opt_out_en": ["(?i)\x08(stop calling)\x08"]})
        check("config validator rejects a backspace pattern", False, "it was accepted")
    except ValidationError:
        check("config validator rejects a backspace pattern", True)


def check_devanagari_tokens(cfg) -> None:
    """Python's \\w excludes combining marks, so a naive tokenizer shreds every
    Devanagari word and the whole Hindi numeral path fails silently."""
    from app.text.numerals import tokenize

    word = "हज़ार"  # हज़ार, with a nukta and a matra
    tokens = [t for t, _, _ in tokenize(word)]
    check("Devanagari word survives tokenization", tokens == [word], str(tokens))

    from app.compliance import _lexicon
    from app.text.numerals import find_word_numbers

    sentence = "आपका बकाया दो लाख पैंसठ हज़ार रुपये है"
    values = [v for v, _, _, _ in find_word_numbers(sentence, _lexicon(cfg))]
    check("Hindi amount parses to its value", 265000 in values, str(values))


def check_money_marker_substring(cfg) -> None:
    """"rs" matched inside "understand", tripping the money marker and blocking
    ordinary sentences that mention no figure at all."""
    from datetime import date

    from app.compliance import build_allowed_facts, screen_outbound

    facts = build_allowed_facts(26500000, cfg=cfg, today=date.today())
    for sentence in ("Thanks, I understand completely.",
                     "Let me reverse that for you.",
                     "Of course, no worries."):
        decision = screen_outbound(sentence, facts, "en", cfg=cfg)
        check(f"innocent sentence passes: {sentence[:28]!r}", decision.ok, decision.code)


def check_adjacent_number_misread(cfg) -> None:
    """A symmetric proximity window read the "one" in "one-time settlement" as
    a rupee amount, blocking a sentence whose only real figure was correct."""
    from datetime import date

    from app.compliance import build_allowed_facts, screen_outbound

    facts = build_allowed_facts(26500000, cfg=cfg, today=date.today())
    ok = screen_outbound("I can offer you Rs 2,38,500 as a one-time settlement.", facts, "en", cfg=cfg)
    check("valid settlement figure is not blocked", ok.ok, f"{ok.code} {ok.evidence!r}")
    bad = screen_outbound("You must pay three lakh rupees.", facts, "en", cfg=cfg)
    check("invented figure is still blocked", not bad.ok, bad.code)


# ---- the tool-loop defects ----------------------------------------------------

def check_tool_call_after_question(cfg) -> None:
    """`break` on the first question closed the generator, discarding any tool
    call the model emitted afterwards. gpt-oss puts text first, so real calls
    were lost with no error anywhere."""
    from app.pipeline import run_agent_turn

    executed: list[str] = []

    class Llm:
        last_usage = None
        request_count = 0

        async def stream_chat(self, messages, *, max_tokens, temperature, tools=None):
            self.request_count += 1
            if tools:
                yield TextDelta("Can you confirm the amount? ")  # trips stop_after_question
                yield ToolCall(id="c1", name="record_promise_to_pay", arguments="{}")
                yield Done(finish_reason="tool_calls")
            else:
                yield TextDelta("Recorded.")
                yield Done()

    class Mcp:
        available = True

        def schemas(self):
            return [{"type": "function", "function": {
                "name": "record_promise_to_pay", "description": "x",
                "parameters": {"type": "object", "properties": {}}}}]

        async def call(self, name, args, timeout=2.5):
            executed.append(name)
            return {"ok": True}

        async def aclose(self):
            pass

    async def run():
        s = _session(cfg, Llm())
        s.mcp = Mcp()
        await run_agent_turn(s, 1, b"\x00\x00" * 8000, transcript="about the payment")
        s.call_logger.finalize()
        shutil.rmtree(s.call_logger.dir, ignore_errors=True)

    asyncio.run(run())
    check("tool call after a question is not swallowed", executed == ["record_promise_to_pay"], str(executed))


def check_history_window(cfg) -> None:
    """The cut landed one message late, so the context window opened on an
    assistant reply whose question was no longer present."""
    s = _session(cfg, None)
    for i in range(8):
        s.history.append({"role": "user", "content": f"u{i}"})
        s.history.append({"role": "assistant", "content": f"a{i}"})
    s.trim_history()
    opens_on_user = s.history[1]["role"] == "user"
    exchanges = sum(1 for m in s.history if m["role"] == "user")
    shutil.rmtree(s.call_logger.dir, ignore_errors=True)
    check("history window opens on a caller turn", opens_on_user, s.history[1]["role"])
    check("history keeps the configured exchange count",
          exchanges == cfg.llm.history_max_turns, f"kept {exchanges}")


def check_round_two_shape(cfg) -> None:
    """Measured live: with tools=None after a tool has been used, gpt-oss emits
    one anyway and Groq rejects the whole request - every retry failed. Round
    two must therefore DECLARE the tools and discard what it asks for."""
    from app.pipeline import run_agent_turn

    rounds: list[bool] = []   # whether each round was given tools
    executed: list[str] = []

    class Llm:
        last_usage = None
        request_count = 0

        async def stream_chat(self, messages, *, max_tokens, temperature, tools=None):
            self.request_count += 1
            rounds.append(bool(tools))
            if len(rounds) == 1:
                yield TextDelta("Recording that. ")
                yield ToolCall(id="c1", name="record_promise_to_pay", arguments="{}")
                yield Done(finish_reason="tool_calls")
            else:
                # round two answers with another tool call and no words at all
                yield ToolCall(id="c2", name="record_promise_to_pay", arguments="{}")
                yield Done(finish_reason="tool_calls")

    class Mcp:
        available = True

        def schemas(self):
            return [{"type": "function", "function": {
                "name": "record_promise_to_pay", "description": "x",
                "parameters": {"type": "object", "properties": {}}}}]

        async def call(self, name, args, timeout=2.5):
            executed.append(name)
            return {"ok": True}

        async def aclose(self):
            pass

    async def run():
        s = _session(cfg, Llm())
        s.mcp = Mcp()
        await run_agent_turn(s, 1, b"\x00\x00" * 8000, transcript="I will pay")
        s.call_logger.finalize()
        shutil.rmtree(s.call_logger.dir, ignore_errors=True)
        return s

    s = asyncio.run(run())
    check("two rounds were made", len(rounds) == 2, str(rounds))
    check("round two still declares tools", rounds[-1] is True, str(rounds))
    check("round two's tool call is discarded, not executed",
          executed == ["record_promise_to_pay"], str(executed))
    check("a silent round two still speaks a confirmation",
          any("recorded" in line.lower() for line in s.transport.spoken), str(s.transport.spoken))
    check("what was spoken is recorded in history",
          any("recorded" in m["content"].lower()
              for m in s.history if m["role"] == "assistant"),
          str([m["content"] for m in s.history if m["role"] == "assistant"]))


def check_round_two_speaks(cfg) -> None:
    """`producer` assigned queued_text without declaring it nonlocal, so it
    bound a local that the round function never wrote to. The "did round two
    say anything?" test was then permanently False and the fixed confirmation
    was spoken on top of a perfectly good reply."""
    from app.pipeline import run_agent_turn

    class Llm:
        last_usage = None

        def __init__(self):
            self.request_count = 0
            self.n = 0

        async def stream_chat(self, messages, *, max_tokens, temperature, tools=None):
            self.request_count += 1
            self.n += 1
            if self.n == 1:
                yield TextDelta("Let me record that. ")
                yield ToolCall(id="c1", name="record_promise_to_pay", arguments="{}")
                yield Done(finish_reason="tool_calls")
            else:
                yield TextDelta("Your payment is noted for Friday.")
                yield Done()

    class Mcp:
        available = True

        def schemas(self):
            return [{"type": "function", "function": {
                "name": "record_promise_to_pay", "description": "x",
                "parameters": {"type": "object", "properties": {}}}}]

        async def call(self, name, args, timeout=2.5):
            await asyncio.sleep(0.5)  # slow enough that the filler is used
            return {"ok": True}

        async def aclose(self):
            pass

    async def run():
        s = _session(cfg, Llm())
        s.mcp = Mcp()
        await run_agent_turn(s, 1, SILENCE, transcript="I will pay Friday")
        s.call_logger.finalize()
        shutil.rmtree(s.call_logger.dir, ignore_errors=True)
        return s

    s = asyncio.run(run())
    spoken = " ".join(s.transport.spoken)
    history = " ".join(m["content"] for m in s.history if m["role"] == "assistant")
    check("a speaking round two gets no extra confirmation",
          "That is recorded" not in spoken, spoken)
    check("round two's own words reach the caller",
          "noted for Friday" in spoken, spoken)
    check("interstitial filler stays out of history",
          "One moment" not in history, history)


def check_mcp_error_flag(cfg) -> None:
    """MCP reports a failure as ordinary text with an error flag set. Ignoring
    the flag turned a failed tool into {"ok": true}, which reads to the model
    as success and has the agent tell the caller the action went through."""
    from app.mcp_client import _unwrap

    class Block:
        def __init__(self, text):
            self.text = text

    class Result:
        def __init__(self, is_error, text):
            self.is_error = is_error
            self.content = [Block(text)]
            self.structured_content = None

    failed = _unwrap(Result(True, "Unknown tool: nonexistent_tool"))
    check("a failed tool does not report ok", failed.get("ok") is False, str(failed))
    fine = _unwrap(Result(False, json.dumps({"ok": True, "date": "2026-11-05"})))
    check("a successful tool still parses", fine.get("ok") is True, str(fine))


def check_streaming_send(cfg) -> None:
    """A streaming provider must deliver every byte, re-framed, not dropped.

    _speak_text buffers partial frames across iterations, so a piece that does
    not divide evenly into the 500 ms send frame leaves a remainder. Dropping
    that remainder clips the end of every sentence by a few milliseconds -
    inaudible on one line, and a progressively worse stutter as a call goes on.
    Reported duration must match the bytes actually sent, since the server
    sleeps on it before listening again.
    """
    import asyncio

    from app import pipeline

    session = _session(cfg, None)
    session.tts = _TTSStreaming()
    timing: dict = {}
    duration = asyncio.run(pipeline._speak_text(session, 1, "hello there", timing))
    sent = b"".join(session.transport.audio)
    expected = 30 * 1700
    check("streaming send loses no bytes", len(sent) == expected,
          f"{len(sent)} of {expected}")
    check("streaming duration matches the bytes sent",
          abs(duration - len(sent) / 2 / session.tts.sample_rate) < 1e-9)
    check("streaming records first-audio latency", "tts_first_ms" in timing)
    frame = int(session.tts.sample_rate * pipeline._SEND_CHUNK_MS / 1000) * 2
    check("streaming frames respect the send chunk size",
          all(len(c) <= frame for c in session.transport.audio))
    check("streaming exercises both a full frame and a remainder",
          any(len(c) == frame for c in session.transport.audio)
          and any(len(c) < frame for c in session.transport.audio))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="show detail for passing checks too")
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    cfg = load_config_or_die()

    # These checks drive real CallSessions, so without redirection they write
    # into live state: scripted lines go through the real tts_cache, and every
    # session opens a real directory under logs/calls. The fake TTS returns
    # silence and the fake config never resolves a voice, so the cache entries
    # are silent audio filed under whatever voice name happened to be the
    # default - a file a real call could later serve instead of the line - and
    # the log directories are indistinguishable from real calls in the admin
    # Call Logs page. Both go to a throwaway directory for the duration.
    with tempfile.TemporaryDirectory(prefix="vca-regress-") as tmp:
        tts_cache.CACHE_DIR = Path(tmp) / "cache"
        cfg = cfg.model_copy(
            update={"call": cfg.call.model_copy(update={"log_dir": str(Path(tmp) / "calls")})}
        )
        _run_checks(cfg)


def _run_checks(cfg) -> None:
    for fn in (
        check_toml_control_chars,
        check_devanagari_tokens,
        check_money_marker_substring,
        check_adjacent_number_misread,
        check_tool_call_after_question,
        check_history_window,
        check_round_two_shape,
        check_round_two_speaks,
        check_mcp_error_flag,
        check_streaming_send,
    ):
        try:
            fn(cfg)
        except Exception as e:
            check(f"{fn.__name__} ran", False, f"{type(e).__name__}: {e}")

    failed = 0
    for passed, name, detail in _RESULTS:
        if passed and not VERBOSE:
            print(f"  ok   {name}")
        elif passed:
            print(f"  ok   {name}  {detail}")
        else:
            failed += 1
            print(f"  FAIL {name}  {detail}")

    print(f"\n{len(_RESULTS) - failed}/{len(_RESULTS)} checks passed")
    if failed:
        sys.exit(f"error: {failed} regression check(s) failed")


if __name__ == "__main__":
    main()
