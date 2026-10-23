"""One agent turn: STT -> streaming LLM -> sentence chunking -> TTS -> transport.

The LLM producer and TTS consumer run concurrently over an asyncio.Queue so the
first sentence starts playing while the LLM is still generating later ones.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass

from . import compliance, mcp_client, tts_cache
from .audio.frames import PIPELINE_SAMPLE_RATE, pcm_duration_s
from .call_state import State
from .providers.base import Done, ToolCall
from .text.sentences import SentenceAssembler

log = logging.getLogger("agent")

_SEND_CHUNK_MS = 500  # slice agent audio into <=500ms frames so playback can start early
_MIN_UTTERANCE_S = 0.3  # utterances shorter than this are almost certainly noise/clicks
_SENTINEL = object()

# A tool call that takes longer than this has already lost the caller's
# patience: the filler line is roughly 1.5s of audio, so a longer budget buys
# dead air after it rather than a result. Fail to a fixed line instead.
_TOOL_TIMEOUT_S = 2.5
# Don't speak "one moment" for a tool that returns in under this - a cached or
# local lookup finishes faster than the filler would take to say.
_FILLER_GRACE_S = 0.35


@dataclass(frozen=True)
class _Filler:
    """A fixed line queued mid-turn, spoken from cache.

    Distinct from a plain sentence so the consumer knows not to treat it as
    something the model said. `terminal` separates the two kinds:

    - False ("one moment, let me check") covers a gap and is forgettable. It
      must NOT reach history, or the model learns it wrote it.
    - True (a confirmation, or a failure notice) is the last thing the caller
      hears this turn, so it MUST reach history. Leaving it out makes the model
      believe it never confirmed, and it confirms again on the next turn.
    """

    text: str
    terminal: bool = False


def _chunk_pcm(pcm: bytes, sample_rate: int, chunk_ms: int = _SEND_CHUNK_MS) -> list[bytes]:
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


def _tools_for(session) -> list[dict]:
    """Tool schemas for this turn, or [] when the call has no tool server.

    The list never varies by bot. A per-bot subset would be a different prompt
    per bot for no behavioural gain, since a disallowed tool is already
    rejected at execution - see McpClient.call.
    """
    client = getattr(session, "mcp", None)
    if client is None or not client.available:
        return []
    return client.schemas()


def _tool_exchange(calls: list, results: list[dict]) -> list[dict]:
    """The assistant/tool message pair for round two.

    Built transiently and thrown away: these never enter session.history. That
    keeps trim_history's tail slicing safe by construction rather than by a
    careful rule the next edit forgets, and stops a stale tool result from
    being replayed to the model four turns after it stopped being true.
    """
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]},
        *[
            {"role": "tool", "tool_call_id": c.id, "name": c.name,
             "content": mcp_client.truncate_result(r)}
            for c, r in zip(calls, results)
        ],
        # Round two has to produce speech, not another tool call. The model can
        # still emit one and it will be ignored, but asking plainly makes that
        # much rarer - and a round that returns only a tool call leaves the
        # caller having heard nothing but the filler.
        {"role": "user", "content": "Tell the caller the outcome in one short sentence. Do not call a tool."},
    ]


def _result_key(session, call) -> str:
    """Idempotency key, derived in code and never from the model.

    A caller who barges in during a tool call usually did so *because* they
    want the thing, so the model is likely to request it again next turn. The
    first call may already have completed unheard. Replaying the stored result
    stops a second write.

    Different arguments are deliberately a different key: a promise of Rs 5,000
    followed by one of Rs 5,500 is two genuine calls, not a duplicate.
    """
    try:
        args = json.dumps(json.loads(call.arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        args = call.arguments or ""
    raw = f"{session.session_id}|{call.name}|{args}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def _dispatch(session, turn_id: int, calls: list) -> list[dict]:
    """Run every requested tool concurrently and return their results."""
    async def one(call):
        key = _result_key(session, call)
        cached = session.tool_results.get(key)
        if cached is not None:
            log.info("turn %d: replaying %s (already run this call)", turn_id, call.name)
            return cached
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "arguments were not understood"}
        session.call_logger.log_event({
            "type": "tool_call", "turn_id": turn_id, "call_id": call.id,
            "name": call.name, "args": args,
        })
        result = await session.mcp.call(call.name, args, timeout=_TOOL_TIMEOUT_S)
        session.call_logger.log_event({
            "type": "tool_result", "turn_id": turn_id, "call_id": call.id,
            "name": call.name, "ok": bool(result.get("ok", True)), "result": result,
        })
        session.tool_results[key] = result
        if session.flow is not None:
            session.flow.note_tool_result(call.name, result)
        return result

    return list(await asyncio.gather(*(one(c) for c in calls)))


async def _run_tool_phase(session, turn_id: int, queue, calls: list, timing: dict) -> list[dict]:
    """Execute tools, covering the wait with a fixed filler line.

    The task is detached and shielded so that a barge-in cancels *this turn*
    without cancelling a tool that is already writing to the database. The
    caller does not hear the result - the only consumer of it is this
    coroutine, which is gone - but the write completes and the flow still
    learns of it, because a commitment that was recorded is recorded whether or
    not anyone heard the confirmation.
    """
    timing["tool_names"] = [c.name for c in calls]
    task = session.spawn_detached(_dispatch(session, turn_id, calls))

    # Give a fast tool a moment to finish before filling silence that is not
    # there. With nothing spoken yet there is nothing covering the gap, so the
    # filler goes out immediately instead.
    grace = _FILLER_GRACE_S if session.spoken_anything else 0.0
    done, _ = await asyncio.wait({task}, timeout=grace)
    if not done:
        await queue.put(_Filler(_script_line(session, "tool_filler")))

    try:
        results = await asyncio.wait_for(asyncio.shield(task), timeout=_TOOL_TIMEOUT_S)
    except (asyncio.TimeoutError, Exception) as e:
        if isinstance(e, asyncio.CancelledError):
            raise
        log.warning("turn %d: tool phase did not complete in time (%s)", turn_id, type(e).__name__)
        await queue.put(_Filler(_script_line(session, "tool_failed"), terminal=True))
        # No round two. Spending a request so the model can compose an apology
        # we already have written is the most expensive way to say nothing.
        return []

    if all(r.get("ok") is False for r in results):
        await queue.put(_Filler(_script_line(session, "tool_failed"), terminal=True))
        return []
    return results


def _script_line(session, key: str) -> str:
    return session.cfg.collections.script(session.cfg.stt.language, key)


def _screen(session, sentence: str):
    """Compliance check for one sentence, or None if it may be spoken.

    Inert unless the session carries allowed_facts, which only a collections
    call sets. An ordinary assistant bot has no account behind it, so there is
    nothing to check a figure against and this costs one attribute lookup.

    Never raises: a screening failure must not take the call down. It fails
    OPEN, deliberately. The screen is the second line of defence behind a
    prompt that already says not to invent figures, and a crash here that
    silently muted the agent would be a worse outcome than one unchecked
    sentence - but it is logged loudly so it cannot pass unnoticed.
    """
    allowed = getattr(session, "allowed_facts", None)
    if allowed is None:
        return None
    try:
        decision = compliance.screen_outbound(
            sentence, allowed, session.cfg.stt.language, cfg=session.cfg
        )
    except Exception:
        log.exception("outbound screen failed - speaking the sentence unchecked")
        return None
    return None if decision.ok else decision


def _recovery_line(session) -> str:
    """Spoken in place of a blocked sentence. Fixed text, never model-generated,
    and safe to say at any point in any call."""
    lines = session.cfg.compliance.recovery_line or {}
    return lines.get(session.cfg.stt.language) or lines.get("en") or "Let me confirm that and come back to you."


def _mark_first_audio(timing: dict) -> None:
    """Record when the caller's first audio byte went out, relative to the turn.

    Logged next to llm_done_ms so the two can be compared directly: audio
    leaving before the model has finished generating is the whole point of
    chunking replies into sentences, and it is the kind of property that
    regresses silently when the TTS path changes.
    """
    t0 = timing.get("_turn_t0")
    if t0 is not None:
        timing.setdefault("first_audio_sent_ms", (time.perf_counter() - t0) * 1000)


async def _speak_text(session, turn_id: int, text: str, timing: dict, *, cache: bool = False) -> float:
    """Synthesize + send one sentence; return its audio duration in seconds.

    `cache` is for fixed script lines only - a disclosure, a verification
    prompt, a recovery line. Model-generated replies are unique, so caching
    them would grow the directory without ever hitting.
    """
    t0 = time.perf_counter()
    hit = False
    rate = session.tts.sample_rate
    if cache:
        pcm, hit = await tts_cache.synthesize_cached(
            session.tts, text,
            provider=session.cfg.tts.provider,
            voice=session.cfg.tts.voice,
            speaking_rate=session.cfg.tts.speaking_rate,
            sample_rate=rate,
        )
        timing.setdefault("tts_first_ms", (time.perf_counter() - t0) * 1000)
        _mark_first_audio(timing)
        for chunk in _chunk_pcm(pcm, rate):
            await session.transport.send_audio(turn_id, chunk)
        total_bytes = len(pcm)
    elif session.tts.supports_streaming():
        # Send each slice as it decodes rather than waiting for the sentence to
        # finish. On a provider whose first byte costs more than the rest of the
        # utterance put together, this is most of the perceived latency.
        total_bytes = 0
        pending = bytearray()
        frame_bytes = int(rate * _SEND_CHUNK_MS / 1000) * 2
        async for piece in session.tts.synthesize_stream(text):
            timing.setdefault("tts_first_ms", (time.perf_counter() - t0) * 1000)
            _mark_first_audio(timing)
            total_bytes += len(piece)
            pending += piece
            while len(pending) >= frame_bytes:
                await session.transport.send_audio(turn_id, bytes(pending[:frame_bytes]))
                del pending[:frame_bytes]
        if pending:
            await session.transport.send_audio(turn_id, bytes(pending))
    else:
        pcm = await session.tts.synthesize(text)
        timing.setdefault("tts_first_ms", (time.perf_counter() - t0) * 1000)
        _mark_first_audio(timing)
        for chunk in _chunk_pcm(pcm, rate):
            await session.transport.send_audio(turn_id, chunk)
        total_bytes = len(pcm)
    await session.transport.send_control({"type": "agent_text", "turn_id": turn_id, "text": text})
    duration_s = total_bytes / 2 / rate
    # A cache hit made no request, so it bills no characters. Counting them
    # anyway would quietly inflate every call's cost estimate by the length of
    # the fixed script, which is the one part that is genuinely free.
    session.call_logger.record_tts(0 if hit else len(text), duration_s)
    return duration_s


async def _drain_and_return_to_listening(session, speaking_started: float, total_audio_s: float) -> None:
    # The server only knows when it *sent* the last byte, not when the client
    # finished playing it - hold SPEAKING until the estimated playback drains.
    elapsed = time.monotonic() - speaking_started
    remaining = total_audio_s - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)
    await session.set_state(State.LISTENING)


async def speak_greeting(session, turn_id: int, text: str) -> None:
    session.call_logger.turn_count += 1
    await session.set_state(State.SPEAKING)
    await session.transport.send_control({"type": "turn_start", "turn_id": turn_id})
    speaking_started = time.monotonic()
    dur = await _speak_text(session, turn_id, text, {})
    session.history.append({"role": "assistant", "content": text})
    session.call_logger.log_event({"type": "greeting", "turn_id": turn_id, "text": text})
    await session.transport.send_control({"type": "turn_end", "turn_id": turn_id})
    await _drain_and_return_to_listening(session, speaking_started, dur)


async def speak_notice(session, turn_id: int, text: str) -> None:
    """Speak a short code-generated line (e.g. an apology after a provider
    failure). Deliberately not appended to session.history - it isn't part of
    the script the agent is following, and feeding it back to the LLM would just
    muddy the next turn. turn_start is (re)sent because the client drops audio
    frames whose turn_id it isn't currently expecting."""
    await session.set_state(State.SPEAKING)
    await session.transport.send_control({"type": "turn_start", "turn_id": turn_id})
    speaking_started = time.monotonic()
    dur = await _speak_text(session, turn_id, text, {})
    session.call_logger.log_event({"type": "agent_notice", "turn_id": turn_id, "text": text})
    await session.transport.send_control({"type": "turn_end", "turn_id": turn_id})
    await _drain_and_return_to_listening(session, speaking_started, dur)


async def speak_script(session, turn_id: int, text: str, stage: str) -> None:
    """Speak a fixed, pre-approved script line from the audio cache.

    Logged as a normal agent_turn so the existing transcript parser and the
    admin UI render it with no changes, carrying `scripted` and `stage` as
    extra fields for the renderer that wants them.

    turn_start is (re)sent for the same reason speak_notice does it: the client
    drops audio frames whose turn_id it is not currently expecting.

    Script lines deliberately bypass the outbound screen. They are fixed text
    reviewed in advance - screening them would only ever produce a false
    positive against an account's own figures.
    """
    session.call_logger.turn_count += 1
    await session.set_state(State.SPEAKING)
    await session.transport.send_control({"type": "turn_start", "turn_id": turn_id})
    started = time.monotonic()
    dur = await _speak_text(session, turn_id, text, {}, cache=True)
    session.history.append({"role": "assistant", "content": text})
    session.call_logger.log_event({
        "type": "agent_turn", "turn_id": turn_id, "text": text,
        "interrupted": False, "scripted": True, "stage": stage,
    })
    await session.transport.send_control({"type": "turn_end", "turn_id": turn_id})
    await _drain_and_return_to_listening(session, started, dur)


async def run_agent_turn(session, turn_id: int, utterance_pcm: bytes,
                         transcript: str | None = None) -> None:
    """One agent turn.

    `transcript` lets a caller that has already transcribed this utterance pass
    the text in rather than have it recognised a second time - the collections
    flow screens the caller's words before deciding whether a model should see
    them at all, so by the time it delegates here the STT request is spent.
    Passing it also means the caller owns the user_utterance log line.
    """
    session.call_logger.turn_count += 1
    timing: dict = {}
    t_turn0 = time.perf_counter()
    timing["_turn_t0"] = t_turn0
    await session.set_state(State.PROCESSING)

    duration_s = pcm_duration_s(utterance_pcm, PIPELINE_SAMPLE_RATE)
    if duration_s < _MIN_UTTERANCE_S:
        log.info("turn %d: utterance too short (%.2fs) - ignoring", turn_id, duration_s)
        await session.set_state(State.LISTENING)
        return

    if transcript is None:
        t0 = time.perf_counter()
        text = (await session.stt.transcribe(utterance_pcm, PIPELINE_SAMPLE_RATE)).strip()
        timing["stt_ms"] = (time.perf_counter() - t0) * 1000
        session.call_logger.record_stt(duration_s)
        if not text:
            log.info("turn %d: empty transcript - ignoring", turn_id)
            await session.set_state(State.LISTENING)
            return
        session.call_logger.log_event({"type": "user_utterance", "turn_id": turn_id, "text": text, "duration_s": round(duration_s, 2)})
        await session.transport.send_control({"type": "transcript", "role": "user", "text": text})
    else:
        text = transcript

    session.history.append({"role": "user", "content": text})
    session.trim_history()
    await session.transport.send_control({"type": "turn_start", "turn_id": turn_id})

    queue: asyncio.Queue = asyncio.Queue()
    assembler = SentenceAssembler()
    reply_parts: list[str] = []
    speaking_started: float | None = None
    total_audio_s = 0.0
    # Set once a sentence fails the outbound screen; the remainder of the turn
    # is then drained unspoken (see consumer()).
    blocked = False
    # Whether any round has queued speech. A tool turn whose second round
    # returns only another tool call would otherwise leave the caller with the
    # filler line and then silence.
    queued_text = False
    # The last fixed line actually spoken this turn. A turn can end having said
    # only fixed lines - a tool confirmation, or a failure notice - and history
    # has to record what the caller heard, or the next turn reasons from a gap.
    spoke_fixed: str | None = None
    # One question per turn, enforced in code rather than trusted to the prompt
    # alone - the model sometimes stacks multiple questions (or re-asks one
    # already answered) into a single reply. Once a sentence containing "?" has
    # been queued, stop pulling more from the LLM stream so nothing after it is
    # ever spoken.
    stop_after_question = False

    async def _round(messages: list[dict], tools, *, execute_tools: bool = True) -> list:
        """One LLM round. Queues sentences, returns any tool calls to execute.

        `execute_tools` False still DECLARES the tools to the provider but
        discards anything it asks for. That split exists because of a measured
        gpt-oss behaviour: with no tools declared on a conversation that has
        just used one, it emits a tool call anyway, Groq rejects the request
        outright, and the retry fails every attempt - three requests spent to
        produce nothing. Declaring the tools makes the request legal; ignoring
        the result is what actually enforces one tool round-trip per turn.
        """
        nonlocal stop_after_question, queued_text
        calls: list = []
        # A fresh assembler per round. It has no reset(), and its _is_first flag
        # is what lets the first sentence flush at any length - reusing the
        # round-1 instance would make round 2's first sentence wait for 25
        # characters, which is exactly the moment latency is most visible.
        round_assembler = SentenceAssembler()
        llm_t0 = time.perf_counter()
        first_token = True
        async for event in session.llm.stream_chat(
            messages,
            max_tokens=session.cfg.llm.max_tokens,
            temperature=session.cfg.llm.temperature,
            tools=tools,
        ):
            if isinstance(event, ToolCall):
                if execute_tools:
                    calls.append(event)
                else:
                    log.info("turn %d: ignoring a second-round tool call to %s", turn_id, event.name)
                continue
            if isinstance(event, Done):
                continue
            if stop_after_question:
                # Keep draining rather than breaking. Breaking closes the async
                # generator at its yield point, so a tool call the model emits
                # after its question - gpt-oss usually puts text first - is
                # thrown away without ever being seen.
                continue
            if first_token:
                timing.setdefault("llm_ttft_ms", (time.perf_counter() - llm_t0) * 1000)
                first_token = False
            for sentence in round_assembler.feed(event.text):
                queued_text = True
                await queue.put(sentence)
                if "?" in sentence:
                    stop_after_question = True
                    break
        if not stop_after_question:
            tail = round_assembler.flush()
            if tail:
                queued_text = True
                await queue.put(tail)
        return calls

    async def producer() -> None:
        # nonlocal, not a fresh local: this function ASSIGNS queued_text below,
        # which without the declaration would bind a local that _round never
        # writes to - leaving the "did round two say anything?" test permanently
        # False, and the fixed confirmation spoken on top of a reply that was
        # perfectly fine.
        nonlocal queued_text
        try:
            if session.llm.request_count >= session.cfg.llm.max_requests_per_call:
                log.error("call %s: request ceiling (%d) reached - not calling the model again",
                          session.session_id, session.cfg.llm.max_requests_per_call)
                await queue.put(_Filler(_script_line(session, "tool_failed"), terminal=True))
                return
            tools = _tools_for(session)
            calls = await _round(session.history, tools or None)
            if calls and not blocked:
                results = await _run_tool_phase(session, turn_id, queue, calls, timing)
                if results:
                    queued_text = False  # judge round two on its own output
                    # Round two declares no tools. One tool round-trip per turn,
                    # enforced by the parameter rather than by asking the prompt
                    # nicely - this is the single biggest lever on the request
                    # budget, and it is not left to the model's discretion.
                    await _round(
                        session.history + _tool_exchange(calls, results),
                        tools or None,
                        execute_tools=False,
                    )
                    if not queued_text:
                        # The model answered with another tool call instead of
                        # words. Say something fixed rather than leaving the
                        # caller on a filler line followed by nothing.
                        log.info("turn %d: round two produced no speech - using the fixed confirmation", turn_id)
                        await queue.put(_Filler(_script_line(session, "tool_done"), terminal=True))
        finally:
            await queue.put(_SENTINEL)

    async def consumer() -> None:
        nonlocal speaking_started, total_audio_s, blocked, spoke_fixed
        while True:
            sentence = await queue.get()
            if sentence is _SENTINEL:
                break
            if blocked:
                continue  # drain the queue; nothing more from this turn is spoken

            if isinstance(sentence, _Filler):
                # Fixed text: pre-approved, so it skips the screen, and kept out
                # of reply_parts so it never reaches history.
                if speaking_started is None:
                    await session.set_state(State.SPEAKING)
                    speaking_started = time.monotonic()
                session.spoken_anything = True
                if sentence.terminal:
                    spoke_fixed = sentence.text
                total_audio_s += await _speak_text(session, turn_id, sentence.text, timing, cache=True)
                continue

            decision = _screen(session, sentence)
            if decision is not None:
                blocked = True
                session.call_logger.log_event({
                    "type": "blocked_output",
                    "turn_id": turn_id,
                    "text": sentence,            # verbatim: the point is to show what was almost said
                    "code": decision.code,
                    "detail": decision.detail,
                    "evidence": decision.evidence,
                })
                log.warning("turn %d: blocked (%s) %r", turn_id, decision.code, sentence)
                # The rest of the reply is abandoned rather than spoken around.
                # A model that just went off-script produces follow-on sentences
                # that assume the blocked one was said, so they are incoherent
                # at best and corroborating at worst.
                recovery = _recovery_line(session)
                if speaking_started is None:
                    await session.set_state(State.SPEAKING)
                    speaking_started = time.monotonic()
                total_audio_s += await _speak_text(session, turn_id, recovery, timing)
                continue

            reply_parts.append(sentence)
            if speaking_started is None:
                await session.set_state(State.SPEAKING)
                speaking_started = time.monotonic()
                timing["e2e_first_audio_ms"] = (time.perf_counter() - t_turn0) * 1000
            session.spoken_anything = True
            total_audio_s += await _speak_text(session, turn_id, sentence, timing)

    async def timed_producer() -> None:
        await producer()
        timing["llm_done_ms"] = (time.perf_counter() - t_turn0) * 1000

    try:
        await asyncio.gather(timed_producer(), consumer())
    except asyncio.CancelledError:
        # barge-in: record whatever the agent managed to say before being cut off
        reply_text = " ".join(reply_parts).strip()
        if reply_text:
            session.history.append(
                {"role": "assistant", "content": f"{reply_text} (interrupted)"}
            )
        if session.llm.last_usage:
            session.call_logger.record_llm(
                session.llm.last_usage.input_tokens,
                session.llm.last_usage.output_tokens,
                session.llm.last_usage.cached_tokens,
            )
        session.call_logger.log_event(
            {"type": "agent_turn", "turn_id": turn_id, "text": reply_text, "interrupted": True}
        )
        log.info("turn %d: cancelled by barge-in after %r", turn_id, reply_text)
        raise

    reply_text = " ".join(reply_parts).strip()
    if reply_text:
        # Append any closing fixed line too. History has to match what the
        # caller heard end to end - recording only the model's half leaves the
        # agent believing it never confirmed, so it confirms again next turn.
        heard = f"{reply_text} {spoke_fixed}".strip() if spoke_fixed else reply_text
        session.history.append({"role": "assistant", "content": heard})
    elif blocked:
        # History must match what the caller actually heard, not what the model
        # wanted to say. Recording the recovery line keeps the next turn honest:
        # leaving the turn out entirely reads to the model as "my last reply was
        # never delivered", and it tends to say the blocked thing again.
        session.history.append({"role": "assistant", "content": _recovery_line(session)})
    elif spoke_fixed:
        session.history.append({"role": "assistant", "content": spoke_fixed})
    else:
        log.warning("turn %d: the model produced no text", turn_id)

    if session.llm.last_usage:
        prompt_tokens = session.llm.last_usage.input_tokens
        session.call_logger.log_event({
            "type": "prompt_tokens", "turn_id": turn_id, "tokens": prompt_tokens,
            "requests_so_far": session.llm.request_count,
        })
        if prompt_tokens > session.cfg.llm.warn_prompt_tokens:
            log.warning("turn %d: prompt was %d tokens (warn above %d)",
                        turn_id, prompt_tokens, session.cfg.llm.warn_prompt_tokens)
        timing["llm_input_tokens"] = session.llm.last_usage.input_tokens
        timing["llm_output_tokens"] = session.llm.last_usage.output_tokens
        timing["llm_cached_tokens"] = session.llm.last_usage.cached_tokens
        session.call_logger.record_llm(
            session.llm.last_usage.input_tokens,
            session.llm.last_usage.output_tokens,
            session.llm.last_usage.cached_tokens,
        )
    session.call_logger.log_event(
        {"type": "agent_turn", "turn_id": turn_id,
         "text": (f"{reply_text} {spoke_fixed}".strip() if (reply_text and spoke_fixed) else
                  reply_text or (_recovery_line(session) if blocked else (spoke_fixed or ""))),
         "interrupted": False, "blocked": blocked}
    )

    log.info(
        "turn %d timing: %s",
        turn_id,
        {k: (round(v, 1) if isinstance(v, float) else v)
         for k, v in timing.items() if not k.startswith("_")},
    )
    await session.transport.send_control({"type": "turn_end", "turn_id": turn_id})

    if speaking_started is None:
        await session.set_state(State.LISTENING)
    else:
        await _drain_and_return_to_listening(session, speaking_started, total_audio_s)
