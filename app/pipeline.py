"""One agent turn: STT -> streaming LLM -> sentence chunking -> TTS -> transport.

The LLM producer and TTS consumer run concurrently over an asyncio.Queue so the
first sentence starts playing while the LLM is still generating later ones.
"""

import asyncio
import logging
import time

from . import compliance, tts_cache
from .audio.frames import PIPELINE_SAMPLE_RATE, pcm_duration_s
from .call_state import State
from .text.sentences import SentenceAssembler

log = logging.getLogger("agent")

_SEND_CHUNK_MS = 500  # slice agent audio into <=500ms frames so playback can start early
_MIN_UTTERANCE_S = 0.3  # utterances shorter than this are almost certainly noise/clicks
_SENTINEL = object()


def _chunk_pcm(pcm: bytes, sample_rate: int, chunk_ms: int = _SEND_CHUNK_MS) -> list[bytes]:
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


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


async def _speak_text(session, turn_id: int, text: str, timing: dict, *, cache: bool = False) -> float:
    """Synthesize + send one sentence; return its audio duration in seconds.

    `cache` is for fixed script lines only - a disclosure, a verification
    prompt, a recovery line. Model-generated replies are unique, so caching
    them would grow the directory without ever hitting.
    """
    t0 = time.perf_counter()
    hit = False
    if cache:
        pcm, hit = await tts_cache.synthesize_cached(
            session.tts, text,
            voice=session.cfg.tts.voice,
            speaking_rate=session.cfg.tts.speaking_rate,
            sample_rate=session.tts.sample_rate,
        )
    else:
        pcm = await session.tts.synthesize(text)
    timing.setdefault("tts_first_ms", (time.perf_counter() - t0) * 1000)
    for chunk in _chunk_pcm(pcm, session.tts.sample_rate):
        await session.transport.send_audio(turn_id, chunk)
    await session.transport.send_control({"type": "agent_text", "turn_id": turn_id, "text": text})
    duration_s = pcm_duration_s(pcm, session.tts.sample_rate)
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
    # One question per turn, enforced in code rather than trusted to the prompt
    # alone - the model sometimes stacks multiple questions (or re-asks one
    # already answered) into a single reply. Once a sentence containing "?" has
    # been queued, stop pulling more from the LLM stream so nothing after it is
    # ever spoken.
    stop_after_question = False

    async def producer() -> None:
        nonlocal stop_after_question
        llm_t0 = time.perf_counter()
        first_token = True
        async for delta in session.llm.stream_chat(
            session.history,
            max_tokens=session.cfg.llm.max_tokens,
            temperature=session.cfg.llm.temperature,
        ):
            if stop_after_question:
                break
            if first_token:
                timing["llm_ttft_ms"] = (time.perf_counter() - llm_t0) * 1000
                first_token = False
            for sentence in assembler.feed(delta):
                await queue.put(sentence)
                if "?" in sentence:
                    stop_after_question = True
                    break
            if stop_after_question:
                break
        if not stop_after_question:
            tail = assembler.flush()
            if tail:
                await queue.put(tail)
        await queue.put(_SENTINEL)

    async def consumer() -> None:
        nonlocal speaking_started, total_audio_s, blocked
        while True:
            sentence = await queue.get()
            if sentence is _SENTINEL:
                break
            if blocked:
                continue  # drain the queue; nothing more from this turn is spoken

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
            total_audio_s += await _speak_text(session, turn_id, sentence, timing)

    try:
        await asyncio.gather(producer(), consumer())
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
        session.history.append({"role": "assistant", "content": reply_text})
    elif blocked:
        # History must match what the caller actually heard, not what the model
        # wanted to say. Recording the recovery line keeps the next turn honest:
        # leaving the turn out entirely reads to the model as "my last reply was
        # never delivered", and it tends to say the blocked thing again.
        session.history.append({"role": "assistant", "content": _recovery_line(session)})
    else:
        log.warning("turn %d: LLM produced no text", turn_id)

    if session.llm.last_usage:
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
         "text": reply_text or (_recovery_line(session) if blocked else ""),
         "interrupted": False, "blocked": blocked}
    )

    log.info(
        "turn %d timing: %s",
        turn_id,
        {k: (round(v, 1) if isinstance(v, float) else v) for k, v in timing.items()},
    )
    await session.transport.send_control({"type": "turn_end", "turn_id": turn_id})

    if speaking_started is None:
        await session.set_state(State.LISTENING)
    else:
        await _drain_and_return_to_listening(session, speaking_started, total_audio_s)
