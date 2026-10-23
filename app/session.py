"""CallSession: per-connection state machine wiring VAD events to agent turns."""

import asyncio
import logging
import time
import uuid

from .audio.frames import FrameBuffer
from .audio.vad import SileroVAD, SpeechEnd, SpeechStart, VADGate
from .call_state import State
from .calllog import CallLogger
from .config import AppConfig
from .pipeline import run_agent_turn, speak_greeting, speak_notice
from .providers.base import LLMProvider, STTProvider, TTSProvider
from .transport.base import AudioFrame, AudioTransport, ControlMessage, Disconnected

log = logging.getLogger("agent")

_WRAP_UP_MESSAGE = "We're at the time limit for this call - thanks for chatting, goodbye!"

# Spoken when a turn fails outright - most often the LLM provider rate-limiting
# us (Groq returns 429 once the daily token cap is hit). Without this the caller
# just gets dead air and assumes the line dropped. Hindi is written in
# Devanagari on purpose: the Google voices mispronounce romanized Hindi
# ("main" comes out as the English word "mayn", not "मैं").
_TURN_FAILED_MESSAGES = {
    "en": "Sorry, I'm having a bit of trouble at my end - could you say that again?",
    "hi": "माफ़ कीजिए, मेरी तरफ़ से थोड़ी दिक्कत आ रही है। क्या आप दोबारा कह सकते हैं?",
}


class CallSession:
    def __init__(
        self,
        transport: AudioTransport,
        cfg: AppConfig,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        user_id: int | None = None,
        username: str | None = None,
        flow=None,
    ):
        self.transport = transport
        self.cfg = cfg
        self.stt = stt
        self.llm = llm
        self.tts = tts

        # A CollectionsFlow for a collections-mode bot, None for an ordinary
        # assistant bot - which then takes exactly the path it always has.
        self.flow = flow
        #: set by the flow once an account is bound; the outbound compliance
        #: screen in pipeline.py is inert while this is None
        self.allowed_facts = None
        #: McpClient for a collections call, None otherwise
        self.mcp = None
        #: idempotency cache, keyed in code - see pipeline._result_key
        self.tool_results: dict[str, dict] = {}
        #: whether any audio has gone out this turn, so the tool filler is only
        #: used to cover a gap that the caller would otherwise hear as silence
        self.spoken_anything = False
        #: tool executions that must outlive the turn that started them
        self._detached: set[asyncio.Task] = set()
        self.session_id = uuid.uuid4().hex[:8]
        self.state = State.LISTENING
        self.history: list[dict] = [{"role": "system", "content": cfg.persona.system_prompt}]
        self.turn_id = 0
        self.gate = VADGate(SileroVAD(), cfg.vad)
        self.frame_buffer = FrameBuffer()
        self.turn_task: asyncio.Task | None = None
        self.call_logger = CallLogger(
            self.session_id, cfg, user_id=user_id, username=username,
            recorder=getattr(transport, "recorder", None),
        )
        self._deadline = time.monotonic() + cfg.call.max_call_minutes * 60

    async def run(self) -> None:
        log.info("call %s started", self.session_id)
        await self.transport.send_control(
            {
                "type": "ready",
                "session_id": self.session_id,
                "out_sample_rate": self.transport.out_sample_rate,
            }
        )
        if self.flow is not None:
            # A collections call opens before the caller says anything: the
            # automated-assistant and recording disclosure has to come first.
            await self.flow.start(self)
        else:
            # no auto-played greeting - the agent waits for the caller to speak
            # first, and introduces itself in its first reply (see system_prompt)
            await self.set_state(State.LISTENING)

        try:
            while True:
                event = await self.transport.recv()
                if isinstance(event, Disconnected):
                    break
                if isinstance(event, ControlMessage):
                    if event.data.get("type") == "stop":
                        break
                    continue
                if isinstance(event, AudioFrame):
                    await self._on_audio(event.pcm)
                    if time.monotonic() >= self._deadline:
                        await self._wrap_up_and_close()
                        break
        finally:
            if self.turn_task and not self.turn_task.done():
                self.turn_task.cancel()
                try:
                    await self.turn_task  # let it fully unwind before the transport goes away
                except asyncio.CancelledError:
                    pass
                except Exception:
                    log.exception("call %s: error while cancelling the in-flight turn", self.session_id)
            if self._detached:
                # Let tool writes finish before the process stops caring. This
                # is what makes the barge-in guarantee real.
                log.info("call %s: waiting for %d in-flight tool call(s)",
                         self.session_id, len(self._detached))
                await asyncio.wait(set(self._detached), timeout=5.0)
            if self.mcp is not None:
                await self.mcp.aclose()
            log.info("call %s ended", self.session_id)
            meta = self.call_logger.finalize()
            log.info("call %s cost estimate: Rs %.2f (%s)", self.session_id, meta["cost_inr"], meta["usage"])
            await self.transport.close()

    async def _wrap_up_and_close(self) -> None:
        log.info("call %s: reached max_call_minutes, wrapping up", self.session_id)
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task
            except asyncio.CancelledError:
                pass
        try:
            self.turn_id += 1
            await speak_greeting(self, self.turn_id, _WRAP_UP_MESSAGE)
        except Exception:
            log.exception("call %s: failed to speak the wrap-up message", self.session_id)
        await self.transport.send_control({"type": "call_ending", "reason": "max_duration"})

    async def _on_audio(self, pcm: bytes) -> None:
        for frame in self.frame_buffer.push(pcm):
            # VADGate uses a stricter threshold/duration while the agent is talking,
            # since browser echo cancellation isn't perfect at rejecting its own voice
            self.gate.strict = self.state in (State.PROCESSING, State.SPEAKING)
            for event in self.gate.push(frame):
                if isinstance(event, SpeechStart):
                    if self.state == State.LISTENING:
                        await self.set_state(State.USER_SPEAKING)
                    elif self.state in (State.PROCESSING, State.SPEAKING):
                        await self._barge_in()
                elif isinstance(event, SpeechEnd):
                    if self.state != State.USER_SPEAKING:
                        continue  # discard: e.g. a stray VAD trigger with nothing tracking it
                    self.turn_id += 1
                    turn_id = self.turn_id
                    self.call_logger.dump_utterance_wav(event.utterance)
                    self.turn_task = asyncio.create_task(self._run_turn(turn_id, event.utterance))

    async def _barge_in(self) -> None:
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task  # wait for its cancellation cleanup (partial history, etc.)
            except asyncio.CancelledError:
                pass
        log.info("call %s: barge-in during turn %d", self.session_id, self.turn_id)
        self.call_logger.log_event({"type": "barge_in", "turn_id": self.turn_id})
        await self.transport.send_control({"type": "barge_in", "turn_id": self.turn_id})
        await self.set_state(State.USER_SPEAKING)

    async def _run_turn(self, turn_id: int, utterance_pcm: bytes) -> None:
        try:
            if self.flow is not None:
                await self.flow.handle_turn(self, turn_id, utterance_pcm)
            else:
                await run_agent_turn(self, turn_id, utterance_pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("call %s: turn %d failed", self.session_id, turn_id)
            await self.transport.send_control(
                {"type": "error", "message": "something went wrong processing that - please try again"}
            )
            await self._speak_turn_failed(turn_id)

    async def _speak_turn_failed(self, turn_id: int) -> None:
        """Say something out loud after a failed turn so the caller hears an
        apology rather than silence. Best-effort: if TTS is unavailable too,
        just fall back to listening again instead of tearing down the call."""
        message = _TURN_FAILED_MESSAGES.get(self.cfg.stt.language, _TURN_FAILED_MESSAGES["en"])
        try:
            await speak_notice(self, turn_id, message)
        except asyncio.CancelledError:
            raise  # caller barged in over the apology - normal, let it unwind
        except Exception:
            log.exception("call %s: failed to speak the turn-failure notice", self.session_id)
            await self.set_state(State.LISTENING)

    def spawn_detached(self, coro) -> asyncio.Task:
        """Run something that must survive the turn being cancelled.

        A barge-in cancels the whole turn task. A tool call in flight is
        already writing to the database, so cancelling it mid-write would
        leave the caller having made a commitment the system has no record of.
        The task is tracked here and awaited during teardown; without that,
        connection close would cancel it anyway and "the tool finishes" would
        be a claim rather than a guarantee.
        """
        task = asyncio.create_task(coro)
        self._detached.add(task)

        def _done(t: asyncio.Task) -> None:
            self._detached.discard(t)
            if not t.cancelled() and t.exception() is not None:
                # Retrieved deliberately: an unretrieved exception prints a
                # warning from the event loop at some arbitrary later point,
                # with no context about which call produced it.
                log.error("call %s: detached tool task failed", self.session_id,
                          exc_info=t.exception())

        task.add_done_callback(_done)
        return task

    async def set_state(self, state: State) -> None:
        self.state = state
        await self.transport.send_control({"type": "state", "state": state.value})

    def trim_history(self) -> None:
        """Keep the system prompt and the most recent exchanges.

        Walks back to a user message rather than slicing a fixed number off the
        tail, so the window always opens on a caller turn and never starts
        mid-exchange with an assistant reply to a question that is no longer
        there.

        Tool messages are deliberately never in history (see
        pipeline._tool_exchange), which is what keeps this safe: a blind tail
        slice would otherwise be free to separate a tool result from the
        assistant message that requested it, and every provider rejects that.
        """
        system = self.history[0]
        rest = self.history[1:]
        max_exchanges = self.cfg.llm.history_max_turns

        seen = 0
        cut = 0
        for i in range(len(rest) - 1, -1, -1):
            if rest[i].get("role") == "user":
                seen += 1
                if seen == max_exchanges:
                    # cut AT this user message, not after it: the window has to
                    # begin with the caller turn that opens the exchange
                    cut = i
                    break
        self.history = [system] + rest[cut:]
