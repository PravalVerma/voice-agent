"""run_live_call: the "gemini_live" pipeline - Gemini's Live API (a persistent
bidirectional audio-to-audio websocket, wss://.../BidiGenerateContent) instead
of the cascaded STT->LLM->TTS turns session.py/pipeline.py drive for the
"standard" pipeline. Turn detection, interruption, and audio
generation all happen server-side over one continuous stream, so this is a
parallel, separate call path rather than a mode bolted onto CallSession - it
reuses AudioTransport, CallLogger, and the call_state.State control-message
vocabulary the frontend already understands, and nothing else from the
cascaded pipeline.

Uses the official `google-genai` SDK (not this project's usual plain-httpx
style) - Google's own docs steer Python callers to it for the Live API, since
hand-rolling the session-setup/activity-signal/resumption message shapes is
materially more complex than the single-shot generateContent calls the other
Gemini providers in this codebase make.
"""

import asyncio
import logging
import os
import time
import uuid

from google import genai
from google.genai import types

from . import tts_catalog
from .audio.frames import pcm_duration_s
from .call_state import State
from .calllog import CallLogger
from .config import AppConfig
from .transport.base import AudioTransport, ControlMessage, Disconnected

log = logging.getLogger("agent")

_MODEL = "gemini-3.1-flash-live-preview"
_IN_SAMPLE_RATE = 16000  # matches AudioTransport.in_sample_rate / PIPELINE_SAMPLE_RATE - no resampling needed
_OUT_SAMPLE_RATE = 24000  # matches AudioTransport.out_sample_rate and every other TTS provider's sample_rate

_WRAP_UP_PROMPT = (
    "We've reached the time limit for this call - say a brief goodbye now and end the conversation."
)


async def run_live_call(
    transport: AudioTransport,
    cfg: AppConfig,
    gender: str | None,
    voice_name: str | None,
    user_id: int | None = None,
    username: str | None = None,
) -> None:
    # cfg already has everything else it needs baked in by config.apply_language()
    # (system_prompt resolved for the bot/language/call_vars) - only cfg.tts is
    # left untouched for this pipeline (no GeminiTTS provider is built), so the
    # actual voice name is resolved here, the same call apply_language() itself
    # made to pick the gender used in the system prompt's "the voice speaking
    # for you is {gender}" instruction.
    session_id = uuid.uuid4().hex[:8]
    resolved_voice, _ = tts_catalog.resolve_gemini_voice(gender, voice_name)
    call_logger = CallLogger(
        session_id, cfg, pipeline="gemini_live", user_id=user_id, username=username,
        recorder=getattr(transport, "recorder", None),
    )
    deadline = time.monotonic() + cfg.call.max_call_minutes * 60

    log.info("live call %s started (voice=%s)", session_id, resolved_voice)
    await transport.send_control(
        {"type": "ready", "session_id": session_id, "out_sample_rate": transport.out_sample_rate}
    )

    state = State.LISTENING

    async def set_state(new_state: State) -> None:
        nonlocal state
        if state != new_state:
            state = new_state
            await transport.send_control({"type": "state", "state": new_state.value})

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=cfg.persona.system_prompt,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=resolved_voice)
            )
        ),
        # enabled so the existing transcript UI (transcript/agent_text control
        # messages) keeps working without any protocol-specific frontend changes
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Gemini's default VAD is tuned for snappy barge-in, which also makes it
        # prone to firing on the agent's own voice leaking back into the mic
        # (browser echo cancellation is never perfect, especially over
        # speakers) - seen as the agent getting cut off mid-sentence with no
        # actual caller utterance ever transcribed. Requiring a bit more
        # sustained speech before committing start-of-speech cuts down on that
        # false-positive barge-in while still letting a real interruption
        # through quickly.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                prefix_padding_ms=300,
            )
        ),
    )

    wrap_up_requested = False
    turn_id = 0
    turn_open = False

    async def pump_caller_audio(live_session) -> None:
        nonlocal wrap_up_requested
        while True:
            event = await transport.recv()
            if isinstance(event, Disconnected):
                return
            if isinstance(event, ControlMessage):
                if event.data.get("type") == "stop":
                    return
                continue
            # only AudioFrame is left once Disconnected/ControlMessage are handled above
            call_logger.record_live_audio(input_s=pcm_duration_s(event.pcm, _IN_SAMPLE_RATE))
            await live_session.send_realtime_input(
                audio=types.Blob(data=event.pcm, mime_type=f"audio/pcm;rate={_IN_SAMPLE_RATE}")
            )
            if not wrap_up_requested and time.monotonic() >= deadline:
                wrap_up_requested = True
                log.info("live call %s: reached max_call_minutes, wrapping up", session_id)
                # sending client content out-of-band interrupts whatever the model
                # is currently saying (the same mechanism that powers barge-in),
                # so this reliably becomes the next - and last - turn
                await live_session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=_WRAP_UP_PROMPT)]),
                    turn_complete=True,
                )
                await transport.send_control({"type": "call_ending", "reason": "max_duration"})
                return  # stop accepting caller audio; let the model's goodbye play out

    async def pump_model_events(live_session) -> None:
        nonlocal turn_id, turn_open
        while True:
            got_message = False
            async for message in live_session.receive():
                got_message = True
                sc = message.server_content
                if sc is None:
                    continue
                if sc.interrupted:
                    await transport.send_control({"type": "barge_in", "turn_id": turn_id})
                    call_logger.log_event({"type": "barge_in", "turn_id": turn_id})
                    turn_open = False
                    await set_state(State.LISTENING)
                if sc.input_transcription and sc.input_transcription.text:
                    await transport.send_control(
                        {"type": "transcript", "role": "user", "text": sc.input_transcription.text}
                    )
                    call_logger.log_event(
                        {"type": "user_utterance", "text": sc.input_transcription.text}
                    )
                has_audio = message.data is not None
                has_text = bool(sc.output_transcription and sc.output_transcription.text)
                if not turn_open and (has_audio or has_text):
                    turn_id += 1
                    turn_open = True
                    await transport.send_control({"type": "turn_start", "turn_id": turn_id})
                if has_text:
                    await transport.send_control(
                        {"type": "agent_text", "turn_id": turn_id, "text": sc.output_transcription.text}
                    )
                    call_logger.log_event(
                        {"type": "agent_turn", "turn_id": turn_id, "text": sc.output_transcription.text}
                    )
                if has_audio:
                    await set_state(State.SPEAKING)
                    await transport.send_audio(turn_id, message.data)
                    call_logger.record_live_audio(output_s=pcm_duration_s(message.data, _OUT_SAMPLE_RATE))
                if sc.turn_complete:
                    if turn_open:
                        await transport.send_control({"type": "turn_end", "turn_id": turn_id})
                    turn_open = False
                    if wrap_up_requested:
                        return  # the goodbye turn finished playing out - end the call
                    await set_state(State.LISTENING)
            if not got_message:
                return  # the server closed the stream

    try:
        async with client.aio.live.connect(model=_MODEL, config=live_config) as live_session:
            await set_state(State.LISTENING)
            task_in = asyncio.create_task(pump_caller_audio(live_session))
            task_out = asyncio.create_task(pump_model_events(live_session))
            done, pending = await asyncio.wait({task_in, task_out}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            for t in done:
                t.result()  # re-raise so the outer except logs/reports a real failure
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("live call %s: failed", session_id)
        await transport.send_control({"type": "error", "message": f"call failed: {e}"})
    finally:
        log.info("live call %s ended", session_id)
        meta = call_logger.finalize()
        log.info("live call %s cost estimate: Rs %.2f (%s)", session_id, meta["cost_inr"], meta["usage"])
        await transport.close()
