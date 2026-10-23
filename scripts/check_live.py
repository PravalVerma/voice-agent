"""Open a Gemini Live session, speak one line, and write the reply as a WAV.

Proves the Studio tier end to end - that GEMINI_API_KEY is valid, that this key
may use the Live model, and that audio actually comes back - without needing a
microphone or the browser UI.

Usage: uv run python scripts/check_live.py ["say something short"] [--out live.wav]
"""

import argparse
import asyncio
import sys
import time

from _bootstrap import load_config_or_die, need_env

from app.audio.frames import pcm_duration_s, wav_wrap
from app.live_session import _MODEL, _OUT_SAMPLE_RATE

# Imported late: the google-genai SDK is only needed for this pipeline, and the
# error if it's missing should be the one clear line the other checks give.
try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError:
    sys.exit("error: the google-genai SDK is not installed - run: uv sync")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! Please reply with one short sentence.",
        help="what to say to the model",
    )
    parser.add_argument("--out", default="live.wav", help="output WAV path (default: live.wav)")
    parser.add_argument("--voice", default="Kore", help="Gemini voice name (default: Kore)")
    args = parser.parse_args()

    load_config_or_die()  # loads .env as a side effect, so GEMINI_API_KEY is visible
    need_env("GEMINI_API_KEY", "the Studio (Gemini Live) tier")

    # Same model and response shape run_live_call uses, so a pass here means the
    # real call path is reachable too - just driven by text instead of a mic.
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="You are a test harness. Answer in one short sentence.",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    print(f"connecting to {_MODEL} (voice {args.voice}) ...")
    t0 = time.perf_counter()
    audio = bytearray()
    said = []

    try:
        async with client.aio.live.connect(model=_MODEL, config=config) as session:
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=args.text)]),
                turn_complete=True,
            )
            first_audio_ms = None
            async for message in session.receive():
                if message.data:
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - t0) * 1000
                    audio.extend(message.data)
                sc = message.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
                if sc and sc.turn_complete:
                    break
    except Exception as e:
        # Auth, quota and model-access failures all surface here. Keep it to one
        # line like the other checks - the message from the API is the useful part.
        sys.exit(f"error: Live session failed - {type(e).__name__}: {e}")

    if not audio:
        sys.exit("error: the session opened but returned no audio - the key may not have Live access")

    with open(args.out, "wb") as f:
        f.write(wav_wrap(bytes(audio), _OUT_SAMPLE_RATE))

    print(f"first audio: {first_audio_ms:.0f} ms")
    if said:
        print(f"model said: {''.join(said).strip()}")
    print(
        f"wrote {args.out} - {pcm_duration_s(bytes(audio), _OUT_SAMPLE_RATE):.2f}s "
        f"of audio @ {_OUT_SAMPLE_RATE} Hz"
    )
    print("OK - this key can use the Live model, so the Studio tier will work.")


if __name__ == "__main__":
    asyncio.run(main())
