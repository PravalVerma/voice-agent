"""Run a WAV file through the VAD gate and print detected speech segments.

Usage: uv run python scripts/check_vad.py path\\to\\sample.wav [--dump]
"""

import argparse
from pathlib import Path

from _bootstrap import load_config_or_die

from app.audio.frames import FrameBuffer, read_wav_as_pipeline_pcm, wav_wrap
from app.audio.vad import FRAME_MS, SileroVAD, SpeechEnd, SpeechStart, VADGate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="path to a WAV file")
    parser.add_argument("--dump", action="store_true", help="write each segment as a WAV next to the input")
    args = parser.parse_args()

    cfg = load_config_or_die()
    pcm = read_wav_as_pipeline_pcm(args.wav)
    print(f"audio: {len(pcm) / 2 / 16000:.2f}s @ 16 kHz mono")

    gate = VADGate(SileroVAD(), cfg.vad)
    fb = FrameBuffer()
    segments = []
    t_ms = 0
    for frame in fb.push(pcm):
        for event in gate.push(frame):
            if isinstance(event, SpeechStart):
                print(f"  {t_ms / 1000:7.2f}s  speech start")
            elif isinstance(event, SpeechEnd):
                start_s = t_ms / 1000 - event.duration_s
                flag = "  (forced: max_utterance_s)" if event.forced else ""
                print(
                    f"  {t_ms / 1000:7.2f}s  speech end - segment "
                    f"{max(start_s, 0):.2f}s..{t_ms / 1000:.2f}s "
                    f"({event.duration_s:.2f}s){flag}"
                )
                segments.append(event)
        t_ms += FRAME_MS

    print(f"{len(segments)} segment(s) detected")
    if args.dump:
        base = Path(args.wav)
        for i, seg in enumerate(segments, 1):
            out = base.with_name(f"{base.stem}_seg{i}.wav")
            out.write_bytes(wav_wrap(seg.utterance))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
