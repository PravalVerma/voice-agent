"""Read-only access to the per-call logs CallLogger writes under logs/calls/ -
backs the admin-only Call Logs page (list + full transcript), so seeing what a
caller/client said and what the agent replied never requires SSH/file access
to the server, just an admin login."""

import json
import re
from pathlib import Path

from . import s3_archive
from .config import ROOT, get_config

# call directories are always "{yyyymmdd}_{hhmmss}_{8 hex chars}" (see
# CallLogger.__init__) - validated strictly since the directory name comes
# straight from the URL path in api_get_call_log (main.py) and is used to
# build a filesystem path, so this is what stands between that endpoint and a
# path-traversal read of arbitrary files on the server
_DIR_NAME_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")

RECORDING_NAME = "recording.wav"


def _log_root() -> Path:
    return ROOT / get_config().call.log_dir


def _read_first_event(transcript_path: Path) -> dict | None:
    try:
        with transcript_path.open(encoding="utf-8") as f:
            first_line = f.readline()
        return json.loads(first_line) if first_line else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_last_event(transcript_path: Path) -> dict | None:
    # transcript files are append-only JSONL, at most a few thousand short
    # lines for even a very long call - reading the whole file to grab the
    # last line is simpler than a tail-seek and plenty fast at this scale
    try:
        last = None
        with transcript_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        return json.loads(last) if last else None
    except (OSError, json.JSONDecodeError):
        return None


def _summarize(dir_path: Path) -> dict | None:
    transcript_path = dir_path / "transcript.jsonl"
    meta_path = dir_path / "meta.json"
    # dir name is "{yyyymmdd}_{hhmmss}_{session_id}"
    ts_part = dir_path.name[:15]
    started_at = f"{ts_part[:4]}-{ts_part[4:6]}-{ts_part[6:8]} {ts_part[9:11]}:{ts_part[11:13]}:{ts_part[13:15]}"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
    else:
        meta = None

    if meta is not None:
        # a recording written by an older build (or a call that predates the
        # feature) has no meta entry - fall back to the file actually being there
        recording = meta.get("recording")
        if recording is None and (dir_path / RECORDING_NAME).is_file():
            recording = {"file": RECORDING_NAME}
        return {
            "dir": dir_path.name,
            "started_at": started_at,
            "user_id": meta.get("user_id"),
            "username": meta.get("username"),
            "language": meta.get("language"),
            "tts_tier": meta.get("tts_tier"),
            "duration_s": meta.get("duration_s"),
            "cost_inr": meta.get("cost_inr"),
            "turn_count": meta.get("turn_count"),
            "complete": True,
            "has_recording": recording is not None,
            "recording_s3_key": (recording or {}).get("s3_key"),
        }

    # no meta.json - the process died/restarted mid-call (see CallLogger)
    # before finalize() ran. user attribution still survives because it's
    # logged as the very first transcript line, not just at the end.
    if not transcript_path.exists():
        return None
    first = _read_first_event(transcript_path)
    last = _read_last_event(transcript_path)
    user_id = first.get("user_id") if first else None
    username = first.get("username") if first else None
    return {
        "dir": dir_path.name,
        "started_at": started_at,
        "user_id": user_id,
        "username": username,
        "language": None,
        "tts_tier": None,
        "duration_s": last.get("t") if last else None,
        "cost_inr": None,
        "turn_count": None,
        "complete": False,
        # finalize() never ran, so there's no mixdown - the recording only gets
        # written at the end of a call
        "has_recording": False,
        "recording_s3_key": None,
    }


def list_call_logs(limit: int = 200) -> list[dict]:
    root = _log_root()
    if not root.is_dir():
        return []
    # directory names sort chronologically (yyyymmdd_hhmmss prefix) - newest first
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    out = []
    for d in dirs[:limit]:
        summary = _summarize(d)
        if summary is not None:
            out.append(summary)
    return out


def _parse_transcript(transcript_path: Path) -> list[dict]:
    """Turns raw per-event JSONL into the same shape the live call's transcript
    panel shows: one entry per user utterance, one per agent turn (with any
    agent_turn text fragments for that turn_id joined together), tagged
    interrupted if a barge_in event landed on that turn_id."""
    order: list[dict] = []
    by_turn: dict[int, dict] = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec_type = rec.get("type")
        if rec_type == "user_utterance":
            order.append({"role": "user", "text": rec.get("text", ""), "t": rec.get("t")})
        elif rec_type == "agent_turn":
            turn_id = rec.get("turn_id")
            entry = by_turn.get(turn_id)
            if entry is None:
                entry = {"role": "agent", "turn_id": turn_id, "text": "", "t": rec.get("t"), "interrupted": False}
                by_turn[turn_id] = entry
                order.append(entry)
            entry["text"] += rec.get("text", "")
        elif rec_type == "barge_in":
            entry = by_turn.get(rec.get("turn_id"))
            if entry is not None:
                entry["interrupted"] = True
    return order


def recording_path(dir_name: str) -> Path | None:
    """Local path to a call's recording, or None if the name is invalid or no
    recording exists. Goes through the same strict dir-name validation as
    get_call_log - the name arrives straight from a URL path."""
    if not _DIR_NAME_RE.match(dir_name):
        return None
    path = _log_root() / dir_name / RECORDING_NAME
    return path if path.is_file() else None


def recording_download_name(dir_name: str) -> str:
    """Filename the browser saves as. Mirrors the S3 key's naming so a file
    downloaded from the UI and one pulled from the bucket are recognisably the
    same recording."""
    summary = _summarize(_log_root() / dir_name) or {}
    return s3_archive.build_key(
        dir_name, summary.get("username"), summary.get("language"), summary.get("tts_tier")
    ).rsplit("/", 1)[-1]


def get_call_log(dir_name: str) -> dict | None:
    if not _DIR_NAME_RE.match(dir_name):
        return None
    dir_path = _log_root() / dir_name
    if not dir_path.is_dir():
        return None
    summary = _summarize(dir_path)
    if summary is None:
        return None
    transcript_path = dir_path / "transcript.jsonl"
    summary["transcript"] = _parse_transcript(transcript_path) if transcript_path.exists() else []
    return summary
