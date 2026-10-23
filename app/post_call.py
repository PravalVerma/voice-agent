"""Work that happens after a call ends: a summary, a compliance audit, retention.

**The queue is the log directory itself.** A call needs work when its meta.json
exists, its `post.<job>.status` is missing or pending, and it has been still
long enough to have finished writing. There is no queue file and no job table.

That matters more than it looks. An in-memory queue is lost on restart, which
in a compliance system means silently dropping audits - the one failure mode
you cannot detect later, because a dropped audit looks exactly like a call that
passed. Scanning the directories is idempotent, survives a restart, and
self-heals: if a job dies half-done, meta.json was never amended and the next
sweep picks it up again.

**Most of the audit needs no model.** Four of the five checks - missing
disclosure, threat language, calling outside the window, and spoken figures
that do not match the account - are decidable from the transcript and the
account with the code that already exists. They run locally, for free, on every
call. Only third-party disclosure genuinely needs judgement, so only that is
sent to a model, and only when one is configured.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from . import compliance
from .config import ROOT, AppConfig, get_config

log = logging.getLogger("agent")

_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")

#: Phrases that satisfy the opening disclosure. Matched case-insensitively
#: against the greeting; each element is a set of alternatives, all of which
#: must be present somewhere in it.
_DISCLOSURE_REQUIRED = (
    ("automated", "automatic", "recorded message", "ऑटोमेटेड", "automated assistant"),
    ("record", "recording", "recorded", "रिकॉर्ड"),
)


def _log_root(cfg: AppConfig) -> Path:
    return ROOT / cfg.call.log_dir


def _read_meta(path: Path) -> dict | None:
    try:
        return json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_events(path: Path) -> list[dict]:
    out = []
    try:
        for line in (path / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return out


def amend_meta(path: Path, job: str, payload: dict) -> None:
    """Merge a job's result into meta.json, atomically and additively.

    Never touches an existing top-level key: call_logs._summarize reads them
    and has to keep working on amended and un-amended files alike.
    """
    meta = _read_meta(path)
    if meta is None:
        return
    meta.setdefault("post", {})[job] = payload
    tmp = path / "meta.json.tmp"
    try:
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, path / "meta.json")  # atomic, including on Windows
    except OSError:
        log.exception("could not amend meta.json for %s", path.name)


def pending(cfg: AppConfig, job: str) -> list[Path]:
    """Call directories still needing `job`, oldest first."""
    root = _log_root(cfg)
    if not root.is_dir():
        return []
    cutoff = time.time() - cfg.llm_post.settle_seconds
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not _DIR_RE.match(d.name):
            continue
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue  # call never finished; nothing settled to audit
        try:
            # Settle on the TRANSCRIPT's mtime, not meta.json's. The jobs amend
            # meta.json, so using it would mean the audit's own write pushes the
            # summary back past the settle window - each job starving the other
            # and neither finishing in the same pass.
            if (d / "transcript.jsonl").stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        meta = _read_meta(d)
        if meta is None:
            continue
        entry = (meta.get("post") or {}).get(job) or {}
        status = entry.get("status")
        if status in ("done", "skipped", "error"):
            continue
        out.append(d)
    return out


# ---- the deterministic audit --------------------------------------------------

def audit_locally(path: Path, cfg: AppConfig) -> list[dict]:
    """Every check that can be decided without a model. Runs on every call."""
    meta = _read_meta(path) or {}
    events = _read_events(path)
    flags: list[dict] = []

    def flag(code: str, severity: str, detail: str, evidence: str = "", t=None) -> None:
        flags.append({"code": code, "severity": severity, "detail": detail,
                      "evidence": evidence[:200], "t": t})

    if meta.get("refused"):
        return []  # the call never happened; there is nothing to audit

    agent_lines = [e for e in events if e.get("type") == "agent_turn" and e.get("text")]
    customer_id = meta.get("customer_id")

    # 1. opening disclosure
    if customer_id is not None:
        greeting = next((e["text"] for e in events
                         if e.get("type") == "agent_turn" and e.get("stage") == "greeting"),
                        agent_lines[0]["text"] if agent_lines else "")
        low = greeting.lower()
        if not greeting:
            flag("no_disclosure", "high", "the call opened with nothing spoken")
        else:
            missing = [alts[0] for alts in _DISCLOSURE_REQUIRED
                       if not any(a.lower() in low for a in alts)]
            if missing:
                flag("no_disclosure", "high",
                     "the opening did not state " + " and ".join(missing), greeting)

    # 2. threat language, and 3. figures the account does not support - both
    # replayed through the same screen the live call used
    allowed = None
    if customer_id is not None:
        from . import customers_store

        customer = customers_store.get_customer(customer_id)
        if customer is not None:
            started = meta.get("started_at")
            try:
                day = datetime.strptime(path.name[:8], "%Y%m%d").date()
            except ValueError:
                day = date.today()
            allowed = compliance.build_allowed_facts(
                customer["amount_due_paise"], cfg=cfg, today=day
            )
    language = meta.get("language") or "en"
    for e in agent_lines:
        if e.get("scripted"):
            continue  # fixed text, approved in advance
        for sentence in re.split(r"(?<=[.!?।])\s+", e["text"]):
            if not sentence.strip():
                continue
            if allowed is not None:
                decision = compliance.screen_outbound(sentence, allowed, language, cfg=cfg)
                if not decision.ok:
                    severity = "high" if decision.code == "threat" else "medium"
                    flag(decision.code, severity, decision.detail, decision.evidence, e.get("t"))
            else:
                # no account bound, so only the threat half is decidable
                threats = cfg.compliance.threats or {}
                for lang in {language, "en"}:
                    for pat in threats.get(lang, []):
                        m = re.search(pat, sentence)
                        if m and not compliance._hedged(
                            sentence, m.start(), m.end(),
                            threats.get("hedge_markers_" + lang, []),
                            int(threats.get("hedge_window_chars", 60)),
                        ):
                            flag("threat", "high",
                                 "the agent asserted a consequence it may not state",
                                 m.group(0), e.get("t"))

    # 4. calling window, recomputed from when the call actually started
    if customer_id is not None:
        from . import customers_store

        customer = customers_store.get_customer(customer_id)
        if customer is not None:
            try:
                started = datetime.strptime(path.name[:15], "%Y%m%d_%H%M%S").astimezone()
                decision = compliance.check_calling_window(customer["timezone"], started, cfg=cfg)
                if not decision.ok:
                    flag("outside_window", "high", decision.detail, decision.evidence)
            except ValueError:
                pass

    # anything the live call already blocked is reported here too, so the
    # reviewer sees it without reading the raw transcript
    for e in events:
        if e.get("type") == "blocked_output":
            flag("blocked_in_call", "medium",
                 "a reply was blocked before it was spoken: " + str(e.get("detail", "")),
                 e.get("text", ""), e.get("t"))
    return flags


# ---- the one check that needs a model -----------------------------------------

_THIRD_PARTY_PROMPT = (
    "You are auditing a recorded debt-collection call for one thing only: did the "
    "agent discuss the debt, the balance, or the fact that an account is overdue "
    "with someone who was NOT the verified borrower? Answer with JSON only: "
    '{\"third_party_disclosure\": true|false, \"evidence\": \"<quote or empty>\"}'
)


def _transcript_for_audit(events: list[dict], flags: list[dict]) -> str:
    """A reduced transcript: what the agent said, plus only the caller turns
    already flagged.

    The auditor's question is about what the AGENT disclosed, so the caller's
    side is mostly noise - and dropping it typically halves the token count,
    which is what makes this fit a 2,000 token-per-minute ceiling at all.
    """
    flagged_times = {f.get("t") for f in flags}
    lines = []
    for e in events:
        if e.get("type") == "agent_turn" and e.get("text"):
            lines.append(f"AGENT: {e['text']}")
        elif e.get("type") == "compliance_flag":
            lines.append(f"[flag {e.get('code')}]")
        elif e.get("type") == "user_utterance" and (
            e.get("t") in flagged_times or e.get("text", "").strip()
        ):
            lines.append(f"CALLER: {e.get('text', '')}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Chars-per-token approximation, matching the convention already used for
    cost estimates. Pessimistic on Devanagari, which tokenises far worse than
    Latin on an English-tuned vocabulary."""
    ascii_n = sum(1 for c in text if ord(c) < 128)
    return int(ascii_n / 4 + (len(text) - ascii_n) / 1.5) + 8


async def audit_with_model(path: Path, cfg: AppConfig, flags: list[dict]) -> list[dict]:
    """Third-party disclosure only. Returns extra flags, or [] on any problem."""
    events = _read_events(path)
    body = _transcript_for_audit(events, flags)
    if not body.strip():
        return []
    if estimate_tokens(body) > cfg.llm_post.batch_token_budget:
        body = body[: cfg.llm_post.batch_token_budget * 3]

    from .providers import build_llm
    from .providers.base import TextDelta

    post_cfg = cfg.model_copy(update={
        "llm": cfg.llm.model_copy(update={
            "provider": cfg.llm_post.provider, "model": cfg.llm_post.model,
        })
    })
    try:
        llm = build_llm(post_cfg)
        text = ""
        async for ev in llm.stream_chat(
            [{"role": "system", "content": _THIRD_PARTY_PROMPT},
             {"role": "user", "content": body}],
            max_tokens=200, temperature=0.0,
        ):
            if isinstance(ev, TextDelta):
                text += ev.text
    except Exception as e:
        log.info("post-call audit: model check unavailable (%s)", type(e).__name__)
        return []

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if verdict.get("third_party_disclosure"):
        return [{"code": "third_party_disclosure", "severity": "high",
                 "detail": "the agent appears to have discussed the account with someone other than the borrower",
                 "evidence": str(verdict.get("evidence", ""))[:200], "t": None}]
    return []


# ---- summary ------------------------------------------------------------------

_SUMMARY_PROMPT = (
    "Summarise this collection call in one or two sentences, then give the outcome. "
    "Reply as JSON only: {\"summary\": \"...\", \"disposition\": \"promise_to_pay|callback|"
    "dispute|opt_out|refused|no_contact|verification_failed|welfare_concern|completed\"}"
)


async def summarise(path: Path, cfg: AppConfig) -> dict:
    events = _read_events(path)
    body = _transcript_for_audit(events, [])
    if not body.strip():
        return {"summary": "", "disposition": "no_contact"}

    from .providers import build_llm
    from .providers.base import TextDelta

    post_cfg = cfg.model_copy(update={
        "llm": cfg.llm.model_copy(update={
            "provider": cfg.llm_post.provider, "model": cfg.llm_post.model,
        })
    })
    llm = build_llm(post_cfg)
    text = ""
    async for ev in llm.stream_chat(
        [{"role": "system", "content": _SUMMARY_PROMPT}, {"role": "user", "content": body}],
        max_tokens=250, temperature=0.0,
    ):
        if isinstance(ev, TextDelta):
            text += ev.text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"summary": text.strip()[:400], "disposition": "completed"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"summary": text.strip()[:400], "disposition": "completed"}


# ---- the sweep ----------------------------------------------------------------

def _attempts(path: Path, job: str) -> int:
    return int(((_read_meta(path) or {}).get("post") or {}).get(job, {}).get("attempts", 0))


async def run_once(cfg: AppConfig | None = None, *, dry_run: bool = False) -> dict:
    """One pass over everything outstanding. Safe to call repeatedly."""
    cfg = cfg or get_config()
    done = {"audited": 0, "summarised": 0, "skipped": 0}

    for path in pending(cfg, "audit"):
        attempts = _attempts(path, "audit") + 1
        if dry_run:
            print(f"  would audit {path.name}")
            done["audited"] += 1
            continue
        try:
            flags = audit_locally(path, cfg)
            if cfg.llm_post.enabled:
                flags += await audit_with_model(path, cfg, flags)
            amend_meta(path, "audit", {
                "status": "done", "at": datetime.now().isoformat(timespec="seconds"),
                "attempts": attempts, "flags": flags,
            })
            done["audited"] += 1
        except Exception:
            log.exception("audit failed for %s", path.name)
            status = "error" if attempts >= cfg.llm_post.max_attempts else "pending"
            amend_meta(path, "audit", {"status": status, "attempts": attempts, "flags": []})

    if cfg.llm_post.enabled:
        for path in pending(cfg, "summary"):
            attempts = _attempts(path, "summary") + 1
            if dry_run:
                print(f"  would summarise {path.name}")
                done["summarised"] += 1
                continue
            try:
                result = await summarise(path, cfg)
                amend_meta(path, "summary", {
                    "status": "done", "at": datetime.now().isoformat(timespec="seconds"),
                    "attempts": attempts,
                    "text": str(result.get("summary", ""))[:1000],
                    "disposition": result.get("disposition", "completed"),
                })
                done["summarised"] += 1
            except Exception:
                log.info("summary failed for %s", path.name, exc_info=True)
                status = "error" if attempts >= cfg.llm_post.max_attempts else "pending"
                amend_meta(path, "summary", {"status": status, "attempts": attempts})
            # the token ceiling binds long before the request ceiling, but pace
            # anyway so a backlog cannot burst through the per-minute limit
            await asyncio.sleep(max(0.0, 60.0 / max(cfg.llm_post.rpm, 1)))
    else:
        done["skipped"] = len(pending(cfg, "summary"))
    return done


def sweep_retention(cfg: AppConfig | None = None, *, dry_run: bool = False) -> dict:
    """Delete aged audio, and much later the record itself.

    Deleting a recording leaves a marker in meta.json rather than removing the
    key. "Audio deleted per retention policy" and "this call was never
    recorded" are very different facts to anyone reviewing it later.
    """
    cfg = cfg or get_config()
    out = {"recordings": 0, "records": 0}
    if not cfg.retention.enabled and not dry_run:
        return out
    root = _log_root(cfg)
    if not root.is_dir():
        return out

    today = date.today()
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not _DIR_RE.match(d.name):
            continue
        try:
            day = datetime.strptime(d.name[:8], "%Y%m%d").date()
        except ValueError:
            continue
        age = (today - day).days

        if age >= cfg.retention.record_keep_days:
            out["records"] += 1
            if dry_run:
                print(f"  would DELETE WHOLE RECORD {d.name} ({age} days old)")
            else:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
            continue

        wav = d / "recording.wav"
        if age >= cfg.retention.recording_keep_days and wav.is_file():
            out["recordings"] += 1
            if dry_run:
                print(f"  would delete audio for {d.name} ({age} days old)")
            else:
                meta = _read_meta(d) or {}
                recording = meta.get("recording") or {}
                try:
                    wav.unlink()
                except OSError:
                    continue
                recording["deleted_at"] = datetime.now().isoformat(timespec="seconds")
                recording.pop("file", None)
                amend_meta(d, "_recording_placeholder", {})  # ensure post exists
                meta = _read_meta(d) or {}
                meta["recording"] = recording
                meta.get("post", {}).pop("_recording_placeholder", None)
                tmp = d / "meta.json.tmp"
                tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                os.replace(tmp, d / "meta.json")
    return out


async def loop(cfg: AppConfig | None = None) -> None:
    """Background sweep, started from main.startup when in_process is set."""
    cfg = cfg or get_config()
    last_retention = 0.0
    while True:
        try:
            await run_once(cfg)
            if time.time() - last_retention > 86400:
                sweep_retention(cfg)
                last_retention = time.time()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("post-call sweep failed")
        await asyncio.sleep(cfg.llm_post.poll_seconds)
