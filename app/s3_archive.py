"""Optional S3 archival for call recordings.

Entirely opt-in: with S3_RECORDINGS_BUCKET unset the app behaves exactly as it
did before, recordings just stay on local disk. Uploads are best-effort and
never propagate - losing the archive copy of a recording must not fail a call
that has already completed successfully.
"""

import logging
import os
import re
import threading

log = logging.getLogger("agent")

_client = None
_client_lock = threading.Lock()

# Top-level folder inside the bucket. Overridable per deployment with
# S3_RECORDINGS_PREFIX.
DEFAULT_PREFIX = "arcil"

# S3 keys are far easier to work with (and to paste into a shell) without
# spaces, "@" or "/" in them - usernames here are email addresses.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def bucket() -> str | None:
    return os.environ.get("S3_RECORDINGS_BUCKET", "").strip() or None


def is_configured() -> bool:
    return bucket() is not None


def _safe(value: str | None, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", (value or "").strip()).strip("_")
    return cleaned or fallback


def build_key(dir_name: str, username: str | None, language: str | None, tier: str | None) -> str:
    """Recording key, e.g.

        arcil/2026/07/27/alex_example.com_hi_Core_20260727_114856_442f3c09.wav

    Date-partitioned so a day's calls can be listed/lifecycled without scanning
    the bucket, and the filename alone identifies who called, in which language,
    on which voice tier, when, and which session - so a downloaded file is still
    identifiable once it's detached from the console.
    """
    prefix = os.environ.get("S3_RECORDINGS_PREFIX", DEFAULT_PREFIX).strip("/") or DEFAULT_PREFIX
    # dir_name is "{yyyymmdd}_{hhmmss}_{session_id}" (validated in call_logs.py)
    yyyy, mm, dd = dir_name[0:4], dir_name[4:6], dir_name[6:8]
    name = f"{_safe(username, 'unknown')}_{_safe(language, 'xx')}_{_safe(tier, 'na')}_{dir_name}.wav"
    return f"{prefix}/{yyyy}/{mm}/{dd}/{name}"


def _imds_region() -> str | None:
    """Region from the EC2 instance metadata service (IMDSv2).

    boto3 finds the instance role's *credentials* on its own but does not
    resolve the *region* from IMDS by default, so on a box with only an
    attached role and no AWS_REGION / ~/.aws/config the client would be built
    region-less and fail to sign S3 requests. Kept deliberately short-timeout
    so a non-EC2 host (or a blocked IMDS) costs a fraction of a second, once.
    """
    import urllib.error
    import urllib.request

    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_req, timeout=1) as resp:
            token = resp.read().decode()
        region_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(region_req, timeout=1) as resp:
            return resp.read().decode().strip() or None
    except Exception:
        return None


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            import boto3  # deferred: only paid for when S3 is actually configured

            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or boto3.Session().region_name
                or _imds_region()
            )
            _client = boto3.client("s3", region_name=region or None)
            log.info("s3: client ready (region=%s, bucket=%s)", region, bucket())
        return _client


def upload(data: bytes, key: str) -> str | None:
    """Upload and return the s3:// URI, or None if S3 isn't configured or the
    upload failed. Never raises."""
    target = bucket()
    if not target:
        return None
    try:
        _get_client().put_object(
            Bucket=target, Key=key, Body=data, ContentType="audio/wav"
        )
    except Exception:
        log.exception("s3: failed to upload recording to s3://%s/%s", target, key)
        return None
    log.info("s3: uploaded recording to s3://%s/%s (%.1f KB)", target, key, len(data) / 1024)
    return f"s3://{target}/{key}"


def presigned_url(key: str, expires_s: int = 3600) -> str | None:
    target = bucket()
    if not target:
        return None
    try:
        return _get_client().generate_presigned_url(
            "get_object", Params={"Bucket": target, "Key": key}, ExpiresIn=expires_s
        )
    except Exception:
        log.exception("s3: failed to presign s3://%s/%s", target, key)
        return None
