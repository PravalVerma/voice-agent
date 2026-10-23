"""Run the post-call jobs once: compliance audit, then summary.

Usage: uv run python scripts/run_post_jobs.py [--once] [--dry-run] [--retention]

The same sweep the server runs in the background when [llm_post] in_process is
true. Set that to false and schedule this instead if the work should not share
the web process.

--retention additionally ages out old audio. That is destructive, so pair it
with --dry-run first.
"""

import argparse
import asyncio

from _bootstrap import load_config_or_die

from app import post_call


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one pass (the default)")
    parser.add_argument("--dry-run", action="store_true", help="report what would happen, change nothing")
    parser.add_argument("--retention", action="store_true", help="also run the retention sweep")
    args = parser.parse_args()

    cfg = load_config_or_die()

    audit = post_call.pending(cfg, "audit")
    summary = post_call.pending(cfg, "summary")
    print(f"pending: {len(audit)} to audit, {len(summary)} to summarise")

    result = await post_call.run_once(cfg, dry_run=args.dry_run)
    print(f"audited {result['audited']}, summarised {result['summarised']}"
          + (f", skipped {result['skipped']} (post-call model disabled)" if result["skipped"] else ""))

    if args.retention:
        if not cfg.retention.enabled and not args.dry_run:
            print("retention is disabled in config.toml - nothing deleted")
        else:
            counts = post_call.sweep_retention(cfg, dry_run=args.dry_run)
            verb = "would delete" if args.dry_run else "deleted"
            print(f"{verb} {counts['recordings']} recording(s) and {counts['records']} whole record(s)")


if __name__ == "__main__":
    asyncio.run(main())
