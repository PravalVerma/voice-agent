"""Spawn the collections tool server, list its tools, and exercise them.

Usage: uv run python scripts/check_mcp.py --customer 1 [--opt-out]

Prints each tool's converted schema and what it costs in prompt tokens, since
that list is sent on every NEGOTIATE turn.

--opt-out runs the loop the whole compliance design turns on: register the
opt-out through the tool, then ask the pre-call gate whether the next call may
be placed. It must refuse. This WRITES to the database - the customer really is
opted out afterwards.
"""

import argparse
import asyncio
import json
import sys

from _bootstrap import load_config_or_die

from app import customers_store
from app.compliance import check_contact_eligibility
from app.config import ConfigError
from app.mcp_client import McpClient


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", type=int, required=True, help="customer id to bind the server to")
    parser.add_argument("--opt-out", action="store_true", help="register an opt-out and prove the next call is refused")
    parser.add_argument("--allow", help="comma-separated tool allowlist to enforce (default: all allowed)")
    args = parser.parse_args()

    cfg = load_config_or_die()
    customer = customers_store.get_customer(args.customer)
    if customer is None:
        sys.exit(f"error: no customer with id {args.customer} - add one from Advanced Settings first")
    print(f"customer: {customer['name']} ({customer['customer_ref']}) outstanding Rs {customer['rupees']}")

    try:
        schemas = await McpClient.warm()
    except ConfigError as e:
        sys.exit(f"error: {e}")
    if not schemas:
        sys.exit("error: the tool server did not start - run it directly to see why: uv run python -m mcp_server")

    blob = json.dumps(schemas, ensure_ascii=False)
    print(f"\n{len(schemas)} tools declared, ~{len(blob) // 3} prompt tokens per request:")
    for s in schemas:
        fn = s["function"]
        params = ", ".join(fn["parameters"].get("properties", {}))
        print(f"  {fn['name']}({params}) - {fn['description']}")
        for leaked in ("customer_id", "session_id", "phone"):
            if leaked in fn["parameters"].get("properties", {}):
                sys.exit(f"error: {fn['name']} exposes {leaked} as a parameter - it must be injected server-side")
    print("  (no tool exposes customer_id, session_id or phone - all injected server-side)")

    allowed = [t.strip() for t in args.allow.split(",")] if args.allow else None
    client = McpClient(args.customer, "checkmcp", allowed=allowed)
    await client.connect()
    if not client.available:
        sys.exit("error: could not start a bound tool server")

    try:
        print("\nverify_identity with a wrong answer:")
        print("  ->", await client.call("verify_identity", {"answer": "not the right answer"}))

        print("record_promise_to_pay with a date in the past:")
        print("  ->", await client.call("record_promise_to_pay", {"date": "2020-01-01", "amount_rupees": 1000}))

        print("record_promise_to_pay below the settlement floor:")
        low = round(customer["rupees"] * 0.10, 2)
        print("  ->", await client.call("record_promise_to_pay", {"date": "2026-11-05", "amount_rupees": low}))

        if allowed is not None:
            print("a tool outside the allowlist:")
            print("  ->", await client.call("end_call", {"disposition": "completed"}))

        if args.opt_out:
            print("\n-- opt-out loop --")
            before = check_contact_eligibility(args.customer, cfg=cfg)
            print(f"  eligibility before: {'PASS' if before.ok else before.code}")
            # register_opt_out is not declared on the 3-tool server yet; write it
            # through the same policy function the tool will call.
            from mcp_server import policy

            print("  register_opt_out ->", policy.register_opt_out(args.customer, "checkmcp"))
            after = check_contact_eligibility(args.customer, cfg=cfg)
            print(f"  eligibility after:  {'PASS' if after.ok else after.code} - {after.detail}")
            if after.ok:
                sys.exit("error: the opt-out did not block the next call - the loop is broken")
            print("  OK - the next call to this customer is refused before dialling.")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
