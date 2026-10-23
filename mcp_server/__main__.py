"""The tool server process. Launched over stdio, one per call.

Usage: uv run python -m mcp_server
  VCA_CUSTOMER_ID and VCA_SESSION_ID must be set; they bind this process to one
  call and are the only way a tool learns whose account it is acting on.

Tool descriptions are one terse line each. They are sent to the model on every
turn of every NEGOTIATE exchange, so each extra clause is a cost paid
repeatedly against a per-minute token budget - not a one-off.
"""

import logging
import os
import sys

from mcp.server.mcpserver import MCPServer

from app.customers_store import get_verify_value
from mcp_server import policy

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)-7s mcp_server: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mcp_server")


def _bound(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"error: {name} is not set - this server must be launched bound to one call")
    return value


CUSTOMER_ID = int(_bound("VCA_CUSTOMER_ID"))
SESSION_ID = _bound("VCA_SESSION_ID")

server = MCPServer("voice-agent-tools")


@server.tool(description="Check the caller's answer to the identity question.")
def verify_identity(answer: str) -> dict:
    """Returns whether the answer matched. Never returns the stored value.

    The correct answer is read here and compared here; it is never returned,
    never echoed in an error, and never reaches the model. strikes_left is the
    only thing that comes back besides the verdict, and it carries no
    information about what the answer is.
    """
    stored = get_verify_value(CUSTOMER_ID)
    if stored is None:
        return {"verified": False}
    # Imported lazily: app.collections_flow pulls in the pipeline, which the
    # tool process has no use for beyond this one comparison.
    from app.collections_flow import verify_answer

    return {"verified": bool(verify_answer(stored[0], stored[1], answer or ""))}


@server.tool(description="Record a commitment to pay an amount on a date.")
def record_promise_to_pay(date: str, amount_rupees: float) -> dict:
    return policy.record_promise(CUSTOMER_ID, SESSION_ID, date, amount_rupees)


@server.tool(description="End the call with an outcome.")
def end_call(disposition: str) -> dict:
    allowed = {"promise_to_pay", "refused", "dispute", "callback", "no_contact", "completed"}
    if disposition not in allowed:
        return {"ok": False, "error": "unknown disposition"}
    return {"ok": True, "disposition": disposition}


if __name__ == "__main__":
    log.info("bound to customer %s, session %s", CUSTOMER_ID, SESSION_ID)
    server.run(transport="stdio")
