"""Standalone tool server for collections calls, spoken to over MCP.

Runs as its own process, one per call, launched over stdio by app/mcp_client.py.

**How a tool knows whose account it is acting on.** It does not ask, and the
model is never able to say. The customer id, the phone number and the session
id are bound into this process at spawn time from the environment
(VCA_CUSTOMER_ID, VCA_SESSION_ID) and are read once, here, into module state.
No tool takes them as a parameter and no schema mentions them. There is nothing
for a model to forge, no identifier to confuse with another call's, and no
lookup table to poison - the binding is a property of the process, and the
process dies with the call.

That is also why there is no get_account_details tool: the facts are known
before the call starts and are injected once into the first message. A tool
round-trip would spend a request and about a second of silence fetching data
already in hand.

This package may import app.db, app.config and app.compliance - one schema and
one set of rules, rather than a second copy that drifts. It must never import
app.main, which would construct a FastAPI application inside a subprocess.
"""
