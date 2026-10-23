"""Client for the collections tool server.

Two lifetimes, and keeping them apart is the whole design:

- **Schemas** are listed once per *process* and cached for its lifetime. They
  are identical for every bot and every call, so re-listing them per call would
  spend time discovering a constant.
- **Execution** gets its own server process per *call*, bound at spawn to one
  customer and session through the environment. That binding is why no tool
  takes a customer id: there is nothing for the model to supply, and nothing it
  could supply would be believed.

The per-bot allowlist is enforced here, at execution, and never by sending a
bot a different set of schemas. Varying the declared tools per bot would make
the prompt vary per bot for no behavioural gain.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack

from .config import ConfigError, ROOT

log = logging.getLogger("agent")

#: Converted once per process. See the module docstring.
_SCHEMAS: list[dict] | None = None
_SCHEMA_LOCK = asyncio.Lock()

#: Tool results are fed back into the prompt, so an unbounded one would blow
#: the per-turn budget on its own.
_MAX_RESULT_CHARS = 400


def _import_mcp():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ModuleNotFoundError as e:
        raise ConfigError(
            "collections mode needs the tool-server runtime: uv sync --extra collections"
        ) from e
    return ClientSession, StdioServerParameters, stdio_client


def _server_params(StdioServerParameters, customer_id: int, session_id: str):
    """Launch parameters, with the call binding passed through the environment.

    sys.executable rather than "uv run" so the child is the interpreter already
    running this process - it inherits the resolved virtualenv instead of
    re-resolving one, which matters when the server is started from a systemd
    unit with no uv on PATH.
    """
    env = dict(os.environ)
    env["VCA_CUSTOMER_ID"] = str(customer_id)
    env["VCA_SESSION_ID"] = session_id
    return StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"], env=env, cwd=str(ROOT)
    )


def _to_openai_schema(tool) -> dict:
    """MCP inputSchema is already JSON Schema, so this is a rename, not a translation."""
    # mcp 2.x exposes input_schema; 1.x called it inputSchema. Accept either so
    # the client is not tied to one minor of the SDK.
    raw = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    params = raw or {"type": "object", "properties": {}}
    if params.get("type") != "object":
        raise ConfigError(
            f"tool {tool.name!r} has a non-object parameter schema, which the model API rejects"
        )
    # Groq rejects these alongside a function definition.
    params = {k: v for k, v in params.items() if k not in ("$schema", "additionalProperties")}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": params,
        },
    }


class McpClient:
    """One live tool-server process, bound to one call."""

    def __init__(self, customer_id: int, session_id: str, allowed: list[str] | None = None):
        self.customer_id = customer_id
        self.session_id = session_id
        #: None means every tool is allowed, matching user_bots.allowed_tiers.
        self.allowed = allowed
        self._stack: AsyncExitStack | None = None
        self._session = None
        self.available = False

    # ---- schemas (process-wide) ----------------------------------------------

    @classmethod
    async def warm(cls) -> list[dict]:
        """List the tools once and cache their converted schemas.

        Uses a throwaway binding: this connection is only ever asked what tools
        exist, which does not depend on whose account is bound.
        """
        global _SCHEMAS
        async with _SCHEMA_LOCK:
            if _SCHEMAS is not None:
                return _SCHEMAS
            probe = cls(customer_id=0, session_id="schemawarm")
            await probe.connect()
            try:
                if not probe.available:
                    return []
                tools = (await probe._session.list_tools()).tools
                _SCHEMAS = [_to_openai_schema(t) for t in tools]
                log.info("tool schemas cached: %s", [t.name for t in tools])
                return _SCHEMAS
            finally:
                await probe.aclose()

    @classmethod
    def schemas(cls) -> list[dict]:
        """The cached schema list, or [] if warming never succeeded."""
        return list(_SCHEMAS or [])

    # ---- connection ----------------------------------------------------------

    async def connect(self) -> None:
        """Spawn the server. Never raises: a tool server that will not start
        must degrade the call, not end it."""
        ClientSession, StdioServerParameters, stdio_client = _import_mcp()
        self._stack = AsyncExitStack()
        try:
            params = _server_params(StdioServerParameters, self.customer_id, self.session_id)
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            self.available = True
        except Exception:
            log.exception("tool server failed to start - the call will run without tools")
            await self.aclose()

    async def aclose(self) -> None:
        self.available = False
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                log.debug("tool server shutdown was not clean", exc_info=True)
            self._stack = None
        self._session = None

    # ---- dispatch ------------------------------------------------------------

    async def call(self, name: str, arguments: dict, timeout: float = 2.5) -> dict:
        """Run one tool. Always returns a dict; never raises into the turn."""
        if self.allowed is not None and name not in self.allowed:
            # Phrased as a capability, not a permission. Told it is "not
            # allowed", the model apologises to the caller and sometimes
            # suggests they ring back; told the tool is unavailable, it simply
            # carries on with the conversation.
            return {"ok": False, "error": "not available on this call"}
        if not self.available or self._session is None:
            return {"ok": False, "error": "not available right now"}

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments or {}), timeout=timeout
            )
        except asyncio.TimeoutError:
            log.warning("tool %s timed out after %.1fs", name, timeout)
            return {"ok": False, "error": "took too long"}
        except Exception:
            log.exception("tool %s failed", name)
            return {"ok": False, "error": "not available right now"}

        return _unwrap(result)


def _unwrap(result) -> dict:
    """Pull a plain dict out of an MCP tool result.

    The error flag is checked first and is not optional. A failing tool returns
    its message as ordinary text content, so treating text as success would
    hand the model {"ok": true, "text": "Unknown tool: ..."} and have it tell
    the caller the action succeeded.
    """
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)

    data = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if isinstance(data, dict) and not is_error:
        # MCPServer wraps a bare return value under "result"
        inner = data.get("result")
        return inner if isinstance(inner, dict) else data

    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        if is_error:
            log.warning("tool returned an error: %s", text[:200])
            return {"ok": False, "error": "could not be completed"}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"ok": True, "text": text[:_MAX_RESULT_CHARS]}
    return {"ok": False, "error": "no result"}


def truncate_result(result: dict) -> str:
    """Serialise a tool result for the model, hard-capped.

    A single unbounded result would otherwise spend the whole per-turn token
    budget on its own.
    """
    return json.dumps(result, ensure_ascii=False)[:_MAX_RESULT_CHARS]
