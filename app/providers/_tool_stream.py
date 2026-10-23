"""Reassembly of streamed tool calls.

Groq and OpenAI emit a tool call as a series of fragments keyed by index: the
first carries the id and function name, later ones carry slices of the argument
JSON and nothing else. Both use the identical shape, so the accumulator lives
here once rather than in each provider.

Calls are only emitted once the stream ends. `arguments` is not parseable until
every fragment has arrived, and a half-received call is indistinguishable from
a complete one while it is still growing.
"""

from .base import ToolCall


class ToolCallAccumulator:
    def __init__(self) -> None:
        self._by_index: dict[int, dict] = {}

    def feed(self, fragments) -> None:
        for frag in fragments or ():
            index = getattr(frag, "index", 0) or 0
            slot = self._by_index.setdefault(index, {"id": None, "name": None, "args": []})
            # The guards matter: id and name arrive on the first fragment and
            # are None on every one after it, so an unguarded assignment would
            # wipe them out partway through the call.
            if getattr(frag, "id", None):
                slot["id"] = frag.id
            fn = getattr(frag, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"].append(fn.arguments)

    def drain(self) -> list[ToolCall]:
        calls = []
        for index, slot in sorted(self._by_index.items()):
            if not slot["name"]:
                continue  # fragments arrived but the call never named a function
            calls.append(ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments="".join(slot["args"]) or "{}",
            ))
        self._by_index.clear()
        return calls

    def __bool__(self) -> bool:
        return bool(self._by_index)
