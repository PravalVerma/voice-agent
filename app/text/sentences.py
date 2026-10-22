"""Incremental token -> sentence chunker, for streaming LLM output to TTS."""

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "e.g", "i.e", "etc", "approx", "no", "inc", "ltd",
}
_BOUNDARY_CHARS = ".!?…"  # . ! ? …


def _word_before(text: str, idx: int) -> str:
    j = idx
    while j > 0 and not text[j - 1].isspace():
        j -= 1
    return text[j:idx]


class SentenceAssembler:
    """Feed LLM token deltas in; get complete sentences out as soon as it's
    safe to speak them. The first sentence of a turn flushes at the very
    first boundary found, however short, for the fastest possible first
    audio. Later sentences need >= min_chars so short interjections like
    "Sure." don't each trigger their own TTS call.
    """

    def __init__(self, min_chars: int = 25):
        self._buf = ""
        self._is_first = True
        self._min_chars = min_chars

    def feed(self, token: str) -> list[str]:
        self._buf += token
        sentences = []
        while True:
            sentence = self._pop_sentence()
            if sentence is None:
                break
            sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        """Call once the source stream has ended to collect any trailing text."""
        text = self._buf.strip()
        self._buf = ""
        if not text:
            return None
        self._is_first = False
        return text

    def _pop_sentence(self) -> str | None:
        buf = self._buf
        min_len = 1 if self._is_first else self._min_chars
        for i, ch in enumerate(buf):
            if ch == "\n":
                candidate = buf[:i].strip()
                if candidate and len(candidate) >= min_len:
                    self._buf = buf[i + 1 :].lstrip()
                    self._is_first = False
                    return candidate
                continue
            if ch not in _BOUNDARY_CHARS:
                continue
            nxt = buf[i + 1] if i + 1 < len(buf) else None
            if nxt is None:
                continue  # boundary is at the buffer's current end - could still grow ("3.5"); wait
            if not nxt.isspace():
                continue  # mid-word/number punctuation ("3.5", "e.g.x") - not a sentence boundary
            if ch == "." and i > 0 and buf[i - 1].isdigit():
                continue  # decimal point
            if _word_before(buf, i).lower().rstrip(".") in _ABBREVIATIONS:
                continue
            candidate = buf[: i + 1].strip()
            if len(candidate) >= min_len:
                self._buf = buf[i + 1 :].lstrip()
                self._is_first = False
                return candidate
        return None
