"""Measure Groq's prompt-cache floor: the smallest fixed prefix that actually caches.

The collections pipeline freezes its system prompt + tool schemas so Groq can
serve them from cache (cached tokens don't count against the rate limits). That
only pays off above the model's minimum cacheable prompt length, which Groq
documents as varying from 128 to 1024 tokens without publishing a figure per
model - so measure it rather than assume it.

Sends the same prefix twice at each size, with a different tail each time, and
reports cached_tokens from the second request.

Usage: uv run python scripts/check_cache.py [--sizes 300,600,1100] [--model openai/gpt-oss-20b]
"""

import argparse
import asyncio

from groq import RateLimitError

from _bootstrap import build_or_die, load_config_or_die, need_env

from app.providers import build_llm

# ~4 chars/token of ordinary English prose. Deterministic, so the prefix is
# byte-identical between the two requests of a pair - which is the whole point.
_FILLER = (
    "The agent follows the lender's documented recovery policy at all times. "
    "It keeps every reply short, states only figures it has been given, and "
    "never speculates about consequences it cannot confirm. "
)


def _prefix_of_about(n_tokens: int) -> str:
    target_chars = n_tokens * 4
    reps = max(1, target_chars // len(_FILLER) + 1)
    return (_FILLER * reps)[:target_chars]


async def _probe(llm, prefix: str, tail: str) -> tuple[int, int]:
    """One request. Returns (prompt_tokens, cached_tokens)."""
    messages = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": tail},
    ]
    async for _ in llm.stream_chat(messages, max_tokens=8, temperature=0.0):
        pass
    if llm.last_usage is None:
        return 0, 0
    return llm.last_usage.input_tokens, llm.last_usage.cached_tokens


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="300,600,1100", help="approx prefix sizes in tokens (default: 300,600,1100)")
    parser.add_argument("--model", help="override llm.model from config.toml")
    # the free tier is 8000 TPM and a probe pair spends 2x the prefix, so back-to-back
    # sizes trip a 429 well before the interesting sizes are reached
    parser.add_argument("--delay", type=float, default=20.0, help="seconds to wait between sizes (default: 20)")
    args = parser.parse_args()

    cfg = load_config_or_die()
    if args.model:
        cfg.llm.model = args.model
    if cfg.llm.provider != "groq":
        raise SystemExit(f"error: this check only applies to llm.provider 'groq' (config says {cfg.llm.provider!r})")
    need_env("GROQ_API_KEY", "Groq LLM")

    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        raise SystemExit("error: --sizes must be a comma-separated list of integers")

    llm = build_or_die(build_llm, cfg)
    print(f"model: {cfg.llm.model}")
    print(f"{'target':>8} {'prompt_tok':>11} {'cached_1st':>11} {'cached_2nd':>11}  verdict")

    floor = None
    for i, n in enumerate(sizes):
        if i:
            await asyncio.sleep(args.delay)
        prefix = _prefix_of_about(n)
        # different tails: we are measuring PREFIX reuse, not whole-request reuse
        try:
            tokens, cached_first = await _probe(llm, prefix, "Reply with the single word A.")
            _, cached_second = await _probe(llm, prefix, "Reply with the single word B.")
        except RateLimitError as e:
            raise SystemExit(f"error: hit the per-minute token limit at size {n} - raise --delay and retry ({e.body['error']['message'][:80]}...)")
        hit = cached_second > 0
        if hit and floor is None:
            floor = n
        print(f"{n:>8} {tokens:>11} {cached_first:>11} {cached_second:>11}  {'CACHED' if hit else 'no cache'}")

    print()
    if floor is None:
        print("no cache hit at any size tested - freezing the prefix buys nothing here.")
        print("the binding constraint is raw token count, so shrink the prompt instead.")
    else:
        print(f"cache floor is at or below ~{floor} tokens.")
        print("size the fixed prefix above this; a smaller prefix that never caches is worse than a larger one that does.")


if __name__ == "__main__":
    asyncio.run(main())
