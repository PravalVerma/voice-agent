"""Run the in-call compliance screens over a sentence, without placing a call.

Usage: uv run python scripts/check_compliance.py "sentence" [--language hi] [--amount 265000] [--direction out]

--amount is the account's outstanding balance in RUPEES; the allowed figures
are derived from it the same way a real call derives them, so a settlement
percentage of it passes while an invented figure does not.

This is the main tool for growing the Hindi numeral lexicon: feed it a line
that was wrongly blocked, and the evidence field names the span that failed to
resolve.
"""

import argparse
from datetime import date

from _bootstrap import load_config_or_die

from app.compliance import build_allowed_facts, extract_figures, screen_inbound, screen_outbound
from app.compliance import _lexicon  # noqa: E402  - deliberately exercising the real lexicon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", help="the text to screen")
    parser.add_argument("--language", default="en", help="call language code (default: en)")
    parser.add_argument("--amount", type=float, default=265000.0, help="outstanding balance in rupees (default: 265000)")
    parser.add_argument("--direction", choices=("out", "in", "both"), default="both",
                        help="which screen to run (default: both)")
    args = parser.parse_args()

    cfg = load_config_or_die()
    facts = build_allowed_facts(int(round(args.amount * 100)), cfg=cfg, today=date.today())

    if args.direction in ("out", "both"):
        figures, residue = extract_figures(args.sentence, _lexicon(cfg))
        print("outbound:")
        for f in figures:
            shown = f.value / 100 if f.kind == "amount" else f.value
            print(f"  figure {f.kind:8} {str(shown):>14}  from {f.text!r}")
        for r in residue:
            print(f"  unresolved {r!r}")
        decision = screen_outbound(args.sentence, facts, args.language, cfg=cfg)
        if decision.ok:
            print("  PASS")
        else:
            print(f"  BLOCK {decision.code} - {decision.detail}")
            print(f"  evidence: {decision.evidence!r}")

    if args.direction in ("in", "both"):
        flags = screen_inbound(args.sentence, args.language, cfg=cfg)
        print("inbound:")
        if not flags:
            print("  no flags")
        for fl in flags:
            print(f"  {fl.severity:6} {fl.code:14} evidence: {fl.evidence!r}")


if __name__ == "__main__":
    main()
