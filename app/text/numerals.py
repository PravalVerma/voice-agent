"""Spoken-number parsing, for the compliance output filter and identity checks.

The agent is told to speak every amount as words, never digits (see the number
rules in config.py), so "did it say a figure it was not given?" cannot be
answered by looking for digits. This turns written-out numbers back into
integers so they can be compared against the facts the call was actually given.

Two lexicons, one accumulator:

- English is regular and compositional, so the lexicon here is complete: ~45
  words plus scale terms, and anything expressible is parseable.
- Hindi 1-99 are irregular lexical items - "painsath" (65) and "chhiyaanve"
  (96) are not built from smaller words the way "sixty-five" is. There is no
  algorithm for them, only a table, and the table lives in config.toml
  ([compliance.numerals.hi]) so it can be extended without a code change. It
  is seeded with the scale words and the Devanagari 1-100 set and is expected
  to grow from real blocked_output evidence - romanised Hindi in particular has
  several spellings per number and no authority to enumerate them.

Parsing is therefore best-effort by construction. Callers must treat a span
that tokenises as numeric but yields no value as *unresolved*, not as absent -
see compliance.screen_outbound, which blocks an unresolved span only when it
sits next to a currency or date marker.
"""

import re
import unicodedata

#: Words that may appear inside a number without contributing a value.
_FILLERS = frozenset({"and", "aur", "और", "व"})

_UNITS_EN = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS_EN = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# "hundred" multiplies what precedes it; the rest flush it into the running
# total. Indian scales are included because the calls are Indian-market and the
# agent says "lakh"/"crore" far more often than "million".
_SCALES_EN = {
    "hundred": 100, "thousand": 1000, "lakh": 100_000, "lakhs": 100_000,
    "lac": 100_000, "lacs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
}

#: Complete English lexicon. Hindi is supplied by the caller from config.
EN_LEXICON: dict[str, int] = {**_UNITS_EN, **_TENS_EN, **_SCALES_EN}

# The explicit Devanagari range is load-bearing and not redundant with \w.
# Python's \w follows str.isalnum(), which is False for combining marks - every
# matra and the nukta. Relying on \w alone shreds "हज़ार" into "हज" + "र" and
# silently breaks the entire Hindi path, since no fragment is ever in the
# lexicon. Digits are matched first so a bare "65" is never taken for a word.
_TOKEN_RE = re.compile(r"\d+|[\wऀ-ॿ]+", re.UNICODE)


def normalize(text: str) -> str:
    """NFC, so one spelling of a Devanagari word matches another.

    "ज़" reaches us either precomposed (U+095B) or as "ज" + nukta, depending on
    what produced the text. Devanagari's precomposed forms are Unicode
    composition exclusions, so NFC settles every input on the decomposed form -
    which makes it a consistent comparison key as long as BOTH the lexicon and
    the text being scanned go through here.
    """
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """(token, start, end) for every word or digit run, lowercased.

    Offsets index the string as given, so callers can slice it directly.
    """
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def combine(values: list[int]) -> int | None:
    """Fold a run of number-word values into one integer.

    Handles both the Western and Indian grouping systems with the same rule:
    "hundred" multiplies the pending value, any larger scale flushes it into
    the total. "two lakh sixty five thousand" -> 2*100000 + 65*1000 = 265000.
    """
    if not values:
        return None
    total = 0
    current = 0
    for value in values:
        if value == 100:
            current = (current or 1) * 100
        elif value >= 1000:
            total += (current or 1) * value
            current = 0
        else:
            current += value
    return total + current


def find_word_numbers(text: str, lexicon: dict[str, int]) -> list[tuple[int, str, int, int]]:
    """Every written-out number in `text` as (value, span_text, start, end).

    A span runs while tokens keep resolving in `lexicon`, hopping over fillers
    like "and". Trailing fillers are not included in the reported span.
    """
    # Normalised per token rather than over the whole string, so the offsets
    # tokenize() produced still index the caller's text exactly.
    tokens = [(normalize(t), a, b) for t, a, b in tokenize(text)]
    out: list[tuple[int, str, int, int]] = []
    i = 0
    while i < len(tokens):
        if tokens[i][0] not in lexicon:
            i += 1
            continue
        values = []
        start = tokens[i][1]
        end = tokens[i][2]
        j = i
        while j < len(tokens):
            token = tokens[j][0]
            if token in lexicon:
                values.append(lexicon[token])
                end = tokens[j][2]
                j += 1
            elif token in _FILLERS and j + 1 < len(tokens) and tokens[j + 1][0] in lexicon:
                j += 1  # "one hundred and five" - skip the filler, keep the span open
            else:
                break
        value = combine(values)
        if value is not None:
            out.append((value, text[start:end], start, end))
        i = j
    return out


def has_numeric_token(text: str, lexicon: dict[str, int]) -> bool:
    """Cheap pre-check: does this text contain a digit or any number word?

    The outbound filter runs on every sentence the agent is about to speak, so
    the common case - no number at all - must not pay for a full parse.
    """
    for token, _, _ in tokenize(text):
        if token.isdigit() or normalize(token) in lexicon:
            return True
    return False


#: Ordinals a date is spoken with. The regular ones are reached by stripping
#: the suffix; these are the forms that change the stem.
_IRREGULAR_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fifth": 5, "eighth": 8, "ninth": 9,
    "twelfth": 12, "twentieth": 20, "thirtieth": 30,
}
_ORDINAL_SUFFIX = re.compile(r"^(.*?)(?:st|nd|rd|th)$")


def _ordinal_value(token: str, lexicon: dict[str, int]) -> int | None:
    if token in _IRREGULAR_ORDINALS:
        return _IRREGULAR_ORDINALS[token]
    m = _ORDINAL_SUFFIX.match(token)
    if m:
        stem = m.group(1)
        # "fifteenth" -> "fifteen", "twentieth" handled above, "sixtieth" -> "sixty"
        for candidate in (stem, stem + "e", stem[:-1] + "y" if stem.endswith("ie") else stem):
            if candidate in lexicon:
                return lexicon[candidate]
    return None


def spoken_values(text: str, lexicon: dict[str, int]) -> set[int]:
    """Every integer a sentence could be stating, generously.

    Built for comparison rather than extraction: it returns all plausible
    readings instead of committing to one, so a caller checking "is 1988 in
    here?" gets a yes whether the year was said as digits, as "nineteen eighty
    eight", or as "nineteen eighty-eight".

    This is why it is separate from find_word_numbers(), which must commit to a
    single value because the compliance screen compares amounts exactly. Years
    break that function's rule deliberately - "nineteen ninety" is 1990 as a
    year and 109 as a sum, and only the caller knows which it is reading.
    """
    values: set[int] = set()
    tokens = [normalize(t) for t, _, _ in tokenize(text)]

    for token in tokens:
        if token.isdigit():
            values.add(int(token))
        # Each word on its own, as well as the span it belongs to. "March
        # fifteen nineteen eighty-eight" is one contiguous run of number words,
        # so the span reading is 15+19+80+8; only the per-token reading
        # recovers the 15 that is actually the day.
        if token in lexicon:
            values.add(lexicon[token])
        ordinal = _ordinal_value(token, lexicon)
        if ordinal is not None:
            values.add(ordinal)

    # whole word-number spans, as find_word_numbers reads them
    for value, _, _, _ in find_word_numbers(text, lexicon):
        values.add(value)

    # year readings: two adjacent values spoken as centuries, "nineteen" +
    # "eighty eight" -> 1988. Also accepts the parts arriving separately.
    numeric = [lexicon[t] for t in tokens if t in lexicon]
    for i in range(len(numeric) - 1):
        head = numeric[i]
        if not (10 <= head <= 99):
            continue
        tail = numeric[i + 1]
        if 0 <= tail <= 99:
            values.add(head * 100 + tail)
        # "nineteen eighty eight" arrives as 19, 80, 8
        if i + 2 < len(numeric) and tail < 100:
            third = numeric[i + 2]
            if third < 10 and tail % 10 == 0:
                values.add(head * 100 + tail + third)
    return values


def digit_sequence(text: str, lexicon: dict[str, int]) -> str:
    """Read the text as a run of individual digits, in order.

    People read account numbers, PINs and phone numbers digit by digit -
    "three two one zero" - and the ordinary number parse sums that run to 6.
    This is the other reading of the same words, so a caller can check both.
    """
    out = []
    for token, _, _ in tokenize(text):
        if token.isdigit():
            out.append(token)
            continue
        value = lexicon.get(normalize(token))
        if value is not None and 0 <= value <= 9:
            out.append(str(value))
    return "".join(out)


def spell_to_digits(text: str, lexicon: dict[str, int]) -> str:
    """Replace written-out numbers with their digits, leaving the rest alone.

    Used when checking a spoken answer against a stored one (a date of birth
    read aloud, the last four digits of a phone number) - the caller may say
    either form and STT transcribes whichever they said.

    Years are NOT handled here: spoken years use a different rule from amounts
    ("nineteen ninety" is 1990, not 19+90), and applying that rule generally
    would turn "twenty five" into 2005. A caller comparing a spoken date of
    birth needs a date-aware parse on top of this, not instead of it.
    """
    spans = find_word_numbers(text, lexicon)
    out = []
    cursor = 0
    for value, _, start, end in spans:
        out.append(text[cursor:start])
        out.append(str(value))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
