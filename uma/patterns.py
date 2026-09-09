from __future__ import annotations

import re


# Match common scale formats.
SCALE_RE = re.compile(r"""
      \bSCALE\b
    | \b\d+\s*:\s*\d+\b
    | (?:\d+\s+)?\d+(?:/\d+)?\s*"\s*=\s*\d+\s*'
    | \b(?:AS\s+SHOWN|AS\s+NOTED|FULL\s+SIZE|N\.?T\.?S\.?|NOT\s+TO\s+SCALE|VARIOUS)\b
""", re.IGNORECASE | re.VERBOSE)


# Match vague scale descriptions.
VAGUE_SCALE_RE = re.compile(
    r"AS\s+SHOWN|AS\s+NOTED|VARIOUS",
    re.IGNORECASE
)


# Match common drawing captions.
CAPTION_RE = re.compile(r"""
      ^\s*(?:NORTH|SOUTH|EAST|WEST|FRONT|REAR|SIDE)?\s*
        (?:ELEVATION|PLAN|PERSPECTIVE)\b
    | ^\s*SECTIONS?\s+[A-Z0-9]{1,2}\s*[-–—]\s*[A-Z0-9]{1,2}\b
    | ^\s*DETAIL\s+\w
""", re.IGNORECASE | re.VERBOSE)


# Common month abbreviations.
MONTHS = ("JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC")


# Match common date formats.
DATE_RE = re.compile(rf"""
      \b\d{{1,2}}\s*[./-]\s*\d{{1,2}}\s*[./-]\s*\d{{2,4}}\b
    | \b(?:{MONTHS})[A-Z]*\.?\s+\d{{4}}\b
    | \b\d{{1,2}}\s+(?:{MONTHS})[A-Z]*\.?\s+\d{{2,4}}\b
    | \b(?:19|20)\d{{2}}\b
""", re.IGNORECASE | re.VERBOSE)


# Match standalone years.
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


# Match common drawing number formats.
DRAWING_NO_RE = re.compile(r"""
      \b\d{2,4}\s*/\s*[A-Z]{1,2}\s*\d{1,3}[A-Z]?\b
    | \b[A-Z]{1,3}\s*-?\s*\d{2,5}\b
    | \b\d{3,6}\s*[-/]\s*\d{1,4}\b
""", re.IGNORECASE | re.VERBOSE)


# Check whether the text is a valid drawing number.
def is_drawing_number(text: str | None) -> bool:
    value = " ".join((text or "").upper().split()).strip()
    value = value.rstrip(".,;:")

    if not value:
        return False

    if DATE_RE.fullmatch(value):
        return False

    return DRAWING_NO_RE.fullmatch(value) is not None


# Match dimension-like text.
DIMENSION_RE = re.compile(r"""^[\d\s'"/\-.,ØøRr]+$""")


# Common anchor labels for metadata fields.
ANCHORS: dict[str, list[str]] = {
    "scale":          ["SCALE", "SCALES"],
    "date":           ["DATE", "DATED"],
    "draughtsperson": ["DRAWN", "DRAWN BY", "DFTSMN", "TRACED", "TRACED BY"],
    "drawing_number": ["DRAWING NO", "DRAWING No.", "DWG NO", "DRG NO",
                       "SHEET NO", "DRAWING NUMBER"],
    "checked":        ["CHECKED", "APPROVED", "CHKD"],
    "job_number":     ["JOB NO", "JOB NUMBER", "PROJECT NO"],
}


# Match labels that should not be treated as draughtsperson names.
DRAUGHTSPERSON_CONFLICT_RE = re.compile(r"""
    ^\s*(?:
        DATE(?=\b|\d)
      | SCALES?\b
      | DRAWN(?:\s+BY)?\b
      | CHKD\b
      | CHECKED\b
      | APPD\b
      | APPROVED\b
      | CLIENT\b
      | JOB\b
      | PROJECT\b
      | TITLE\b
      | DRAWING\b
      | DWG\b
      | DRG\b
      | SHEET\b
      | REVISION\b
    )
""", re.IGNORECASE | re.VERBOSE)


# Common revision-related keywords.
REVISION_KEYS = (
    "REVISION",
    "REVISIONS",
    "AMENDMENT",
    "AMENDMENTS",
    "REV."
)


# Check whether the scale description is vague.
def is_vague_scale(text: str | None) -> bool:
    return bool(text) and bool(VAGUE_SCALE_RE.search(text))


# Calculate the proportion of alphabetic characters.
def alpha_ratio(text: str) -> float:
    s = text.replace(" ", "")

    if not s:
        return 0.0

    return sum(c.isalpha() for c in s) / len(s)


# Common OCR letter-digit confusions.
CONFUSION = {
    "I": "1",
    "l": "1",
    "O": "0",
    "o": "0",
    "S": "5",
    "B": "8",
    "D": "0"
}


# Fields that commonly contain numeric values.
NUMERIC_FIELDS = {
    "drawing_number",
    "date",
    "scale",
    "job_number"
}


# Words that should not be modified by OCR confusion correction.
CONFUSION_SAFE_WORDS = {
    "TO", "NO", "OF", "ON", "IN", "IS", "AS", "BY", "OR", "SO", "DO",
    "DIA", "DO.", "NO.", "SEC", "ELEV", "SCALE", "SITE", "PLAN",
}


# Correct likely OCR character confusions.
def normalize_confusions(text: str, field: str | None = None) -> str:
    if field is not None and field not in NUMERIC_FIELDS:
        return text

    lenient = field in NUMERIC_FIELDS

    words = [w for w in re.split(r"[\s]+", text) if w]
    alpha_only = sum(
        1 for w in words
        if w.replace(".", "").isalpha() and len(w) > 2
    )
    has_digit = any(c.isdigit() for c in text)

    if not lenient and alpha_only >= 2 and not has_digit:
        return text

    out, n = list(text), len(text)

    for i, ch in enumerate(out):
        if ch not in CONFUSION:
            continue

        prev = text[i - 1] if i > 0 else " "
        nxt = text[i + 1] if i + 1 < n else " "

        near_digit = prev.isdigit() or nxt.isdigit()
        isolated = (not prev.isalpha()) and (not nxt.isalpha())

        seg = re.split(r"[\s]", text)
        pos, cur = 0, ""

        for w in seg:
            if pos <= i < pos + len(w):
                cur = w
                break
            pos += len(w) + 1

        if cur.upper().strip(".,;:") in CONFUSION_SAFE_WORDS:
            continue

        in_alnum_word = any(c.isdigit() for c in cur) and len(cur) <= 12

        if (
            near_digit
            or isolated
            or in_alnum_word
            or (lenient and len(cur) <= 3)
        ):
            out[i] = CONFUSION[ch]

    return "".join(out)


# Generate possible corrected OCR variants.
def confusion_variants(
    text: str,
    field: str | None = None,
    max_n: int = 8
) -> list[str]:

    out = [text]

    for i, c in enumerate(text):
        if c in CONFUSION:
            out.append(
                text[:i] + CONFUSION[c] + text[i + 1:]
            )

            if len(out) >= max_n:
                break

    full = normalize_confusions(text, field)

    if full not in out:
        out.append(full)

    return out


# Characters that are easy for OCR to miss.
NARROW_CHARS = set("1IiljL|/\\'\"`.,-_ ")


# Maximum number of additional characters in an alternative candidate.
MAX_EXTRA_CHARS = 4


# Check whether an alternative OCR result is more complete.
def more_complete(main: str, alt: str) -> tuple[bool, str]:
    m, a = (main or "").strip(), (alt or "").strip()

    if not m or not a or len(a) <= len(m) or m not in a:
        return False, ""

    extra = a.replace(m, "", 1).strip()

    if not extra or len(extra) > MAX_EXTRA_CHARS:
        return False, ""

    core = [ch for ch in extra if not ch.isspace()]

    if not core:
        return False, ""

    narrow = sum(
        1 for ch in core
        if ch in NARROW_CHARS or ch.isdigit()
    )

    if narrow / len(core) < 0.6:
        return False, ""

    return True, extra


# Find a more complete OCR candidate if available.
def scan_candidates(token) -> tuple[object, str] | tuple[None, str]:

    for c in getattr(token, "candidates", []):
        if c.text.strip() == (token.text or "").strip():
            continue

        ok, extra = more_complete(token.text, c.text)

        if ok:
            return c, extra

    return None, ""