from __future__ import annotations

import re
from statistics import median

from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
import numpy as np

from .config import RuleCfg
from .models import FieldValue, Line, Token
from .patterns import (
    ANCHORS,
    CAPTION_RE,
    DATE_RE,
    DRAWING_NO_RE,
    SCALE_RE,
    scan_candidates,
    YEAR_RE,
    alpha_ratio,
    confusion_variants,
    is_vague_scale,
)


# Metadata fields handled by the rule-based extractor.
FIELDS = [
    "project_name",
    "drawing_title",
    "architect",
    "draughtsperson",
    "drawing_number",
    "date",
    "scale",
    "drawing_type",
]


# Check whether a line contains useful metadata-like text.
def keep_line(line: Line) -> bool:
    from .patterns import CAPTION_RE

    t = line.text.strip()

    if CAPTION_RE.search(t) or SCALE_RE.search(t):
        return True

    return t.isupper() and 4 <= len(t) <= 40 and alpha_ratio(t) > 0.70


# Expected patterns for selected metadata fields.
FIELD_PATTERNS = {
    "scale": SCALE_RE,
    "date": DATE_RE,
    "drawing_number": DRAWING_NO_RE,
}


# Normalize text for comparison.
def _norm(t: str) -> str:
    return " ".join((t or "").upper().split())


# Check whether an alternative OCR candidate may be better than the primary one.
def analyze_candidates(
    tok: Token,
    field: str | None = None
) -> dict | None:

    alts = tok.alternatives()

    if not alts:
        return None

    main = _norm(tok.text)
    pat = FIELD_PATTERNS.get(field) if field else None

    best, reasons = None, []

    for c in alts:
        alt = _norm(c.text)
        why = []

        if main and main != alt and main in alt:
            why.append(
                "superset: the primary candidate is its substring, so "
                "this candidate is more complete"
            )

        if pat:
            if pat.search(alt) and not pat.search(main):
                why.append(
                    f"it matches the expected {field} format, while the "
                    "primary candidate does not"
                )
            else:
                for v in confusion_variants(alt, field):
                    if pat.search(v) and not pat.search(main):
                        why.append(
                            f"after correcting OCR confusions to {v!r}, it "
                            f"matches the expected {field} format"
                        )
                        break

        if why and (best is None or len(why) > len(reasons)):
            best, reasons = c, why

    if best is None:
        return None

    return {
        "preferred": best.text,
        "engine": best.engine,
        "conf": round(best.conf, 3),
        "reasons": reasons,
    }


# Labels that should not be treated as scale values.
_BARE_LABEL = re.compile(
    r"^\s*SCALES?\s*[:.]?\s*$",
    re.IGNORECASE
)


# Common anchor words that indicate the start of another field.
ANCHOR_WORDS = re.compile(
    r"^\s*(DRAWN|DATE|CHECKED|APPROVED|DRAWING|DWG|JOB|SHEET)\b",
    re.IGNORECASE,
)


# Extend from a token to nearby tokens on the same row.
def extend_row(
    seed: Token,
    tokens: list[Token],
    max_gap_ratio: float = 1.8
) -> list[Token]:

    row = [
        t for t in tokens
        if abs(t.cy - seed.cy) <= max(seed.h, 1) * 0.6
    ]

    row.sort(key=lambda t: t.x0)

    try:
        i = row.index(seed)
    except ValueError:
        return [seed]

    group = [seed]

    for nxt in row[i + 1:]:
        if nxt.x0 - group[-1].x1 > max(seed.h, 1) * max_gap_ratio:
            break

        if _BARE_LABEL.match(nxt.text) or ANCHOR_WORDS.search(nxt.text):
            break

        group.append(nxt)

    return group


# Find a scale value using scale patterns and nearby tokens.
def match_scale(
    tokens: list[Token]
) -> tuple[list[Token], float, str | None]:

    for i, t in enumerate(tokens):

        if _BARE_LABEL.match(t.text):
            continue

        m = SCALE_RE.search(t.text)

        if not m:
            continue

        group = extend_row(t, tokens)
        whole = " ".join(x.text for x in group)

        clean = SCALE_RE.search(whole)
        trimmed = clean.group(0).strip() if clean else None

        noisy = (
            trimmed
            and len(trimmed) < len(whole.strip()) * 0.8
        )

        return group, (0.72 if noisy else 0.94), trimmed

    return [], 0.0, None


# Find a valid date within the configured year range.
def match_date(
    tokens: list[Token],
    cfg: RuleCfg
) -> tuple[Token | None, float, str | None]:

    lo, hi = cfg.year_range

    for t in tokens:

        if not DATE_RE.search(t.text):
            continue

        m = YEAR_RE.search(t.text)

        if not m:
            continue

        year = int(m.group(1))

        if not (lo <= year <= hi):
            continue

        return t, 0.89, normalize_date(t.text, year)

    return None, 0.0, None


# Normalize recognized dates into a consistent format.
def normalize_date(text: str, year: int) -> str:

    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    up = text.upper()

    for k, v in months.items():
        if k in up:
            return f"{year:04d}-{v:02d}"

    m = re.search(
        r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*\d{2,4}\b",
        text,
    )

    if m:
        d, mo = int(m.group(1)), int(m.group(2))

        if 1 <= mo <= 12:
            return f"{year:04d}-{mo:02d}-{d:02d}"

    return str(year)


# Match OCR text against the configured architect list.
def match_architect(
    tokens: list[Token],
    cfg: RuleCfg
) -> tuple[list[Token], float, str | None]:

    joined = " ".join(t.text.upper() for t in tokens)

    best, best_score = None, 0

    for name in cfg.architects:
        s = fuzz.partial_ratio(name, joined)

        if s > best_score:
            best, best_score = name, s

    if best_score < 80:
        return [], 0.0, None

    key = best.split()[0]

    hit = [
        t for t in tokens
        if key in t.text.upper()
    ]

    return hit, min(best_score / 100, 0.96), best.title()


# Find the most reliable drawing number candidate.
def match_drawing_no(
    tokens: list[Token]
) -> tuple[Token | None, float]:

    hit = [
        t for t in tokens
        if DRAWING_NO_RE.search(t.text)
    ]

    if hit:
        best = max(
            hit,
            key=lambda t: (t.agreement, t.conf)
        )

        return best, 0.88 if best.agreement >= 0.99 else 0.52

    # Check alternative OCR candidates if the primary result does not match.
    for t in tokens:

        better, _ = scan_candidates(t)

        if (
            better is not None
            and DRAWING_NO_RE.search(better.text)
        ):
            t.notes_hint = (
                f"Alternative {better.engine}:{better.text!r} matches "
                "the drawing-number format"
            )

            return t, 0.60

    return None, 0.0


# Normalize drawing numbers for comparison.
def _drawing_no_key(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        (text or "").upper()
    ).rstrip(".,;:")


# Find metadata anchor labels using fuzzy matching.
def find_anchors(
    tokens: list[Token],
    fuzz_threshold: int
) -> dict[str, Token]:

    out: dict[str, Token] = {}

    for field, variants in ANCHORS.items():

        best, best_s = None, 0

        for t in tokens:

            up = t.text.upper().rstrip(".:")

            for v in variants:
                s = fuzz.ratio(v.upper(), up)

                if s > best_s:
                    best, best_s = t, s

        if best_s >= fuzz_threshold:
            out[field] = best

    return out


# Score the spatial relationship between an anchor and a value candidate.
def pair_score(anchor: Token, cand: Token) -> float:

    if cand is anchor:
        return 0.0

    dx, dy = (
        cand.cx - anchor.cx,
        cand.cy - anchor.cy,
    )

    # Prefer values directly to the right of the anchor.
    if (
        cand.x0 >= anchor.x1
        and abs(dy) < max(anchor.h, 1) * 0.6
    ):
        return 1.00 - min(
            cand.x0 - anchor.x1,
            300
        ) / 300

    # Also allow values positioned below the anchor.
    if (
        abs(dx) < max(anchor.w, 1)
        and 0 < dy < max(anchor.h, 1) * 3
    ):
        return 0.90 - dy / (
            max(anchor.h, 1) * 3
        )

    return 0.0


# Globally assign candidate tokens to metadata anchors.
def assign_globally(
    anchors: dict[str, Token],
    tokens: list[Token]
) -> dict[str, tuple[Token, float]]:

    if not anchors:
        return {}

    fields = list(anchors)

    anchor_ids = {
        id(a) for a in anchors.values()
    }

    cands = [
        t for t in tokens
        if id(t) not in anchor_ids
    ]

    if not cands:
        return {}

    # Build the anchor-to-candidate score matrix.
    mat = np.zeros(
        (len(fields), len(cands))
    )

    for i, f in enumerate(fields):
        for j, c in enumerate(cands):
            mat[i, j] = pair_score(
                anchors[f],
                c
            )

    # Find the globally optimal assignment.
    rows, cols = linear_sum_assignment(-mat)

    out = {}

    for i, j in zip(rows, cols):

        field = fields[i]

        min_score = (
            0.30
            if field == "drawing_number"
            else 0.35
        )

        if mat[i, j] > min_score:
            out[field] = (
                cands[j],
                float(mat[i, j]),
            )

    return out


# Keywords used to infer the drawing type.
TYPE_WORDS = [
    ("PERSPECTIVE", "perspective"),
    ("ELEVATION", "elevation"),
    ("SECTION", "section"),
    ("DETAIL", "detail"),
    ("PLAN", "plan"),
]


# Infer drawing type from the drawing title.
def infer_drawing_type(title: str) -> str | None:

    up = (title or "").upper()

    hit = [
        v for k, v in TYPE_WORDS
        if k in up
    ]

    if not hit:
        return None

    return (
        hit[0]
        if len(set(hit)) == 1
        else "mixed"
    )


# Extract metadata fields using rule-based matching.
def extract(
    tokens: list[Token],
    cfg: RuleCfg
) -> tuple[dict[str, FieldValue], list[str]]:

    locked: dict[str, FieldValue] = {}
    rejected: dict[str, str] = {}

    # Check whether a value passes field format validation.
    def gate(
        name: str,
        value: str
    ) -> tuple[bool, str]:

        from .verify import format_check
        from .models import FieldValue as _FV

        probe = _FV(
            name=name,
            verbatim=value
        )

        r = format_check(probe)

        if r is None:
            return True, ""

        if r >= 1.0:
            return True, ""

        return False, (
            f"Value {value!r} does not match the expected format for "
            f"{name}; deferred to the model"
        )

    # Store a reliable rule-based field result.
    def lock(
        name,
        toks,
        score,
        verbatim=None,
        corrected=None,
        basis=None
    ):

        toks = (
            toks
            if isinstance(toks, list)
            else [toks]
        )

        text = (
            verbatim
            or " ".join(t.text for t in toks)
        )

        ok, why = gate(name, text)

        if not ok:
            rejected[name] = why
            return

        locked[name] = FieldValue(
            name=name,
            tokens=[t.tid for t in toks],
            verbatim=(
                verbatim
                or " ".join(t.text for t in toks)
            ),
            corrected=corrected,
            correction_basis=(
                basis if corrected else None
            ),
            source="rule",
            confidence=score,
            status="locked",
            signals={
                "rule_score": score,
                "agreement": min(
                    (t.agreement for t in toks),
                    default=1.0,
                ),
                "ocr_conf": min(
                    (t.conf for t in toks),
                    default=0.0,
                ),
            },
        )

    # Extract scale.
    group, s, trimmed = match_scale(tokens)

    if group:

        if s >= cfg.lock_threshold:
            lock(
                "scale",
                group,
                s,
                corrected=trimmed,
                basis=(
                    "Matched the value pattern and extracted the scale segment "
                    "from the full line"
                ),
            )

        else:
            whole = " ".join(
                t.text for t in group
            )

            rejected["scale"] = (
                f"Value {whole!r} contains non-scale content; only {trimmed!r} "
                "resembles a scale, so its boundary was deferred to the model"
            )

    # Extract date.
    tok, s, norm = match_date(tokens, cfg)

    if tok and s >= cfg.lock_threshold:
        lock(
            "date",
            tok,
            s,
            corrected=norm,
            basis=(
                "Recognized the month and year, then converted them to ISO format"
            ),
        )

    # Extract architect.
    hits, s, norm = match_architect(
        tokens,
        cfg
    )

    if hits:

        read = " ".join(
            t.text for t in hits
        )

        expand = len(norm or "") / max(
            len(read),
            1
        )

        if expand > 2.0:
            rejected["architect"] = (
                f"Only {read!r} was read from the architect list; expanding it "
                f"to {norm!r} increases its length by a factor of {expand:.1f}, "
                "so confirmation "
                "using the layout was deferred to the model"
            )

        elif s >= cfg.lock_threshold:
            lock(
                "architect",
                hits,
                min(
                    s,
                    0.90 if expand > 1.3 else s
                ),
                corrected=norm,
                basis=(
                    f"Matched against the architect list and expanded {read!r} "
                    "to the standard form"
                ),
            )

    # Find drawing number using its value pattern.
    value_drawing_tok, value_drawing_score = match_drawing_no(tokens)

    # Find anchor labels and assign nearby values.
    anchors = find_anchors(
        tokens,
        cfg.anchor_fuzz
    )

    paired = assign_globally(
        anchors,
        tokens
    )

    # Combine drawing-number pattern evidence with anchor evidence.
    anchor_drawing = paired.pop(
        "drawing_number",
        None
    )

    if (
        value_drawing_tok is not None
        and anchor_drawing is not None
    ):
        anchor_tok, anchor_sc = anchor_drawing

        if (
            _drawing_no_key(value_drawing_tok.text)
            == _drawing_no_key(anchor_tok.text)
        ):
            score = max(
                value_drawing_score,
                0.80 + 0.15 * anchor_sc,
            )

            if score >= cfg.lock_threshold:
                lock(
                    "drawing_number",
                    anchor_tok,
                    round(score, 3),
                )

        else:
            rejected["drawing_number"] = (
                "Evidence conflict: the highest-scoring value match is "
                f"[{value_drawing_tok.tid}] {value_drawing_tok.text!r}, while "
                "the DRAWING NUMBER anchor points to "
                f"[{anchor_tok.tid}] {anchor_tok.text!r} "
                f"(anchor score {anchor_sc:.3f}); deferred to the model"
            )

    elif value_drawing_tok is not None:

        if value_drawing_score >= cfg.lock_threshold:
            lock(
                "drawing_number",
                value_drawing_tok,
                value_drawing_score,
            )

    elif anchor_drawing is not None:

        anchor_tok, anchor_sc = anchor_drawing

        score = 0.80 + 0.15 * anchor_sc

        if score >= cfg.lock_threshold:
            lock(
                "drawing_number",
                anchor_tok,
                round(score, 3),
            )

    # Fields that may contain multiple neighbouring tokens.
    MULTI_TOKEN_FIELDS = {
        "scale",
        "project_name",
        "drawing_title",
        "architect",
    }

    # Extract remaining fields using anchor-value assignments.
    for field, (tok, sc) in paired.items():

        if field in locked or field not in FIELDS:
            continue

        toks = (
            extend_row(tok, tokens)
            if field in MULTI_TOKEN_FIELDS
            else [tok]
        )

        score = 0.80 + 0.15 * sc

        if score >= cfg.lock_threshold:
            lock(
                field,
                toks,
                round(score, 3),
            )

    # Infer drawing type from the extracted drawing title.
    title = locked.get("drawing_title")

    if title and title.verbatim:

        t = infer_drawing_type(
            title.verbatim
        )

        if t:
            locked["drawing_type"] = FieldValue(
                name="drawing_type",
                tokens=list(title.tokens),
                verbatim=title.verbatim,
                corrected=t,
                correction_basis=(
                    f"Classified as {t} from type words in the drawing title"
                ),
                source="rule",
                confidence=0.88,
                status="locked",
                signals={
                    "rule_score": 0.88,
                    "inferred_from": "drawing_title",
                },
                notes=[
                    f"Classified from drawing title {title.verbatim!r}"
                ],
            )

    # Fields not locked by rules remain pending for the model.
    pending = [
        f for f in FIELDS
        if f not in locked
    ]

    # Log fields that require model handling.
    if rejected:
        import logging

        for k, why in rejected.items():
            logging.getLogger(__name__).info(
                "Rule extraction deferred %s to the model: %s",
                k,
                why,
            )

    return locked, pending, rejected