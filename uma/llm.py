from __future__ import annotations

import json
import logging
import os
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LlmCfg
from .models import FieldValue, Token

log = logging.getLogger(__name__)


# Resolve the API key setting used by the LLM client.
def resolve_api_key(cfg: LlmCfg, override: str | None = None) -> str:

    return cfg.api_key_env


# Classify whether corrected text is an OCR fix or only normalization.
def classify_correction(verbatim: str | None, corrected: str | None,
                        basis: str | None) -> str | None:
    if not corrected:
        return None
    flat = lambda s: "".join(ch for ch in (s or "").upper() if ch.isalnum())
    if flat(verbatim) == flat(corrected):
        return None
    return "fix" if basis else "normalize"


# Describe each metadata field for the model prompt.
FIELD_DESC = {
    "project_name":   ("Project name. Extract the complete contents of the PROJECT "
                       "block, which may contain a building name (for example, "
                       "\"CABANA\"), an owner (for example, MRS. A.S. PIERCE), "
                       "and a location (for example, TRAWALLA AVENUE, TOORAK).\n"
                       "                  **Include the address**; do not return only "
                       "a person's name. Format corrected as \"building name, for "
                       "owner, address\"."),
    "drawing_title":  ("Drawing title: what this drawing depicts, such as SITE PLAN "
                       "or AMENDED FOUNDATION DETAILS.\n"
                       "                  Distinguish it from the project name: the "
                       "title says what the drawing shows, not whose property it is."),
    "architect":      "Name of the architect or architectural practice",
    "draughtsperson": "Draughtsperson, usually shown as initials",
    "drawing_number": ("Drawing number. Common forms include 562/D8 and 603/A1 "
                       "(digits, slash, letter and digits), as well as A-1234 and "
                       "12345-678. There is only one component on each side of the "
                       "slash; do not append extra digits after it. If the evidence "
                       "cannot form one of these patterns, return null rather than "
                       "forcing a value."),
    "date":           "Drawing date, not a date from the revision table",
    "scale":          "Sheet-level drawing scale",
    "drawing_type":   ("Drawing type. Choose elevation, perspective, plan, section, "
                       "or detail.\n"
                       "                  Base the choice on words in the drawing "
                       "title (SITE PLAN → plan; FOUNDATION DETAILS → detail).\n"
                       "                  Use mixed when the sheet contains multiple "
                       "types. If the title does not reveal the type, return null "
                       "with reason code requires_visual. The text-only workflow has "
                       "no graphical evidence, so do not guess."),
}

# System instructions for metadata extraction and OCR correction.
SYSTEM = """You are cataloguing architectural drawings for a university archive. You will be shown two views of a drawing title block.

View 1 is a character grid that restores text to its approximate original position so you can understand the layout.
View 2 is a numbered token index with one text block per line and a row number. Values may come only from this index.

Non-negotiable requirements:
1. Every field value must be composed from token IDs in View 2. Do not introduce any text that is absent from the index.
2. verbatim is the exact text of the selected tokens joined with single spaces. corrected is the cleaned form, used only for standardized capitalization or punctuation, or for an evidence-backed OCR correction. Keep them separate and never alter verbatim.
3. When uncertain, return null with one of these reason codes: illegible, not_present, ambiguous, or low_resolution. Prefer an empty field to a plausible guess.
4. Do not infer missing facts. If no architect appears on the drawing, leave it empty; do not infer one from the project name.
5. A date inside the small REVISION / No / DATE / BY table is not the drawing date.
6. For a text block marked ← alternatives, choose the most plausible supplied candidate and explain the basis. Do not output text outside those candidates. **The primary candidate is not necessarily correct.** Fine OCR has higher resolution, but its detector can omit isolated narrow characters such as 1, I, or a slash, so an alternative may be more complete. When a location is marked ⚠ prefer, the rules have identified a better-supported alternative and supplied the reason. Follow it unless stronger evidence contradicts it.
7. The ⚠ marker often indicates a superset: one candidate fully contains the primary candidate and adds only one or two characters. This usually means the detector missed isolated narrow characters such as 1, I, a slash, or a quotation mark at high resolution while the lower-resolution pass captured them. **Prefer the more complete candidate by default** unless the added portion is clearly inconsistent with the context. Put the complete value in corrected and state in correction_basis which candidate you selected and why.
   Example: primary 'FOOT' + more complete 'I FOOT' → in a scale, I represents 1, so use '1 FOOT'.
            primary '603' + more complete '603/A1' → the drawing number should be '603/A1'.
8. You may correct clear OCR confusions in corrected, including I/1, O/0, S/5, B/8, comma/period, and a vertical bar | or digit 1 that represents a slash in a drawing number. **Every such correction must include correction_basis.** correction_basis is the sole distinction between two uses of corrected: include it when the interpreted characters change; omit it when only capitalization or punctuation styling changes without changing any character identity. **Disagreement candidates are alternative readings of the same pixels: choose one and never add them together.** For example, if the primary candidate is '562 |D66' and an alternative is '562 1D6', the | and 1 represent the same pixel, so determine that it is a slash and write '562/D66', not '562/1D66'. verbatim must still preserve the chosen OCR text exactly. The source transcription and the interpreted value are different kinds of archival data and must not be mixed. If the correction is uncertain, leave it empty rather than guess.
9. This rule overrides rule 8 when a candidate spans multiple token IDs. If an alternative already contains the complete text of an adjacent token on the same row, it may cover multiple token IDs. When selecting that alternative, omit the adjacent token IDs already covered by it to avoid duplicate text.
   Example:
   [26] I" INCH
   [27] TO
   [28] FOOT ← alternatives paddle@coarse:TO I FOOT
   The alternative for [28] already contains TO from [27], so use tokens=[26,28] and verbatim "I\" INCH TO I FOOT"; do not also cite [27].
10. corrected may apply one-to-one corrections to existing OCR characters, such as I→1 or O→0, but it must not insert a digit, fraction bar, or slash that appears in neither the primary candidate nor any alternative. Common notation, field formats, and domain knowledge may help choose among candidates, but they are not evidence for adding a new character. Claiming in correction_basis that OCR omitted a character is not evidence by itself; identify the specific primary or alternative candidate that contains it.
    Example: if [27] is 8 INCH and no candidate contains 1/, do not correct it to 1/8 INCH. Preserve 8 INCH and lower the confidence or use reason code ambiguous. If [29] FOOT has an alternative I FOOT, you may select that alternative and correct I to the digit 1 in corrected because explicit candidate evidence exists.
Return JSON only, with no explanatory text or Markdown code fences."""

# Build the model prompt from OCR views and rule-based results.
def build_prompt(grid: str, index: str, pending: list[str],
                 locked: dict[str, FieldValue],
                 rejected: dict[str, str] | None = None) -> str:
    known = "\n".join(f"  {k} = {v.verbatim}" for k, v in locked.items()) or "  (none)"
    want = "\n".join(f"  {f}: {FIELD_DESC.get(f, '')}" for f in pending
                      if f in FIELD_DESC)
    example = {
        "drawing_title": {"tokens": [9, 10, 11],
                          "verbatim": "AMENDED FOUNDATION DETAILS",
                          "corrected": "Amended foundation details",
                          "confidence": "high"},
        "draughtsperson": {"tokens": [], "verbatim": None,
                           "reason": "not_present"},
    }
    warn = ""
    if rejected:
        lines = "\n".join(f"  {k}: {v}" for k, v in rejected.items())
        warn = ("\nThe rules evaluated the following fields but did not have "
                "enough confidence to lock them. Verify them carefully for the "
                f"reasons below:\n{lines}\n")

    return f"""[DOCUMENT]

View 1 · Layout grid
```
{grid}
```

View 2 · Token index
```
{index}
```

[TASK]
The rules have already resolved these fields; do not extract them again:
{known}

Extract the following fields:
{want}
{warn}
[FORMAT]
Spaces and line breaks in View 1 restore the relative positions of text blocks. Use them to determine which label belongs to which value.
Row numbers in View 2 correspond directly to rows in View 1. Use them to convert positions in the grid into token IDs.

[OUTPUT]
Return a JSON object whose keys are field names. Example:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


# Wrap the LLM client with availability checks and retry handling.
class LlmClient:
    def __init__(self, cfg: LlmCfg):
        self.cfg = cfg
        self._client = None

    # Check whether LLM support is enabled and the package is available.
    def available(self) -> bool:
        if not self.cfg.enabled:
            return False
        try:
            import openai  # noqa: F401
            return True
        except Exception as e:
            log.warning("The openai package is unavailable: %s", e)
            return False

    # Create the API client only when it is first needed.
    def _lazy(self):
        if self._client is None:
            from openai import OpenAI
            # key = resolve_api_key(self.cfg)
            key = resolve_api_key(self.cfg)
            if not key:
                log.warning("No API key was found. api_key_env specifies the "
                            "environment variable %r, but that variable is not "
                            "set. The model request will probably return 401.",
                            self.cfg.api_key_env)
            self._client = OpenAI(
                base_url=self.cfg.base_url,
                api_key=key or "EMPTY",
                timeout=self.cfg.timeout,
                max_retries=0)
        return self._client

    # Run a completion with optional exponential-backoff retries.
    def complete(self, system: str, user: str) -> tuple[str, dict]:
        n = max(int(self.cfg.max_retries or 0), 0)
        if n == 0:
            return self._once(system, user)     # Preserve the original exception.
        fn = retry(stop=stop_after_attempt(n + 1),
                   wait=wait_exponential(multiplier=1, min=1, max=8),
                   # Preserve the original exception instead of wrapping it in RetryError.
                   # This keeps errors such as 401s, invalid model names, and timeouts visible.
                   reraise=True)(self._once)
        return fn(system, user)

    # Send one model request and collect usage diagnostics.
    def _once(self, system: str, user: str) -> tuple[str, dict]:
        import time
        t0 = time.time()
        resp = self._lazy().chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,     # Use deterministic output when set to 0.
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        u = getattr(resp, "usage", None)
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                 "completion_tokens": getattr(u, "completion_tokens", None),
                 "total_tokens": getattr(u, "total_tokens", None),
                 "elapsed_sec": round(time.time() - t0, 2),
                 "model": self.cfg.model} if u else {
                     "elapsed_sec": round(time.time() - t0, 2)}

        usage["finish_reason"] = getattr(choice, "finish_reason", None)
        usage["system_fingerprint"] = getattr(resp, "system_fingerprint", None)
        usage["response_chars"] = len(text)

        det = getattr(u, "completion_tokens_details", None) if u else None
        rt = getattr(det, "reasoning_tokens", None) if det else None
        reasoning = str(getattr(msg, "reasoning_content", "") or "")
        if rt or reasoning:
            usage["reasoning_tokens"] = rt
            usage["reasoning_chars"] = len(reasoning) or None
            log.warning("The endpoint enabled reasoning mode: %s reasoning "
                        "tokens / %d characters. These tokens are billed but "
                        "do not appear in the response; include them in cost "
                        "projections.",
                        rt if rt is not None else "not reported", len(reasoning))

        if usage.get("finish_reason") == "length":
            log.error("The response was truncated at max_tokens=%d and its JSON "
                      "will probably not parse. Increase llm.max_tokens.",
                      self.cfg.max_tokens)
        return text, usage


# Extract and parse a JSON object from the model response.
def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        t = m.group(0)
    return json.loads(t)


# Normalize evidence text for token comparison.
def _evidence_text(text: str | None) -> str:
    return "".join((text or "").upper().split())


# Match a field value back to token candidates and image regions.
def resolve_evidence_regions(fv: FieldValue, by_id: dict[int, Token]) -> list[dict] | None:
    if not fv.tokens or any(token_id not in by_id for token_id in fv.tokens):
        return None

    target = _evidence_text(fv.verbatim)
    paths: dict[int, list[dict]] = {0: []}
    for token_id in fv.tokens:
        token = by_id[token_id]
        main_norm = _evidence_text(token.text)
        main_engine = next(
            (candidate.engine for candidate in token.candidates
             if _evidence_text(candidate.text) == main_norm),
            "token",
        )
        choices = [(main_norm, token.text, main_engine, token.bbox)]
        seen = {main_norm}
        for candidate in token.candidates:
            normalized = _evidence_text(candidate.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            choices.append((normalized, candidate.text, candidate.engine,
                            candidate.source_bbox or token.bbox))

        next_paths: dict[int, list[dict]] = {}
        for position, path in paths.items():
            for normalized, text, engine, bbox in choices:
                if not target.startswith(normalized, position):
                    continue
                end = position + len(normalized)
                if end in next_paths:
                    continue
                next_paths[end] = path + [{
                    "tid": token_id,
                    "text": text,
                    "engine": engine,
                    "bbox": [int(value) for value in bbox],
                }]
        if not next_paths:
            return None
        paths = next_paths
    return paths.get(len(target))


# Store matched evidence regions and their combined bounding box.
def record_evidence_regions(fv: FieldValue, by_id: dict[int, Token]) -> bool:
    fv.signals.pop("evidence_regions", None)
    fv.signals.pop("evidence_bbox", None)
    regions = resolve_evidence_regions(fv, by_id)
    if regions is None:
        return False

    fv.signals["evidence_regions"] = regions
    boxes = [region["bbox"] for region in regions]
    fv.signals["evidence_bbox"] = {
        "x0": min(box[0] for box in boxes),
        "y0": min(box[1] for box in boxes),
        "x1": max(box[2] for box in boxes),
        "y1": max(box[3] for box in boxes),
    }
    return True


# Verify that returned text is supported by the referenced OCR tokens.
def verify_evidence(fv: FieldValue, by_id: dict[int, Token]) -> tuple[bool, str, list[int]]:
    fv.signals.pop("evidence_regions", None)
    fv.signals.pop("evidence_bbox", None)
    if not fv.tokens:
        return True, "", []
    missing = [i for i in fv.tokens if i not in by_id]
    if missing:
        return False, f"Referenced nonexistent token IDs: {missing}", []
    flat = _evidence_text
    joined = " ".join(by_id[i].text for i in fv.tokens)
    normalized_verbatim = flat(fv.verbatim)
    if record_evidence_regions(fv, by_id):
        return True, "", []

    if flat(joined) == normalized_verbatim:
        return True, "", []

    n_join, n_verb = flat(joined), normalized_verbatim
    if not n_verb.startswith(n_join):
        return False, (f"Verbatim evidence check failed: joining the referenced "
                       f"tokens produced {joined!r}, but the model returned "
                       f"{fv.verbatim!r}"), []

    tail = n_verb[len(n_join):]
    if not tail:
        return True, "", []

    used = set(fv.tokens)
    rest = tail
    extra: list[int] = []
    for tid in sorted(by_id):
        if tid in used or not rest:
            continue
        t = flat(by_id[tid].text)
        if t and rest.startswith(t):
            extra.append(tid)
            rest = rest[len(t):].strip()

    if rest:
        return False, (f"Verbatim evidence check failed: the response contains "
                       f"the extra text {tail!r}, of which {rest!r} has no "
                       f"matching token in the index; classified as fabricated"), []
    return False, (f"Token references were incomplete: the extra text {tail!r} "
                   f"matches token IDs {extra}, which were added automatically"), extra



# Ask the model to fill unresolved fields and validate its evidence.
def assign_fields(grid: str, index: str, pending: list[str],
                  locked: dict[str, FieldValue], flat: list[Token],
                  client: LlmClient,
                  verbose: bool = True,
                  rejected: dict[str, str] | None = None
                  ) -> tuple[dict[str, FieldValue], dict]:
    by_id = {t.tid: t for t in flat}
    diag: dict = {"called": False}

    if not pending or not client.available():
        diag["skipped"] = "no_pending" if not pending else "llm_unavailable"
        return {}, diag

    prompt = build_prompt(grid, index, pending, locked, rejected)
    diag["called"] = True
    diag["prompt_chars"] = len(prompt)
    diag["prompt"] = prompt

    log.info("→ Model request: extracting %d fields %s; prompt has %d "
             "characters (~%d tokens)",
             len(pending), pending, len(prompt), len(prompt) // 3)
    if verbose:
        log.info("─── system ───\n%s", SYSTEM)
        log.info("─── user ───\n%s\n─── end of input ───", prompt)
    else:
        log.info("--quiet-prompt: prompt omitted from the log; it remains stored "
                 "in sidecar diagnostics.llm.prompt")

    try:
        raw, usage = client.complete(SYSTEM, prompt)
    except Exception as e:
        log.error("Model request failed: %s", e)
        diag["error"] = str(e)
        return {}, diag

    diag["usage"] = usage
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is not None:
        log.info("← Model response: %s input tokens, %s output tokens, %s "
                 "total; %.1fs elapsed",
                 pt, ct, usage.get("total_tokens"), usage.get("elapsed_sec", 0))
    else:
        log.info("← Model response: %d characters; %.1fs elapsed (the "
                 "endpoint did not report token usage)",
                 len(raw), usage.get("elapsed_sec", 0))
    log.info("─── raw model response ───\n%s\n─── end of response ───",
             raw.strip())

    diag["raw_response"] = raw
    try:
        data = parse_json(raw)
    except Exception as e:
        log.error("Failed to parse JSON: %s", e)
        diag["parse_error"] = str(e)
        return {}, diag

    out: dict[str, FieldValue] = {}
    for name, item in data.items():
        if not isinstance(item, dict):
            continue
        fv = FieldValue(
            name=name,
            tokens=[int(i) for i in (item.get("tokens") or [])],
            verbatim=item.get("verbatim"),
            corrected=item.get("corrected"),
            correction_basis=item.get("correction_basis"),
            reason=item.get("reason"),
            source="llm",
            notes=[item["note"]] if item.get("note") else [])
        ok, msg, extra = verify_evidence(fv, by_id)
        if extra:
            fv.tokens = sorted(set(fv.tokens) | set(extra))
            fv.signals["tokens_repaired"] = extra
            fv.notes.append(msg)
            ok, msg, _ = verify_evidence(fv, by_id)
        fv.signals["evidence_ok"] = ok
        if not ok:
            if msg:
                fv.notes.append(msg)
            fv.status = "needs_review"
        if fv.tokens:
            fv.signals["agreement"] = min(by_id[i].agreement for i in fv.tokens)
            fv.signals["ocr_conf"] = min(by_id[i].conf for i in fv.tokens)
        fv.signals["llm_confidence"] = item.get("confidence")
        if item.get("note") or fv.correction_basis or fv.corrected:
            fv.signals["adjudicated"] = True

        kind = classify_correction(fv.verbatim, fv.corrected, fv.correction_basis)
        if kind == "fix":
            fv.signals["corrected"] = True
            fv.notes.append(f"OCR correction: {fv.verbatim!r} → {fv.corrected!r} "
                            f"(basis: {fv.correction_basis})")
        elif kind == "normalize":
            log.warning("[%s] The model returned corrected=%r without a "
                        "correction_basis; treating it as normalization without "
                        "the confidence penalty for a correction",
                        name, fv.corrected)
            fv.notes.append(f"Normalized form: {fv.verbatim!r} → {fv.corrected!r} "
                            "(the model supplied no basis)")
        out[name] = fv

    diag["fields_returned"] = list(out)
    diag["evidence_failures"] = [k for k, v in out.items()
                                 if not v.signals.get("evidence_ok", True)]
    return out, diag

