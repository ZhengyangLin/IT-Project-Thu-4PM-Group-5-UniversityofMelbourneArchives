from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import locate as loc
from . import trace as tr
from . import triage
from . import layout, preprocess, rules, verify, views, viz
from .config import Config
from .llm import (SYSTEM as SYSTEM_TEXT, LlmClient, assign_fields,
                  record_evidence_regions)
from .models import FieldValue, SheetRecord
from .ocr.engines import build_engines
from .ocr.fusion import merge_passes, read_all
from .patterns import is_vague_scale

log = logging.getLogger(__name__)

PROMPT_VERSION = "v3.1"


# Calculate the SHA-256 hash of an input file.
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Use the parent folder name as the default node ID.
def node_id_from_path(path: Path) -> str:
    return path.parent.name


# Run the complete metadata extraction pipeline.
class Pipeline:
    _last_debug: dict = {}

    # Initialize the pipeline configuration, OCR engines, and LLM client.
    def __init__(self, cfg: Config, force: bool = False,
                 verbose_prompt: bool = True):
        self.cfg = cfg
        self.force = force
        self.verbose_prompt = verbose_prompt
        self.engines = build_engines(cfg.ocr.engines, cfg.ocr.lang,
                                     cfg.ocr.paddle_max_side_limit,
                                     cfg.ocr.tesseract_psm_coarse)
        self.llm = LlmClient(cfg.llm)
        log.info("Loaded OCR engines: %s", [e.name for e in self.engines])

    # Process one drawing through the full extraction pipeline.
    def process(self, image_path: str, node_id: str | None = None,
                out_dir: str = "output") -> SheetRecord:
        t0 = time.time()
        p = Path(image_path)
        node = node_id or node_id_from_path(p)
        rec = SheetRecord(node_id=node, path=str(p))
        diag: dict = {"prompt_version": PROMPT_VERSION}
        tracer = tr.Tracer(node, out_dir, self.cfg.write_trace)

        # Preprocess the input image before OCR.
        gray, plain, meta = preprocess.run(str(p), self.cfg.preprocess)
        diag["preprocess"] = meta

        # Assess image readability and infer the drawing series.
        tri = triage.assess(gray)
        diag["triage"] = tri
        rec.series = triage.infer_series(p)

        if tri["verdict"] != "reliable":
            log.warning("[%s] 90th-percentile character height: %.0f px (%s): %s", node,
                        tri["char_height_p90_px"], tri["verdict"], tri["note"])
            diag["triage_warning"] = tri["note"]

        # Run coarse OCR on the full drawing.
        ch0, _ = triage.char_height(gray)
        coarse_up = self.cfg.ocr.coarse_upscale
        if ch0 > 0:
            coarse_up = self.cfg.ocr.coarse_target_char_px / ch0
        tokens, d = read_all(gray, self.engines, coarse_up,
                             self.cfg.ocr.iou_merge, self.cfg.ocr.min_conf,
                             max_side=self.cfg.ocr.max_long_side)
        d["coarse_scale_used"] = round(coarse_up, 3)
        d["page_char_height_p90"] = round(ch0, 1)
        nr = triage.noise_ratio([t.text for t in tokens])
        d["noise_ratio"] = round(nr, 3)
        d["anchor_hits"] = triage.probe_anchors([t.text for t in tokens])
        diag["coarse_ocr"] = d

        # Stop if no OCR engine returns any result.
        if not tokens:
            rec.diagnostics = {**diag, "fatal":
                f"No OCR engine returned a result (engines used: "
                f"{d.get('engines_used') or 'none'}). This is an engine failure, "
                f"not an image-quality issue; check the preceding errors."}
            return rec

        if self.cfg.debug_viz:
            self._save_tokens_only(node, out_dir, plain, tokens)

        # Reject highly noisy OCR unless force processing is enabled.
        if nr > 0.75 and d["anchor_hits"] == 0 and not self.force:
            rec.diagnostics = {**diag, "fatal":
                f"OCR output is {nr:.0%} noise and contains none of the SCALE, "
                f"DATE, DRAWING, or other anchor labels; the image is considered "
                f"unreadable. The 90th-percentile character height is "
                f"{tri['char_height_p90_px']} px. Use --force to process it "
                f"anyway and inspect the actual OCR output."}
            return rec

        # Locate the title block using OCR and layout information.
        lr = loc.locate(plain, tokens, self.cfg.layout.line_ink_percentile,
                        self.cfg.rule.anchor_fuzz,
                        self.cfg.layout.line_min_ratio)
        bx = lr.bbox
        rec.title_bbox = bx
        rec.localization_method = lr.method
        rec.localization_conf = lr.confidence
        diag["title_block"] = {"method": lr.method, "score": lr.confidence,
                               **lr.detail}

        if lr.method == "fallback":
            diag.setdefault("degraded_reasons", []).append(
                "All localization stages failed; fell back to the lower-right "
                "quadrant, so the result is unreliable")

        if self.cfg.debug_viz:
            self._save_debug(node, out_dir, plain, gray, tokens, lr)

        # Crop the detected title block and prepare for fine OCR.
        crop = gray[bx[1]:bx[3], bx[0]:bx[2]]
        diag["crop_size"] = [int(bx[2] - bx[0]), int(bx[3] - bx[1])]
        ch, _ = triage.char_height(crop, roi=(0.0, 0.0, 1.0, 1.0))
        up = self.cfg.ocr.fine_upscale

        if ch > 0:
            up = min(max(self.cfg.ocr.target_char_px / ch,
                         self.cfg.ocr.fine_upscale), self.cfg.ocr.max_upscale)

        up = min(up, self.cfg.ocr.max_long_side / max(crop.shape))
        up = max(up, 0.1)

        # Run fine OCR on the title-block crop.
        fine, d = read_all(crop, self.engines, up,
                           self.cfg.ocr.iou_merge, self.cfg.ocr.min_conf,
                           max_side=self.cfg.ocr.max_long_side)
        d["fine_upscale_used"] = round(up, 2)
        d["crop_char_height_p90"] = round(ch, 1)
        diag["fine_ocr"] = d

        # Convert crop-relative bounding boxes back to full-image coordinates.
        for t in fine:
            t.bbox = (t.bbox[0] + bx[0], t.bbox[1] + bx[1],
                      t.bbox[2] + bx[0], t.bbox[3] + bx[1])

        tracer.section("① Fine OCR result (before merge_passes)",
                       "Raw output after enlarging and rereading the title-block crop.\n"
                       "Coarse OCR has not yet been merged, so narrow characters "
                       "missed by the detector are absent here.")
        tracer.tokens(fine)

        # Select coarse OCR tokens located inside the title block.
        coarse_in_crop = [t for t in tokens
                          if bx[0] <= t.cx <= bx[2] and bx[1] <= t.cy <= bx[3]]

        tracer.section("② Coarse OCR result (within the crop)",
                       "Full-page coarse OCR blocks that fall inside the title block.\n"
                       "Resolution is lower, but the looser detection boxes often "
                       "retain narrow characters missed by fine OCR.")
        tracer.tokens(coarse_in_crop)

        # Merge fine OCR with coarse OCR evidence.
        fine, mdiag = merge_passes(fine, coarse_in_crop)
        diag["pass_merge"] = mdiag

        tracer.section("③ Fine OCR result (after merge_passes)",
                       f"Result after merging coarse and fine OCR. {mdiag}\n"
                       "Both candidates are retained where their text differs, "
                       "which lowers agreement.\n"
                       "Coarse-only blocks are restored in full, tagged @coarse, "
                       "and downweighted.")
        tracer.tokens(fine)

        # Separate upright and rotated text.
        upright, rotated = layout.split_rotated(fine)

        if rotated:
            log.info("Detected %d vertical labels: %s", len(rotated),
                     [t.text for t in rotated[:6]])

        # Group OCR tokens into text rows.
        fine_lines = layout.group_rows(upright, self.cfg.layout)
        fine_lines += [layout.Line([t], row_no=-1) for t in rotated]
        fine_lines.sort(key=lambda l: (l.y0, l.x0))

        # Assign token IDs before generating candidate hints.
        views.assign_ids(fine_lines)

        # Analyse OCR alternatives and record preferred candidates.
        hints = {}
        for l in fine_lines:
            for t in l.tokens:
                for fld in (None, "drawing_number", "scale", "date"):
                    h = rules.analyze_candidates(t, fld)
                    if h:
                        hints[t.tid] = h
                        break

        if hints:
            log.info("Candidate analysis recommends alternatives in %d locations: %s",
                     len(hints),
                     [h["preferred"] for h in list(hints.values())[:5]])

        diag["candidate_hints"] = {str(k): v for k, v in hints.items()}

        # Build the layout grid, token index, and flattened token list.
        grid, index, flat = views.build_views(fine_lines, self.cfg.view, hints)
        rec.grid, rec.index_view = grid, index

        tracer.section("⑤ View 1 · Layout grid",
                       "Spaces and line breaks restore text to its approximate "
                       "position for the model to interpret the layout.\n"
                       "This cannot be stored as JSON because escaping would "
                       "destroy the layout.")
        tracer.text(grid)

        tracer.section("⑥ View 2 · Token index",
                       "One text block per line, with row numbers and engine "
                       "disagreements, for the model to identify positions.\n"
                       "Token IDs returned by the model refer to this index.")
        tracer.text(index)

        tracer.section("⑦ flat (token ID → text block; internal)",
                       "The third value returned by build_views. After the model "
                       "returns tokens:[33,34],\n"
                       "the program uses these IDs to reconstruct verbatim text "
                       "for the evidence check.")
        tracer.tokens(flat)

        # Store OCR token information by token ID.
        rec.token_map = {
            str(t.tid): {
                "text": t.text,
                "bbox": list(t.bbox),
                "conf": round(t.conf, 3),
                "agreement": t.agreement,
                "candidates": [[c.text, c.engine, round(c.conf, 3)]
                               for c in t.candidates],
            } for t in flat}

        # Extract reliable metadata using rule-based matching.
        locked, pending, rejected = rules.extract(flat, self.cfg.rule)
        diag["rules"] = {"locked": list(locked), "pending": pending,
                         "rejected": rejected}

        # Use the LLM to resolve fields that remain uncertain.
        llm_fields, d = assign_fields(grid, index, pending, locked, flat,
                                      self.llm, verbose=self.verbose_prompt,
                                      rejected=rejected)
        diag["llm"] = d

        # Record model failures without discarding rule-derived results.
        llm_failure = None
        if d.get("error"):
            llm_failure = f"Model request failed: {d['error']}"
        elif d.get("parse_error"):
            llm_failure = f"Failed to parse model response: {d['parse_error']}"
        elif (self.cfg.llm.enabled and pending
              and d.get("skipped") == "llm_unavailable"):
            llm_failure = ("The model is enabled but unavailable; pending fields "
                           "were not processed")

        if llm_failure:
            diag["processing_status"] = "failed"
            diag["processing_error"] = llm_failure
            log.error("[%s] %s; rule-derived values will be saved as a partial result",
                      node, llm_failure)

        # Store the model input and raw response in the trace.
        if d.get("prompt"):
            tracer.section("⑧ Input sent to the model",
                           f"Pending fields: {pending}\n"
                           "Full system and user messages follow.")
            tracer.text("── system ──\n" + SYSTEM_TEXT)
            tracer.text("\n── user ──\n" + d["prompt"])

        if d.get("raw_response"):
            tracer.section("⑨ Raw model response",
                           f"Unmodified response. Usage: {d.get('usage')}")
            tracer.text(d["raw_response"])

        # Merge rule and model results, giving rule results priority.
        merged: dict[str, FieldValue] = {**locked}
        for k, v in llm_fields.items():
            if k in merged:
                merged[k].notes.append(
                    "The model also returned a value; the rule-derived value takes precedence")
                continue
            merged[k] = v

        # Build a token lookup table for evidence checking.
        by_id = {t.tid: t for t in flat}

        # Record the OCR evidence regions used by each field.
        for fv in merged.values():
            record_evidence_regions(fv, by_id)

        # Combine confidence signals and assign final field status.
        for fv in merged.values():
            fv.signals["localization_conf"] = lr.confidence
            verify.fuse(fv, self.cfg.conf)

            if lr.confidence < 0.5:
                fv.confidence = min(fv.confidence, self.cfg.conf.t_low)
                fv.status = "needs_review"

        # Flag vague sheet-level scale values.
        sc = merged.get("scale")
        if sc and is_vague_scale(sc.verbatim):
            sc.notes.append(
                "The sheet-level scale is non-specific and must not be inherited by figures")

        # Downgrade results when the pipeline reports degraded conditions.
        degraded = list(diag.get("degraded_reasons", []))

        if lr.confidence < 0.5 and lr.method != "fallback":
            degraded.append(
                f"Localization method {lr.method} has confidence of only {lr.confidence}")

        if degraded:
            diag["degraded_reasons"] = degraded
            for fv in merged.values():
                fv.confidence = min(fv.confidence, self.cfg.conf.t_low)
                fv.status = "needs_review"
                fv.notes.extend(degraded)

        # Log detailed evidence for each extracted field.
        if merged:
            log.info("─── field evidence details ───")

            for name, fv in merged.items():
                if not fv.tokens:
                    log.info("  %-16s (no tokens) %s", name,
                             f"reason={fv.reason}" if fv.reason else "")
                    continue

                log.info("  %-16s tokens=%s  verbatim=%r%s", name, fv.tokens,
                         fv.verbatim,
                         f"  corrected={fv.corrected!r}" if fv.corrected else "")

                if fv.correction_basis:
                    log.info("  %-16s   basis: %s", "", fv.correction_basis)

                for i in fv.tokens:
                    m = rec.token_map.get(str(i))

                    if not m:
                        log.warning("      [%s] Token ID does not exist; the "
                                    "model referenced an out-of-range ID", i)
                        continue

                    alt = " | ".join(f"{c[1]}:{c[0]}" for c in m["candidates"][1:])

                    log.info("      [%s] %-24r bbox=%s conf=%.3f agreement=%.2f%s",
                             i, m["text"], m["bbox"], m["conf"], m["agreement"],
                             f"  alternatives: {alt}" if alt else "")

            log.info("─── end of details ───")

        # Write final extracted fields to the trace.
        tracer.section("⑩ Final fields",
                       "Result after merging rule and model output, checking "
                       "evidence, and combining confidence signals.")

        for name, fv in merged.items():
            val = fv.corrected or fv.verbatim or f"(empty: {fv.reason})"
            tracer.text(f"{name:18s} {val!r}")

            if fv.correction_basis:
                tracer.text(f"{'':18s} correction basis: {fv.correction_basis}")

            tracer.text(f"{'':18s} tokens={fv.tokens} verbatim={fv.verbatim!r} "
                        f"confidence={fv.confidence} status={fv.status} "
                        f"source={fv.source}")

            for n in fv.notes:
                tracer.text(f"{'':18s} note: {n}")

        tp = tracer.close()

        if self.cfg.write_trace:
            log.info("★ Intermediate artifacts written to: %s", tp)
            diag["trace_file"] = tp

        rec.fields = merged
        diag.setdefault("processing_status", "success")
        diag["elapsed_sec"] = round(time.time() - t0, 2)
        rec.diagnostics = diag

        rec.diagnostics.setdefault("debug_images", self._last_debug)

        return rec

    # Save an image showing OCR token bounding boxes.
    def _save_tokens_only(self, node, out_dir, plain, tokens) -> None:
        try:
            Path(out_dir, "debug").mkdir(parents=True, exist_ok=True)
            p = str(Path(out_dir, "debug", f"{node}_0_tokens.jpg"))
            viz.draw_tokens(plain, tokens, p)
            log.info("★ OCR bounding-box image saved to: %s", p)
        except Exception as e:
            log.error("Failed to save OCR bounding-box image: %s", e)

    # Save title-block localization debug images.
    def _save_debug(self, node, out_dir, plain, gray, tokens, lr) -> dict:
        out = {}

        try:
            Path(out_dir, "debug").mkdir(parents=True, exist_ok=True)

            vp = str(Path(out_dir, "debug", f"{node}_1_overview.jpg"))
            viz.draw_v2(plain, tokens, lr, vp)
            out["overview"] = vp

            cp = str(Path(out_dir, "debug", f"{node}_2_titleblock.jpg"))
            viz.save_titleblock_crop(gray, lr.bbox, cp)
            out["titleblock"] = cp

            log.info("★ Debug image saved (localization=%s, score=%.2f): %s",
                     lr.method, lr.confidence, cp)

        except Exception as e:
            log.error("Failed to save debug image (main processing is unaffected): %s", e)

        self._last_debug = out
        return out


FIELD_ORDER = ["project_name", "drawing_title", "architect", "draughtsperson",
               "drawing_number", "date", "scale", "drawing_type"]


# Convert one extraction record into a CSV-ready row.
def record_to_row(rec: SheetRecord) -> dict:
    row = {"node_id": rec.node_id, "file_path": rec.path, "series": rec.series}
    worst = "auto_accept"

    processing_error = (rec.diagnostics.get("processing_error")
                        or rec.diagnostics.get("fatal") or "")

    for f in FIELD_ORDER:
        fv = rec.fields.get(f)

        if fv is None:
            row[f] = ""
            row[f + "_verbatim"] = ""
            row[f + "_corrected"] = ""
            row[f + "_correction_basis"] = ""
            row[f + "_conf"] = ""
            row[f + "_status"] = ""
            row[f + "_reason"] = ""
            continue

        row[f] = fv.corrected or fv.verbatim or ""
        row[f + "_verbatim"] = fv.verbatim or ""
        row[f + "_corrected"] = fv.corrected or ""
        row[f + "_correction_basis"] = fv.correction_basis or ""
        row[f + "_conf"] = fv.confidence
        row[f + "_status"] = fv.status
        row[f + "_reason"] = fv.reason or ""

        if fv.status != "auto_accept":
            worst = "needs_review"

    row["localization_method"] = rec.localization_method
    row["localization_conf"] = rec.localization_conf
    row["processing_status"] = "failed" if processing_error else "success"
    row["processing_error"] = processing_error
    row["review_status"] = "needs_review" if processing_error else worst
    row["prompt_version"] = PROMPT_VERSION

    return row


# Write sidecar JSON files and the summary CSV.
def write_outputs(records: list[SheetRecord], out_dir: str) -> dict[str, str]:
    out = Path(out_dir)
    (out / "sidecar").mkdir(parents=True, exist_ok=True)

    for rec in records:
        payload = {
            "node_id": rec.node_id,
            "path": rec.path,
            "title_bbox": rec.title_bbox,
            "localization": {
                "method": rec.localization_method,
                "score": rec.localization_conf
            },
            "grid": rec.grid,
            "index_view": rec.index_view,
            "token_map": rec.token_map,
            "fields": {k: asdict(v) for k, v in rec.fields.items()},
            "diagnostics": rec.diagnostics,
        }

        (out / "sidecar" / f"{rec.node_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    csv_path = Path(write_summary_rows(
        [record_to_row(r) for r in records], str(out)
    ))

    return {
        "csv": str(csv_path),
        "sidecar_dir": str(out / "sidecar")
    }


# Write summary rows to sheets.csv.
def write_summary_rows(rows: list[dict], out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    return str(_write_csv(
        pd.DataFrame(rows),
        out / "sheets.csv"
    ))


# Write the CSV and use a timestamped fallback if the file is locked.
def _write_csv(df, path: Path) -> Path:
    import time as _t

    candidates = [
        path,
        path.with_name(
            f"{path.stem}_{_t.strftime('%H%M%S')}{path.suffix}"
        )
    ]

    last = None

    for i, p in enumerate(candidates):
        try:
            with open(p, "w", encoding="utf-8-sig", newline="") as f:
                df.to_csv(f, index=False)

            if i:
                log.warning(
                    "%s is in use (probably open in Excel); saved to %s instead",
                    path.name,
                    p.name
                )

            return p

        except PermissionError as e:
            last = e
            continue

    raise last