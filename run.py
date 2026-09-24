from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from uma.config import Config
from uma.logging_utils import log_values, setup_logging
from uma.pipeline import Pipeline, node_id_from_path, write_outputs

IMG_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
log = logging.getLogger(__name__)


# Write console output through the application logger.
def _out(*values: object) -> None:
    log_values(log, *values)


# Collect input images and assign a node ID to each one.
def collect(args) -> list[tuple[str, str]]:
    if args.image:
        p = Path(args.image)
        if not p.exists():
            sys.exit(f"Image not found: {p}")
        return [(str(p), args.node_id or node_id_from_path(p))]

    root = Path(args.dir)
    if not root.exists():
        sys.exit(f"Directory not found: {root}")

    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            out.append((str(p), node_id_from_path(p)))
    if args.limit:
        out = out[:args.limit]
    return out


# Parse command-line arguments and run the extraction pipeline.
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract metadata from UMA drawing title blocks")
    src = ap.add_mutually_exclusive_group(required=False)

    # Input source options.
    src.add_argument("--image", help="Path to a single image")
    src.add_argument(
        "--dir",
        help="Root image directory; each parent folder name is used as the Node ID",
    )
    # src.add_argument(
    #     "--check", action="store_true", help="Run environment checks only")
    src.add_argument("--demo", action="store_true",
                     help="Generate a synthetic drawing and run the full pipeline")

    # General processing options.
    ap.add_argument(
        "--node-id", help="Set the Node ID manually in single-image mode")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="output")
    ap.add_argument("--limit", type=int, help="Process only the first N images")
    ap.add_argument(
        "--no-llm", action="store_true",
        help="Run rule-based extraction only; do not call the model")
    ap.add_argument(
        "--viz-only", action="store_true",
        help="Generate debug images only; do not extract metadata")
    ap.add_argument(
        "--engines",
        help="Override the comma-separated OCR engine list, e.g. paddle,tesseract",
    )
    ap.add_argument("--quiet-prompt", action="store_true",
                    help="Do not print the full prompt sent to the model")
    ap.add_argument("--force", action="store_true",
                    help="Process images classified as unreadable by triage")
    ap.add_argument("--triage-only", action="store_true",
                    help="Run step 0 triage only and output a feasibility report")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # Initialize logging and runtime tracking.
    log_handler = setup_logging(Path(args.out) / "logs", args.verbose)
    started = time.time()
    log.info("Application started: log_dir=%s verbose=%s", Path(args.out) / "logs",
             args.verbose)
    log.info("Runtime arguments: %s", vars(args))


    # Generate and process a synthetic demo image when requested.
    if args.demo:

        p = make_demo(out_dir=str(Path(args.out) / "demo"))
        _out(f"Synthetic drawing generated: {p}\n")
        args.dir, args.image = str(Path(p).parent.parent), None

    # Require an image or directory unless another mode provides input.
    if not (args.image or args.dir):
        ap.error("--image or --dir is required (unless using --check or --demo)")

    # Load configuration and apply command-line overrides.
    cfg = Config.load(args.config)
    if args.no_llm or args.viz_only:
        cfg.llm.enabled = False
    if args.engines:
        cfg.ocr.engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    cfg.debug_viz = True

    # Build the list of images to process.
    jobs = collect(args)
    if not jobs:
        sys.exit("No images found")

    # Run image triage only and export the feasibility report.
    if args.triage_only:
        from uma.triage import build_manifest, summarize
        import json as _json
        root = args.dir or str(Path(args.image).parent)
        rows = build_manifest(root)
        rep = summarize(rows)
        _out(_json.dumps(rep, ensure_ascii=False, indent=2))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        import pandas as _pd
        _pd.DataFrame(rows).to_csv(Path(args.out, "manifest.csv"),
                                   index=False, encoding="utf-8-sig")
        _out(f"\nManifest  {args.out}/manifest.csv")
        return

    _out(f"Images queued: {len(jobs)}\n")
    log.info("Queued tasks: %d image(s)", len(jobs))

    # Create the pipeline and process each queued image.
    pipe = Pipeline(cfg, force=args.force, verbose_prompt=not args.quiet_prompt)
    records = []
    succeeded = 0
    failed = 0
    for i, (path, node) in enumerate(jobs, 1):
        _out(f"[{i}/{len(jobs)}] node={node}  {Path(path).name}")
        try:
            rec = pipe.process(path, node, out_dir=args.out)
        except Exception as e:
            failed += 1
            logging.exception("Failed to process image: %s", path)
            _out(f"    ✗ {e}")
            continue
        records.append(rec)

        # Track success and failure counts.
        failure = (rec.diagnostics.get("processing_error")
                   or rec.diagnostics.get("fatal"))
        if failure:
            failed += 1
            log.error("Task failed: node=%s path=%s reason=%s",
                      node, path, failure)
        else:
            succeeded += 1
        _summarize(rec)

    # Summarize LLM token usage across processed images.
    used = [r.diagnostics.get("llm", {}).get("usage") for r in records]
    used = [u for u in used if u and u.get("total_tokens")]
    if used:
        pin = sum(u["prompt_tokens"] for u in used)
        pout = sum(u["completion_tokens"] for u in used)
        n = len(used)
        _out(f"\nModel usage  {n} call(s)  input {pin:,}  output {pout:,}  "
             f"total {pin+pout:,} tokens")
        _out(f"             Per-image average: input {pin//n:,}  output {pout//n:,}"
             f"  →  about {(pin+pout)*1000//n:,} tokens per 1,000 images")

    # Write final CSV and audit outputs.
    if records and not args.viz_only:
        try:
            paths = write_outputs(records, args.out)
            _out(f"\nResults table  {paths['csv']}")
            _out(f"Audit records  {paths['sidecar_dir']}/")
        except PermissionError as e:
            _out(f"\n✗ Failed to write the results table: {e}")
            _out("  Another application is using the file "
                 "(most likely sheets.csv is open in Excel).")
            _out(f"  Per-image audit records have already been saved: "
                 f"{args.out}/sidecar/")
            _out("  Close the application using the file, then rerun only the "
                 "table export; recognition does not need to run again.")

    # Report debug output location and total runtime.
    _out(f"Debug images  {args.out}/debug/")
    elapsed = time.time() - started
    log.info("Run finished: total=%d succeeded=%d failed=%d elapsed=%.2fs log=%s",
             len(jobs), succeeded, failed, elapsed,
             log_handler.current_path or Path(args.out) / "logs")

    # Return a failure exit code if any image failed.
    if failed:
        log.error("This run had %d failed task(s); setting the process exit code to 1",
                  failed)
        raise SystemExit(1)


# Print a short summary of one processed record.
def _summarize(rec) -> None:
    d = rec.diagnostics

    # Stop early when a fatal processing error was recorded.
    if "fatal" in d:
        log.error("    ✗ %s", d["fatal"])
        return

    # Display basic triage, OCR, and localization information.
    tb = d.get("title_block", {})
    tri = d.get("triage", {})
    co = d.get("coarse_ocr", {})
    crop = d.get("crop_size")
    _out(f"    Character height {tri.get('char_height_p90_px', '?')}px"
          f" ({tri.get('verdict', '?')})"
          f"  OCR blocks {co.get('token_count', 0)}"
          f"  Localization {tb.get('method', '?')} score {tb.get('score', 0)}"
          + (f"  Crop {crop[0]}x{crop[1]}" if crop else ""))

    # Display any degraded processing conditions.
    if d.get("degraded_reasons"):
        for r in d["degraded_reasons"]:
            _out(f"    ! {r}")

    # Stop if no metadata fields were extracted.
    if not rec.fields:
        _out("    No fields extracted")
        return

    # Display each extracted field with confidence and source.
    for name, fv in rec.fields.items():
        mark = {"auto_accept": "✓", "needs_review": "?", "rejected": "✗",
                "locked": "✓"}.get(fv.status, "·")
        val = (fv.corrected or fv.verbatim or f"(empty: {fv.reason})")[:46]
        _out(f"      {mark} {name:16s} {val:48s} {fv.confidence:.2f} [{fv.source}]")

    # Warn when only partial rule-based results are available.
    if d.get("processing_error"):
        log.error("    ✗ Overall processing failed: %s", d["processing_error"])
        log.warning("    The displayed and saved fields are only partial rule-based "
                    "results and must be reviewed")


# Run the command-line application.
if __name__ == "__main__":
    main()