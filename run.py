"""Command-line test entry for OCR, title block localization, and visualization.

Single image:
    python run.py --image /path/to/605_A64.tif --node-id 12847

Batch:
    python run.py --dir /path/to/images --limit 20

Select OCR engines:
    python run.py --image sample.tif --engines paddle,tesseract
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from uma import locate, preprocess, viz
from uma.config import Config
from uma.logging_utils import log_values, setup_logging
from uma.ocr.engines import build_engines
from uma.ocr.fusion import read_all

IMG_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

log = logging.getLogger(__name__)


def _out(*values: object) -> None:
    """Write output through the configured logger."""
    log_values(log, *values)


def node_id_from_path(path: Path) -> str:
    """Use the parent folder name as the node ID."""
    return path.parent.name


def collect(args) -> list[tuple[str, str]]:
    """Collect image paths and node IDs."""
    if args.image:
        p = Path(args.image)

        if not p.exists():
            sys.exit(f"Image does not exist: {p}")

        return [(str(p), args.node_id or node_id_from_path(p))]

    root = Path(args.dir)

    if not root.exists():
        sys.exit(f"Directory does not exist: {root}")

    jobs = []

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            jobs.append((str(p), node_id_from_path(p)))

    if args.limit:
        jobs = jobs[:args.limit]

    return jobs


def process_image(
    path: str,
    node: str,
    cfg: Config,
    engines,
    out_dir: str,
) -> dict:
    """Run preprocessing, coarse OCR, localization, and debug visualization."""

    started = time.time()

    log.info("Starting image processing: node=%s file=%s", node, path)

    # Step 1: preprocess the source image.
    gray, plain, preprocess_meta = preprocess.run(
        path,
        cfg.preprocess,
    )

    log.info(
        "Preprocessing completed: image size=%dx%d",
        gray.shape[1],
        gray.shape[0],
    )

    # Step 2: run coarse OCR on the whole drawing.
    coarse_upscale = cfg.ocr.coarse_upscale

    tokens, ocr_diag = read_all(
        gray,
        engines,
        coarse_upscale,
        cfg.ocr.iou_merge,
        cfg.ocr.min_conf,
        max_side=cfg.ocr.max_long_side,
    )

    if not tokens:
        raise RuntimeError("OCR returned no text tokens.")

    log.info(
        "Coarse OCR completed: %d tokens detected",
        len(tokens),
    )

    # Step 3: locate the title block.
    locate_result = locate.locate(
        plain,
        tokens,
    )

    log.info(
        "Title block localization completed: method=%s confidence=%.3f bbox=%s",
        locate_result.method,
        locate_result.confidence,
        locate_result.bbox,
    )

    # Step 4: prepare the debug output directory.
    debug_dir = Path(out_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    token_path = debug_dir / f"{node}_0_tokens.jpg"
    overview_path = debug_dir / f"{node}_1_overview.jpg"
    titleblock_path = debug_dir / f"{node}_2_titleblock.jpg"

    # Step 5: save OCR token visualization.
    viz.draw_tokens(
        plain,
        tokens,
        str(token_path),
    )

    log.info(
        "OCR token visualization saved: %s",
        token_path,
    )

    # Step 6: save title block localization visualization.
    viz.draw_v2(
        plain,
        tokens,
        locate_result,
        str(overview_path),
    )

    log.info(
        "Localization visualization saved: %s",
        overview_path,
    )

    # Step 7: save the detected title block crop.
    viz.save_titleblock_crop(
        gray,
        locate_result.bbox,
        str(titleblock_path),
    )

    log.info(
        "Title block crop saved: %s",
        titleblock_path,
    )

    elapsed = time.time() - started

    result = {
        "node_id": node,
        "image": path,
        "token_count": len(tokens),
        "localization_method": locate_result.method,
        "localization_confidence": locate_result.confidence,
        "title_bbox": locate_result.bbox,
        "token_visualization": str(token_path),
        "overview_visualization": str(overview_path),
        "titleblock_crop": str(titleblock_path),
        "elapsed_sec": round(elapsed, 2),
        "preprocess": preprocess_meta,
        "ocr": ocr_diag,
    }

    log.info(
        "Image processing completed: node=%s elapsed=%.2fs",
        node,
        elapsed,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR title block localization and visualization test"
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--image",
        help="Path to a single image",
    )

    source.add_argument(
        "--dir",
        help="Root directory containing images",
    )

    parser.add_argument(
        "--node-id",
        help="Manual node ID for single-image mode",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path",
    )

    parser.add_argument(
        "--out",
        default="output",
        help="Output directory",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N images",
    )

    parser.add_argument(
        "--engines",
        help="Override OCR engines, for example: paddle,tesseract",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_handler = setup_logging(
        Path(args.out) / "logs",
        args.verbose,
    )

    started = time.time()

    log.info(
        "Program started: output=%s verbose=%s",
        args.out,
        args.verbose,
    )

    log.info(
        "Arguments: %s",
        vars(args),
    )

    # Load project configuration.
    cfg = Config.load(args.config)

    if args.engines:
        cfg.ocr.engines = [
            engine.strip()
            for engine in args.engines.split(",")
            if engine.strip()
        ]

    # Build the configured OCR engines.
    engines = build_engines(
        cfg.ocr.engines,
        cfg.ocr.lang,
        cfg.ocr.paddle_max_side_limit,
        cfg.ocr.tesseract_psm_coarse,
    )

    log.info(
        "OCR engines loaded: %s",
        [engine.name for engine in engines],
    )

    jobs = collect(args)

    if not jobs:
        sys.exit("No images were found.")

    _out(f"Images to process: {len(jobs)}")

    succeeded = 0
    failed = 0

    for index, (path, node) in enumerate(jobs, 1):
        _out(
            f"[{index}/{len(jobs)}] "
            f"node={node} "
            f"file={Path(path).name}"
        )

        try:
            result = process_image(
                path,
                node,
                cfg,
                engines,
                args.out,
            )

        except Exception as exc:
            failed += 1

            log.exception(
                "Image processing failed: node=%s file=%s",
                node,
                path,
            )

            _out(f"    Failed: {exc}")

            continue

        succeeded += 1

        _out(
            f"    OCR tokens: {result['token_count']}"
        )

        _out(
            f"    Localization: "
            f"{result['localization_method']} "
            f"confidence={result['localization_confidence']:.3f}"
        )

        _out(
            f"    Bounding box: {result['title_bbox']}"
        )

        _out(
            f"    Token visualization: "
            f"{result['token_visualization']}"
        )

        _out(
            f"    Overview visualization: "
            f"{result['overview_visualization']}"
        )

        _out(
            f"    Title block crop: "
            f"{result['titleblock_crop']}"
        )

    elapsed = time.time() - started

    log.info(
        "Program finished: total=%d succeeded=%d failed=%d elapsed=%.2fs log=%s",
        len(jobs),
        succeeded,
        failed,
        elapsed,
        log_handler.current_path or Path(args.out) / "logs",
    )

    _out("")
    _out(f"Total: {len(jobs)}")
    _out(f"Succeeded: {succeeded}")
    _out(f"Failed: {failed}")
    _out(f"Debug images: {Path(args.out) / 'debug'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()