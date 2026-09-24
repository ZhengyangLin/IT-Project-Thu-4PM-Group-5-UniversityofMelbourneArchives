from __future__ import annotations

import cv2
import numpy as np

from .models import Block, Cluster

# Colors used for different visualization elements.
C_BLOCK = (200, 160, 60)
C_RESPLIT = (200, 90, 190)
C_CLUSTER = (60, 160, 240)
C_TITLE = (60, 60, 230)
C_LINE = (150, 150, 150)


# Draw blocks, clusters, text lines, and the detected title block on the image.
def draw(gray: np.ndarray, blocks: list[Block], clusters: list[Cluster],
         title_bbox: tuple | None, out_path: str,
         max_width: int = 2200, show_lines: bool = False) -> str:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = cv2.addWeighted(canvas, 0.45, np.full_like(canvas, 255), 0.55, 0)

    # Optionally draw the bounding box of each text line.
    if show_lines:
        for b in blocks:
            for l in b.lines:
                x0, y0, x1, y1 = l.bbox
                cv2.rectangle(canvas, (x0, y0), (x1, y1), C_LINE, 1)

    # Draw each detected block with a label.
    for i, b in enumerate(blocks, 1):
        x0, y0, x1, y1 = b.bbox
        color = C_RESPLIT if b.source == "caption_resplit" else C_BLOCK
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 5)
        cv2.putText(canvas, f"B{i}({len(b.lines)})", (x0 + 10, y0 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    # Draw detected clusters.
    for c in clusters:
        x0, y0, x1, y1 = c.bbox
        cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), C_CLUSTER, 2)

    # Highlight the detected title block if available.
    if title_bbox:
        x0, y0, x1, y1 = title_bbox
        cv2.rectangle(canvas, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), C_TITLE, 7)
        cv2.putText(canvas, "TITLE BLOCK", (x0, max(y0 - 22, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, C_TITLE, 4)

    _legend(canvas)

    # Resize very large images before saving.
    h, w = canvas.shape[:2]
    if w > max_width:
        f = max_width / w
        canvas = cv2.resize(canvas, (int(w * f), int(h * f)),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, canvas)
    return out_path


# Draw a legend explaining the visualization colors.
def _legend(canvas: np.ndarray) -> None:
    items = [(C_BLOCK, "block (xy-cut)"), (C_RESPLIT, "block (caption resplit)"),
             (C_CLUSTER, "cluster"), (C_TITLE, "title block")]
    x, y = 24, 30
    cv2.rectangle(canvas, (x - 10, y - 22), (x + 640, y + 38 * len(items)),
                  (255, 255, 255), -1)
    for color, label in items:
        cv2.rectangle(canvas, (x, y), (x + 44, y + 24), color, -1)
        cv2.putText(canvas, label, (x + 58, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2)
        y += 38



# Visualize OCR tokens, detected frame lines, and the title-block localization result.
def draw_v2(plain, tokens, locate_result, out_path: str,
            max_width: int = 2400) -> str:
    from . import locate as _loc

    canvas = cv2.cvtColor(plain, cv2.COLOR_GRAY2BGR)
    canvas = cv2.addWeighted(canvas, 0.40, np.full_like(canvas, 255), 0.60, 0)
    H, W = plain.shape

    # Detect and draw horizontal and vertical frame lines.
    horiz, vert = _loc.line_masks(plain)
    ys, xs = _loc.line_positions(horiz, vert)
    for y in ys:
        cv2.line(canvas, (0, y), (W, y), (230, 190, 110), 2)
    for x in xs:
        cv2.line(canvas, (x, 0), (x, H), (230, 190, 110), 2)

    # Draw bounding boxes for all OCR tokens.
    for t in tokens:
        x0, y0, x1, y1 = t.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (150, 150, 150), 1)

    # Draw the detected title-block bounding box and localization information.
    x0, y0, x1, y1 = locate_result.bbox
    cv2.rectangle(canvas, (x0, y0), (x1, y1), C_TITLE, max(4, W // 500))
    label = f"{locate_result.method}  {locate_result.confidence:.2f}"
    cv2.putText(canvas, label, (x0, max(y0 - 14, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, W / 1600, C_TITLE, 3)

    # Draw a legend containing frame, OCR, and localization statistics.
    items = [((230, 190, 110), f"frame lines  h={len(ys)} v={len(xs)}"),
             ((150, 150, 150), f"ocr tokens  n={len(tokens)}"),
             (C_TITLE, f"title block  [{locate_result.method}]")]
    x, y = 24, 30
    cv2.rectangle(canvas, (x - 10, y - 22), (x + 700, y + 40 * len(items)),
                  (255, 255, 255), -1)
    for color, text in items:
        cv2.rectangle(canvas, (x, y), (x + 44, y + 24), color, -1)
        cv2.putText(canvas, text, (x + 58, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2)
        y += 40

    # Resize the visualization if it exceeds the maximum width.
    if W > max_width:
        f = max_width / W
        canvas = cv2.resize(canvas, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, canvas)
    return out_path


# Save a padded crop of the detected title block for manual inspection.
def save_titleblock_crop(gray, bbox, out_path: str, pad_ratio: float = 0.04,
                         max_width: int = 1800) -> str:
    H, W = gray.shape
    x0, y0, x1, y1 = bbox

    # Add padding around the title-block bounding box.
    px, py = int((x1 - x0) * pad_ratio), int((y1 - y0) * pad_ratio)
    cx0, cy0 = max(x0 - px, 0), max(y0 - py, 0)
    cx1, cy1 = min(x1 + px, W), min(y1 + py, H)

    # Crop the region and highlight the original title-block boundary.
    crop = cv2.cvtColor(gray[cy0:cy1, cx0:cx1], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0),
                  C_TITLE, max(2, (cx1 - cx0) // 400))

    # Resize the crop to keep it easy to inspect visually.
    if crop.shape[1] > max_width:
        f = max_width / crop.shape[1]
        crop = cv2.resize(crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)),
                          interpolation=cv2.INTER_AREA)
    elif crop.shape[1] < 700:
        f = 700 / max(crop.shape[1], 1)
        crop = cv2.resize(crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)),
                          interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, crop)
    return out_path


# Draw all OCR token bounding boxes for quick OCR debugging.
def draw_tokens(plain, tokens, out_path: str, max_width: int = 2400) -> str:
    canvas = cv2.cvtColor(plain, cv2.COLOR_GRAY2BGR)
    canvas = cv2.addWeighted(canvas, 0.45, np.full_like(canvas, 255), 0.55, 0)
    H, W = plain.shape

    # Draw a bounding box around every OCR token.
    for t in tokens:
        x0, y0, x1, y1 = t.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 160, 240), 1)

    # Display the total number of detected OCR tokens.
    cv2.rectangle(canvas, (14, 8), (700, 56), (255, 255, 255), -1)
    cv2.putText(canvas, f"ocr tokens: {len(tokens)}", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2)

    # Resize the visualization if necessary before saving.
    if W > max_width:
        f = max_width / W
        canvas = cv2.resize(canvas, (int(W * f), int(H * f)),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, canvas)
    return out_path