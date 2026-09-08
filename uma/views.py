from __future__ import annotations

from statistics import median

from .config import ViewCfg
from .patterns import scan_candidates
from .models import Line, Token


# Assign a unique token ID and row number to each OCR token.
def assign_ids(lines: list[Line]) -> list[Token]:

    # Sort lines from top to bottom and then from left to right.
    ordered = sorted(lines, key=lambda l: (l.y0, l.x0))
    tid = 0
    flat: list[Token] = []

    # Assign row numbers and unique token IDs.
    for row_no, line in enumerate(ordered, 1):
        line.row_no = row_no

        # Process tokens in each line from left to right.
        for t in sorted(line.tokens, key=lambda x: x.x0):
            t.tid = tid
            tid += 1
            flat.append(t)

    return flat


# Build a text grid that approximately preserves the original spatial layout.
def build_grid(lines: list[Line], cfg: ViewCfg) -> str:
    if not lines:
        return ""

    # Sort lines by their spatial position.
    ordered = sorted(lines, key=lambda l: (l.y0, l.x0))
    toks = [t for l in ordered for t in l.tokens]

    # Estimate the average width of one character in image coordinates.
    widths = []
    for t in toks:
        n = max(len(t.text), 1)
        if t.h > 0 and (t.w / t.h) < 60:
            widths.append(t.w / n)

    # Use the median character width as the grid spacing unit.
    unit = max(median(widths) if widths else 8.0, 1.0)

    # Use the leftmost token as the starting reference position.
    x_min = min(t.x0 for t in toks)
    out: list[str] = []
    prev_bottom = None

    for line in ordered:

        # Insert blank rows when there is a large vertical gap between text lines.
        if prev_bottom is not None:
            gap = int((line.y0 - prev_bottom) / max(line.h, 1))
            out.extend([""] * min(max(gap - 1, 0), cfg.grid_max_blank_rows))

        row = ""

        # Place each token according to its horizontal position.
        for t in sorted(line.tokens, key=lambda x: x.x0):
            col = int((t.x0 - x_min) / unit)

            # Add spaces to approximately preserve horizontal alignment.
            if col > len(row):
                row += " " * (col - len(row))

            row += t.text if not row or row.endswith(" ") else " " + t.text

        out.append(row[:cfg.grid_max_cols].rstrip())
        prev_bottom = line.y1

    return "\n".join(out)


# Build an indexed text view containing token IDs, row numbers, and OCR alternatives.
def build_index(lines: list[Line], cfg: ViewCfg,
                hints: dict[int, dict] | None = None) -> str:

    hints = hints or {}

    # Sort lines by their spatial position.
    ordered = sorted(lines, key=lambda l: (l.y0, l.x0))
    rows: list[str] = []

    for line in ordered:

        # Process tokens from left to right within each line.
        for t in sorted(line.tokens, key=lambda x: x.x0):
            prefix = f"[{t.tid}]"

            # Optionally include the row number in the output.
            if cfg.show_row_no:
                prefix += f" row {line.row_no}"

            row = f"{prefix}  {t.text}"

            # Display alternative OCR candidates when available.
            alts = t.alternatives()
            if alts:
                row += "   ← alternatives " + " | ".join(
                    f"{a.engine}:{a.text}" for a in alts)

            # Display a preferred candidate hint when one is available.
            h = hints.get(t.tid)
            if h:
                row += f"\n{'':>{len(prefix)}}   ⚠ prefer {h['preferred']!r}"
                row += ": " + "; ".join(h["reasons"])

            rows.append(row)

    return "\n".join(rows)


# Build both the grid view and index view from the same OCR lines.
def build_views(lines: list[Line], cfg: ViewCfg,
                hints: dict[int, dict] | None = None
                ) -> tuple[str, str, list[Token]]:

    # Assign IDs before building the views.
    flat = assign_ids(lines)

    # Automatically inspect alternative OCR candidates when no hints are provided.
    if hints is None:
        hints = {}

        for t in flat:
            better, extra = scan_candidates(t)

            if better is None:
                continue

            # Record a suggestion when an alternative candidate appears more complete.
            hints[t.tid] = {
                "preferred": better.text,
                "reasons": [f"alternative {better.engine} fully contains the "
                            f"primary candidate and adds {extra!r}, suggesting "
                            "the detector missed narrow characters"],
            }

    # Return the spatial grid view, indexed view, and flattened token list.
    return build_grid(lines, cfg), build_index(lines, cfg, hints), flat