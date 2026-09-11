from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

# Separator lines used to format the trace output.
SEP = "═" * 78
SUB = "─" * 78


# Write intermediate processing information to a trace text file.
class Tracer:
    # Initialize the tracer and create the output trace file.
    def __init__(self, node_id: str, out_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.node_id = node_id
        self.path = Path(out_dir) / "trace" / f"{node_id}.txt"
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self._head()

    # Write the trace header with the node ID and current timestamp.
    def _head(self) -> None:
        self.f.write(f"{SEP}\n")
        self.f.write(f"node_id  {self.node_id}\n")
        self.f.write(f"timestamp {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.f.write(f"{SEP}\n")
        self.f.flush()

    # Write a new section title and optional notes to the trace file.
    def section(self, title: str, note: str = "") -> None:
        if not self.enabled:
            return
        self.f.write(f"\n\n{SEP}\n{title}\n")
        if note:
            for i, line in enumerate(note.strip().split("\n")):
                self.f.write(f"  Note: {line}\n" if i == 0 else f"        {line}\n")
        self.f.write(f"{SEP}\n")
        self.f.flush()

    # Write plain text content to the trace file.
    def text(self, body: str) -> None:
        if not self.enabled:
            return
        self.f.write(body.rstrip() + "\n")
        self.f.flush()

    # Write detailed OCR token information, including position, confidence, and candidates.
    def tokens(self, items: Iterable, note: str = "") -> None:
        if not self.enabled:
            return
        items = list(items)
        if note:
            self.f.write(f"  {note}\n")
        self.f.write(f"  Total: {len(items)}\n{SUB}\n")

        # Sort tokens from top to bottom and then from left to right.
        for t in sorted(items, key=lambda x: (x.cy, x.x0)):
            tid = f"[{t.tid}]" if getattr(t, "tid", -1) >= 0 else "[-]"

            # Write the token ID and recognized text.
            self.f.write(f"{tid:>6s} {t.text!r}\n")

            # Write the token bounding box, confidence, and OCR agreement.
            self.f.write(f"       bbox={list(t.bbox)}  conf={t.conf:.3f}"
                         f"  agreement={t.agreement:.2f}\n")

            # Write all OCR candidates and mark the selected primary result.
            for c in getattr(t, "candidates", []):
                mark = "←primary" if c.text == t.text else "        "
                self.f.write(f"       {mark} {c.engine:22s} {c.text!r} ({c.conf:.3f})\n")
        self.f.flush()

    # Finish the trace, close the file, and return its path.
    def close(self) -> str:
        if self.enabled:
            self.f.write(f"\n\n{SEP}\nEnd\n{SEP}\n")
            self.f.close()
        return str(self.path)