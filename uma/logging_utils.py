from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO


def log_values(logger: logging.Logger, *values: object,
               sep: str = " ") -> None:
    logger.info(sep.join(str(value) for value in values))


class DailyFileHandler(logging.Handler):

    terminator = "\n"

    def __init__(self, log_dir: str | Path,
                 now: Callable[[], datetime] | None = None) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now().astimezone())
        self._date = ""
        self._stream: TextIO | None = None
        self.current_path: Path | None = None

    def _ensure_stream(self) -> TextIO:
        date = self._now().date().isoformat()
        if self._stream is None or date != self._date:
            if self._stream is not None:
                self._stream.flush()
                self._stream.close()
            self._date = date
            self.current_path = self.log_dir / f"{date}.log"
            self._stream = self.current_path.open(
                "a", encoding="utf-8", newline="")
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = self._ensure_stream()
            stream.write(self.format(record) + self.terminator)
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.flush()
                self._stream.close()
                self._stream = None
        finally:
            self.release()
            super().close()


def setup_logging(log_dir: str | Path, verbose: bool = False
                  ) -> DailyFileHandler:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S")

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    daily = DailyFileHandler(log_dir)
    daily.setLevel(level)
    daily.setFormatter(formatter)
    root.addHandler(daily)
    return daily