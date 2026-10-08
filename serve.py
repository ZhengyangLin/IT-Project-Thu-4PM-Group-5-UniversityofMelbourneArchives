# Entry point for the UMA OCR Web API.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from uma.logging_utils import setup_logging


# Start the UMA OCR Web API.
def main() -> None:
    ap = argparse.ArgumentParser(description="Start the UMA OCR Web API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="output", help="Directory for logs and task outputs")
    ap.add_argument("--reload", action="store_true", help="Enable automatic reload in development mode")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # Configure application logging.
    setup_logging(Path(args.out) / "logs", args.verbose)
    log = logging.getLogger(__name__)

    # Import the web server dependency.
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Missing web backend dependencies. Run: pip install -r requirements.txt") from exc

    # Start the API server.
    log.info("Starting UMA OCR API: http://%s:%d docs=/docs reload=%s",
             args.host, args.port, args.reload)
    uvicorn.run("uma.web.app:app", host=args.host, port=args.port,
                reload=args.reload, log_config=None)


if __name__ == "__main__":
    main()