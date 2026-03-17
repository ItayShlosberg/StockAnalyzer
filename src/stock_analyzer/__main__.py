"""Entry point: python -m stock_analyzer"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from stock_analyzer.config import RESULTS_DIR
from stock_analyzer.pipeline import ScreenerPipeline


def _setup_logging() -> Path:
    """
    Configure dual logging: console (INFO summary) + per-run log file (DEBUG detail).
    Returns the log file path.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"run_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    ))
    root.addHandler(file_handler)

    for noisy in ("httpx", "openai", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.INFO)

    return log_path


def main() -> None:
    log_path = _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log file: %s", log_path)

    pipeline = ScreenerPipeline()
    result = pipeline.run()

    if not result.validation.passed:
        print("\nNo early-stage momentum candidates found in this run.")
    else:
        print(f"\nFound {len(result.validation.passed)} validated early-stage candidates.")

    print(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
