"""logging 設定。批次跑數十支影片時，log 是唯一能回頭查「當時發生什麼」的東西，
所以預設同時輸出到 stderr 與 out/weft.log。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    try:
        from rich.logging import RichHandler

        console: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        console.setFormatter(logging.Formatter("%(name)-22s %(message)s", datefmt=_DATEFMT))
    except ImportError:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logging.getLogger("weft")
