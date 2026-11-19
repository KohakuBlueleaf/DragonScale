"""Structured logger: `log.info("message", key=value)`; rank 0 writes, a key warns once."""

import logging
import os
import sys

ROOT = "dragonscale"


class Logger:
    """`logging` with keyword fields rendered after the message."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        self._once: set[str] = set()

    def _emit(self, level: int, message: str, fields: dict) -> None:
        if os.environ.get("RANK", "0") != "0":
            return
        text = " ".join(f"{k}={v}" for k, v in fields.items())
        self._log.log(level, f"{message} {text}" if text else message)

    def debug(self, message: str, **fields) -> None:
        self._emit(logging.DEBUG, message, fields)

    def info(self, message: str, **fields) -> None:
        self._emit(logging.INFO, message, fields)

    def warning(self, message: str, **fields) -> None:
        self._emit(logging.WARNING, message, fields)

    def warning_once(self, key: str, message: str, **fields) -> None:
        """`warning`, at most once per `key` in this process."""
        if key not in self._once:
            self._once.add(key)
            self.warning(message, **fields)


def get_logger(name: str) -> Logger:
    """A logger under the package root; the first call installs a stderr handler."""
    root = logging.getLogger(ROOT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("DRAGONSCALE_LOG_LEVEL", "INFO"))
    return Logger(f"{ROOT}.{name}")
