from __future__ import annotations

from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path

from semgraph import config

__all__ = [
    "get_logger",
]


def get_formatter(fmt: str | None = None) -> Formatter:
    if fmt is None:
        fmt = config.logging_format
    return Formatter(fmt)


def add_stream_handler(logger: Logger, fmt: str | None = None):
    handler = StreamHandler()
    formatter = get_formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def add_file_handler(
    logger: Logger, fpath: str | Path | None = None, fmt: str | None = None
):
    if fpath is None:
        fpath = config.cache_dir / "logs.txt"
    handler = FileHandler(fpath)
    formatter = get_formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str | None = None, level=INFO, fmt: str | None = None) -> Logger:
    logger = getLogger(name)
    logger.setLevel(level)
    if config.log_to_console:
        add_stream_handler(logger, fmt=fmt)
    if config.log_to_file:
        add_file_handler(logger, fmt=fmt)
    return logger
