import logging
import os

import structlog

# Determine log level from environment
if os.getenv("DEBUG") == "1":
    log_level = logging.DEBUG
elif os.getenv("LOG_LEVEL"):
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
else:
    log_level = logging.INFO

logging.basicConfig(level=logging.WARNING)

# Configure structlog for beautiful console output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
