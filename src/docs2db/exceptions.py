"""Custom exceptions for the Docs2DB application."""


class Docs2DBException(Exception):
    """Base exception for expected/user-facing errors in Docs2DB.

    These exceptions represent "normal" error conditions that should be
    presented to the user without a traceback (e.g., missing files,
    configuration errors, database connection issues).

    All other exceptions will bubble up and show tracebacks for debugging.
    """

    pass


class DatabaseError(Docs2DBException):
    """Database-related errors (connection, missing database, etc.)."""

    pass


class ContentError(Docs2DBException):
    """Content-related errors (missing directories, invalid files, etc.)."""

    pass


class ConfigurationError(Docs2DBException):
    """Configuration-related errors (missing config files, invalid settings, etc.)."""

    pass
