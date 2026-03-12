class WAWDError(Exception):
    """Base exception for all WAWD errors."""


class OracleUnavailableError(WAWDError):
    """Raised when the oracle backend is unreachable."""


class RestorationError(WAWDError):
    """Raised when a file restoration fails."""


class WatcherError(WAWDError):
    """Raised when a watcher operation fails."""
