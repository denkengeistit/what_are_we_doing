class WAWDError(Exception):
    """Base exception for all WAWD errors."""


class OracleUnavailableError(WAWDError):
    """Raised when the oracle backend is unreachable."""


class RestorationError(WAWDError):
    """Raised when a file restoration fails."""


class FUSEError(WAWDError):
    """Raised when a FUSE operation fails."""
