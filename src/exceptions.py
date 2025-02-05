class BaseError(Exception):
    """Base class for exceptions."""


class ConfigurationError(BaseError):
    """Exception raised when encountering an invalid configuration."""
