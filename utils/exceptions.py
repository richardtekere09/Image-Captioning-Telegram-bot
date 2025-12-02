"""
Custom Exceptions for API Service
"""

from typing import Any, Optional


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self, message: str, status_code: int = 500, details: Optional[dict] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ImageValidationError(APIError):
    """Raised when image validation fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            status_code=400,  # Bad Request
            details=details,
        )


class ModelError(APIError):
    """Raised when model inference fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            status_code=500,  # Internal Server Error
            details=details,
        )


class InvalidRequestError(APIError):
    """Raised when request format is invalid."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            status_code=422,  # Unprocessable Entity
            details=details,
        )
