"""
Custom exceptions for runtime stability control.
"""


class StabilityViolation(Exception):
    """
    Raised when a destabilizing update is detected.
    """
    pass
