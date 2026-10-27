"""Explicit optional total-job deadlines; stall and resource checks are separate."""
from __future__ import annotations


def validate_max_seconds(value):
    """JSON null means no elapsed job limit; explicit positive integers remain valid."""
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("max_seconds must be null or a positive integer")
    return value


def elapsed_limit_exceeded(elapsed_seconds, max_seconds):
    maximum = validate_max_seconds(max_seconds)
    return maximum is not None and elapsed_seconds > maximum
