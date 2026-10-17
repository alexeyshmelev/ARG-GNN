"""Cheap queue preflight; authoritative accounting remains in existing guards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class PreflightResult:
    ready: bool
    reason: str | None
    accounting: object | None = None


def idle_preflight(*, available_slots: Callable[[], int],
                   available_rss_bytes: Callable[[], int],
                   required_rss_bytes: int,
                   capture_accounting: Callable[[], T]) -> PreflightResult:
    """Skip expensive accounting when existing slot/RSS checks already block.

    Positive preflight is advisory. Existing admission still performs its full
    fresh capture, binding, capacity and apply checks; nothing is cached here.
    """
    if type(required_rss_bytes) is not int or required_rss_bytes <= 0:
        raise ValueError("A positive original RSS requirement is required")
    slots = available_slots()
    if type(slots) is not int or slots < 0:
        raise ValueError("Invalid existing-slot count")
    if slots == 0:
        return PreflightResult(False, "no_existing_slot")
    rss = available_rss_bytes()
    if type(rss) is not int or rss < 0:
        raise ValueError("Invalid existing available RSS")
    if rss < required_rss_bytes:
        return PreflightResult(False, "insufficient_rss")
    return PreflightResult(True, None, capture_accounting())
