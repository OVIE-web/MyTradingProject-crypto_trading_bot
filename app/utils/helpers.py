"""Small reusable helper functions shared across the application."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import TypeVar

T = TypeVar("T")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_decimal(value: object, *, field_name: str = "value") -> Decimal:
    """Convert a value to Decimal with a clear validation error."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal number") from exc


def quantize_decimal(
    value: Decimal | int | float | str,
    places: str = "0.00000001",
) -> Decimal:
    """Round a numeric value to a fixed decimal precision."""
    return to_decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between an inclusive minimum and maximum."""
    if minimum > maximum:
        raise ValueError("minimum must be less than or equal to maximum")
    return max(minimum, min(maximum, value))


def chunked[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield items in fixed-size chunks."""
    if size < 1:
        raise ValueError("size must be at least 1")

    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk
