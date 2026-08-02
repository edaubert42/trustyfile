"""
Comparing dates that come from different places.

TrustyFile reads dates from two sources that do not agree about timezones:

- **PDF metadata**, via `parse_pdf_date`, which returns timezone-aware UTC
  datetimes because PDF date strings carry an offset (``D:20250115103000+02'00'``).
- **Document text**, via `extract_dates_from_text`, which returns naive
  datetimes — a date printed on an invoice says nothing about a timezone.

Subtracting one from the other raises::

    TypeError: can't subtract offset-naive and offset-aware datetimes

This has now broken three separate call sites, most recently crashing the
whole result page in production for any invoice whose metadata carried an
offset. Each site fixed it locally and the next one repeated the mistake, so
the coercion lives here once and gets tested.

The rule: reduce everything to **naive UTC** before comparing. Naive rather
than aware, because the text-derived side can never be made aware honestly —
we do not know what timezone an invoice was printed in.
"""

from __future__ import annotations

from datetime import datetime, timezone


def to_naive_utc(value: datetime | None) -> datetime | None:
    """Reduce a datetime to naive UTC so it can be compared with any other.

    Aware datetimes are converted to UTC and stripped of their tzinfo. Naive
    datetimes are returned unchanged — they are assumed to already be "wall
    clock" values with no zone.

    Args:
        value: The datetime to normalise, or None.

    Returns:
        A naive datetime in UTC, or None if `value` was None.

    Examples:
        >>> from datetime import timezone, timedelta
        >>> aware = datetime(2025, 1, 15, 10, 30, tzinfo=timezone(timedelta(hours=2)))
        >>> to_naive_utc(aware)
        datetime.datetime(2025, 1, 15, 8, 30)
        >>> to_naive_utc(datetime(2025, 1, 15, 10, 30))
        datetime.datetime(2025, 1, 15, 10, 30)
        >>> to_naive_utc(None) is None
        True
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def to_aware_utc(value: datetime | None) -> datetime | None:
    """Promote a datetime to UTC-aware so it can be compared with `now()`.

    The mirror of `to_naive_utc`, for the other valid direction: when a value
    is being compared against an aware reference such as
    ``datetime.now(timezone.utc)``, everything must be aware instead. Naive
    inputs are assumed to already be UTC.

    Args:
        value: The datetime to normalise, or None.

    Returns:
        A UTC-aware datetime, or None if `value` was None.

    Examples:
        >>> to_aware_utc(datetime(2025, 1, 15, 10, 30))
        datetime.datetime(2025, 1, 15, 10, 30, tzinfo=datetime.timezone.utc)
        >>> to_aware_utc(None) is None
        True
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def seconds_between(first: datetime | None, second: datetime | None) -> float | None:
    """Absolute gap in seconds between two datetimes of any awareness.

    Args:
        first: One datetime, aware or naive, or None.
        second: The other datetime, aware or naive, or None.

    Returns:
        Absolute difference in seconds, or None if either input was None.

    Examples:
        >>> seconds_between(datetime(2025, 1, 15, 12), datetime(2025, 1, 15, 10))
        7200.0
        >>> seconds_between(datetime(2025, 1, 15), None) is None
        True
    """
    left, right = to_naive_utc(first), to_naive_utc(second)
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())


def days_between(first: datetime | None, second: datetime | None) -> int | None:
    """Absolute gap in whole days between two datetimes of any awareness.

    Partial days are truncated — a 36-hour gap is one day.

    Note the order of operations: the timedelta is made absolute *before*
    reading `.days`. `timedelta.days` floors toward negative infinity, so
    `abs((a - b).days)` reports 1 for a gap of minus eight hours, and 2 for a
    gap of minus 36 hours. Taking `abs()` of the timedelta first is what makes
    the result independent of argument order.

    Args:
        first: One datetime, aware or naive, or None.
        second: The other datetime, aware or naive, or None.

    Returns:
        Absolute difference in whole days, or None if either input was None.

    Examples:
        >>> days_between(datetime(2025, 1, 20), datetime(2025, 1, 15))
        5
        >>> days_between(datetime(2025, 1, 15), datetime(2025, 1, 20))
        5
        >>> days_between(None, datetime(2025, 1, 15)) is None
        True
    """
    left, right = to_naive_utc(first), to_naive_utc(second)
    if left is None or right is None:
        return None
    return abs(left - right).days
