"""top_reasons: which findings the results page shows next to the verdict."""

from src.evidence import GENUINE_CODES
from src.models import Flag
from src.summary import MAX_TOP_REASONS, top_reasons


def _flag(severity: str, code: str) -> Flag:
    return Flag(severity=severity, code=code, message=code)


def test_keeps_the_most_severe_and_stops_at_the_limit():
    flags = [
        _flag("critical", "A"),
        _flag("high", "B"),
        _flag("high", "C"),
        _flag("high", "D"),
        _flag("medium", "E"),
    ]
    assert [f.code for f in top_reasons(flags)] == ["A", "B", "C"]
    assert MAX_TOP_REASONS == 3


def test_low_severity_is_never_a_reason():
    flags = [_flag("medium", "A"), _flag("low", "B")]
    assert [f.code for f in top_reasons(flags)] == ["A"]


def test_genuine_direction_codes_are_never_a_reason():
    genuine = next(iter(GENUINE_CODES))
    flags = [_flag("high", genuine), _flag("medium", "A")]
    assert [f.code for f in top_reasons(flags)] == ["A"]


def test_no_flags_gives_no_reasons():
    assert top_reasons([]) == []
