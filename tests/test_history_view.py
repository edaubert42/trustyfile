"""Tests for the modification-history display helpers."""
from src.history_view import fill_covered_values, pair_changes


def test_replaced_value_becomes_one_old_new_pair():
    assert pair_changes(["1 200,00 EUR"], ["2 200,00 EUR"]) == [
        ("1 200,00 EUR", "2 200,00 EUR")
    ]


def test_extra_removed_line_has_no_new_side():
    assert pair_changes(["a", "b"], ["x"]) == [("a", "x"), ("b", None)]


def test_extra_added_line_has_no_old_side():
    assert pair_changes([], ["x"]) == [(None, "x")]


def test_blank_lines_are_ignored_and_do_not_shift_the_pairing():
    assert pair_changes(["", "a"], ["  ", "x"]) == [("a", "x")]


def test_nothing_changed_gives_no_pair():
    assert pair_changes([], []) == []


def test_added_line_gets_its_old_value_from_a_cover_up_finding():
    pairs = fill_covered_values([(None, " 2 200,00 EUR ")], {"2 200,00 EUR": "1 200,00 EUR"})
    assert pairs == [("1 200,00 EUR", " 2 200,00 EUR ")]


def test_pairs_with_no_matching_finding_are_untouched():
    pairs = [(None, "x"), ("a", "b"), ("c", None)]
    assert fill_covered_values(pairs, {"b": "zzz"}) == pairs
