"""find_amount_in_pdf: one printed amount at a time, sign carried by the layout.

The text below reproduces the layout of a real Avis d'impôt 2023 (values
changed): a calculation column with one number per line, an operator minus
separated from the digits by spaces, and the balance repeated in words on
page 1. The old pattern used \\s and glued the column into one number.
"""
import pytest

from src.modules.twod_doc import find_amount_in_pdf

TAX_NOTICE = (
    "Impôt sur le revenu net\n       7530\n___________\n       7530\n"
    "-     10608\n-      3078\n       3078\n      67901\n      13702\n"
    "16/04/2024\n67 901\n1.0\nSomme qui vous est remboursée\n3 078,00 €\n"
)


@pytest.mark.parametrize("amount", [67901, 10608, -3078, 7530, 13702])
def test_values_of_a_calculation_column_are_found(amount):
    assert find_amount_in_pdf(TAX_NOTICE, amount) is not None


def test_numbers_on_consecutive_lines_are_not_glued():
    # "67\n901" is two numbers, not 67 901.
    assert find_amount_in_pdf("ligne 67\n901 fin", 67901) is None


def test_thousands_separator_space_is_still_understood():
    assert find_amount_in_pdf("Revenu fiscal de référence 67 901", 67901) == "67 901"
    assert find_amount_in_pdf("Total 1\u00a0234,56 €", 1234.56) == "1\u00a0234,56"


def test_a_changed_value_is_not_found():
    # The tamper case this check exists for: the visible RFR was lowered.
    assert find_amount_in_pdf("Revenu fiscal de référence 57 901", 67901) is None


def test_a_number_containing_the_value_is_not_a_match():
    assert find_amount_in_pdf("référence 167901", 67901) is None
