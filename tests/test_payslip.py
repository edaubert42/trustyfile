"""
Payslip module (src/modules/payslip.py) and the "payslip" document type.

The synthetic payslip follows the French model layout (bulletin de paie
clarifié, 2019+): gross, contributions, net before tax, withholding tax
(base / rate / amount), net paid. Each test edits one figure the way a forger
would and checks the matching flag fires; the untouched payslip must raise
nothing (zero false positive is the module's first requirement).
"""
import fitz
import pytest

from src.document_type import detect_document_type
from src.modules.payslip import (
    analyze_payslip,
    line_amounts,
    read_pas_line,
)

GENUINE = {
    "period": "Période du 01/03/2024 au 31/03/2024",
    "payment": "Date de paiement : 28/03/2024",
    "before_tax": "Net à payer avant impôt sur le revenu      2 400,00",
    "pas": "Impôt sur le revenu prélevé à la source   2 612,35   7,00 %   182,86",
    "net_paid": "Net payé en euros      2 217,14",
}
# 2 400,00 - 182,86 = 2 217,14 ; 2 612,35 x 7 % = 182,86


def _lines(**overrides) -> list[str]:
    parts = {**GENUINE, **overrides}
    return [
        "BULLETIN DE PAIE",
        "ACME SAS - SIRET 732 829 320 00074 - Convention collective Syntec",
        "Salarié : Jean DUPONT    Matricule 00042    Date d'entrée 01/09/2021",
        parts["period"],
        parts["payment"],
        "Salaire de base   151,67 h   3 100,00",
        "Salaire brut      3 100,00",
        "Total des cotisations salariales   700,00",
        "Net social   2 410,00",
        parts["before_tax"],
        "Net imposable   2 612,35",
        parts["pas"],
        "Taux personnalisé",
        parts["net_paid"],
        "Congés payés : acquis 2,08 jours",
    ]


class _PDF:
    """Minimal stand-in for PDFData: what analyze_payslip reads."""

    def __init__(self, lines: list[str]):
        self.text_by_page = ["\n".join(lines)]
        self.extraction_ok = True
        self.file_path = "synthetic.pdf"


def _codes(**overrides) -> list[str]:
    return [f.code for f in analyze_payslip(_PDF(_lines(**overrides))).flags]


# --- reading amounts ------------------------------------------------------------

def test_amounts_keep_thousands_groups_and_skip_percentages():
    assert line_amounts("Impôt prélevé à la source  2 612,35  4,30 %  112,33") == [2612.35, 112.33]


def test_amounts_with_narrow_no_break_space_thousands():
    assert line_amounts("Net payé 2 217,14") == [2217.14]


def test_pas_line_pairs_base_and_amount_by_arithmetic_not_by_column_order():
    # Amount printed before the base: still read correctly.
    pas = read_pas_line(["Net imposable 2 612,35",
                         "Prélèvement à la source  182,86  7,00 %  2 612,35"])
    assert pas == {"base": 2612.35, "rate": 7.0, "amount": 182.86, "consistent": True}


def test_rate_amount_and_cumul_without_a_base_is_not_misread():
    # "rate | amount | year-to-date cumul": no printed base. Taking the cumul
    # for a base would invent a mismatch on an honest payslip.
    pas = read_pas_line(["Net imposable 2 612,35",
                         "Impôt prélevé à la source   7,00 %   182,86   2 194,32"])
    assert pas is None
    assert _codes(pas="Impôt prélevé à la source   7,00 %   182,86   2 194,32") == []


def test_reduced_base_different_from_taxable_net_is_not_checked():
    # Apprentices / short contracts get an allowance: base < taxable net.
    assert _codes(pas="Impôt sur le revenu prélevé à la source   2 100,00   7,00 %   182,86") == []


# --- the untouched payslip -----------------------------------------------------

def test_genuine_payslip_raises_nothing():
    result = analyze_payslip(_PDF(_lines()))
    assert result.flags == []
    assert result.score == 100


def test_paid_on_the_28th_for_the_current_month_is_normal():
    # Paid before the END of the period: the usual case, never flagged.
    assert _codes(payment="Date de paiement : 25/03/2024") == []


# --- one edited figure each -----------------------------------------------------

def test_edited_net_paid_is_flagged():
    flags = analyze_payslip(_PDF(_lines(net_paid="Net payé en euros      2 817,14"))).flags
    assert [f.code for f in flags] == ["PAYSLIP_NET_MISMATCH"]
    assert flags[0].details["expected"] == 2217.14
    assert flags[0].severity == "critical"


def test_edited_net_before_tax_is_flagged():
    assert _codes(before_tax="Net à payer avant impôt sur le revenu   3 000,00") == ["PAYSLIP_NET_MISMATCH"]


def test_edited_withholding_amount_is_flagged_on_the_rate_not_twice():
    # 2 612,35 x 7 % is 182,86, not 82,86. The rate check fires; the net
    # check stands down because the PAS amount itself is now unreliable.
    assert _codes(pas="Impôt sur le revenu prélevé à la source   2 612,35   7,00 %   82,86") == [
        "PAYSLIP_PAS_RATE_MISMATCH"]


def test_payment_before_the_period_started_is_flagged():
    assert _codes(payment="Date de paiement : 28/02/2024") == ["PAYSLIP_PAID_BEFORE_PERIOD"]


# --- ambiguity means silence, never a guess -------------------------------------

def test_cumulative_column_on_the_net_line_skips_the_check():
    # "2 817,14   26 600,00": which one is this month's? Unknown -> no check.
    assert _codes(net_paid="Net payé en euros   2 817,14   26 600,00") == []


def test_an_advance_already_paid_skips_the_net_check():
    # Net paid = net before tax - PAS - advance: the simple identity does not hold.
    lines = _lines(net_paid="Net payé en euros      1 717,14") + ["Acompte versé le 15/03   500,00"]
    assert [f.code for f in analyze_payslip(_PDF(lines)).flags] == []


def test_no_withholding_line_checks_nothing():
    lines = [l for l in _lines() if "prélevé" not in l]
    result = analyze_payslip(_PDF(lines))
    assert result.flags == []
    assert result.confidence < 1.0


def test_withholding_amount_alone_still_feeds_the_net_check():
    lines = _lines(pas="Impôt sur le revenu prélevé à la source   182,86",
                   net_paid="Net payé en euros   2 317,14")
    assert [f.code for f in analyze_payslip(_PDF(lines)).flags] == ["PAYSLIP_NET_MISMATCH"]


def test_impossible_dates_are_left_to_the_content_module():
    assert _codes(payment="Date de paiement : 31/02/2024") == []


def test_unreadable_pdf_is_neutral():
    pdf = _PDF([])
    pdf.extraction_ok = False
    assert analyze_payslip(pdf).flags == []


# --- document type ---------------------------------------------------------------

def test_payslip_is_classified_as_payslip():
    assert detect_document_type(_PDF(_lines())) == "payslip"


def test_tax_notice_wording_is_not_a_payslip():
    tax_notice = [
        "Avis d'imposition - impôt sur le revenu 2023",
        "Direction générale des finances publiques",
        "Revenu fiscal de référence 67 901",
        "Prélèvement à la source déjà versé 10 608",
        "Salaires et assimilés 70 000",
    ]
    assert detect_document_type(_PDF(tax_notice)) == "tax_document"


# --- end to end through the analyzer ----------------------------------------------

def _pdf_file(tmp_path, lines: list[str]) -> str:
    path = tmp_path / "payslip.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((40, 60 + i * 18), line, fontsize=10)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_analyzer_runs_the_module_on_payslips_only(tmp_path):
    from src.analyzer import TrustyFileAnalyzer

    analyzer = TrustyFileAnalyzer()
    forged = analyzer.analyze(_pdf_file(tmp_path, _lines(net_paid="Net payé en euros   2 817,14")))
    assert forged.document_type == "payslip"
    payslip_module = next(m for m in forged.modules if m.module == "payslip")
    assert [f.code for f in payslip_module.flags] == ["PAYSLIP_NET_MISMATCH"]

    invoice = analyzer.analyze(_pdf_file(tmp_path, [
        "FACTURE n° 2024-001", "Total HT 100,00", "TVA 20,00", "Total TTC 120,00",
        "Net à payer 120,00", "IBAN FR76 3000 6000 0112 3456 7890 189", "Échéance 30/04/2024",
    ]))
    assert all(m.module != "payslip" for m in invoice.modules)


@pytest.mark.parametrize("overrides", [{}, {"payment": "Date de paiement : 25/03/2024"}])
def test_genuine_payslip_end_to_end_has_no_payslip_flag(tmp_path, overrides):
    from src.analyzer import TrustyFileAnalyzer

    result = TrustyFileAnalyzer().analyze(_pdf_file(tmp_path, _lines(**overrides)))
    codes = [f.code for m in result.modules for f in m.flags]
    assert not [c for c in codes if c.startswith("PAYSLIP_")]


def test_edited_net_paid_is_not_scored_as_genuine(tmp_path):
    """A broken legal identity must at least say "verify this" (REVIEW)."""
    from src.analyzer import TrustyFileAnalyzer

    forged = TrustyFileAnalyzer().analyze(_pdf_file(tmp_path, _lines(net_paid="Net payé en euros   2 817,14")))
    assert forged.band != "GENUINE"


# --- table layouts (what payroll software actually produces) ----------------------

def _table_pdf(tmp_path, rows, name="table.pdf") -> str:
    """Each cell drawn separately, like a payroll table: label at x=40,
    base / rate / amount columns at x=300 / 400 / 480."""
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 40), "BULLETIN DE PAIE   Période du 01/03/2024 au 31/03/2024", fontsize=10)
    for i, row in enumerate(rows):
        for x, cell in zip((40, 300, 400, 480), row):
            if cell:
                page.insert_text((x, 80 + i * 20), cell, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def _table_rows(net_paid="2 217,14", side_note=None):
    return [
        ("Net à payer avant impôt sur le revenu", "", "", "2 400,00"),
        ("Net imposable", "", "", "2 612,35"),
        ("Impôt sur le revenu prélevé à la source", "2 612,35", "7,00 %", "182,86"),
        ("Net payé en euros", side_note or "", "", net_paid),
    ]


class _FilePDF(_PDF):
    def __init__(self, path):
        super().__init__([])
        self.file_path = path


def test_plain_text_splits_table_cells_so_rows_are_rebuilt(tmp_path):
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.payslip import visual_lines

    path = _table_pdf(tmp_path, _table_rows())
    assert "Net payé en euros\n2 217,14" in extract_pdf_data(path).text_by_page[0]
    rows = visual_lines(path)
    assert any(r.startswith("Net payé en euros") and r.endswith("2 217,14") for r in rows)


def test_table_layout_genuine_raises_nothing(tmp_path):
    assert analyze_payslip(_FilePDF(_table_pdf(tmp_path, _table_rows()))).flags == []


def test_table_layout_edited_net_is_flagged(tmp_path):
    flags = analyze_payslip(_FilePDF(_table_pdf(tmp_path, _table_rows(net_paid="2 817,14")))).flags
    assert [f.code for f in flags] == ["PAYSLIP_NET_MISMATCH"]


def test_another_columns_label_on_the_row_rejects_the_value(tmp_path):
    # The net is printed on the row BELOW its label, and the label's row
    # holds another column's "Congés pris 3,00". Reading 3,00 as the net
    # paid would accuse an honest payslip; the foreign words reject it.
    rows = _table_rows(net_paid="", side_note="Congés pris 3,00") + [("", "", "", "2 217,14")]
    assert analyze_payslip(_FilePDF(_table_pdf(tmp_path, rows))).flags == []


def test_column_gap_never_glues_two_numbers_into_one_amount(tmp_path):
    from src.modules.payslip import line_amounts, visual_lines

    path = _table_pdf(tmp_path, [("Heures", "35", "", "100,00")])
    row = next(r for r in visual_lines(path) if r.startswith("Heures"))
    assert line_amounts(row) == [100.0]


def test_french_sentences_use_french_number_and_date_formats():
    from src.summary import SENTENCE_TEMPLATES_FR

    net = analyze_payslip(_PDF(_lines(net_paid="Net payé en euros   2 817,14"))).flags[0]
    sentence = SENTENCE_TEMPLATES_FR["PAYSLIP_NET_MISMATCH"](net)
    assert "2 400,00" in sentence and "2 817,14" in sentence

    paid = analyze_payslip(_PDF(_lines(payment="Date de paiement : 28/02/2024"))).flags[0]
    assert "28/02/2024" in SENTENCE_TEMPLATES_FR["PAYSLIP_PAID_BEFORE_PERIOD"](paid)
