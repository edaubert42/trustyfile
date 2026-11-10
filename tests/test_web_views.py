"""Tests for the HTML views (src/web/views/) and the per-document view model.

The views are plain functions: data and a language in, an HTML string out.
That is what these tests rely on: no Streamlit, no browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analyzer import TrustyFileAnalyzer
from src.batch import BatchItem
from src.i18n import translate
from src.web.viewmodel import DocumentView, _editor_name, build_document_view
from src.web.views import batch as batch_views
from src.web.views import details, errors, findings, landing, properties, verdict
from src.web.virustotal import check_virustotal

DEMO = Path(__file__).resolve().parent.parent / "public" / "static" / "demo" / "facture-exemple.pdf"
HOSTILE = '<img src=x onerror="alert(1)">'


@pytest.fixture(scope="module")
def demo():
    """The sample invoice, analysed once: (result, view)."""
    result = TrustyFileAnalyzer(enable_external=False).analyze(str(DEMO))
    return result, build_document_view(str(DEMO), result, DEMO.name)


def test_view_model_gathers_what_the_page_shows(demo):
    result, view = demo
    assert view.file_name == DEMO.name
    assert view.file_hash == result.file_hash
    assert view.page_count >= 1
    assert view.all_flags, "the sample invoice is tampered: it has findings"
    assert len(view.reasons) == len(view.reason_numbers)
    assert view.virustotal is None          # no lookup function given


def test_view_model_calls_the_virustotal_lookup_once_with_the_hash(demo):
    result, _ = demo
    seen = []
    view = build_document_view(str(DEMO), result, DEMO.name,
                               virustotal_lookup=lambda sha: seen.append(sha) or {"status": "clean"})
    assert seen == [result.file_hash]
    assert view.virustotal == {"status": "clean"}


def test_editor_name_is_read_from_the_last_revision():
    assert _editor_name({"diffs": [{"to_tool": "Word → modified with iLovePDF"}]}) == "iLovePDF"
    assert _editor_name({"diffs": [{"to_tool": "Word"}]}) is None
    assert _editor_name({"diffs": []}) is None


def test_virustotal_without_a_key_makes_no_call():
    assert check_virustotal("abc", None) is None
    assert check_virustotal("abc", "") is None


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_results_views_render_in_both_languages(demo, lang):
    result, view = demo
    from src.summary import generate_rich_summary
    summary = generate_rich_summary(result, lang=lang)

    gauge = verdict.gauge_html(result, summary, lang)
    assert "<svg" in gauge and translate(f"risk_{result.risk_level}", lang) in gauge

    summary_block = findings.summary_html(result, view, lang)
    assert translate("summary", lang) in summary_block

    props = properties.file_properties_html(view, lang)
    assert DEMO.name in props and result.file_hash in props

    fold = details.all_details_html(
        findings.evidence_bars_html(result, lang),
        details.issues_html(view.all_flags, result.document_type, lang),
        details.modules_panel_html(result, lang),
        lang,
    )
    assert fold.startswith('<details class="tf-all-details"')
    assert translate("detected_issues", lang) in fold


def test_file_properties_escape_what_the_pdf_controls(demo):
    """File name, producer and signer come from the upload: never raw HTML."""
    _, view = demo
    hostile = DocumentView(**{**view.__dict__, "file_name": HOSTILE,
                              "signature_info": HOSTILE, "editor_name": HOSTILE})
    out = properties.file_properties_html(hostile, "en")
    assert "<img" not in out
    assert "&lt;img" in out


def test_virustotal_pill_follows_the_status(demo):
    _, view = demo
    link = "https://www.virustotal.com/gui/file/x"
    for status, key in (("clean", "vt_no_threats"), ("unknown", "vt_unknown")):
        shown = DocumentView(**{**view.__dict__, "virustotal":
                                {"status": status, "malicious": 0, "total": 70, "link": link}})
        out = properties.file_properties_html(shown, "en")
        assert translate(key, "en") in out and link in out
    assert link not in properties.file_properties_html(view, "en")


def test_error_card_carries_title_message_and_way_back():
    out = errors.error_card_html("Too big", "12.0 MB", "en")
    assert "Too big" in out and "12.0 MB" in out
    assert translate("back_to_home", "en") in out


def test_landing_lists_the_guides_and_tools_it_is_given():
    class Page:
        def __init__(self, slug, title):
            self.slug, self.title = slug, title

    out = landing.steps_html([Page("g1", "Guide <one>")], [Page("t1", "Tool")], "en")
    assert '/en/guides/g1' in out and '/en/tools/t1' in out
    assert "Guide &lt;one&gt;" in out
    # No guide, no tool: the link rows disappear, the three steps stay.
    bare = landing.steps_html([], [], "fr")
    assert "/guides/" not in bare and translate("step1_title", "fr") in bare


def test_batch_row_of_a_refused_file_says_why():
    item = BatchItem(name="notes.pdf", size_bytes=5, pages=0, problem="not_a_pdf")
    band, score, finding = batch_views.row_cells_html(item, "en")
    assert translate("batch_refused", "en") in band
    assert "—" in score
    assert translate("batch_problem_not_a_pdf", "en") in finding
    assert "&lt;img" in batch_views.file_name_html(HOSTILE)
