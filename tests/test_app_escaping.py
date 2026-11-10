"""
Tests for esc() (src/web/views/common.py): the HTML-escaping helper used before interpolating
attacker-controlled values (PDF metadata, filename) into
st.markdown(..., unsafe_allow_html=True) blocks.

These tests only import the pure `esc` helper — they do not
run Streamlit or exercise the file-properties rendering block.
"""


def test_esc_neutralizes_html():
    from src.web.views import common as app
    out = app.esc('<img src=x onerror="alert(1)">')
    assert "<img" not in out
    assert "&lt;img" in out
    assert '"' not in out or "&quot;" in out


def test_esc_handles_none():
    from src.web.views import common as app
    assert app.esc(None) == ""
