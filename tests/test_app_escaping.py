"""
Tests for app.esc(): the HTML-escaping helper used before interpolating
attacker-controlled values (PDF metadata, filename) into
st.markdown(..., unsafe_allow_html=True) blocks.

These tests only import the pure `esc` helper from app.py — they do not
run Streamlit or exercise the file-properties rendering block.
"""


def test_esc_neutralizes_html():
    import app
    out = app.esc('<img src=x onerror="alert(1)">')
    assert "<img" not in out
    assert "&lt;img" in out
    assert '"' not in out or "&quot;" in out


def test_esc_handles_none():
    import app
    assert app.esc(None) == ""
