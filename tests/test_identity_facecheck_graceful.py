"""
Regression: the face-detection pre-check must DEGRADE GRACEFULLY when OpenCV
lacks CascadeClassifier (headless / opencv 5 preview builds don't expose it).

Before the fix, `_has_face_photo` accessed `cv2.CascadeClassifier` guarded only
by `except ImportError`, so a missing attribute raised AttributeError and
crashed `analyze()` on ANY textless (image-only / scanned) PDF — a core
document family. The "never crash on any PDF" contract requires returning
False (face detection unavailable) instead.
"""
import fitz

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.identity import _has_face_photo, is_likely_identity_doc


def _textless_image_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
    pix.clear_with(220)
    page.insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)  # image only, no text layer
    doc.save(path)
    doc.close()


def test_face_check_degrades_gracefully_on_textless_pdf(tmp_path):
    p = tmp_path / "scan.pdf"
    _textless_image_pdf(str(p))
    data = extract_pdf_data(str(p))
    # Must not raise, even when cv2.CascadeClassifier is unavailable in this env.
    assert _has_face_photo(data) is False
    assert isinstance(is_likely_identity_doc(data), bool)


def test_analyze_does_not_crash_on_textless_pdf(tmp_path):
    from src.analyzer import TrustyFileAnalyzer
    p = tmp_path / "scan.pdf"
    _textless_image_pdf(str(p))
    result = TrustyFileAnalyzer(enable_external=False).analyze(str(p))
    assert result is not None and result.trust_score is not None
