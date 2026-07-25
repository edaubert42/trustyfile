"""PDF preview rendering + evidence-region collection (spec 2026-07-21).

Pure PyMuPDF + dataclass code, zero Streamlit dependency, so every function
here is directly unit-testable. app.py imports from this module instead of
defining its own rendering helper.
"""
from typing import Iterator

from src.models import AnalysisResult


def _iter_geometry(result: AnalysisResult) -> Iterator[tuple[int, tuple[float, float, float, float]]]:
    """Yield (page_index, bbox) for every flag with usable geometry.

    Flags carry geometry in one of two shapes (both display-only additions to
    their modules, never consulted by scoring):
      - flat: details == {..., "page": int, "bbox": (x0,y0,x1,y1), ...}
        (overlay.py's 5 geometry-bearing flags)
      - instances: details == {..., "instances": [{"page": int,
        "bbox": (x0,y0,x1,y1), ...}, ...], ...} (invisible.py, one flag can
        cover several regions)

    A flag with neither shape, or a malformed bbox, yields nothing — never
    raises. This is decoration over an untrusted-input tool's output; it
    must degrade silently, the same posture as every other display path.
    Shared by both public functions below so the shape-normalization logic
    exists exactly once.
    """
    for module_result in result.modules:
        for flag in module_result.flags:
            details = flag.details or {}
            candidates = details.get("instances")
            if candidates is None:
                candidates = [details] if "bbox" in details else []
            for c in candidates:
                page = c.get("page")
                bbox = c.get("bbox")
                if page is None or bbox is None:
                    continue
                try:
                    x0, y0, x1, y1 = (float(v) for v in bbox)
                except (TypeError, ValueError):
                    continue
                yield page, (x0, y0, x1, y1)


def collect_evidence_boxes(result: AnalysisResult, page_index: int) -> list[tuple[float, float, float, float]]:
    """Every tamper-region bbox any flag reports for the given 0-based page."""
    return [bbox for page, bbox in _iter_geometry(result) if page == page_index]


def first_flagged_page(result: AnalysisResult) -> int:
    """Lowest 0-based page index any flag has usable geometry for; 0 if none."""
    pages = {page for page, _bbox in _iter_geometry(result)}
    return min(pages) if pages else 0
