"""
End-to-end verification: the evidence ledger flows through the real,
wired TrustyFileAnalyzer on the forged corpus.

This is not a unit test of any single module — it proves that the
plumbing (analyzer -> modules -> create_analysis_result -> evidence
ledger -> band/score_low/score_high) works together on realistic PDFs
built by scripts/build_corpus.py:

  - A clean, internally-consistent invoice should score high, land in
    the GENUINE band, and have an empty evidence ledger (every module
    abstains — no flags raised).
  - A forged invoice with tampered metadata (ModDate 45 days after
    CreationDate + an online-converter producer) should score lower,
    land outside GENUINE, and carry at least one piece of evidence.

Two other forgeries in the corpus (white-box amount overlay, and an
incremental-update edit) are NOT asserted here because current modules
don't detect the overlay and only weakly flag the incremental edit
(still GENUINE) — those are known Phase-1 gaps tracked for later
calibration work, not wiring bugs. Asserting on them here would be a
false failure.
"""

from scripts.build_corpus import build_corpus
from src.analyzer import TrustyFileAnalyzer


def test_corpus_scores_through_ledger(tmp_path):
    paths = build_corpus(str(tmp_path))
    a = TrustyFileAnalyzer(enable_external=False)

    clean = a.analyze(paths["legitimate"][0])
    assert clean.band == "GENUINE"
    assert clean.risk_level == "LOW"
    assert clean.evidence_ledger == []          # clean doc: all modules abstain
    assert clean.score_low <= clean.trust_score <= clean.score_high

    # The metadata-date-tampered forgery carries real findings → flagged, not GENUINE.
    forged = a.analyze([p for p in paths["suspicious"] if "metadata" in p][0])
    assert forged.band in ("REVIEW", "SUSPICIOUS")
    assert forged.trust_score < clean.trust_score
    assert len(forged.evidence_ledger) >= 1
    assert forged.score_low <= forged.trust_score <= forged.score_high
