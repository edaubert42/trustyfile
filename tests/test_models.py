from src.models import AnalysisResult


def test_analysis_result_has_new_fields_with_defaults():
    r = AnalysisResult(file_hash="h", trust_score=90, risk_level="LOW")
    assert r.band == "GENUINE"
    assert r.score_low == 0 and r.score_high == 100
    assert r.evidence_ledger == []


def test_existing_fields_intact():
    r = AnalysisResult(file_hash="h", trust_score=42, risk_level="HIGH",
                       document_type="invoice", analysis_time_ms=5)
    assert r.trust_score == 42 and r.risk_level == "HIGH"
    assert r.document_type == "invoice" and r.analysis_time_ms == 5
