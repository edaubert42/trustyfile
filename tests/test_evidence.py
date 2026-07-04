from src.models import Flag
from src.evidence import Evidence, flag_to_evidence, SEVERITY_WEIGHTS


def test_flag_to_evidence_defaults():
    f = Flag(severity="high", code="STRUCT_JS", message="js found", details={"x": 1})
    e = flag_to_evidence(f, module="structure")
    assert isinstance(e, Evidence)
    assert e.kind == "soft"
    assert e.direction == "tamper"
    assert e.weight == SEVERITY_WEIGHTS["high"] == 1.5
    assert e.group == "structure"
    assert e.code == "STRUCT_JS"
    assert e.message == "js found"
    assert e.details == {"x": 1}


def test_severity_weight_ordering():
    assert (SEVERITY_WEIGHTS["low"] < SEVERITY_WEIGHTS["medium"]
            < SEVERITY_WEIGHTS["high"] < SEVERITY_WEIGHTS["critical"])
