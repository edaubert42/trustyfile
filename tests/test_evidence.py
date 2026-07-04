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


from src.evidence import sigmoid, aggregate_soft, PRIOR_LOG_ODDS, Evidence


def _soft(code, weight, direction="tamper", group="g"):
    return Evidence(code=code, weight=weight, kind="soft",
                    direction=direction, group=group, message=code)


def test_no_evidence_returns_prior():
    assert aggregate_soft([]) == PRIOR_LOG_ODDS


def test_one_strong_signal_not_diluted_by_many_clean():
    # The core failure of averaging: a strong tamper signal must survive
    # even when it is the ONLY module that fired (others are clean = no evidence).
    strong = [_soft("BAD", 3.0)]  # one critical-weight finding
    logit = aggregate_soft(strong)
    assert logit > PRIOR_LOG_ODDS + 2.5   # the +3.0 tamper weight is preserved


def test_genuine_evidence_lowers_logit():
    logit = aggregate_soft([_soft("GOOD", 1.5, direction="genuine")])
    assert logit < PRIOR_LOG_ODDS


def test_sigmoid_monotonic():
    assert sigmoid(-10) < sigmoid(0) < sigmoid(10)
    assert 0.0 < sigmoid(0) < 1.0
