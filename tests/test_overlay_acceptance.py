import pytest
from scripts.build_corpus import build_corpus, corpus_expectations
from src.analyzer import TrustyFileAnalyzer


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    by_name = {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}
    a = TrustyFileAnalyzer(enable_external=False)
    return {name: a.analyze(p) for name, p in by_name.items()}


_HARD_OVERLAY = {"OVERLAY_COVER_AND_RETYPE_CONFIRMED", "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"}


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_overlay")])
def test_truly_benign_legit_stays_genuine(analyzed, name):
    # expect_no_overlay fixtures cover no real value (or a placeholder) → zero overlay
    # flags AND GENUINE. This is the strict equal-priority "clear legit files" gate.
    r = analyzed[name]
    overlay_flags = [f.code for m in r.modules if m.module == "overlay" for f in m.flags]
    assert overlay_flags == [], f"{name}: unexpected overlay flags {overlay_flags}"
    assert r.band == "GENUINE", f"{name}: {r.band} score={r.trust_score}"


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_hard")])
def test_ambiguous_legit_never_hard(analyzed, name):
    # expect_no_hard fixtures legitimately changed a real value but WITHOUT editor
    # provenance: a SOFT flag / REVIEW is acceptable (honest uncertainty), but a HARD
    # *_CONFIRMED* code (a fraud accusation) must NEVER fire. No false accusations.
    r = analyzed[name]
    overlay_flags = {f.code for m in r.modules if m.module == "overlay" for f in m.flags}
    assert not (_HARD_OVERLAY & overlay_flags), f"{name}: falsely HARD-flagged {overlay_flags}"
    assert r.band != "SUSPICIOUS", f"{name}: {r.band} — legit doc must not be SUSPICIOUS"


def test_incremental_forgery_is_critical(analyzed):
    r = analyzed["forged_incremental_edit.pdf"]
    assert r.trust_score <= 15 and r.band == "SUSPICIOUS"


def test_full_save_overlay_at_least_review(analyzed):
    r = analyzed["forged_amount_overlay.pdf"]
    assert r.band in ("REVIEW", "SUSPICIOUS")   # detected; SOFT geometry → ~REVIEW per policy
    assert r.trust_score < 80


def test_one_edit_not_quadruple_penalized(analyzed):
    # the multiple overlay faces of a single edit share group="overlay" and correlation-cap
    r = analyzed["forged_amount_overlay.pdf"]
    overlay = [m for m in r.modules if m.module == "overlay"][0]
    assert len(overlay.flags) >= 1
    # score reflects one strong finding (REVIEW band), not a runaway multi-count
    assert r.trust_score >= 15
