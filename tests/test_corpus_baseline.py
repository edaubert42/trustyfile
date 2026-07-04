from scripts.build_corpus import build_corpus
from src.analyzer import TrustyFileAnalyzer


def test_forged_variants_score_below_clean(tmp_path):
    paths = build_corpus(str(tmp_path))
    analyzer = TrustyFileAnalyzer(enable_external=False)

    legit_scores = [analyzer.analyze(p).trust_score for p in paths["legitimate"]]
    forged_scores = [analyzer.analyze(p).trust_score for p in paths["suspicious"]]

    assert legit_scores and forged_scores
    # The clean invoice should score higher than the average forged variant.
    assert (sum(legit_scores) / len(legit_scores)) > (sum(forged_scores) / len(forged_scores))
    # At least one forged variant must be clearly flagged.
    assert min(forged_scores) < 50
