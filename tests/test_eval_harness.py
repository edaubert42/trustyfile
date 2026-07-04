from scripts.eval_harness import run_evaluation


def test_empty_corpus_is_graceful(tmp_path):
    (tmp_path / "legitimate").mkdir()
    (tmp_path / "suspicious").mkdir()
    report = run_evaluation(str(tmp_path))
    assert report["total"] == 0
    assert report["false_positives"] == 0
    assert report["false_negatives"] == 0
    assert isinstance(report["flag_code_counts"], dict)


def test_missing_corpus_dir_notes_it(tmp_path):
    report = run_evaluation(str(tmp_path / "does_not_exist"))
    assert report["total"] == 0
    assert any("no corpus" in n.lower() for n in report["notes"])
