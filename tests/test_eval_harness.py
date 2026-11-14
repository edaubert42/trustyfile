import shutil
import subprocess
from pathlib import Path

import fitz
import pytest

from scripts.eval_harness import code_version, corpus_fingerprint, run_evaluation, subcorpus


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


def test_report_includes_band_counts_and_weights(tmp_path):
    from scripts.eval_harness import run_evaluation
    (tmp_path / "legitimate").mkdir(); (tmp_path / "suspicious").mkdir()
    report = run_evaluation(str(tmp_path))
    assert "band_counts" in report
    assert isinstance(report["band_counts"], dict)
    assert "code_weights" in report


# --- Sub-corpus, fingerprint, code version, rewrite resistance -------------

def test_subcorpus_is_the_folder_relative_to_the_corpus():
    root = Path("samples")
    assert subcorpus(root / "legitimate/web/zugferd/a.pdf", root) == "legitimate/web/zugferd"
    assert subcorpus(root / "legitimate/b.pdf", root) == "legitimate"


def test_fingerprint_ignores_read_order_but_sees_any_change():
    base = {"a.pdf": "11", "b.pdf": "22"}
    assert corpus_fingerprint(base) == corpus_fingerprint({"b.pdf": "22", "a.pdf": "11"})
    assert corpus_fingerprint(base) != corpus_fingerprint({"a.pdf": "11", "b.pdf": "23"})
    assert corpus_fingerprint(base) != corpus_fingerprint({"a.pdf": "11", "c.pdf": "22"})
    assert corpus_fingerprint(base) != corpus_fingerprint({"a.pdf": "11"})


def test_code_version_without_git_repository(tmp_path):
    assert code_version(tmp_path) == {"commit": None, "dirty": None}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_code_version_reports_commit_and_tracked_changes(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "init")

    clean = code_version(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False

    (tmp_path / "untracked.pdf").write_bytes(b"%PDF")   # like the samples/ link
    assert code_version(tmp_path)["dirty"] is False

    (tmp_path / "code.py").write_text("x = 2\n")        # a tracked file changed
    assert code_version(tmp_path)["dirty"] is True


def _pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_report_groups_by_subcorpus_fingerprints_and_rewrites(tmp_path):
    """Structure only: which band each file gets is the rules' business."""
    _pdf(tmp_path / "legitimate/web/ref/one.pdf", "Facture FA-1 Total 10,00 EUR")
    _pdf(tmp_path / "legitimate/two.pdf", "Facture FA-2 Total 20,00 EUR")
    _pdf(tmp_path / "suspicious/editors/bad.pdf", "Facture FA-3 Total 30,00 EUR")

    report = run_evaluation(str(tmp_path))

    assert set(report["by_subcorpus"]) == {"legitimate/web/ref", "legitimate", "suspicious/editors"}
    for group in report["by_subcorpus"].values():
        assert group["total"] == 1 and sum(group["bands"].values()) == 1
    assert report["false_positives"] + report["false_negatives"] == len(report["misses"])

    assert report["corpus"]["files"] == 3
    assert len(report["corpus"]["sha256"]) == 64

    assert set(report["rewrite"]) == {"pymupdf", "qpdf", "pymupdf-forged-header"}
    for entry in report["rewrite"].values():
        assert entry["total"] == 1   # the one forgery, rewritten without error
        assert entry["still_caught"] + len(entry["cleared"]) == 1
    assert report["notes"] == []

    assert run_evaluation(str(tmp_path), rewrite=False)["rewrite"] == {}
