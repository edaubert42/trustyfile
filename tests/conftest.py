"""
Make a run without the sample corpus impossible to mistake for a full one.

The gates that check detection quality (forgeries stay SUSPICIOUS, no false
positive on legitimate documents) read samples/, which is gitignored because
it holds real documents. Without it those tests skip: a dozen "skipped" lines
under ~1900 passes, easy to read as "all green".

- Any run without the corpus ends with a warning saying what was not checked.
- TRUSTYFILE_REQUIRE_CORPUS=1 turns the missing corpus into an error before
  any test runs. Use it for the full local run that clears a PR for merge.
  CI leaves it unset: it never has the corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REQUIRE_ENV = "TRUSTYFILE_REQUIRE_CORPUS"


def corpus_present() -> bool:
    """True when samples/ holds at least one PDF (a symlink to it counts)."""
    if not SAMPLES_DIR.is_dir():
        return False
    first_pdf = next(SAMPLES_DIR.rglob("*.pdf"), None)
    return first_pdf is not None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop at once when the corpus is required and missing."""
    if os.environ.get(REQUIRE_ENV) == "1" and not corpus_present():
        raise pytest.UsageError(
            f"{REQUIRE_ENV}=1 but {SAMPLES_DIR} has no PDF. In a worktree, link it first: "
            "ln -s <main checkout>/samples samples"
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Last lines of a run without the corpus: say what was not checked."""
    if corpus_present():
        return
    terminalreporter.write_sep("!", "CORPUS ABSENT", red=True, bold=True)
    terminalreporter.write_line(
        "samples/ is missing: the detection-quality gates were SKIPPED "
        "(forgeries stay SUSPICIOUS, no false positive on legitimate documents).",
        red=True,
    )
    terminalreporter.write_line(
        f"Green here does not clear a detection change. Rerun with the corpus and {REQUIRE_ENV}=1.",
        red=True,
    )
