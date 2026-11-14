"""
Local evaluation harness for TrustyFile: an internal regression tool.

Runs the analyzer over a labelled corpus and reports, per sub-corpus, how many
documents land in the expected band. Corpus layout (searched RECURSIVELY):
    <corpus_dir>/legitimate/**/*.pdf   → expected GENUINE
    <corpus_dir>/suspicious/**/*.pdf   → expected NOT GENUINE

WHAT THESE FIGURES ARE NOT. The rules were tuned until this corpus passed,
and most forgeries in it were made by us, against our own detectors. So the
counts say "nothing regressed since the last run", never "this is how well
TrustyFile detects fraud". Do not publish them as detection or false-alarm
rates: that needs a held-out set never used for tuning (decision 2026-11-14).

A miss is judged on the BAND, not on a raw score cutoff. REVIEW means "a human
must look at this", which is the tool working; only GENUINE on a known-bad
document is a real miss, and anything but GENUINE on a legitimate one.

The report also records what makes two runs comparable:
  - the code version (git commit, and whether tracked files were modified);
  - the corpus fingerprint (one sha256 over every file's path and sha256);
  - rewrite resistance: each forgery is analysed again after each rewriter of
    scripts/corpus_rewrite.py wiped its file history.

No corpus is bundled (real documents never leave the machine). The --json
output names those documents: keep it out of the repository.

Usage:
    venv/bin/python -m scripts.eval_harness [corpus_dir] [--json out.json] [--no-rewrite]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Label directory -> does a band count as correct for a document of that label.
EXPECTED_BAND = {
    "legitimate": lambda band: band == "GENUINE",
    "suspicious": lambda band: band != "GENUINE",
}


def subcorpus(pdf: Path, root: Path) -> str:
    """Name of the sub-corpus a document belongs to: its folder, relative to root.

    Args:
        pdf: Path of a document inside the corpus.
        root: The corpus directory.

    Returns:
        The folder path with forward slashes.

    Example:
        subcorpus(Path("samples/legitimate/web/zugferd/a.pdf"), Path("samples"))
        -> "legitimate/web/zugferd"
        subcorpus(Path("samples/legitimate/b.pdf"), Path("samples")) -> "legitimate"
    """
    return pdf.parent.relative_to(root).as_posix()


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes.

    Example:
        file_sha256(Path("empty.pdf")) for an empty file
        -> "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_fingerprint(hashes: dict[str, str]) -> str:
    """One sha256 standing for the whole corpus.

    Two runs measured the same documents only if their fingerprints are equal.
    Lines are sorted first, so the order files were read in does not matter;
    the path is part of each line, so a renamed, added or removed file changes
    the fingerprint just like a changed byte does.

    Args:
        hashes: Relative path -> sha256 of that file.

    Example:
        corpus_fingerprint({"b.pdf": "22", "a.pdf": "11"})
        == corpus_fingerprint({"a.pdf": "11", "b.pdf": "22"})
    """
    lines = [f"{path}  {digest}\n" for path, digest in sorted(hashes.items())]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def code_version(repo: Path = REPO) -> dict:
    """The commit the analyzer ran from, and whether tracked files differ from it.

    Untracked files are ignored on purpose: the samples/ and venv/ links of a
    worktree are untracked, and a new file changes nothing until a tracked
    file imports it.

    Returns:
        {"commit": "697edc3...", "dirty": False}, or None for both when git is
        unavailable (a copy of the code outside a repository).
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def rewrite_resistance(analyzer, forgeries: list[Path], root: Path,
                       notes: list[str]) -> dict:
    """Analyse every forgery again after each rewriter wiped its file history.

    Args:
        analyzer: A TrustyFileAnalyzer.
        forgeries: Paths of the known-bad documents.
        root: The corpus directory (to report relative paths).
        notes: Where a failed rewrite or analysis is recorded.

    Returns:
        Rewriter name -> {"total", "still_caught", "cleared"}, where "cleared"
        lists the rewritten forgeries that reached GENUINE.

    Example:
        {"qpdf": {"total": 33, "still_caught": 33, "cleared": []}, ...}
    """
    from scripts.corpus_rewrite import REWRITERS

    results = {name: {"total": 0, "still_caught": 0, "cleared": []} for name in REWRITERS}
    with tempfile.TemporaryDirectory() as tmp:
        copy = str(Path(tmp) / "rewritten.pdf")  # one file, overwritten each time
        for pdf in forgeries:
            rel = pdf.relative_to(root).as_posix()
            for name, rewrite in REWRITERS.items():
                try:
                    rewrite(str(pdf), copy)
                    band = analyzer.analyze(copy).band
                except Exception as e:  # one unreadable file must not stop the run
                    notes.append(f"Rewrite {name} failed for {rel}: {e}")
                    continue
                entry = results[name]
                entry["total"] += 1
                if band == "GENUINE":
                    entry["cleared"].append(rel)
                else:
                    entry["still_caught"] += 1
    return results


def run_evaluation(corpus_dir: str, rewrite: bool = True) -> dict:
    """Analyse every labelled document and build the report.

    Args:
        corpus_dir: Directory holding legitimate/ and suspicious/.
        rewrite: Also measure rewrite resistance (about 3x the time spent on
            the suspicious documents).

    Returns:
        The report dict; see _print_report for how it reads.
    """
    from src.analyzer import TrustyFileAnalyzer

    root = Path(corpus_dir)
    notes: list[str] = []
    report = {
        "code": code_version(),
        "corpus": {"files": 0, "sha256": None},
        "total": 0,
        "by_label": {"legitimate": 0, "suspicious": 0},
        "by_subcorpus": {},
        "flag_code_counts": {},
        "false_positives": 0,   # legitimate doc that did not reach GENUINE
        "false_negatives": 0,   # known-bad doc cleared as GENUINE
        "band_counts": {},
        "code_weights": {},
        "misses": [],           # the offending files, so a regression is actionable
        "rewrite": {},
        "notes": notes,
    }

    if not root.exists():
        notes.append(f"No corpus at {root} — nothing to evaluate.")
        return report

    analyzer = TrustyFileAnalyzer(enable_external=False)
    flag_counter: Counter = Counter()
    band_counter: Counter = Counter()
    hashes: dict[str, str] = {}
    forgeries: list[Path] = []

    for label, band_ok in EXPECTED_BAND.items():
        subdir = root / label
        if not subdir.exists():
            notes.append(f"Missing subdir: {subdir}")
            continue
        for pdf in sorted(subdir.rglob("*.pdf")):
            rel = pdf.relative_to(root).as_posix()
            hashes[rel] = file_sha256(pdf)
            try:
                result = analyzer.analyze(str(pdf))
            except Exception as e:  # harness must survive one bad file
                notes.append(f"Analyze failed for {rel}: {e}")
                continue
            if label == "suspicious":
                forgeries.append(pdf)

            report["total"] += 1
            report["by_label"][label] += 1
            band_counter[result.band] += 1
            for m in result.modules:
                for f in m.flags:
                    flag_counter[f.code] += 1

            # Per sub-corpus: how many, how many as expected, in which bands.
            group = report["by_subcorpus"].setdefault(
                subcorpus(pdf, root), {"label": label, "total": 0, "ok": 0, "bands": {}})
            group["total"] += 1
            group["bands"][result.band] = group["bands"].get(result.band, 0) + 1

            if band_ok(result.band):
                group["ok"] += 1
                continue
            key = "false_positives" if label == "legitimate" else "false_negatives"
            report[key] += 1
            report["misses"].append(f"{rel} -> {result.band} ({result.trust_score})")

    report["corpus"] = {"files": len(hashes), "sha256": corpus_fingerprint(hashes)}
    report["flag_code_counts"] = dict(flag_counter)
    report["band_counts"] = dict(band_counter)
    if rewrite and forgeries:
        report["rewrite"] = rewrite_resistance(analyzer, forgeries, root, notes)

    from src.evidence import SEVERITY_WEIGHTS, HARD_NEGATIVE_CODES
    report["code_weights"] = {
        **{c: "HARD_NEG" for c in HARD_NEGATIVE_CODES},
        "_soft_severity_weights": dict(SEVERITY_WEIGHTS),
    }
    return report


def _print_report(report: dict) -> None:
    code = report["code"]
    dirty = " (+ uncommitted changes)" if code["dirty"] else ""
    print(f"Code: {code['commit']}{dirty}")
    print(f"Corpus: {report['corpus']['files']} files, sha256 {report['corpus']['sha256']}")
    print("Internal regression figures: tuned on this corpus, not a detection rate.")
    print(f"Documents evaluated: {report['total']}")
    print(f"  legitimate: {report['by_label']['legitimate']}  "
          f"suspicious: {report['by_label']['suspicious']}")
    print(f"False positives (legit not GENUINE): {report['false_positives']}")
    print(f"False negatives (bad doc cleared):   {report['false_negatives']}")
    for miss in report.get("misses", []):
        print(f"    {miss}")
    if report["by_subcorpus"]:
        print("By sub-corpus (as expected / total, bands):")
        for name, group in sorted(report["by_subcorpus"].items()):
            bands = ", ".join(f"{b} {n}" for b, n in sorted(group["bands"].items()))
            print(f"  {group['ok']:4d}/{group['total']:<4d} {name}  [{bands}]")
    if report["rewrite"]:
        print("Rewrite resistance (forgeries still not GENUINE after the rewrite):")
        for name, entry in report["rewrite"].items():
            print(f"  {entry['still_caught']:4d}/{entry['total']:<4d} {name}")
            for rel in entry["cleared"]:
                print(f"    cleared: {rel}")
    if report["flag_code_counts"]:
        print("Flag code frequencies:")
        for code_name, n in sorted(report["flag_code_counts"].items(),
                                   key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {code_name}")
    if report["band_counts"]:
        print("Band distribution:")
        for band, n in sorted(report["band_counts"].items(),
                              key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {band}")
    soft_weights = report["code_weights"].get("_soft_severity_weights")
    if soft_weights:
        weights_str = ", ".join(f"{sev}={w}" for sev, w in soft_weights.items())
        print(f"Soft severity weights: {weights_str}")
    for note in report["notes"]:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("corpus", nargs="?", default="samples")
    parser.add_argument("--json", type=Path, help="also write the full report here "
                        "(names private documents: keep it out of the repo)")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="skip rewrite resistance (faster)")
    args = parser.parse_args()
    # MuPDF prints every content-stream syntax error it recovers from (hundreds
    # on the ZUGFeRD files), which buries the report. Analysis is unaffected.
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)
    result = run_evaluation(args.corpus, rewrite=not args.no_rewrite)
    _print_report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Report written to {args.json}")
    sys.exit(0)
