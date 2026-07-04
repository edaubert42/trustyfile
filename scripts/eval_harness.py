"""
Calibration / evaluation harness for TrustyFile (spec §6).

Runs the analyzer over a labelled corpus and reports per-flag-code counts and
FP/FN rates. Corpus layout:
    <corpus_dir>/legitimate/*.pdf   → expected GENUINE (score >= 50)
    <corpus_dir>/suspicious/*.pdf   → expected SUSPICIOUS (score < 50)

No corpus is bundled (evaluation-only licences + org no-leak rule). Point it at
a local directory the user provides. Usage:
    .venv/bin/python -m scripts.eval_harness /path/to/corpus
"""
import sys
from collections import Counter
from pathlib import Path


def run_evaluation(corpus_dir: str) -> dict:
    from src.analyzer import TrustyFileAnalyzer

    root = Path(corpus_dir)
    notes: list[str] = []
    report = {
        "total": 0,
        "by_label": {"legitimate": 0, "suspicious": 0},
        "flag_code_counts": {},
        "false_positives": 0,   # legitimate doc scored < 50
        "false_negatives": 0,   # suspicious doc scored >= 50
        "band_counts": {},
        "code_weights": {},
        "notes": notes,
    }

    if not root.exists():
        notes.append(f"No corpus at {root} — nothing to evaluate.")
        return report

    analyzer = TrustyFileAnalyzer(enable_external=False)
    flag_counter: Counter = Counter()
    band_counter: Counter = Counter()

    for label, threshold_ok in (("legitimate", lambda s: s >= 50),
                                ("suspicious", lambda s: s < 50)):
        subdir = root / label
        if not subdir.exists():
            notes.append(f"Missing subdir: {subdir}")
            continue
        for pdf in sorted(subdir.glob("*.pdf")):
            try:
                result = analyzer.analyze(str(pdf))
            except Exception as e:  # harness must survive one bad file
                notes.append(f"Analyze failed for {pdf.name}: {e}")
                continue
            report["total"] += 1
            report["by_label"][label] += 1
            band_counter[result.band] += 1
            for m in result.modules:
                for f in m.flags:
                    flag_counter[f.code] += 1
            if label == "legitimate" and not threshold_ok(result.trust_score):
                report["false_positives"] += 1
            if label == "suspicious" and not threshold_ok(result.trust_score):
                report["false_negatives"] += 1

    report["flag_code_counts"] = dict(flag_counter)
    report["band_counts"] = dict(band_counter)

    from src.evidence import SEVERITY_WEIGHTS, HARD_NEGATIVE_CODES
    report["code_weights"] = {
        **{c: "HARD_NEG" for c in HARD_NEGATIVE_CODES},
        "_soft_severity_weights": dict(SEVERITY_WEIGHTS),
    }
    return report


def _print_report(report: dict) -> None:
    print(f"Documents evaluated: {report['total']}")
    print(f"  legitimate: {report['by_label']['legitimate']}  "
          f"suspicious: {report['by_label']['suspicious']}")
    print(f"False positives (legit flagged): {report['false_positives']}")
    print(f"False negatives (fraud missed):  {report['false_negatives']}")
    if report["flag_code_counts"]:
        print("Flag code frequencies:")
        for code, n in sorted(report["flag_code_counts"].items(),
                              key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {code}")
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
    corpus = sys.argv[1] if len(sys.argv) > 1 else "samples"
    _print_report(run_evaluation(corpus))
