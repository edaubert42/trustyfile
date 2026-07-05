from scripts.build_corpus import build_corpus, corpus_expectations


def test_corpus_builds_all_fixtures(tmp_path):
    result = build_corpus(str(tmp_path))
    built = {p.split("/")[-1] for p in result["legitimate"] + result["suspicious"]}
    # every fixture named in the expectations table must actually be built
    for name in corpus_expectations:
        assert name in built, f"{name} declared in corpus_expectations but not built"


def test_expectations_table_well_formed():
    tamper = [n for n, e in corpus_expectations.items() if e["label"] == "tamper"]
    legit = [n for n, e in corpus_expectations.items() if e["label"] == "legit"]
    assert len(tamper) >= 4 and len(legit) >= 10
    for name, e in corpus_expectations.items():
        assert e["label"] in ("legit", "tamper")
        if e["label"] == "legit":
            # a legit fixture is either truly benign (no overlay flags, GENUINE)
            # or legit-but-ambiguous (a real value changed w/o editor provenance:
            # REVIEW ok, but never HARD).
            assert e.get("expect_no_overlay") is True or e.get("expect_no_hard") is True
