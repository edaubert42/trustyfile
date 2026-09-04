"""One contract, two implementations: the checksum validators in Python and JS.

The /outils pages (IBAN, SIRET, SIREN checkers) run in the visitor's browser,
so their validators are a JavaScript *reimplementation* of the Python ones in
src/modules/content.py. Two implementations of the same rule drift silently —
until the same number gets two different answers on the same site. This file
pins both to tests/fixtures/checksum_vectors.json:

- the Python tests run every vector through is_valid_iban / is_valid_siret /
  is_valid_siren;
- the Node test runs the same vectors through tools/checksums.js (the exact
  script the pages inline) and is skipped, not failed, when `node` is absent
  locally. GitHub's ubuntu runners ship Node, so CI always runs it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.modules.content import (
    is_valid_french_vat,
    is_valid_iban,
    is_valid_nir,
    is_valid_siren,
    is_valid_siret,
    validate_rib_key,
    validate_siret_checksum,
)

ROOT = Path(__file__).resolve().parent.parent
VECTORS_PATH = ROOT / "tests" / "fixtures" / "checksum_vectors.json"
CHECKSUMS_JS = ROOT / "tools" / "checksums.js"

VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
PYTHON_VALIDATORS = {
    "iban": is_valid_iban,
    "siret": is_valid_siret,
    "siren": is_valid_siren,
    "vat": is_valid_french_vat,
    "nir": is_valid_nir,
}


def _vector_id(vector: dict) -> str:
    return f"{vector['kind']}:{vector['input'] or '<empty>'}"


@pytest.mark.parametrize("vector", VECTORS, ids=_vector_id)
def test_python_validator_agrees_with_the_contract(vector):
    validator = PYTHON_VALIDATORS[vector["kind"]]
    assert validator(vector["input"]) is vector["valid"], vector["note"]
    if "rib_ok" in vector:
        # French RIB key: True / False, or None when the check does not apply.
        assert validate_rib_key(vector["input"]) is vector["rib_ok"], vector["note"]


def test_contract_covers_the_rib_key_both_ways():
    outcomes = {v["rib_ok"] for v in VECTORS if "rib_ok" in v}
    assert outcomes == {True, False, None}


def test_contract_covers_every_kind_both_ways():
    """A fixture with only valid rows (or only one kind) would prove nothing."""
    for kind in PYTHON_VALIDATORS:
        outcomes = {v["valid"] for v in VECTORS if v["kind"] == kind}
        assert outcomes == {True, False}, f"{kind}: need both valid and invalid vectors"


class TestLaPosteException:
    """SIREN 356 000 000 has its own SIRET rule (INSEE): digit sum % 5 == 0.

    Only the head office 35600000000048 happens to satisfy Luhn as well.
    Before 2026-08-28 the validator knew nothing of this, so every La Poste
    invoice with an establishment SIRET would have been flagged as forged.
    """

    def test_head_office_passes_luhn(self):
        assert validate_siret_checksum("35600000000048") is True

    def test_establishment_passes_only_through_the_mod_5_rule(self):
        # Luhn total for this number is 48 (not a multiple of 10); digit sum 45.
        assert validate_siret_checksum("35600000049837") is True

    def test_mod_5_rule_still_rejects_a_mutated_number(self):
        assert validate_siret_checksum("35600000049838") is False

    def test_mod_5_rule_does_not_leak_to_other_companies(self):
        # 55208131700003 (an EDF-prefixed number) fails Luhn; its digit sum is
        # 35, a multiple of 5 — and it must stay invalid because
        # it is not a La Poste number.
        assert sum(int(c) for c in "55208131700003") % 5 == 0
        assert validate_siret_checksum("55208131700003") is False


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_validator_agrees_with_the_contract():
    """Run tools/checksums.js over the same vectors under Node."""
    assert CHECKSUMS_JS.exists(), "run scripts/build_guides.py to generate tools/checksums.js"
    harness = """
        const checks = require(process.argv[1]);
        const vectors = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8')).vectors;
        const fn = { iban: checks.ibanCheck, siret: checks.siretCheck, siren: checks.sirenCheck, vat: checks.vatCheck, nir: checks.nirCheck };
        const out = vectors.map(v => {
          const r = fn[v.kind](v.input);
          return { kind: v.kind, input: v.input, valid: r.ok, rib_ok: r.ribKeyOk === undefined ? 'absent' : r.ribKeyOk };
        });
        process.stdout.write(JSON.stringify(out));
    """
    completed = subprocess.run(
        ["node", "-e", harness, str(CHECKSUMS_JS), str(VECTORS_PATH)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    results = json.loads(completed.stdout)
    mismatches = [
        (r["kind"], r["input"], r["valid"], v["valid"])
        for r, v in zip(results, VECTORS)
        if r["valid"] is not v["valid"]
    ]
    assert not mismatches, f"JS disagrees with the contract (kind, input, js, expected): {mismatches}"
    rib_mismatches = [
        (r["input"], r["rib_ok"], v["rib_ok"])
        for r, v in zip(results, VECTORS)
        if "rib_ok" in v and r["rib_ok"] is not v["rib_ok"]
    ]
    assert not rib_mismatches, f"JS RIB key disagrees (input, js, expected): {rib_mismatches}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_iban_details_are_consistent():
    """The extras the IBAN page shows (country, length, RIB key) are sane."""
    harness = """
        const c = require(process.argv[1]);
        const a = c.ibanCheck('FR76 3000 6000 0112 3456 7890 189');
        const b = c.ibanCheck('DE89370400440532013000');
        const s = c.sirenCheck('552081317');
        const v = c.vatCheck('FR04 552 081 317');
        process.stdout.write(JSON.stringify({a, b, s, v}));
    """
    out = json.loads(subprocess.run(
        ["node", "-e", harness, str(CHECKSUMS_JS)], capture_output=True, text=True, timeout=30, check=True,
    ).stdout)
    assert out["a"]["country"] == "FR" and out["a"]["lengthOk"] is True and out["a"]["ribKeyOk"] is True
    assert out["a"]["formatted"] == "FR76 3000 6000 0112 3456 7890 189"
    assert out["b"]["country"] == "DE" and out["b"]["lengthOk"] is True and out["b"]["ribKeyOk"] is None
    # FR VAT key for EDF's SIREN is 03 (FR03552081317), the formula content.py uses
    assert out["s"]["vatKey"] == "03"
    # The VAT page explains a wrong key by showing the one the SIREN implies.
    assert out["v"]["ok"] is False and out["v"]["sirenOk"] is True and out["v"]["expectedKey"] == "03"
