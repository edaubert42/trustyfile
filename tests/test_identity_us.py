"""
Tests for US document support in the identity module.

Covers:
- US Passport Card (TD1, prefix "IP", issuer "USA")
- US Passport (TD3, prefix "P<", issuer "USA")
- Cross-validation: IP prefix with wrong issuer → HIGH flag
- Validity period check for US documents

Note: MRZ lines in these tests use real ICAO checksums computed via
mrz_check_digit() to ensure structural tests work without OCR errors.
The coherence checks (_check_mrz_coherence) do NOT verify checksums,
so lines with minor checksum errors still test the business logic.
"""

import pytest
from src.modules.identity import (
    _check_mrz_coherence,
    _validate_mrz_structure_only,
    mrz_check_digit,
    US_PASSPORT_CARD_PREFIX,
    US_ISSUER_CODE,
    US_PASSPORT_VALIDITY_ADULT,
    US_PASSPORT_VALIDITY_MINOR,
)


# =============================================================================
# HELPERS — Construction de lignes MRZ de test
# =============================================================================

def make_td1_line1(doc_type: str, issuer: str, doc_number: str) -> str:
    """
    Construit la ligne 1 d'une MRZ TD1 (30 chars).

    Format :
      [0:2]  type document (2 chars, ex: "IP", "IR", "ID")
      [2:5]  code pays émetteur (3 chars, ex: "USA", "FRA")
      [5:14] numéro de document (9 chars, complété avec '<')
      [14]   chiffre de contrôle du numéro
      [15:30] données optionnelles (complété avec '<')

    Args:
        doc_type: préfixe type, ex "IP" ou "IR" (2 chars)
        issuer: code pays, ex "USA" (3 chars)
        doc_number: numéro de document, complété avec '<' jusqu'à 9 chars

    Returns:
        Ligne MRZ TD1 de 30 caractères
    """
    doc_type = doc_type[:2].ljust(2, "<")
    issuer = issuer[:3].ljust(3, "<")
    doc_num_padded = doc_number[:9].ljust(9, "<")
    cd = str(mrz_check_digit(doc_num_padded))
    # Données optionnelles : remplissage '<' (15 chars restants)
    optional = "<" * 15
    line1 = doc_type + issuer + doc_num_padded + cd + optional
    return line1[:30]  # TD1 : exactement 30 chars


def make_td1_line2(dob: str, sex: str, expiry: str, nationality: str) -> str:
    """
    Construit la ligne 2 d'une MRZ TD1 (30 chars).

    Format :
      [0:6]  date naissance YYMMDD
      [6]    checksum DDN
      [7]    sexe (M/F/<)
      [8:14] date expiration YYMMDD
      [14]   checksum expiration
      [15:18] nationalité (3 chars)
      [18:29] données optionnelles (11 chars)
      [29]   checksum composite final

    Pour les tests, on calcule les checksums réels avec mrz_check_digit().
    Le checksum composite (position 29) est calculé sur les positions 0-9 et
    la ligne 1 complète — on simplifie ici en mettant '<' pour éviter
    la dépendance à line1 (les tests de cohérence n'utilisent pas ce CD).

    Args:
        dob: date de naissance YYMMDD (ex: "901215")
        sex: sexe ("M", "F", ou "<")
        expiry: date d'expiration YYMMDD (ex: "341215")
        nationality: code nationalité 3 chars (ex: "USA", "FRA")

    Returns:
        Ligne MRZ TD1 de 30 caractères
    """
    cd_dob = str(mrz_check_digit(dob))
    cd_exp = str(mrz_check_digit(expiry))
    nat = nationality[:3].ljust(3, "<")
    optional2 = "<" * 11
    # Checksum final (composite) : on met '<' dans les tests, _validate_mrz_structure_only
    # ne vérifie pas ce checksum (chemin OCR)
    line2 = dob + cd_dob + sex + expiry + cd_exp + nat + optional2 + "<"
    return line2[:30]


def make_td1_line3(surname: str, given: str) -> str:
    """
    Construit la ligne 3 d'une MRZ TD1 (30 chars) avec le nom.

    Format : NOM<<PRÉNOM<<<<<<<<<<<<<<<< (30 chars)

    Args:
        surname: nom de famille en majuscules
        given: prénom en majuscules

    Returns:
        Ligne MRZ TD1 de 30 caractères
    """
    name_field = f"{surname}<<{given}"
    return name_field[:30].ljust(30, "<")


def make_td3_line1(issuer: str, surname: str, given: str) -> str:
    """
    Construit la ligne 1 d'une MRZ TD3 (44 chars).

    Format :
      [0:2]  type doc ("P<" pour passeport standard)
      [2:5]  code pays émetteur
      [5:44] NOM<<PRÉNOM (complété avec '<')

    Args:
        issuer: code pays 3 chars (ex: "USA")
        surname: nom de famille en majuscules
        given: prénom en majuscules

    Returns:
        Ligne MRZ TD3 de 44 caractères
    """
    name_field = f"{surname}<<{given}"
    line1 = "P<" + issuer[:3] + name_field
    return line1[:44].ljust(44, "<")


def make_td3_line2(doc_number: str, nationality: str, dob: str, sex: str, expiry: str) -> str:
    """
    Construit la ligne 2 d'une MRZ TD3 (44 chars).

    Format :
      [0:9]  numéro document (9 chars)
      [9]    checksum numéro
      [10:13] nationalité
      [13:19] DDN
      [19]   checksum DDN
      [20]   sexe
      [21:27] expiration
      [27]   checksum expiration
      [28:42] données optionnelles (14 chars)
      [42]   checksum données optionnelles
      [43]   checksum composite final

    Args:
        doc_number: numéro doc 9 chars (complété avec '<')
        nationality: code pays 3 chars
        dob: date naissance YYMMDD
        sex: "M", "F", ou "<"
        expiry: date expiration YYMMDD

    Returns:
        Ligne MRZ TD3 de 44 caractères
    """
    doc_num_padded = doc_number[:9].ljust(9, "<")
    cd_doc = str(mrz_check_digit(doc_num_padded))
    nat = nationality[:3]
    cd_dob = str(mrz_check_digit(dob))
    cd_exp = str(mrz_check_digit(expiry))
    optional = "<" * 14
    cd_opt = str(mrz_check_digit(optional))
    # Checksum composite (simplifié pour les tests de cohérence)
    composite_data = doc_num_padded + cd_doc + dob + cd_dob + expiry + cd_exp + optional + cd_opt
    cd_composite = str(mrz_check_digit(composite_data))
    line2 = doc_num_padded + cd_doc + nat + dob + cd_dob + sex + expiry + cd_exp + optional + cd_opt + cd_composite
    return line2[:44]


# =============================================================================
# TESTS — US Passport Card (TD1, préfixe "IP", émetteur "USA")
# =============================================================================

class TestUSPassportCard:
    """Tests pour la US Passport Card (format TD1, préfixe 'IP', émetteur 'USA')."""

    def _make_valid_us_passport_card_lines(
        self,
        doc_number: str = "C11111111",
        dob: str = "900615",      # 15 juin 1990 → adulte
        expiry: str = "300615",   # 15 juin 2030 → 10 ans après délivrance 2020
        nationality: str = "USA",
    ) -> list[str]:
        """Construit 3 lignes MRZ TD1 valides pour une Passport Card américaine."""
        line1 = make_td1_line1("IP", "USA", doc_number)
        line2 = make_td1_line2(dob, "M", expiry, nationality)
        line3 = make_td1_line3("SMITH", "JOHN")
        return [line1, line2, line3]

    def test_valid_us_passport_card_no_flags(self):
        """
        Une US Passport Card avec préfixe 'IP' et émetteur 'USA' valides
        ne doit générer aucun flag lié aux spécificités US.
        """
        mrz_lines = self._make_valid_us_passport_card_lines()

        # Étape 1 : extraction des champs structurels (sans checksums)
        _, fields = _validate_mrz_structure_only(mrz_lines)

        # Étape 2 : vérifications de cohérence
        flags, coherence = _check_mrz_coherence(mrz_lines, fields)

        # Aucun flag US ne doit apparaître
        us_flag_codes = {
            "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH",
            "MRZ_US_VALIDITY_PERIOD_UNEXPECTED",
        }
        actual_codes = {f.code for f in flags}
        assert not (us_flag_codes & actual_codes), (
            f"Flags US inattendus sur une Passport Card valide : "
            f"{us_flag_codes & actual_codes}"
        )

    def test_us_passport_card_subtype_in_coherence(self):
        """
        Quand le préfixe est 'IP', coherence['document_subtype'] doit être
        'US Passport Card'.
        """
        mrz_lines = self._make_valid_us_passport_card_lines()
        _, fields = _validate_mrz_structure_only(mrz_lines)
        _, coherence = _check_mrz_coherence(mrz_lines, fields)

        assert coherence.get("document_subtype") == "US Passport Card", (
            f"Sous-type attendu 'US Passport Card', obtenu : "
            f"{coherence.get('document_subtype')!r}"
        )

    def test_ip_prefix_with_wrong_issuer_raises_high_flag(self):
        """
        Si le préfixe est 'IP' (US Passport Card) mais que l'émetteur est
        autre chose que 'USA' (ex: 'FRA'), c'est une incohérence grave.
        On doit avoir un flag HIGH avec le code MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH.
        """
        # On construit une ligne 1 avec préfixe "IP" mais émetteur "FRA" (frauduleux)
        line1 = make_td1_line1("IP", "FRA", "C22222222")
        line2 = make_td1_line2("900615", "M", "300615", "FRA")
        line3 = make_td1_line3("DUPONT", "JEAN")
        mrz_lines = [line1, line2, line3]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, coherence = _check_mrz_coherence(mrz_lines, fields)

        flag_codes = [f.code for f in flags]
        assert "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH" in flag_codes, (
            f"Flag attendu 'MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH' absent. "
            f"Flags obtenus : {flag_codes}"
        )

        # Vérifier la sévérité HIGH
        mismatch_flags = [f for f in flags if f.code == "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH"]
        assert mismatch_flags[0].severity == "high", (
            f"Sévérité attendue 'high', obtenue : {mismatch_flags[0].severity!r}"
        )

    def test_ip_prefix_with_wrong_issuer_details(self):
        """
        Le flag MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH doit inclure les détails
        expected_issuer et found_issuer pour faciliter le débogage.
        """
        line1 = make_td1_line1("IP", "GBR", "C33333333")
        line2 = make_td1_line2("850101", "F", "350101", "GBR")
        line3 = make_td1_line3("JONES", "ALICE")
        mrz_lines = [line1, line2, line3]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, _ = _check_mrz_coherence(mrz_lines, fields)

        mismatch_flags = [f for f in flags if f.code == "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH"]
        assert mismatch_flags, "Flag MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH attendu mais absent"

        details = mismatch_flags[0].details
        assert details is not None
        assert details.get("expected_issuer") == "USA"
        assert details.get("found_issuer") == "GBR"

    def test_non_ip_td1_does_not_set_subtype(self):
        """
        Un document TD1 avec préfixe 'IR' (CNI française) ne doit pas avoir
        document_subtype = 'US Passport Card'.
        """
        line1 = make_td1_line1("IR", "FRA", "1234567890")
        line2 = make_td1_line2("800101", "F", "300101", "FRA")
        line3 = make_td1_line3("MARTIN", "MARIE")
        mrz_lines = [line1, line2, line3]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        _, coherence = _check_mrz_coherence(mrz_lines, fields)

        assert coherence.get("document_subtype") != "US Passport Card", (
            "Un document français (IR/FRA) ne doit pas être identifié comme US Passport Card"
        )


# =============================================================================
# TESTS — US Passport (TD3, préfixe "P<", émetteur "USA")
# =============================================================================

class TestUSPassport:
    """Tests pour le passeport américain (format TD3, préfixe 'P<', émetteur 'USA')."""

    def _make_valid_us_passport_lines(
        self,
        doc_number: str = "A12345678",
        dob: str = "850115",      # 15 jan 1985
        expiry: str = "350115",   # 15 jan 2035 → ~10 ans
        nationality: str = "USA",
    ) -> list[str]:
        """Construit 2 lignes MRZ TD3 valides pour un passeport américain."""
        line1 = make_td3_line1("USA", "JOHNSON", "MICHAEL")
        line2 = make_td3_line2(doc_number, nationality, dob, "M", expiry)
        return [line1, line2]

    def test_valid_us_passport_no_us_flags(self):
        """
        Un passeport américain TD3 valide ne doit générer aucun flag US.
        La validation ICAO TD3 fonctionne déjà sans modification.
        """
        mrz_lines = self._make_valid_us_passport_lines()

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, coherence = _check_mrz_coherence(mrz_lines, fields)

        us_flag_codes = {
            "MRZ_US_PASSPORT_CARD_ISSUER_MISMATCH",
            "MRZ_US_VALIDITY_PERIOD_UNEXPECTED",
        }
        actual_codes = {f.code for f in flags}
        assert not (us_flag_codes & actual_codes), (
            f"Flags US inattendus sur un passeport valide : {us_flag_codes & actual_codes}"
        )

    def test_us_passport_doc_type_valid(self):
        """
        Le passeport US (TD3, préfixe 'P<') doit avoir doc_type_valid=True
        dans le dict de cohérence.
        """
        mrz_lines = self._make_valid_us_passport_lines()
        _, fields = _validate_mrz_structure_only(mrz_lines)
        _, coherence = _check_mrz_coherence(mrz_lines, fields)

        assert coherence.get("doc_type_valid") is True, (
            f"doc_type_valid attendu True pour 'P<', obtenu : "
            f"{coherence.get('doc_type_valid')!r}"
        )

    def test_us_passport_issuer_in_coherence(self):
        """
        L'émetteur extrait doit être 'USA' pour un passeport américain.
        """
        mrz_lines = self._make_valid_us_passport_lines()
        _, fields = _validate_mrz_structure_only(mrz_lines)
        _, coherence = _check_mrz_coherence(mrz_lines, fields)

        # coherence["issuer"] = valeur brute depuis la ligne 1 (3 chars)
        assert coherence.get("issuer") == "USA", (
            f"Émetteur attendu 'USA', obtenu : {coherence.get('issuer')!r}"
        )


# =============================================================================
# TESTS — Période de validité US
# =============================================================================

class TestUSValidityPeriod:
    """
    Tests pour la vérification de la durée de validité des documents US.

    Règles :
    - Adulte (≥ 16 ans à la délivrance) → 10 ans
    - Mineur (< 16 ans à la délivrance) → 5 ans
    - Tolérance ± 1.5 an
    - Vérification uniquement si issuance_date est fourni
    """

    def test_adult_us_passport_correct_validity_no_flag(self):
        """
        Un passeport US adulte avec exactement 10 ans de validité ne doit
        pas générer MRZ_US_VALIDITY_PERIOD_UNEXPECTED.
        """
        # Adulte : né en 1985, délivré le 2020-01-15, expire le 2030-01-15 → 10 ans ✓
        line1 = make_td3_line1("USA", "BROWN", "ROBERT")
        line2 = make_td3_line2("B98765432", "USA", "850115", "M", "300115")
        mrz_lines = [line1, line2]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, _ = _check_mrz_coherence(mrz_lines, fields, issuance_date="2020-01-15")

        validity_flags = [f for f in flags if f.code == "MRZ_US_VALIDITY_PERIOD_UNEXPECTED"]
        assert not validity_flags, (
            f"Flag inattendu sur validité correcte (10 ans adulte) : {validity_flags}"
        )

    def test_minor_us_passport_correct_validity_no_flag(self):
        """
        Un passeport US mineur avec exactement 5 ans de validité ne doit
        pas générer MRZ_US_VALIDITY_PERIOD_UNEXPECTED.
        """
        # Mineur : né en 2015, délivré le 2020-06-01, expire le 2025-06-01 → 5 ans ✓
        # À la délivrance (2020-06-01), le titulaire a ~5 ans → mineur → 5 ans attendus
        line1 = make_td3_line1("USA", "WILSON", "EMMA")
        line2 = make_td3_line2("W11111111", "USA", "150601", "F", "250601")
        mrz_lines = [line1, line2]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, _ = _check_mrz_coherence(mrz_lines, fields, issuance_date="2020-06-01")

        validity_flags = [f for f in flags if f.code == "MRZ_US_VALIDITY_PERIOD_UNEXPECTED"]
        assert not validity_flags, (
            f"Flag inattendu sur validité correcte (5 ans mineur) : {validity_flags}"
        )

    def test_adult_us_passport_wrong_validity_flags(self):
        """
        Un passeport US adulte avec une durée de validité anormale (ex: 7 ans)
        doit générer MRZ_US_VALIDITY_PERIOD_UNEXPECTED (LOW).
        """
        # Adulte né en 1985, délivré le 2020-01-15, expire le 2027-01-15 → 7 ans ≠ 10 ans
        line1 = make_td3_line1("USA", "DAVIS", "KAREN")
        line2 = make_td3_line2("D55555555", "USA", "850115", "F", "270115")
        mrz_lines = [line1, line2]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, _ = _check_mrz_coherence(mrz_lines, fields, issuance_date="2020-01-15")

        validity_flags = [f for f in flags if f.code == "MRZ_US_VALIDITY_PERIOD_UNEXPECTED"]
        assert validity_flags, (
            "Flag MRZ_US_VALIDITY_PERIOD_UNEXPECTED attendu pour validité de 7 ans (adulte)"
        )
        assert validity_flags[0].severity == "low"

    def test_no_validity_check_without_issuance_date(self):
        """
        Sans issuance_date, la vérification de validité US ne doit pas s'exécuter
        (on ne peut pas calculer la durée sans savoir quand le doc a été émis).
        """
        line1 = make_td3_line1("USA", "MILLER", "JAMES")
        line2 = make_td3_line2("M44444444", "USA", "001201", "M", "200201")
        mrz_lines = [line1, line2]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        # Pas d'issuance_date → pas de flag de validité
        flags, _ = _check_mrz_coherence(mrz_lines, fields, issuance_date=None)

        validity_flags = [f for f in flags if f.code == "MRZ_US_VALIDITY_PERIOD_UNEXPECTED"]
        assert not validity_flags, (
            "Aucun flag de validité ne doit être généré sans issuance_date"
        )

    def test_passport_card_adult_validity_no_flag(self):
        """
        Une US Passport Card adulte (TD1, préfixe 'IP') avec 10 ans de validité
        doit passer sans flag de validité.
        """
        # Adulte né en 1988, délivré le 2021-03-01, expire le 2031-03-01 → 10 ans ✓
        line1 = make_td1_line1("IP", "USA", "C77777777")
        line2 = make_td1_line2("880301", "M", "310301", "USA")
        line3 = make_td1_line3("TAYLOR", "MARK")
        mrz_lines = [line1, line2, line3]

        _, fields = _validate_mrz_structure_only(mrz_lines)
        flags, _ = _check_mrz_coherence(mrz_lines, fields, issuance_date="2021-03-01")

        validity_flags = [f for f in flags if f.code == "MRZ_US_VALIDITY_PERIOD_UNEXPECTED"]
        assert not validity_flags, (
            f"Flag inattendu sur Passport Card adulte valide (10 ans) : {validity_flags}"
        )


# =============================================================================
# TESTS — Constantes et valeurs attendues
# =============================================================================

class TestUSConstants:
    """Vérifie que les constantes US sont correctement définies."""

    def test_us_passport_card_prefix(self):
        assert US_PASSPORT_CARD_PREFIX == "IP"

    def test_us_issuer_code(self):
        assert US_ISSUER_CODE == "USA"

    def test_adult_validity_years(self):
        assert US_PASSPORT_VALIDITY_ADULT == 10

    def test_minor_validity_years(self):
        assert US_PASSPORT_VALIDITY_MINOR == 5
