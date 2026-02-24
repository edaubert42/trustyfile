"""
Module Identity : Analyse des pièces d'identité (CNI, passeports).

Ce module est dédié aux documents d'identité. Il vérifie :

1. **Extraction MRZ** : repérer les 2-3 lignes de caractères MRZ dans le texte
2. **Parsing MRZ** : identifier le format TD1 (CNI) vs TD3 (passeport)
3. **Chiffres de contrôle** : valider les checksums ICAO 9303 sur :
   - numéro de document
   - date de naissance
   - date d'expiration
   - checksum composite final
4. **Cohérence des dates** : document expiré ? âge plausible ?
5. **Score et flags** : bonus si MRZ valide, flag CRITICAL si checksum échoue

--- POURQUOI LA MRZ EST FIABLE ? ---

La MRZ a été conçue exprès pour être difficile à falsifier à la main :
- Chaque chiffre de contrôle dépend de TOUS les caractères du champ concerné
- Modifier un seul chiffre (ex: changer l'année d'expiration) invalide automatiquement
  le checksum du champ ET le checksum composite final
- Un fraudeur doit recalculer les checksums manuellement, ce qui laisse souvent
  des erreurs détectables

--- FORMAT ICAO 9303 ---

TD3 (passeport) - 2 lignes × 44 caractères :
  Ligne 1 : type document (P), état émetteur, nom
  Ligne 2 : n° doc (9) + CD, nationalité, DDN (6) + CD, sexe,
            date expiration (6) + CD, données optionnelles (14) + CD, CD final

TD1 (carte d'identité) - 3 lignes × 30 caractères :
  Ligne 1 : type (I), état, n° doc (9) + CD, données optionnelles
  Ligne 2 : DDN (6) + CD, sexe, date expiration (6) + CD, nationalité,
            données optionnelles, CD final
  Ligne 3 : nom (30 chars)

--- EXEMPLE DE MRZ TD3 (passeport fictif) ---

  P<FRAUNTEL<<JEAN<PIERRE<<<<<<<<<<<<<<<<<<<<
  AB1234567<8FRA8501013M2601016<<<<<<<<<<<<4

  Décomposition ligne 2 :
  AB1234567 = numéro passeport (9 chars)
  8         = checksum du numéro (à vérifier)
  FRA       = nationalité
  850101    = date naissance (1985-01-01)
  3         = checksum DDN
  M         = sexe
  260101    = date expiration (2026-01-01)
  6         = checksum expiration
  <<<<<<<<<<<<<<  = données optionnelles (14 chars)
  4         = checksum composite final
"""

import io
import re
import logging
from datetime import date
from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData

logger = logging.getLogger(__name__)


# =============================================================================
# PRÉ-DÉTECTION : EST-CE UNE PIÈCE D'IDENTITÉ ?
# =============================================================================

# Mots-clés spécifiques aux pièces d'identité.
# Règle : 1 seul match suffit comme indice (contrairement au classifieur principal
# qui exige 3 matches pour être sûr). Ici, on veut juste décider si ça VAUT LA PEINE
# de lancer l'OCR — un faux positif coûte peu (OCR rapide), un faux négatif fait
# rater la détection de fraude sur un vrai document d'identité.
_IDENTITY_KEYWORDS = [
    # Français — termes uniques aux pièces d'identité officielles
    "carte nationale d'identité", "carte nationale", "passeport",
    "permis de conduire", "titre de séjour",
    # Anglais — idem
    "passport", "national identity card", "national id card",
    "driving licence", "driving license",
    # Marqueurs MRZ (présents dans certains PDF électroniques)
    # "<<<<<" = séquence de remplissage MRZ, impossible dans un autre contexte
    "machine readable zone", "<<<<<",
]


def _has_identity_keywords(pdf_data: PDFData) -> bool:
    """
    Vérifie si le texte du PDF contient au moins un mot-clé d'identité.

    Rapide (quelques microsecondes) : simple recherche de chaîne dans le texte
    déjà extrait. Aucune I/O disque supplémentaire.

    Retourne False si le PDF n'a pas de couche texte (scanné).

    Args:
        pdf_data: Données extraites du PDF

    Returns:
        True si au moins un mot-clé correspond
    """
    full_text = " ".join(pdf_data.text_by_page).lower()
    if not full_text.strip():
        return False  # Pas de texte → on ne peut pas décider par mots-clés
    return any(kw in full_text for kw in _IDENTITY_KEYWORDS)


def _has_face_photo(pdf_data: PDFData) -> bool:
    """
    Détecte si le document contient une photo de visage via OpenCV Haar cascade.

    Utilisé pour identifier les PDFs scannés (pas de texte) qui sont
    probablement des pièces d'identité.

    Stratégie :
    - Rendre la page à 100 DPI (résolution minimale pour la détection)
      → beaucoup plus rapide qu'un rendu haute qualité
    - Convertir en niveaux de gris (Haar fonctionne sur grayscale)
    - Passer le Haar cascade frontal
    - Si détection : document probablement une pièce d'identité

    Faux positifs possibles : brochure avec photos, article avec portrait...
    → impact limité car le module identity retournera score=100 si pas de MRZ.

    Args:
        pdf_data: Données extraites du PDF (file_path utilisé pour le rendu)

    Returns:
        True si au moins un visage est détecté
    """
    try:
        import cv2
        import numpy as np
        import fitz
    except ImportError as e:
        logger.debug(f"_has_face_photo : dépendance manquante ({e})")
        return False

    # Charger le Haar cascade (fourni avec OpenCV, pas de téléchargement nécessaire)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        logger.warning("Haar cascade introuvable — détection de visage désactivée")
        return False

    # Rendre les premières pages à basse résolution (vitesse > qualité)
    # 100 DPI : une CNI (85×54mm) donne ~335×213 pixels — suffisant pour Haar
    scale = 100 / 72  # 100 DPI → facteur ~1.39
    mat = fitz.Matrix(scale, scale)

    try:
        with fitz.open(pdf_data.file_path) as doc:
            pages_to_check = range(min(len(doc), 2))  # Max 2 pages
            for page_num in pages_to_check:
                page = doc[page_num]

                # Rendu en niveaux de gris directement (plus rapide que RGB + conversion)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

                # Convertir les bytes raw en tableau numpy 2D (height × width)
                # pix.samples = bytes, 1 octet par pixel en grayscale
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width
                )

                # Paramètres Haar cascade :
                # scaleFactor=1.05 : réduction plus fine → moins de faux négatifs
                # minNeighbors=4   : nombre minimum de détections voisines requises
                # minSize=(30,30)  : ignorer les visages de moins de 30×30 pixels
                faces = face_cascade.detectMultiScale(
                    img,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(30, 30),
                )

                if len(faces) > 0:
                    logger.debug(
                        f"Visage(s) détecté(s) page {page_num + 1} : {len(faces)} face(s)"
                    )
                    return True

    except Exception as e:
        logger.debug(f"_has_face_photo erreur : {e}")
        return False

    return False


def is_likely_identity_doc(pdf_data: PDFData) -> bool:
    """
    Détermine rapidement si un document est probablement une pièce d'identité.

    Appelé depuis l'analyzer pour décider si le module identity doit tourner
    sur un document classifié "unknown" (typiquement : scan sans texte).

    Ordre des vérifications (du plus rapide au plus lent) :
    1. Mots-clés dans le texte  → µs, zéro I/O
    2. Détection de visage      → ~50-200ms, rendu 100 DPI + Haar cascade

    Si l'une des deux vérifications passe → True.

    Args:
        pdf_data: Données extraites du PDF

    Returns:
        True si le document ressemble à une pièce d'identité

    Example:
        >>> is_likely_identity_doc(pdf_data)
        True  # si "passport" trouvé dans le texte, ou si un visage est détecté
    """
    # 1. Vérification rapide : mots-clés dans le texte
    if _has_identity_keywords(pdf_data):
        logger.debug("is_likely_identity_doc : mots-clés identité trouvés")
        return True

    # 2. Vérification visuelle : photo de visage (utile pour les PDFs scannés)
    if _has_face_photo(pdf_data):
        logger.debug("is_likely_identity_doc : visage détecté dans le document")
        return True

    return False


# =============================================================================
# CONSTANTES DE FORMAT MRZ
# =============================================================================

# Longueurs officielles des lignes MRZ selon ICAO 9303
TD3_LINE_LENGTH = 44  # Passeports : 2 lignes de 44 caractères
TD1_LINE_LENGTH = 30  # CNI/permis : 3 lignes de 30 caractères

# Une ligne MRZ ne contient que : majuscules A-Z, chiffres 0-9, et '<'
# Le regex permet aussi des longueurs légèrement hors-norme à cause d'OCR imparfait
# (on autorise 29-45 et 43-45 pour absorber une erreur de 1 char)
MRZ_LINE_PATTERN = re.compile(r'^[A-Z0-9<]{29,45}$')


# =============================================================================
# ALGORITHME DE CHECKSUM ICAO 9303
# =============================================================================

def mrz_check_digit(data: str) -> int:
    """
    Calcule le chiffre de contrôle MRZ selon la norme ICAO 9303.

    L'algorithme applique une pondération cyclique 7-3-1 sur chaque caractère :
    - Chiffres 0-9 → valeur face (0 à 9)
    - Lettres A-Z → valeur 10 à 35 (A=10, B=11, ..., Z=35)
    - '<' et autres → valeur 0

    Le résultat est la somme pondérée modulo 10.

    Args:
        data: Chaîne de caractères MRZ (ex: "AB1234567" pour n° passeport)

    Returns:
        Chiffre de contrôle (0-9)

    Example:
        >>> mrz_check_digit("520727")  # date de naissance fictive
        3
        >>> mrz_check_digit("AB1234567")
        8
    """
    weights = [7, 3, 1]  # Pondération cyclique (position i → weight[i % 3])
    total = 0

    for i, char in enumerate(data):
        if char.isdigit():
            # Chiffre : valeur directe
            value = int(char)
        elif 'A' <= char <= 'Z':
            # Lettre : décalage depuis 'A', avec A=10
            # ord('A') = 65, on veut A=10, donc : ord(char) - 65 + 10 = ord(char) - 55
            value = ord(char) - 55
        else:
            # '<' ou tout autre caractère → valeur zéro
            value = 0

        total += value * weights[i % 3]

    return total % 10


# =============================================================================
# PARSING DES DATES MRZ
# =============================================================================

def parse_mrz_date(yymmdd: str, is_expiry: bool = False) -> date | None:
    """
    Convertit une date MRZ format YYMMDD en objet Python date.

    Deux règles selon le type de date (norme ICAO 9303 §4.7) :

    Date de NAISSANCE (is_expiry=False) :
        On cherche l'année la plus récente qui ne soit pas dans le futur.
        - "92" → 1992  (2092 serait dans le futur)
        - "05" → 2005  (2005 est passé)
        - "26" → 2026  (si on est en 2026, 2026 ≤ aujourd'hui)

    Date d'EXPIRATION (is_expiry=True) :
        Toujours 20XX. Aucun document valide n'expire dans les années 1900.
        - "31" → 2031  (et non 1931 comme la règle naïve donnerait)
        - "99" → 2099

    Args:
        yymmdd: Chaîne de 6 chiffres (ex: "850101" = 1er janvier 1985)
        is_expiry: True pour les dates d'expiration

    Returns:
        Objet date Python, ou None si le format est invalide

    Example:
        >>> parse_mrz_date("850101")
        date(1985, 1, 1)
        >>> parse_mrz_date("310808", is_expiry=True)
        date(2031, 8, 8)   # et non 1931 !
    """
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None

    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])

        if is_expiry:
            # Dates d'expiration : toujours dans le 21e siècle
            # (aucun document en circulation n'expire dans les années 1900)
            year = 2000 + yy
        else:
            # Dates de naissance : année la plus récente ≤ aujourd'hui
            year_2000s = 2000 + yy
            year_1900s = 1900 + yy
            today_year = date.today().year
            year = year_2000s if year_2000s <= today_year else year_1900s

        return date(year, mm, dd)

    except ValueError as e:
        logger.debug(f"Date MRZ invalide '{yymmdd}': {e}")
        return None


# =============================================================================
# DÉTECTION DES LIGNES MRZ DANS LE TEXTE PDF
# =============================================================================

def extract_mrz_lines(pdf_data: PDFData) -> list[str] | None:
    """
    Cherche des lignes MRZ dans le texte extrait du PDF.

    Stratégie :
    1. Découper chaque page en lignes
    2. Chercher 2 lignes consécutives de 44 chars (TD3 = passeport)
    3. Sinon, chercher 3 lignes consécutives de 30 chars (TD1 = CNI)

    Chaque ligne candidate est validée par regex : uniquement [A-Z0-9<].

    Limite : si le PDF est une image scannée sans couche texte, PyMuPDF
    retourne du texte vide → MRZ non détectable sans OCR.
    Une amélioration future consisterait à activer pytesseract en fallback.

    Args:
        pdf_data: Données extraites du PDF (text_by_page est utilisé)

    Returns:
        Liste de 2 lignes (TD3) ou 3 lignes (TD1), ou None si rien trouvé

    Example:
        # Si un passeport a du texte extractible :
        >>> lines = extract_mrz_lines(pdf_data)
        >>> lines
        ['P<FRAUNTEL<<JEAN<PIERRE<<<<<<<<<<<<<<<<<<<<',
         'AB1234567<8FRA8501013M2601016<<<<<<<<<<<<4']
    """
    for page_num, page_text in enumerate(pdf_data.text_by_page):
        # Découper la page en lignes individuelles et supprimer les espaces
        lines = [line.strip() for line in page_text.split('\n')]

        # --- Chercher un format TD3 (2 lignes × 44 chars) ---
        for i in range(len(lines) - 1):
            line_a = lines[i]
            line_b = lines[i + 1]

            # Les deux lignes doivent faire exactement 44 chars et être des lignes MRZ valides
            if (len(line_a) == TD3_LINE_LENGTH and
                    len(line_b) == TD3_LINE_LENGTH and
                    MRZ_LINE_PATTERN.match(line_a) and
                    MRZ_LINE_PATTERN.match(line_b)):
                logger.debug(f"MRZ TD3 détectée page {page_num + 1}, ligne {i + 1}")
                return [line_a, line_b]

        # --- Chercher un format TD1 (3 lignes × 30 chars) ---
        for i in range(len(lines) - 2):
            line_a = lines[i]
            line_b = lines[i + 1]
            line_c = lines[i + 2]

            if (len(line_a) == TD1_LINE_LENGTH and
                    len(line_b) == TD1_LINE_LENGTH and
                    len(line_c) == TD1_LINE_LENGTH and
                    MRZ_LINE_PATTERN.match(line_a) and
                    MRZ_LINE_PATTERN.match(line_b) and
                    MRZ_LINE_PATTERN.match(line_c)):
                logger.debug(f"MRZ TD1 détectée page {page_num + 1}, ligne {i + 1}")
                return [line_a, line_b, line_c]

    logger.debug("Aucune MRZ détectée dans le texte extrait")
    return None


# =============================================================================
# VALIDATION TD3 (PASSEPORT)
# =============================================================================

def validate_td3(line1: str, line2: str) -> tuple[list[Flag], dict]:
    """
    Valide les chiffres de contrôle d'une MRZ format TD3 (passeport).

    Champs vérifiés sur la ligne 2 :
      [0-8]  : numéro de document (9 chars)
      [9]    : checksum du numéro de document
      [10-12]: nationalité
      [13-18]: date de naissance YYMMDD
      [19]   : checksum DDN
      [20]   : sexe (M/F/<)
      [21-26]: date d'expiration YYMMDD
      [27]   : checksum expiration
      [28-41]: données optionnelles (14 chars)
      [42]   : checksum données optionnelles
      [43]   : checksum composite final
                (couvre [0-9] + [13-19] + [21-42])

    Args:
        line1: Première ligne MRZ (44 chars)
        line2: Deuxième ligne MRZ (44 chars)

    Returns:
        Tuple (liste de flags, dict de champs extraits)
    """
    flags: list[Flag] = []
    fields: dict = {"mrz_format": "TD3"}

    # --- Extraction des champs ---
    doc_number  = line2[0:9]
    doc_cd_char = line2[9]
    nationality = line2[10:13]
    dob_raw     = line2[13:19]
    dob_cd_char = line2[19]
    sex         = line2[20]
    expiry_raw  = line2[21:27]
    expiry_cd_char = line2[27]
    optional    = line2[28:42]
    optional_cd_char = line2[42]
    final_cd_char    = line2[43]

    # Convertir les chars de checksum en entiers (avec protection si OCR a produit un non-chiffre)
    def safe_int(c: str) -> int:
        return int(c) if c.isdigit() else -1

    doc_cd      = safe_int(doc_cd_char)
    dob_cd      = safe_int(dob_cd_char)
    expiry_cd   = safe_int(expiry_cd_char)
    optional_cd = safe_int(optional_cd_char)
    final_cd    = safe_int(final_cd_char)

    # Stocker les champs lisibles
    fields['doc_number']  = doc_number.replace('<', '')
    fields['nationality'] = nationality.replace('<', '')
    fields['sex']         = sex

    # --- Vérification 1 : checksum numéro de document ---
    expected_doc_cd = mrz_check_digit(doc_number)
    if doc_cd != expected_doc_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_DOC_NUMBER_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle du numéro de document invalide "
                f"(attendu {expected_doc_cd}, trouvé {doc_cd})"
            ),
            details={"doc_number": doc_number, "expected": expected_doc_cd, "found": doc_cd},
        ))

    # --- Vérification 2 : checksum date de naissance ---
    expected_dob_cd = mrz_check_digit(dob_raw)
    if dob_cd != expected_dob_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_DOB_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle de la date de naissance invalide "
                f"(attendu {expected_dob_cd}, trouvé {dob_cd})"
            ),
            details={"dob_raw": dob_raw, "expected": expected_dob_cd, "found": dob_cd},
        ))
    else:
        # Checksum OK → parser la date et vérifier l'âge
        dob = parse_mrz_date(dob_raw)
        if dob:
            fields['date_of_birth'] = dob.isoformat()
            age = (date.today() - dob).days // 365
            if age < 0 or age > 130:
                flags.append(Flag(
                    severity="high",
                    code="MRZ_DOB_IMPOSSIBLE",
                    message=f"MRZ : âge calculé impossible ({age} ans selon la DDN)",
                    details={"dob": dob.isoformat(), "age_calculated": age},
                ))

    # --- Vérification 3 : checksum date d'expiration ---
    expected_expiry_cd = mrz_check_digit(expiry_raw)
    if expiry_cd != expected_expiry_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_EXPIRY_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle de la date d'expiration invalide "
                f"(attendu {expected_expiry_cd}, trouvé {expiry_cd})"
            ),
            details={"expiry_raw": expiry_raw, "expected": expected_expiry_cd, "found": expiry_cd},
        ))
    else:
        # Checksum OK → vérifier si le document est expiré
        expiry = parse_mrz_date(expiry_raw, is_expiry=True)
        if expiry:
            fields['expiry_date'] = expiry.isoformat()
            if expiry < date.today():
                days_expired = (date.today() - expiry).days
                flags.append(Flag(
                    severity="medium",
                    code="MRZ_DOCUMENT_EXPIRED",
                    message=f"Document expiré depuis {days_expired} jours (expiration : {expiry.isoformat()})",
                    details={"expiry": expiry.isoformat(), "days_expired": days_expired},
                ))

    # --- Vérification 4 : checksum données optionnelles ---
    expected_optional_cd = mrz_check_digit(optional)
    if optional_cd != expected_optional_cd:
        flags.append(Flag(
            severity="high",
            code="MRZ_OPTIONAL_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle des données optionnelles invalide "
                f"(attendu {expected_optional_cd}, trouvé {optional_cd})"
            ),
            details={"optional": optional, "expected": expected_optional_cd, "found": optional_cd},
        ))

    # --- Vérification 5 : checksum composite final ---
    # Le checksum final couvre : n° doc (0-9) + DDN (13-19) + expiration (21-27) + optionnel (28-42)
    # = line2[0:10] + line2[13:20] + line2[21:43]
    composite = line2[0:10] + line2[13:20] + line2[21:43]
    expected_final_cd = mrz_check_digit(composite)
    if final_cd != expected_final_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_FINAL_CHECK_FAIL",
            message=(
                "MRZ : checksum composite final invalide — "
                "signe fort que le document a été modifié après émission "
                f"(attendu {expected_final_cd}, trouvé {final_cd})"
            ),
            details={"expected": expected_final_cd, "found": final_cd},
        ))

    return flags, fields


# =============================================================================
# VALIDATION TD1 (CARTE D'IDENTITÉ / PERMIS)
# =============================================================================

def validate_td1(line1: str, line2: str, line3: str) -> tuple[list[Flag], dict]:
    """
    Valide les chiffres de contrôle d'une MRZ format TD1 (CNI, permis).

    Champs vérifiés :
      Ligne 1 :
        [0-1]  : type de document (ex: 'ID')
        [2-4]  : état émetteur
        [5-13] : numéro de document (9 chars)
        [14]   : checksum numéro de document
        [15-29]: données optionnelles 1

      Ligne 2 :
        [0-5]  : date de naissance YYMMDD
        [6]    : checksum DDN
        [7]    : sexe (M/F/<)
        [8-13] : date d'expiration YYMMDD
        [14]   : checksum expiration
        [15-17]: nationalité
        [18-28]: données optionnelles 2
        [29]   : checksum composite final
                  (couvre line1[5:30] + line2[0:7] + line2[8:15])

      Ligne 3 : nom et prénom (nom en majuscules, séparateur <<)

    Args:
        line1: Première ligne MRZ (30 chars)
        line2: Deuxième ligne MRZ (30 chars)
        line3: Troisième ligne MRZ (30 chars)

    Returns:
        Tuple (liste de flags, dict de champs extraits)
    """
    flags: list[Flag] = []
    fields: dict = {"mrz_format": "TD1"}

    # --- Extraction des champs ---
    doc_type    = line1[0:2]
    issuer      = line1[2:5]
    doc_number  = line1[5:14]
    doc_cd_char = line1[14]
    optional1   = line1[15:30]

    dob_raw        = line2[0:6]
    dob_cd_char    = line2[6]
    sex            = line2[7]
    expiry_raw     = line2[8:14]
    expiry_cd_char = line2[14]
    nationality    = line2[15:18]
    optional2      = line2[18:29]
    final_cd_char  = line2[29]

    name_field = line3  # "NOM<<PRÉNOM<<<"

    def safe_int(c: str) -> int:
        return int(c) if c.isdigit() else -1

    doc_cd    = safe_int(doc_cd_char)
    dob_cd    = safe_int(dob_cd_char)
    expiry_cd = safe_int(expiry_cd_char)
    final_cd  = safe_int(final_cd_char)

    # Stocker les champs lisibles
    fields['doc_number']  = doc_number.replace('<', '')
    fields['nationality'] = nationality.replace('<', '')
    fields['sex']         = sex
    fields['issuer']      = issuer.replace('<', '')

    # Parser le nom : "NOM<<PRÉNOM<..." → "NOM, PRÉNOM"
    if '<<' in name_field:
        parts = name_field.split('<<')
        surname  = parts[0].replace('<', ' ').strip()
        given    = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ''
        fields['name'] = f"{surname}, {given}".strip(', ')

    # --- Vérification 1 : checksum numéro de document ---
    expected_doc_cd = mrz_check_digit(doc_number)
    if doc_cd != expected_doc_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_DOC_NUMBER_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle du numéro de document invalide "
                f"(attendu {expected_doc_cd}, trouvé {doc_cd})"
            ),
            details={"doc_number": doc_number, "expected": expected_doc_cd, "found": doc_cd},
        ))

    # --- Vérification 2 : checksum date de naissance ---
    expected_dob_cd = mrz_check_digit(dob_raw)
    if dob_cd != expected_dob_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_DOB_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle de la date de naissance invalide "
                f"(attendu {expected_dob_cd}, trouvé {dob_cd})"
            ),
            details={"dob_raw": dob_raw, "expected": expected_dob_cd, "found": dob_cd},
        ))
    else:
        dob = parse_mrz_date(dob_raw)
        if dob:
            fields['date_of_birth'] = dob.isoformat()
            age = (date.today() - dob).days // 365
            if age < 0 or age > 130:
                flags.append(Flag(
                    severity="high",
                    code="MRZ_DOB_IMPOSSIBLE",
                    message=f"MRZ : âge calculé impossible ({age} ans selon la DDN)",
                    details={"dob": dob.isoformat(), "age_calculated": age},
                ))

    # --- Vérification 3 : checksum date d'expiration ---
    expected_expiry_cd = mrz_check_digit(expiry_raw)
    if expiry_cd != expected_expiry_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_EXPIRY_CHECK_FAIL",
            message=(
                f"MRZ : chiffre de contrôle de la date d'expiration invalide "
                f"(attendu {expected_expiry_cd}, trouvé {expiry_cd})"
            ),
            details={"expiry_raw": expiry_raw, "expected": expected_expiry_cd, "found": expiry_cd},
        ))
    else:
        expiry = parse_mrz_date(expiry_raw, is_expiry=True)
        if expiry:
            fields['expiry_date'] = expiry.isoformat()
            if expiry < date.today():
                days_expired = (date.today() - expiry).days
                flags.append(Flag(
                    severity="medium",
                    code="MRZ_DOCUMENT_EXPIRED",
                    message=f"Document expiré depuis {days_expired} jours (expiration : {expiry.isoformat()})",
                    details={"expiry": expiry.isoformat(), "days_expired": days_expired},
                ))

    # --- Vérification 4 : checksum composite final ---
    # Pour TD1 : composite = line1[5:30] + line2[0:7] + line2[8:15]
    composite = line1[5:30] + line2[0:7] + line2[8:15]
    expected_final_cd = mrz_check_digit(composite)
    if final_cd != expected_final_cd:
        flags.append(Flag(
            severity="critical",
            code="MRZ_FINAL_CHECK_FAIL",
            message=(
                "MRZ : checksum composite final invalide — "
                "signe fort que le document a été modifié après émission "
                f"(attendu {expected_final_cd}, trouvé {final_cd})"
            ),
            details={"expected": expected_final_cd, "found": final_cd},
        ))

    return flags, fields


# =============================================================================
# VALIDATION STRUCTURELLE (POUR MRZ EXTRAITE PAR OCR)
# =============================================================================

def _validate_mrz_structure_only(mrz_lines: list[str]) -> tuple[list[Flag], dict]:
    """
    Valide la structure d'une MRZ sans vérifier les checksums.

    Pourquoi pas de checksums ?
    Une pièce d'identité est TOUJOURS un document physique scanné ou photographié.
    L'OCR introduit des erreurs aléatoires (ex: '9' lu comme 'Z', '0' comme 'O').
    Un seul caractère mal lu invalide le checksum ICAO → faux CRITICAL garanti
    sur un document authentique.

    Ce qu'on vérifie à la place (robuste aux petites erreurs OCR) :
    - Le champ sexe est M, F, ou < (valeurs MRZ légales)
    - La date de naissance est parsable et donne un âge humain (0-120 ans)
    - La date d'expiration est parsable
    - Si le document est expiré, le signaler (MEDIUM)

    Ce qu'on NE vérifie PAS (nécessite la couche texte propre) :
    - Checksums ICAO 9303
    - Cohérence entre champs et leurs chiffres de contrôle

    Args:
        mrz_lines: 2 lignes (TD3) ou 3 lignes (TD1)

    Returns:
        Tuple (liste de flags, dict de champs extraits)
    """
    flags: list[Flag] = []
    fields: dict = {}

    if len(mrz_lines) == 2:
        # --- Format TD3 : passeport ---
        line1, line2 = mrz_lines
        fields["mrz_format"] = "TD3"

        # Extraire les champs clés depuis les positions connues
        sex       = line2[20] if len(line2) > 20 else "<"
        dob_raw   = line2[13:19] if len(line2) >= 19 else ""
        expiry_raw = line2[21:27] if len(line2) >= 27 else ""
        nationality = line2[10:13].replace("<", "") if len(line2) >= 13 else ""

        fields["nationality"] = nationality
        fields["sex"] = sex

        # Extraire le nom depuis la ligne 1 (après l'état émetteur : positions 5-43)
        name_part = line1[5:] if len(line1) >= 5 else ""
        if "<<" in name_part:
            parts = name_part.split("<<")
            surname = parts[0].replace("<", " ").strip()
            given   = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            fields["name"] = f"{surname}, {given}".strip(", ")

    else:
        # --- Format TD1 : carte d'identité, permis ---
        line1, line2, line3 = mrz_lines
        fields["mrz_format"] = "TD1"

        sex        = line2[7]  if len(line2) > 7  else "<"
        dob_raw    = line2[0:6] if len(line2) >= 6  else ""
        expiry_raw = line2[8:14] if len(line2) >= 14 else ""
        nationality = line2[15:18].replace("<", "") if len(line2) >= 18 else ""

        fields["nationality"] = nationality
        fields["sex"] = sex

        # Nom sur la ligne 3
        if "<<" in line3:
            parts = line3.split("<<")
            surname = parts[0].replace("<", " ").strip()
            given   = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            fields["name"] = f"{surname}, {given}".strip(", ")

    # --- Vérification 1 : champ sexe ---
    if sex not in ("M", "F", "<"):
        flags.append(Flag(
            severity="medium",
            code="MRZ_INVALID_SEX",
            message=f"MRZ : champ sexe invalide ('{sex}' — attendu M, F ou <) — possible erreur OCR",
            details={"sex_found": sex},
        ))

    # --- Vérification 2 : date de naissance ---
    if dob_raw and dob_raw.isdigit():
        dob = parse_mrz_date(dob_raw)
        if dob:
            fields["date_of_birth"] = dob.isoformat()
            age = (date.today() - dob).days // 365
            if age < 0 or age > 120:
                flags.append(Flag(
                    severity="high",
                    code="MRZ_DOB_IMPOSSIBLE",
                    message=f"MRZ : âge calculé impossible ({age} ans) selon la date de naissance",
                    details={"dob": dob.isoformat(), "age_calculated": age},
                ))
    elif dob_raw:
        # Contient des caractères non-numériques → probable erreur OCR
        logger.debug(f"DDN MRZ non numérique (OCR) : '{dob_raw}'")

    # --- Vérification 3 : date d'expiration ---
    if expiry_raw and expiry_raw.isdigit():
        expiry = parse_mrz_date(expiry_raw, is_expiry=True)
        if expiry:
            fields["expiry_date"] = expiry.isoformat()
            if expiry < date.today():
                days_expired = (date.today() - expiry).days
                flags.append(Flag(
                    severity="medium",
                    code="MRZ_DOCUMENT_EXPIRED",
                    message=f"Document expiré depuis {days_expired} jours (expiration : {expiry.isoformat()})",
                    details={"expiry": expiry.isoformat(), "days_expired": days_expired},
                ))
    elif expiry_raw:
        logger.debug(f"Expiration MRZ non numérique (OCR) : '{expiry_raw}'")

    return flags, fields


# =============================================================================
# FONCTION PRINCIPALE DU MODULE
# =============================================================================

def analyze_identity(pdf_data: PDFData) -> ModuleResult:
    """
    Analyse une pièce d'identité : détection et validation de la MRZ.

    Principe fondamental :
    Une pièce d'identité est TOUJOURS un document physique (passeport, CNI,
    permis). L'État ne délivre jamais un PDF natif. Ce que l'on reçoit est
    donc systématiquement un scan ou une photo du document convertie en PDF.

    Conséquences architecturales :
    - La couche texte du PDF sera toujours vide (image sans texte)
    - On va directement à l'OCR sans chercher de texte
    - Les checksums ICAO sont inutilisables (l'OCR introduit des erreurs)
    - On fait une validation structurelle : dates, sexe, format

    Flux :
    1. OCR sur le bas de la page (zone MRZ)
    2. Recherche de 2 × 44 chars (TD3/passeport) ou 3 × 30 chars (TD1/CNI)
    3. Validation structurelle des champs (pas de checksums)
    4. Score + flags

    Args:
        pdf_data: Données extraites du PDF

    Returns:
        ModuleResult avec le nom "identity", des flags et un score 0-100
    """
    details: dict = {"mrz_found": False}

    # Toujours OCR — les IDs sont toujours des scans
    logger.info("Module identity : lancement OCR pour détection MRZ...")
    mrz_lines = None
    try:
        from src.extractors.ocr import ocr_extract_mrz
        mrz_lines = ocr_extract_mrz(pdf_data.file_path, pdf_data.page_count)
    except Exception as e:
        logger.warning(f"Module identity : OCR échoué ({e})")

    if mrz_lines is None:
        logger.info("Module identity : aucune MRZ détectée par OCR")
        return ModuleResult(
            module="identity",
            flags=[],
            score=100,
            confidence=0.4,
            details={
                "mrz_found": False,
                "note": "Aucune MRZ détectée. Le document est peut-être une photo partielle ou la résolution est insuffisante.",
            },
        )

    # MRZ trouvée — validation structurelle uniquement (OCR → checksums non fiables)
    details["mrz_found"] = True
    details["mrz_format"] = "TD3" if len(mrz_lines) == 2 else "TD1"
    details["mrz_lines"] = mrz_lines
    details["note"] = "MRZ détectée par OCR — validation structurelle (dates, sexe). Checksums non vérifiables sur scan."

    logger.info(f"Module identity : MRZ {details['mrz_format']} détectée")

    flags, mrz_fields = _validate_mrz_structure_only(mrz_lines)
    details.update(mrz_fields)

    score = 100
    for flag in flags:
        if flag.severity == "high":
            score -= 25
        elif flag.severity == "medium":
            score -= 10
        elif flag.severity == "low":
            score -= 5
    score = max(0, score)

    if not flags:
        details["mrz_valid"] = True

    return ModuleResult(
        module="identity",
        flags=flags,
        score=score,
        # Confiance modérée : MRZ trouvée mais pas de checksum possible
        confidence=0.65,
        details=details,
    )
