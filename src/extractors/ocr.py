"""
OCR Extractor - Extraction de texte par reconnaissance optique de caractères.

Utilisé en fallback quand le PDF ne contient pas de couche texte (PDF scanné).

Bibliothèques :
- PyMuPDF (fitz) : rend chaque page du PDF en image bitmap (PNG)
- Pillow (PIL)   : manipule l'image en mémoire
- pytesseract    : interface Python vers Tesseract OCR (C++)

Pourquoi ces bibliothèques ?
- fitz est déjà dans le projet (cohérence)
- pytesseract supporte les whitelist de caractères (utile pour la MRZ)
- Tesseract est multi-langue (fra + eng) et open source

Usage typique :
    from src.extractors.ocr import ocr_extract_mrz
    mrz_lines = ocr_extract_mrz("scan_passport.pdf", page_count=1)
"""

import io
import re
import logging
from pathlib import Path

import fitz
from PIL import Image

logger = logging.getLogger(__name__)

# Regex pour valider une ligne MRZ dans les résultats OCR
_MRZ_LINE_RE = re.compile(r'^[A-Z0-9<]{29,45}$')

# Regex pour extraire une sous-séquence MRZ depuis une ligne OCR bruitée
# L'OCR peut ajouter 1-2 caractères parasites autour de la vraie MRZ
_MRZ_CHARS = re.compile(r'[A-Z0-9<]+')


def _extract_mrz_candidate(line: str, expected_len: int) -> str | None:
    """
    Tente d'extraire une ligne MRZ valide depuis une ligne OCR potentiellement bruitée.

    Cas gérés :
    - Correspondance exacte : len(line) == expected_len → retourner directement
    - Ligne trop longue : chercher une fenêtre glissante de expected_len chars
      → utile quand l'OCR ajoute des caractères parasites en début/fin
    - Ligne trop courte : ignorer (données manquantes, on ne peut pas deviner)

    Args:
        line: Ligne nettoyée issue de l'OCR (espaces déjà remplacés par '<')
        expected_len: Longueur attendue (44 pour TD3, 30 pour TD1)

    Returns:
        Sous-séquence MRZ valide de expected_len chars, ou None

    Example:
        >>> _extract_mrz_candidate("XP<FRAAUBERT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", 44)
        'P<FRAAUBERT<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'  # 44 chars, X élagué
    """
    if len(line) == expected_len and _MRZ_LINE_RE.match(line):
        return line

    # Tolérance : ±2 caractères (bruit OCR en début/fin)
    if expected_len <= len(line) <= expected_len + 2:
        # Fenêtre glissante : trouver la sous-séquence de expected_len chars
        # qui ne contient que des caractères MRZ valides
        for start in range(len(line) - expected_len + 1):
            candidate = line[start:start + expected_len]
            if re.match(rf'^[A-Z0-9<]{{{expected_len}}}$', candidate):
                return candidate

    return None


# =============================================================================
# RENDU PAGE → IMAGE
# =============================================================================

def render_page(file_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """
    Rend une page PDF en image PIL (sans OCR).

    PyMuPDF utilise une "Matrix" pour définir le facteur de zoom.
    À 72 DPI natif, scale=1.0. Pour 300 DPI : scale = 300/72 ≈ 4.17.

    Args:
        file_path: Chemin vers le PDF
        page_num: Numéro de page (0-indexé)
        dpi: Résolution de rendu (300 = bonne qualité, 150 = rapide)

    Returns:
        Image PIL en mode RGB

    Example:
        >>> img = render_page("scan.pdf", 0, dpi=300)
        >>> img.size
        (2480, 3508)  # A4 à 300 DPI
    """
    scale = dpi / 72  # Facteur de zoom : 300/72 ≈ 4.17
    mat = fitz.Matrix(scale, scale)

    with fitz.open(file_path) as doc:
        if page_num >= len(doc):
            raise ValueError(f"Page {page_num} n'existe pas dans {file_path}")
        page = doc[page_num]
        # get_pixmap retourne les pixels de la page rendue
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

    # Convertir les bytes raw en image PIL via un buffer mémoire
    # pix.tobytes("png") encode en PNG → Image.open le décode
    return Image.open(io.BytesIO(pix.tobytes("png")))


def render_page_bottom(
    file_path: str,
    page_num: int,
    dpi: int = 300,
    from_y_ratio: float = 0.62,
) -> Image.Image:
    """
    Rend uniquement la partie basse d'une page PDF.

    La MRZ est TOUJOURS en bas du document (norme ICAO 9303).
    Limiter le rendu au bas de la page :
    - réduit la surface OCR → moins de bruit → meilleure détection
    - accélère le traitement (moins de pixels à analyser)

    Le paramètre `from_y_ratio` détermine à quelle hauteur commencer :
    - 0.62 = à partir de 62% de la hauteur de la page (bottom 38%)
    - Pour CNI paysage : la MRZ occupe les ~20% du bas
    - Pour passeport portrait : la MRZ occupe les ~25% du bas

    Args:
        file_path: Chemin vers le PDF
        page_num: Numéro de page (0-indexé)
        dpi: Résolution de rendu
        from_y_ratio: Position de départ (0.0 = haut, 1.0 = bas)

    Returns:
        Image PIL de la bande basse de la page

    Example:
        >>> img = render_page_bottom("cni.pdf", 0, from_y_ratio=0.70)
        >>> img.size  # Seulement le bas de la page
        (2480, 1052)
    """
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)

    with fitz.open(file_path) as doc:
        page = doc[page_num]
        page_rect = page.rect  # Rectangle de la page en coordonnées PDF (72 DPI)

        # Définir le rectangle de rognage : de from_y_ratio × hauteur jusqu'en bas
        # Les coordonnées sont en espace PDF natif (pas en pixels)
        clip_rect = fitz.Rect(
            page_rect.x0,
            page_rect.y0 + page_rect.height * from_y_ratio,
            page_rect.x1,
            page_rect.y1,
        )

        # clip= passe à fitz.get_pixmap indique la zone à rendre
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, colorspace=fitz.csRGB)

    return Image.open(io.BytesIO(pix.tobytes("png")))


# =============================================================================
# OCR SPÉCIALISÉ MRZ
# =============================================================================

def ocr_extract_mrz(
    file_path: str,
    page_count: int,
    dpi: int = 300,
) -> list[str] | None:
    """
    Cherche une MRZ dans un PDF scanné via OCR.

    Stratégie :
    1. Pour chaque page (max 2) : rendre le bas de la page en image
    2. OCR avec whitelist MRZ : uniquement [A-Z0-9<]
       → Tesseract ignore tout ce qui n'est pas un caractère MRZ
       → Réduit le bruit (accents, ponctuation, espaces parasites)
    3. Normaliser : remplacer espaces par '<' (erreur OCR fréquente)
    4. Chercher 2 lignes consécutives de 44 chars (TD3) ou 3×30 (TD1)

    Args:
        file_path: Chemin vers le fichier PDF
        page_count: Nombre de pages dans le PDF
        dpi: Résolution de rendu (300 recommandé pour OCR MRZ)

    Returns:
        Liste de 2 lignes (TD3/passeport) ou 3 lignes (TD1/CNI),
        ou None si aucune MRZ trouvée.

    Example:
        >>> lines = ocr_extract_mrz("passeport.pdf", page_count=1)
        >>> lines
        ['P<FRAUNTEL<<JEAN<<<<<<<<<<<<<<<<<<<<<<<<<<',
         'AB12345671FRA8501019M2601010<<<<<<<<<<<0']
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract non disponible — OCR MRZ impossible")
        return None

    # Whitelist MRZ : Tesseract ne reconnaîtra QUE ces caractères
    # Sans whitelist, Tesseract retournerait aussi espace, ponctuation, etc.
    mrz_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'

    # On ne teste que les 2 premières pages (les pièces d'identité sont 1-2 pages)
    pages_to_check = range(min(page_count, 2))

    for page_num in pages_to_check:
        logger.debug(f"OCR MRZ : tentative sur la page {page_num + 1}")

        try:
            # Rendre le bas de la page (zone MRZ)
            img = render_page_bottom(file_path, page_num, dpi=dpi)

            # OCR avec whitelist MRZ et langue anglaise
            # (la MRZ est toujours en ASCII pur, pas besoin du français)
            raw_text = pytesseract.image_to_string(img, lang='eng', config=mrz_config)

        except Exception as e:
            logger.warning(f"OCR échoué sur page {page_num + 1} : {e}")
            continue

        # --- Nettoyer et chercher des lignes MRZ ---

        lines = []
        for raw_line in raw_text.split('\n'):
            cleaned = raw_line.strip().replace(' ', '<')
            if cleaned:
                lines.append(cleaned)

        logger.debug(f"OCR page {page_num + 1} : {len(lines)} ligne(s) extraite(s)")

        # Extraire des candidats MRZ depuis chaque ligne OCR.
        # L'OCR ajoute parfois 1-2 caractères parasites autour de la vraie MRZ.
        # On cherche donc une sous-séquence [A-Z0-9<] de la bonne longueur
        # à l'intérieur de chaque ligne, pas seulement une correspondance exacte.
        candidates_44 = [_extract_mrz_candidate(line, 44) for line in lines]
        candidates_30 = [_extract_mrz_candidate(line, 30) for line in lines]

        # Chercher TD3 (2 lignes consécutives extraites de longueur 44)
        for i in range(len(candidates_44) - 1):
            if candidates_44[i] and candidates_44[i + 1]:
                logger.info(f"MRZ TD3 trouvée par OCR (page {page_num + 1})")
                return [candidates_44[i], candidates_44[i + 1]]

        # Chercher TD1 (3 lignes consécutives extraites de longueur 30)
        for i in range(len(candidates_30) - 2):
            if candidates_30[i] and candidates_30[i + 1] and candidates_30[i + 2]:
                logger.info(f"MRZ TD1 trouvée par OCR (page {page_num + 1})")
                return [candidates_30[i], candidates_30[i + 1], candidates_30[i + 2]]

        logger.debug(f"Aucune MRZ dans l'OCR de la page {page_num + 1}")

    return None
