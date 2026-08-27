"""Render the static guide pages served at /guides (FR) and /en/guides (EN).

WHY THIS EXISTS
---------------
`seo-fr.html` / `seo-en.html` are served *only to crawlers* — real users get
proxied to Streamlit, whose landing page is built by JavaScript. Serving a
crawler different *content* from what a human sees is cloaking, which Google
can penalise site-wide. Two things keep us on the right side of that line:

1. The crawler pages are generated here (`render_landing`) from the very same
   `src/translations/*.json` strings that `app.py` renders on its landing page
   — headline, tagline, the three steps, both disclaimers, the footer links.
   Same words, static HTML: that is "dynamic rendering", not cloaking. Until
   2026-08-27 they were hand-written and had drifted to ~800 words of prose,
   a feature list and a FAQ that no human ever saw.
2. The guides are real URLs served to **everyone**, linked from the app
   footer. A crawler and a human see the same bytes. The FAQ that used to
   live only on the crawler pages now lives on the guides index, visibly.

They also carry the long-tail queries the single-page app can never rank for
("comment savoir si un PDF a été modifié", "vérifier un justificatif de
domicile"), each landing on a page that answers that one question.

WHY A GENERATOR AND NOT TEN HAND-WRITTEN FILES
----------------------------------------------
Every guide needs the same ~40 lines of head boilerplate: canonical, hreflang,
Open Graph, JSON-LD, the shared stylesheet. Hand-maintaining ten copies means
ten places to forget an update. Here the boilerplate lives in `render_page()`
once and the *content* lives in GUIDES as plain data.

The rendered HTML is committed to the repo — Caddy serves static files and
never runs Python. Regenerate after editing this file:

    python scripts/build_guides.py

`tests/test_guides.py` re-renders and fails if the committed HTML has drifted,
so a forgotten rebuild cannot ship.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

# Repository root — this file lives in scripts/, so one level up.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "guides"

SITE = "https://www.trustyfile.com"

# The app's own UI strings. The crawler landing pages are rendered from these so
# they cannot say anything the Streamlit landing page does not.
TRANSLATIONS_DIR = ROOT / "src" / "translations"


def _translations(lang: str) -> dict[str, str]:
    """Load src/translations/<lang>.json — the strings app.py renders via t()."""
    return json.loads((TRANSLATIONS_DIR / f"{lang}.json").read_text(encoding="utf-8"))


# Bumped by hand when the guide content is substantially rewritten. Feeds
# <lastmod> in the sitemap and dateModified in the Article JSON-LD; lying about
# it (auto-setting it to "today" on every build) is the kind of freshness
# signal Google discounts.
LAST_MODIFIED = "2026-08-13"


@dataclass
class Section:
    """One <h2> block: a heading plus its paragraphs and optional bullets."""

    heading: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


@dataclass
class Guide:
    """A single guide page, in a single language.

    Attributes:
        slug: URL segment. FR pages live at /guides/<slug>, EN at
            /en/guides/<slug>. The two languages use *different* slugs on
            purpose — a French URL should read as French.
        title: <title> and og:title. Kept under ~60 chars so Google does not
            truncate it in the results page.
        description: meta description. ~150-160 chars is the display budget.
        heading: the on-page <h1>. Differs from `title` deliberately: the title
            competes in a results list, the h1 addresses a reader who already
            arrived.
        intro: lead paragraph, rendered right under the h1.
        sections: the body.
        counterpart: slug of the same guide in the other language, for the
            hreflang links. None while a translation does not exist yet.
    """

    slug: str
    title: str
    description: str
    heading: str
    intro: str
    sections: list[Section]
    counterpart: str | None = None


# --- Per-language chrome -----------------------------------------------------
# Everything that wraps the content: labels, the index page, the back link.

CHROME = {
    "fr": {
        "lang": "fr",
        "locale": "fr_FR",
        "app_url": SITE + "/",
        "index_path": "/guides/",
        "guides_path": "/guides",
        "index_title": "Guides - Vérifier un document PDF | TrustyFile",
        "index_description": (
            "Guides pratiques pour vérifier un document PDF : facture falsifiée, "
            "justificatif de domicile, bulletin de salaire, PDF modifié. Méthode, "
            "indices techniques et limites."
        ),
        "index_heading": "Guides de vérification documentaire",
        "index_intro": (
            "Comment reconnaître un document falsifié, quels indices techniques "
            "regarder, et ce qu'une vérification automatique peut ou ne peut pas "
            "prouver. Chaque guide traite une question précise."
        ),
        "index_outro": (
            "Ces guides décrivent la méthode. Pour l'appliquer à un document, "
            "TrustyFile effectue ces contrôles automatiquement et gratuitement."
        ),
        "cta": "Analyser un document gratuitement",
        "nav_home": "Accueil",
        "nav_guides": "Guides",
        "other_lang_label": "English",
        "read_more": "Lire le guide",
        "toc": "Sommaire",
        "related": "Autres guides",
        "footer_contact": "Contact",
        "updated": "Mis à jour le 13 août 2026",
        # Crawler landing page (seo-fr.html). Metadata only — the visible body
        # comes from src/translations/fr.json, see render_landing().
        "landing_path": "/",
        "landing_title": "TrustyFile - Vérificateur de documents PDF gratuit",
        "landing_description": (
            "Vérificateur de documents PDF gratuit en ligne. Déposez un PDF et "
            "obtenez un score de confiance instantané. Détecte les faux documents, "
            "falsifications, incohérences de polices, anomalies de dates et plus encore."
        ),
        "landing_og_description": (
            "Déposez un PDF et obtenez un score de confiance instantané. Détecte les "
            "faux documents, falsifications, incohérences de polices et anomalies de dates."
        ),
        "landing_twitter_description": (
            "Vérificateur de documents PDF gratuit. Déposez un PDF et obtenez un "
            "score de confiance instantané."
        ),
        "landing_og_image_alt": "TrustyFile — détecteur de fraude documentaire",
        "landing_webapp_description": (
            "Détecteur de fraude documentaire gratuit en ligne. Analysez vos PDF pour "
            "détecter les modifications, falsifications et incohérences."
        ),
        "landing_feature_list": [
            "Analyse des métadonnées PDF",
            "Détection des incohérences de dates et de montants",
            "Analyse des polices et des images embarquées",
            "Analyse de la structure interne du PDF",
            "Détection du texte masqué ou recouvert",
            "Vérification SIRET / SIREN / TVA",
            "Lecture et vérification des codes 2D-Doc",
        ],
        "lang_switch": "EN",
        # FAQ shown on the guides index (visible to everyone, hence eligible
        # for FAQPage structured data). Moved here from the crawler-only page.
        "faq_heading": "Questions fréquentes",
        "faq": [
            (
                "Comment fonctionne TrustyFile ?",
                "Déposez un PDF : il est analysé automatiquement par plus d'une "
                "dizaine de modules de détection, et chaque indice trouvé est versé "
                "dans un registre de preuves qui produit un score de confiance de 0 "
                "à 100, accompagné de la liste des anomalies et de leur explication.",
            ),
            (
                "Que vérifie TrustyFile exactement ?",
                "Les métadonnées du PDF (logiciel producteur, dates, convertisseurs en "
                "ligne), la cohérence des dates et des montants, les polices, les images "
                "embarquées (collages, résolutions, EXIF), la structure interne du "
                "fichier (révisions successives, objets supprimés, JavaScript), le texte "
                "masqué ou recouvert, les codes 2D-Doc et leur signature, et les "
                "identifiants d'entreprise (SIRET, SIREN, TVA, IBAN).",
            ),
            (
                "Est-ce que TrustyFile stocke mes documents ?",
                "Non. Votre document est traité temporairement, le temps de l'analyse, "
                "et n'est pas conservé. Seuls le hash du fichier et les identifiants "
                "d'entreprise (SIRET, TVA) peuvent être transmis à des API de "
                "vérification (INSEE, VIES, VirusTotal).",
            ),
            (
                "TrustyFile peut-il certifier qu'un document est authentique ?",
                "Non. TrustyFile est un outil d'aide à la décision, pas une autorité de "
                "certification : il calcule un score de risque à partir d'indices "
                "techniques. Un score élevé signale peu de traces de manipulation mais "
                "ne prouve pas l'authenticité ; un score bas mérite une vérification "
                "manuelle. Un document scanné puis aplati fournit peu de signal : dans ce "
                "cas TrustyFile le déclare non vérifiable plutôt que de le noter "
                "favorablement par défaut.",
            ),
            (
                "TrustyFile est-il gratuit ?",
                "Oui, entièrement gratuit et sans inscription.",
            ),
            (
                "Quels formats et quelles limites ?",
                "Uniquement des fichiers PDF, jusqu'à 10 Mo et 20 pages.",
            ),
        ],
    },
    "en": {
        "lang": "en",
        "locale": "en_US",
        "app_url": SITE + "/en",
        "index_path": "/en/guides/",
        "guides_path": "/en/guides",
        "index_title": "Guides - How to verify a PDF document | TrustyFile",
        "index_description": (
            "Practical guides to verifying a PDF: tampered invoices, proof of "
            "address, pay slips, modified files. Method, technical evidence and "
            "the limits of automated checks."
        ),
        "index_heading": "Document verification guides",
        "index_intro": (
            "How to recognise a forged document, which technical signals to look "
            "at, and what an automated check can and cannot prove. Each guide "
            "answers one question."
        ),
        "index_outro": (
            "These guides describe the method. To apply it to a document, "
            "TrustyFile runs these checks automatically and for free."
        ),
        "cta": "Analyse a document for free",
        "nav_home": "Home",
        "nav_guides": "Guides",
        "other_lang_label": "Français",
        "read_more": "Read the guide",
        "toc": "Contents",
        "related": "More guides",
        "footer_contact": "Contact",
        "updated": "Updated 13 August 2026",
        "landing_path": "/en",
        "landing_title": "TrustyFile - Free Document Fraud Detector",
        "landing_description": (
            "Free online document fraud detector. Upload a PDF and get an instant "
            "trust score. Detects metadata tampering, font inconsistencies, date "
            "anomalies, fake logos and more."
        ),
        "landing_og_description": (
            "Upload a PDF and get an instant trust score. Detects metadata tampering, "
            "font inconsistencies, date anomalies and more."
        ),
        "landing_twitter_description": (
            "Free online document fraud detector. Upload a PDF and get an instant "
            "trust score."
        ),
        "landing_og_image_alt": "TrustyFile — document fraud detection",
        "landing_webapp_description": (
            "Free online PDF fraud detector. Upload a PDF and get an instant trust "
            "score. Detects metadata tampering, font inconsistencies, date anomalies, "
            "fake logos and more."
        ),
        "landing_feature_list": [
            "PDF metadata analysis",
            "Date and amount consistency checks",
            "Font and embedded image analysis",
            "PDF internal structure analysis",
            "Hidden or covered text detection",
            "Company identifier verification (SIRET / VAT)",
            "French 2D-Doc barcode decoding and signature verification",
        ],
        "lang_switch": "FR",
        "faq_heading": "Frequently asked questions",
        "faq": [
            (
                "How does TrustyFile work?",
                "Upload a PDF: it is analysed automatically by more than a dozen "
                "detection modules, and every signal found goes into an evidence ledger "
                "that produces a trust score from 0 to 100, with the list of anomalies "
                "and an explanation of each.",
            ),
            (
                "What exactly does TrustyFile check?",
                "PDF metadata (producer software, dates, online converters), date and "
                "amount consistency, fonts, embedded images (pasting, resolution, EXIF), "
                "the file's internal structure (successive revisions, deleted objects, "
                "JavaScript), hidden or covered text, 2D-Doc barcodes and their "
                "signature, and company identifiers (SIRET, SIREN, VAT, IBAN).",
            ),
            (
                "Does TrustyFile store my documents?",
                "No. Your document is processed temporarily, for the duration of the "
                "analysis, and is not retained. Only the file hash and company "
                "identifiers (SIRET, VAT) may be sent to verification APIs (INSEE, VIES, "
                "VirusTotal).",
            ),
            (
                "Can TrustyFile certify that a document is authentic?",
                "No. TrustyFile is a decision-support tool, not a certification "
                "authority: it computes a risk score from technical indicators. A high "
                "score means few signs of manipulation but does not prove authenticity; "
                "a low score deserves a manual check. A scanned, flattened document "
                "yields little signal: in that case TrustyFile reports it as "
                "unverifiable rather than scoring it well by default.",
            ),
            (
                "Is TrustyFile free?",
                "Yes, completely free, with no registration.",
            ),
            (
                "Which formats and limits?",
                "PDF files only, up to 10 MB and 20 pages.",
            ),
        ],
    },
}


# --- Content -----------------------------------------------------------------

GUIDES: dict[str, list[Guide]] = {
    "fr": [
        Guide(
            slug="verifier-une-facture-falsifiee",
            counterpart="how-to-spot-a-tampered-invoice",
            title="Vérifier si une facture a été falsifiée",
            description=(
                "Montant retouché, date modifiée, IBAN remplacé : les indices "
                "techniques qui trahissent une facture falsifiée dans un PDF, et "
                "comment les contrôler."
            ),
            heading="Comment vérifier qu'une facture n'a pas été falsifiée",
            intro=(
                "La fraude à la facture ne demande plus de compétences "
                "particulières : un éditeur PDF gratuit suffit pour changer un "
                "montant ou un IBAN. En revanche, ces modifications laissent "
                "presque toujours des traces dans le fichier — à condition de "
                "savoir où regarder."
            ),
            sections=[
                Section(
                    heading="1. Regarder qui a produit le fichier",
                    paragraphs=[
                        "Un PDF conserve le nom du logiciel qui l'a écrit, dans un "
                        "champ appelé <em>Producer</em>, et celui de l'application "
                        "d'origine dans <em>Creator</em>. Une facture émise par un "
                        "logiciel de comptabilité porte la signature de ce logiciel. "
                        "Si elle porte celle d'un éditeur PDF grand public ou d'un "
                        "convertisseur en ligne, le fichier a été rouvert et "
                        "réenregistré après émission.",
                        "Ce n'est pas une preuve de fraude en soi : beaucoup de gens "
                        "compressent une pièce jointe trop lourde avant de "
                        "l'envoyer. C'est un signal à croiser avec les autres.",
                    ],
                    bullets=[
                        "<strong>Convertisseurs en ligne</strong> : iLovePDF, "
                        "Smallpdf, Sejda, PDF24, Canva, Nitro. Ils réécrivent "
                        "entièrement le fichier et effacent au passage l'historique "
                        "d'origine.",
                        "<strong>ModDate très postérieure à CreationDate</strong> : "
                        "le document a été modifié longtemps après sa création.",
                        "<strong>Métadonnées vides</strong> sur un document "
                        "prétendument émis par une grande entreprise : les chaînes "
                        "de facturation industrielles remplissent ces champs.",
                    ],
                ),
                Section(
                    heading="2. Compter les révisions du PDF",
                    paragraphs=[
                        "Le format PDF permet d'enregistrer une modification "
                        "<em>en ajoutant</em> des données à la fin du fichier plutôt "
                        "qu'en le réécrivant : c'est la mise à jour incrémentale. "
                        "L'état précédent reste physiquement présent dans le "
                        "fichier.",
                        "Conséquence directe : sur un document à plusieurs "
                        "révisions, on peut souvent afficher la version antérieure "
                        "et comparer. Un montant qui change entre deux révisions "
                        "est l'un des rares indices proches de la preuve.",
                        "Attention aux faux positifs : une signature électronique "
                        "ajoute légitimement une révision, et certains formulaires "
                        "administratifs en ajoutent une à chaque remplissage.",
                    ],
                ),
                Section(
                    heading="3. Examiner les polices ligne par ligne",
                    paragraphs=[
                        "Un document généré par un logiciel utilise un jeu de "
                        "polices cohérent et embarque ces polices dans le fichier. "
                        "Quand quelqu'un retouche un montant avec un éditeur, "
                        "l'éditeur ne dispose pas toujours de la police d'origine et "
                        "substitue une police système : Arial, Calibri, Helvetica.",
                    ],
                    bullets=[
                        "Un changement de police <strong>au milieu d'une ligne</strong> "
                        "ou d'un mot est très difficile à produire accidentellement.",
                        "Une police non embarquée au milieu de polices embarquées "
                        "signale un ajout tardif.",
                        "Une diversité anormale de polices sur une seule page "
                        "(au-delà de cinq ou six sur une facture) mérite un examen.",
                    ],
                ),
                Section(
                    heading="4. Chercher le texte recouvert",
                    paragraphs=[
                        "La retouche la plus courante ne supprime rien : elle "
                        "recouvre. Un rectangle blanc posé sur l'ancien montant, le "
                        "nouveau montant écrit par-dessus. Visuellement, c'est "
                        "parfait. Dans le fichier, l'ancien texte est toujours là, "
                        "sous le rectangle.",
                        "Copier-coller le texte du PDF dans un éditeur de texte "
                        "suffit parfois à faire réapparaître la valeur d'origine. "
                        "Variante : du texte rendu invisible (mode de rendu "
                        "\"invisible\", ou couleur de texte identique au fond) qui "
                        "contredit ce qui est affiché.",
                    ],
                ),
                Section(
                    heading="5. Vérifier ce qui est vérifiable de l'extérieur",
                    paragraphs=[
                        "Certaines données d'une facture se contrôlent hors du "
                        "fichier, et c'est ce qui rend cette étape utile : elle ne "
                        "dépend pas de la bonne foi de l'émetteur.",
                    ],
                    bullets=[
                        "<strong>SIREN / SIRET</strong> : l'INSEE expose un annuaire "
                        "public. Un numéro inexistant, ou rattaché à une entreprise "
                        "sans rapport, est décisif.",
                        "<strong>Numéro de TVA intracommunautaire</strong> : il "
                        "possède une clé de contrôle calculable, et se vérifie "
                        "auprès de VIES.",
                        "<strong>IBAN</strong> : sa clé de contrôle (modulo 97) se "
                        "vérifie sans réseau. Un IBAN retouché à la main échoue "
                        "presque toujours ce test.",
                        "<strong>Cohérence arithmétique</strong> : total HT + TVA = "
                        "total TTC. Une retouche partielle casse l'addition.",
                    ],
                ),
                Section(
                    heading="Le réflexe qui prime sur tous les autres",
                    paragraphs=[
                        "En cas de doute sur un RIB, aucune analyse de fichier ne "
                        "remplace un appel au fournisseur, sur un numéro que vous "
                        "connaissez déjà — jamais celui inscrit sur la facture "
                        "suspecte. La fraude au faux RIB fonctionne parce que la "
                        "victime vérifie l'information avec l'information "
                        "elle-même.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="verifier-un-justificatif-de-domicile",
            counterpart="how-to-verify-a-proof-of-address",
            title="Vérifier un justificatif de domicile",
            description=(
                "Facture d'énergie, de téléphonie ou d'accès Internet fournie en "
                "justificatif de domicile : les contrôles techniques qui révèlent "
                "un document retouché."
            ),
            heading="Comment vérifier un justificatif de domicile",
            intro=(
                "Le justificatif de domicile est l'un des documents les plus "
                "falsifiés, parce qu'il est demandé partout et vérifié presque "
                "nulle part. Dans la majorité des cas, il s'agit d'une facture "
                "authentique appartenant à quelqu'un d'autre, dont on a remplacé "
                "le nom et l'adresse."
            ),
            sections=[
                Section(
                    heading="Le scénario dominant : substitution de nom",
                    paragraphs=[
                        "Contrairement à la facture frauduleuse, le justificatif "
                        "falsifié part rarement de zéro. Le fraudeur reprend une "
                        "vraie facture — la sienne, ou une trouvée en ligne — et "
                        "remplace les seuls champs qui l'intéressent : nom, prénom, "
                        "adresse.",
                        "Cette retouche très localisée est justement ce qui la rend "
                        "détectable : le reste du document est authentique, donc "
                        "toute incohérence se concentre sur la zone modifiée.",
                    ],
                ),
                Section(
                    heading="Comparer le logiciel émetteur au fournisseur annoncé",
                    paragraphs=[
                        "Les grands fournisseurs d'énergie et opérateurs télécoms "
                        "génèrent leurs factures par des chaînes industrielles, avec "
                        "des empreintes techniques très stables : même producteur "
                        "PDF, mêmes polices embarquées, même structure interne d'un "
                        "mois sur l'autre.",
                        "Un document qui se présente comme émis par un grand "
                        "fournisseur mais dont le producteur est un éditeur de "
                        "bureautique, un convertisseur en ligne ou un outil de "
                        "capture d'écran, contredit ce qu'il prétend être.",
                    ],
                ),
                Section(
                    heading="Chercher les traces d'un collage",
                    paragraphs=[
                        "Quand le nom est remplacé par une image plutôt que par du "
                        "texte, le collage laisse des traces mesurables.",
                    ],
                    bullets=[
                        "<strong>Rupture de résolution</strong> : un bloc à 96 dpi "
                        "posé sur une page à 300 dpi.",
                        "<strong>Blanc qui n'est pas le même blanc</strong> : le "
                        "fond du document est rarement du blanc pur, celui d'un "
                        "rectangle collé l'est souvent.",
                        "<strong>Bruit local incohérent</strong> : un scan possède "
                        "un bruit homogène sur toute la page ; une zone recollée est "
                        "anormalement lisse ou anormalement bruitée.",
                        "<strong>Artefacts de compression JPEG</strong> concentrés "
                        "autour d'un seul bloc de texte.",
                    ],
                ),
                Section(
                    heading="Le cas du document scanné : savoir dire \"je ne sais pas\"",
                    paragraphs=[
                        "Un justificatif scanné puis aplati en image ne contient "
                        "plus ni couche texte, ni polices, ni historique de "
                        "révisions. La quasi-totalité des contrôles décrits ici "
                        "deviennent inapplicables.",
                        "C'est un point de méthode important : dans ce cas, "
                        "l'absence d'anomalie ne veut rien dire, puisqu'il n'y avait "
                        "rien à examiner. TrustyFile classe alors le document comme "
                        "<strong>non vérifiable</strong> plutôt que de lui attribuer "
                        "un bon score par défaut. Beaucoup d'outils font l'inverse, "
                        "ce qui est trompeur.",
                        "Demander l'original numérique, plutôt qu'un scan ou une "
                        "photo, transforme une vérification impossible en "
                        "vérification possible. C'est la meilleure chose à faire "
                        "quand on est en position de l'exiger.",
                    ],
                ),
                Section(
                    heading="Contrôles de cohérence non techniques",
                    bullets=[
                        "La date de facture est-elle bien dans la fenêtre exigée "
                        "(souvent trois mois) ?",
                        "La période de consommation est-elle antérieure à la date "
                        "d'émission ? Une facture qui facture le futur est une "
                        "anomalie.",
                        "Le numéro de client suit-il le format du fournisseur ?",
                        "Le montant est-il plausible pour le logement et la saison ?",
                    ],
                ),
            ],
        ),
        Guide(
            slug="savoir-si-un-pdf-a-ete-modifie",
            counterpart="how-to-tell-if-a-pdf-was-modified",
            title="Savoir si un PDF a été modifié",
            description=(
                "Un PDF garde la trace de ses enregistrements successifs : "
                "révisions incrémentales, objets fantômes, métadonnées XMP. Où "
                "regarder pour savoir si un fichier a été retouché."
            ),
            heading="Comment savoir si un PDF a été modifié",
            intro=(
                "Un PDF n'est pas une image figée : c'est un format documenté, qui "
                "conserve beaucoup plus d'informations sur son propre passé que ce "
                "que la page affiche. Voici ce qu'il garde, et ce qu'on peut en "
                "déduire."
            ),
            sections=[
                Section(
                    heading="Les révisions incrémentales",
                    paragraphs=[
                        "Quand un logiciel enregistre une modification, il peut "
                        "réécrire tout le fichier, ou bien ajouter les objets "
                        "modifiés à la fin en laissant intact ce qui précède. Le "
                        "second mode — la mise à jour incrémentale — est le mode par "
                        "défaut de plusieurs éditeurs, et c'est une aubaine pour "
                        "l'analyse : chaque révision est un instantané conservé.",
                        "Un fichier à trois révisions a donc été enregistré trois "
                        "fois. On peut le tronquer à la fin de la première pour "
                        "obtenir l'état d'origine, puis comparer.",
                    ],
                    bullets=[
                        "Une révision unique : le fichier n'a été écrit qu'une fois.",
                        "Plusieurs révisions <strong>avec</strong> signature "
                        "électronique : normal, la signature en ajoute une.",
                        "Plusieurs révisions <strong>sans</strong> signature sur un "
                        "document censé sortir d'une chaîne automatisée : anormal.",
                    ],
                ),
                Section(
                    heading="Les objets fantômes",
                    paragraphs=[
                        "Supprimer un élément dans un éditeur PDF le retire de "
                        "l'affichage, pas nécessairement du fichier. L'objet devient "
                        "orphelin : plus référencé, toujours présent. Retrouver dans "
                        "les octets un ancien montant, une ancienne adresse ou un "
                        "logo qui n'apparaît plus est un indice fort.",
                    ],
                ),
                Section(
                    heading="Les deux jeux de métadonnées",
                    paragraphs=[
                        "Un PDF peut porter ses métadonnées à deux endroits : le "
                        "dictionnaire <em>Info</em>, historique, et un flux "
                        "<em>XMP</em> en XML. Beaucoup d'outils mettent à jour l'un "
                        "et oublient l'autre.",
                        "Une divergence entre les deux — dates différentes, "
                        "producteurs différents — révèle un traitement par un outil "
                        "qui n'a pas nettoyé derrière lui. XMP conserve parfois un "
                        "<em>DocumentID</em> et un historique de versions plus "
                        "bavards que le champ Producer.",
                    ],
                ),
                Section(
                    heading="Ce que les métadonnées ne prouvent pas",
                    paragraphs=[
                        "Les champs de métadonnées sont du texte libre : n'importe "
                        "qui peut écrire le producteur de son choix. Un fraudeur "
                        "averti remet la bonne valeur après modification.",
                        "C'est pourquoi les indices <em>structurels</em> comptent "
                        "davantage : l'ordre des objets, le style de compression, la "
                        "façon dont la table de références croisées est écrite. Ces "
                        "détails-là sont produits par le code de la bibliothèque qui "
                        "a écrit le fichier, et un fraudeur ne les reproduit pas en "
                        "changeant une chaîne de caractères. Un fichier qui annonce "
                        "un producteur mais dont la structure correspond à un autre "
                        "est un cas intéressant.",
                    ],
                ),
                Section(
                    heading="Une modification n'est pas une fraude",
                    paragraphs=[
                        "Il faut le dire clairement : la plupart des PDF modifiés le "
                        "sont pour des raisons parfaitement banales — compression "
                        "avant envoi, ajout d'une signature, fusion de plusieurs "
                        "documents, aplatissement d'un formulaire, passage par un "
                        "outil d'archivage.",
                        "Détecter une modification ne fait donc que déplacer la "
                        "question : <em>qu'est-ce qui</em> a été modifié, et est-ce "
                        "que cela porte sur une donnée qui compte ? C'est la "
                        "conjonction de plusieurs signaux sur la même zone du "
                        "document qui fait un dossier, pas un signal isolé.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="2d-doc-verifier-un-document-officiel",
            counterpart="2d-doc-verifying-official-documents",
            title="2D-Doc : vérifier un document officiel français",
            description=(
                "Le code 2D-Doc des avis d'imposition, bulletins de salaire et "
                "attestations : ce qu'il contient, ce qu'il prouve, et pourquoi "
                "il change la donne."
            ),
            heading="2D-Doc : le seul contrôle qui prouve quelque chose",
            intro=(
                "Tous les contrôles décrits dans les autres guides sont des "
                "faisceaux d'indices : ils rendent une fraude probable ou "
                "improbable, jamais certaine. Le 2D-Doc est différent — il repose "
                "sur de la cryptographie, et permet une réponse tranchée."
            ),
            sections=[
                Section(
                    heading="Ce qu'est un 2D-Doc",
                    paragraphs=[
                        "Le 2D-Doc est un code-barres bidimensionnel, au format "
                        "DataMatrix, défini par l'ANTS (Agence nationale des titres "
                        "sécurisés). On le trouve sur les avis d'imposition, les "
                        "bulletins de salaire, les attestations de la Sécurité "
                        "sociale, les justificatifs de domicile de certains "
                        "fournisseurs, et divers documents administratifs.",
                        "Il contient deux choses : les données clés du document en "
                        "clair (nom, montant, date, référence), et une "
                        "<strong>signature cryptographique</strong> de ces données, "
                        "produite par la clé privée de l'émetteur.",
                    ],
                ),
                Section(
                    heading="Pourquoi c'est plus fort que tout le reste",
                    paragraphs=[
                        "Les métadonnées se réécrivent, les polices se réimitent, un "
                        "scan efface les traces. Une signature cryptographique, non : "
                        "sans la clé privée de l'émetteur, on ne peut pas produire "
                        "une signature valide pour des données modifiées.",
                        "Cela donne deux contrôles distincts, et il faut les "
                        "distinguer :",
                    ],
                    bullets=[
                        "<strong>Le code est-il lisible et cohérent ?</strong> Les "
                        "données encodées correspondent-elles au texte imprimé sur le "
                        "document ? Si le PDF affiche un montant et que le code en "
                        "contient un autre, quelqu'un a retouché le visuel sans "
                        "pouvoir toucher au code.",
                        "<strong>La signature est-elle valide ?</strong> Cela demande "
                        "le certificat de l'émetteur et une vérification ECDSA. "
                        "C'est le seul contrôle qui permet d'affirmer qu'un document "
                        "est authentique, et non simplement \"sans anomalie "
                        "détectée\".",
                    ],
                ),
                Section(
                    heading="Où en est TrustyFile",
                    paragraphs=[
                        "TrustyFile décode aujourd'hui le 2D-Doc et compare les "
                        "données encodées au texte visible. C'est le premier "
                        "contrôle, et il attrape déjà les retouches visuelles.",
                        "La vérification de la signature elle-même n'est pas encore "
                        "implémentée. Nous préférons l'écrire ici plutôt que de "
                        "laisser croire à une garantie qui n'existe pas : "
                        "aujourd'hui, aucun résultat de TrustyFile ne prouve qu'un "
                        "document est authentique. L'outil signale des anomalies, il "
                        "ne délivre pas de certificat.",
                    ],
                ),
                Section(
                    heading="Le contrôle officiel en parallèle",
                    paragraphs=[
                        "Pour un avis d'imposition, l'administration fiscale propose "
                        "un service de vérification en ligne (SVAIR) où saisir le "
                        "numéro fiscal et la référence de l'avis. La réponse vient "
                        "de la source elle-même, ce qui est plus fort que n'importe "
                        "quelle analyse de fichier.",
                        "Règle générale : quand l'émetteur offre un moyen de "
                        "vérifier directement auprès de lui, c'est toujours ce "
                        "moyen-là qu'il faut privilégier.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="verifier-un-bulletin-de-salaire",
            counterpart="how-to-verify-a-pay-slip",
            title="Vérifier un bulletin de salaire",
            description=(
                "Faux bulletins de salaire dans les dossiers de location ou de "
                "crédit : logiciel de paie, calculs, code 2D-Doc — les contrôles "
                "qui révèlent une fiche retouchée."
            ),
            heading="Comment vérifier un bulletin de salaire",
            intro=(
                "Le bulletin de salaire falsifié est devenu un classique des "
                "dossiers de location et des demandes de crédit : un salaire "
                "gonflé sur sa propre fiche, ou un nom remplacé sur celle d'un "
                "tiers. Comme toute retouche, elle laisse des traces — et la "
                "fiche de paie française a une particularité qui la rend plus "
                "vérifiable que la plupart des documents."
            ),
            sections=[
                Section(
                    heading="Chercher d'abord le code 2D-Doc",
                    paragraphs=[
                        "Beaucoup de logiciels de paie français impriment un code "
                        "2D-Doc sur le bulletin : un petit carré de type "
                        "DataMatrix, généralement en haut ou en bas de page. Il "
                        "contient les données clés — nom, période, montant net — "
                        "signées cryptographiquement par l'émetteur.",
                        "Sa présence change tout : si le salaire affiché sur la "
                        "page ne correspond pas à celui encodé dans le code, "
                        "quelqu'un a retouché le visuel sans pouvoir toucher au "
                        "code. C'est l'un des rares contrôles proches de la "
                        "preuve. Le fonctionnement détaillé est décrit dans "
                        "<a href=\"/guides/2d-doc-verifier-un-document-officiel\">"
                        "notre guide sur le 2D-Doc</a>.",
                        "Son absence, en revanche, ne prouve rien : tous les "
                        "employeurs n'utilisent pas un logiciel qui en génère.",
                    ],
                ),
                Section(
                    heading="Identifier le logiciel qui a produit la fiche",
                    paragraphs=[
                        "Les bulletins de salaire sortent de chaînes de paie "
                        "industrielles — Silae, PayFit, ADP, Sage, Cegid — dont "
                        "l'empreinte technique est très stable : même producteur "
                        "PDF, mêmes polices embarquées, même structure d'un mois "
                        "sur l'autre.",
                        "Un bulletin dont les métadonnées annoncent un traitement "
                        "de texte, un convertisseur en ligne ou un outil de "
                        "capture d'écran contredit ce qu'il prétend être. Un "
                        "employeur ne rédige pas les fiches de paie dans Word.",
                    ],
                ),
                Section(
                    heading="Refaire les calculs",
                    paragraphs=[
                        "Une fiche de paie est une addition publique. Le faussaire "
                        "qui gonfle le net à payer oublie presque toujours de "
                        "corriger tout ce qui en dépend :",
                    ],
                    bullets=[
                        "<strong>Brut moins cotisations = net</strong> : chaque "
                        "ligne de cotisation applique un taux à une base ; un net "
                        "modifié sans recalcul des lignes casse l'addition.",
                        "<strong>Le montant net social</strong> et le net "
                        "imposable doivent rester cohérents avec le net à payer.",
                        "<strong>Les cumuls annuels</strong> : un cumul inférieur "
                        "au montant du mois, ou qui ne progresse pas d'un bulletin "
                        "à l'autre, trahit une retouche isolée.",
                        "<strong>Le salaire et le poste</strong> : un montant "
                        "hors de proportion avec l'intitulé et la classification "
                        "mérite une question.",
                    ],
                ),
                Section(
                    heading="Examiner la zone du nom et du montant",
                    paragraphs=[
                        "Comme pour le justificatif de domicile, la fraude "
                        "courante ne fabrique pas une fiche entière : elle "
                        "remplace un nom ou un montant sur une fiche authentique. "
                        "Les indices se concentrent donc sur ces zones : "
                        "changement de police au milieu d'une ligne, rectangle "
                        "blanc recouvrant l'ancienne valeur, bloc d'image collé "
                        "dont la résolution ne correspond pas au reste de la page.",
                    ],
                ),
                Section(
                    heading="Vérifier l'employeur",
                    paragraphs=[
                        "Le bulletin mentionne obligatoirement le SIRET de "
                        "l'employeur, le code APE et la convention collective. "
                        "Tout cela se contrôle de l'extérieur :",
                    ],
                    bullets=[
                        "Le <strong>SIRET</strong> existe-t-il à l'annuaire des "
                        "entreprises (INSEE), et correspond-il au nom affiché ?",
                        "Le <strong>code APE</strong> est-il plausible pour le "
                        "poste occupé ?",
                        "La <strong>convention collective</strong> mentionnée "
                        "est-elle celle du secteur ?",
                        "L'entreprise est-elle toujours en activité à la période "
                        "du bulletin ?",
                    ],
                ),
                Section(
                    heading="Le scan, encore une fois : non vérifiable",
                    paragraphs=[
                        "Un bulletin scanné ou photographié ne conserve ni couche "
                        "texte, ni polices, ni métadonnées de paie. La plupart des "
                        "contrôles ci-dessus deviennent inapplicables, et "
                        "l'absence d'anomalie ne signifie alors rien. Quand vous "
                        "êtes en position d'exiger, demandez le PDF d'origine — "
                        "celui que le salarié télécharge depuis son espace RH — "
                        "plutôt qu'une copie aplatie.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="detecter-un-faux-rib",
            counterpart="how-to-spot-a-fake-iban",
            title="Détecter un faux RIB ou un IBAN falsifié",
            description=(
                "La fraude au faux RIB détourne les virements vers le compte de "
                "l'escroc. Clé IBAN, cohérence bancaire, traces de retouche : "
                "comment vérifier un RIB reçu par e-mail."
            ),
            heading="Comment détecter un faux RIB",
            intro=(
                "La fraude au faux RIB est l'une des plus coûteuses qui soient : "
                "un fournisseur annonce un « changement de coordonnées "
                "bancaires », le virement part vers le compte de l'escroc, et "
                "l'on s'en aperçoit des semaines plus tard, quand le vrai "
                "fournisseur relance. Le document envoyé est souvent authentique "
                "à un détail près : l'IBAN."
            ),
            sections=[
                Section(
                    heading="Comprendre le scénario",
                    paragraphs=[
                        "L'attaque prend trois formes principales : un e-mail "
                        "imitant le fournisseur avec un RIB frauduleux en pièce "
                        "jointe ; une boîte mail réellement compromise, d'où part "
                        "un message parfaitement crédible ; ou une facture "
                        "authentique interceptée puis renvoyée avec le seul IBAN "
                        "modifié.",
                        "Dans les trois cas, le fichier joint est le point de "
                        "contrôle : c'est souvent un vrai document dont un seul "
                        "champ a été retouché, et cette retouche localisée est "
                        "détectable.",
                    ],
                ),
                Section(
                    heading="Vérifier la clé de contrôle de l'IBAN",
                    paragraphs=[
                        "Un IBAN embarque sa propre somme de contrôle : les deux "
                        "chiffres après « FR » se calculent à partir du reste du "
                        "numéro (modulo 97), sans réseau ni base de données. Un "
                        "IBAN modifié à la main — un chiffre changé pour "
                        "ressembler à l'original — échoue presque toujours ce "
                        "test. La clé RIB, en fin de numéro, offre le même "
                        "contrôle au format français.",
                        "Attention au sens de ce test : une clé <em>fausse</em> "
                        "condamne le document, mais une clé <em>valide</em> ne "
                        "l'innocente pas. Les fraudeurs ouvrent de vrais comptes, "
                        "avec de vrais IBAN parfaitement valides.",
                    ],
                ),
                Section(
                    heading="Confronter l'IBAN à la banque affichée",
                    paragraphs=[
                        "Dans un IBAN français, les cinq chiffres qui suivent la "
                        "clé identifient la banque, les cinq suivants le guichet. "
                        "Un RIB qui affiche le logo d'une banque mais dont le "
                        "code établissement appartient à une autre — souvent un "
                        "établissement de paiement en ligne — se contredit "
                        "lui-même. Le BIC, quand il est imprimé, doit désigner le "
                        "même établissement.",
                        "Un IBAN étranger pour un fournisseur français n'est pas "
                        "une preuve de fraude, mais c'est un signal qui justifie "
                        "à lui seul une vérification téléphonique.",
                    ],
                ),
                Section(
                    heading="Chercher la retouche dans le fichier",
                    paragraphs=[
                        "Quand le RIB frauduleux est une facture ou un RIB "
                        "authentique modifié, les indices sont ceux décrits dans "
                        "<a href=\"/guides/verifier-une-facture-falsifiee\">le "
                        "guide sur les factures falsifiées</a> — concentrés sur "
                        "la ligne de l'IBAN :",
                    ],
                    bullets=[
                        "Un <strong>changement de police</strong> au milieu de la "
                        "ligne des coordonnées bancaires.",
                        "Un <strong>rectangle blanc</strong> recouvrant l'ancien "
                        "IBAN, le nouveau écrit par-dessus — l'ancien numéro est "
                        "souvent encore présent sous le cache.",
                        "Une <strong>révision du PDF</strong> où l'IBAN change "
                        "d'une version à l'autre : l'un des rares indices proches "
                        "de la preuve.",
                        "Des <strong>métadonnées d'éditeur PDF</strong> sur un "
                        "document censé sortir d'une chaîne de facturation "
                        "automatisée.",
                    ],
                ),
                Section(
                    heading="Le rappel téléphonique reste la règle",
                    paragraphs=[
                        "Depuis fin 2025, les banques européennes doivent "
                        "vérifier la concordance entre le nom du bénéficiaire et "
                        "l'IBAN au moment du virement (« vérification du "
                        "bénéficiaire »). C'est un filet supplémentaire, pas une "
                        "raison de baisser la garde : l'alerte arrive au moment "
                        "du paiement, pas au moment où la décision se prend.",
                        "La règle d'or ne change pas : tout changement de "
                        "coordonnées bancaires se confirme par téléphone, sur un "
                        "numéro que vous possédiez déjà — jamais celui du message "
                        "qui annonce le changement, ni celui imprimé sur le RIB "
                        "suspect.",
                    ],
                ),
            ],
        ),
    ],
    "en": [
        Guide(
            slug="how-to-spot-a-tampered-invoice",
            counterpart="verifier-une-facture-falsifiee",
            title="How to spot a tampered invoice",
            description=(
                "Edited amounts, altered dates, swapped bank details: the "
                "technical evidence that gives away a tampered invoice PDF, and "
                "how to check for it."
            ),
            heading="How to check whether an invoice has been tampered with",
            intro=(
                "Invoice fraud no longer takes any special skill — a free PDF "
                "editor is enough to change an amount or a bank account number. "
                "The good news is that those edits almost always leave traces in "
                "the file, if you know where to look."
            ),
            sections=[
                Section(
                    heading="1. Look at what produced the file",
                    paragraphs=[
                        "A PDF records the software that wrote it in a field called "
                        "<em>Producer</em>, and the originating application in "
                        "<em>Creator</em>. An invoice issued by accounting software "
                        "carries that software's signature. If it carries a consumer "
                        "PDF editor's instead, the file was reopened and re-saved "
                        "after it was issued.",
                        "That is not proof of fraud on its own — plenty of people "
                        "compress an oversized attachment before emailing it. It is "
                        "a signal to combine with the others.",
                    ],
                    bullets=[
                        "<strong>Online converters</strong>: iLovePDF, Smallpdf, "
                        "Sejda, PDF24, Canva, Nitro. They rewrite the file entirely "
                        "and erase the original history along the way.",
                        "<strong>ModDate far later than CreationDate</strong>: the "
                        "document was changed long after it was created.",
                        "<strong>Empty metadata</strong> on a document supposedly "
                        "issued by a large company: industrial billing pipelines "
                        "fill those fields in.",
                    ],
                ),
                Section(
                    heading="2. Count the PDF revisions",
                    paragraphs=[
                        "The PDF format lets a program save a change by "
                        "<em>appending</em> data to the end of the file instead of "
                        "rewriting it. This is an incremental update, and the "
                        "previous state stays physically present in the file.",
                        "The practical consequence: on a multi-revision document you "
                        "can often render the earlier version and compare. An amount "
                        "that changes between two revisions is one of the few pieces "
                        "of evidence that comes close to proof.",
                        "Watch for false positives: an electronic signature "
                        "legitimately adds a revision, and some administrative forms "
                        "add one every time they are filled in.",
                    ],
                ),
                Section(
                    heading="3. Examine the fonts line by line",
                    paragraphs=[
                        "Software-generated documents use a consistent set of fonts "
                        "and embed them in the file. When someone edits an amount, "
                        "the editor may not have the original font available and "
                        "substitutes a system one: Arial, Calibri, Helvetica.",
                    ],
                    bullets=[
                        "A font change <strong>mid-line</strong> or mid-word is very "
                        "hard to produce by accident.",
                        "A non-embedded font among embedded ones points to a late "
                        "addition.",
                        "Unusual font diversity on a single page — beyond five or "
                        "six on an invoice — is worth a look.",
                    ],
                ),
                Section(
                    heading="4. Look for covered-up text",
                    paragraphs=[
                        "The most common edit deletes nothing: it covers. A white "
                        "rectangle over the old amount, the new amount typed on top. "
                        "Visually flawless. In the file, the old text is still "
                        "there, underneath.",
                        "Copying the PDF's text into a plain text editor is "
                        "sometimes enough to make the original value reappear. A "
                        "variant is text made invisible — invisible render mode, or "
                        "text coloured the same as the background — that contradicts "
                        "what is displayed.",
                    ],
                ),
                Section(
                    heading="5. Check what can be checked from outside",
                    paragraphs=[
                        "Some invoice data can be verified outside the file, which "
                        "is exactly what makes this step valuable: it does not "
                        "depend on the issuer's good faith.",
                    ],
                    bullets=[
                        "<strong>Company registration number</strong> (SIREN/SIRET "
                        "in France): the public registry answers directly. A number "
                        "that does not exist, or belongs to an unrelated company, is "
                        "decisive.",
                        "<strong>EU VAT number</strong>: it has a computable check "
                        "digit and can be validated against VIES.",
                        "<strong>IBAN</strong>: its mod-97 checksum can be verified "
                        "offline. A hand-edited IBAN almost always fails it.",
                        "<strong>Arithmetic consistency</strong>: net + tax = gross. "
                        "A partial edit breaks the sum.",
                    ],
                ),
                Section(
                    heading="The habit that beats every technical check",
                    paragraphs=[
                        "If you have any doubt about bank details, no file analysis "
                        "replaces phoning the supplier on a number you already have "
                        "— never the one printed on the suspicious invoice. Payment "
                        "redirection fraud works precisely because the victim "
                        "verifies the information using the information itself.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-verify-a-proof-of-address",
            counterpart="verifier-un-justificatif-de-domicile",
            title="How to verify a proof of address",
            description=(
                "Utility, telecom and internet bills used as proof of address: the "
                "technical checks that reveal an edited document, and when the "
                "checks simply cannot apply."
            ),
            heading="How to verify a proof of address",
            intro=(
                "Proof of address is among the most forged documents, because it "
                "is requested everywhere and verified almost nowhere. In most "
                "cases it is a genuine bill belonging to somebody else, with the "
                "name and address swapped."
            ),
            sections=[
                Section(
                    heading="The dominant scenario: name substitution",
                    paragraphs=[
                        "Unlike a fraudulent invoice, a forged proof of address "
                        "rarely starts from scratch. The forger takes a real bill — "
                        "their own, or one found online — and replaces only the "
                        "fields they care about: name and address.",
                        "That very localised edit is what makes it detectable: the "
                        "rest of the document is genuine, so every inconsistency "
                        "concentrates on the edited area.",
                    ],
                ),
                Section(
                    heading="Compare the producing software with the claimed issuer",
                    paragraphs=[
                        "Large utilities and telecom operators generate bills "
                        "through industrial pipelines with very stable technical "
                        "fingerprints: same PDF producer, same embedded fonts, same "
                        "internal structure month after month.",
                        "A document claiming to come from a major supplier whose "
                        "producer is an office suite, an online converter or a "
                        "screenshot tool contradicts what it claims to be.",
                    ],
                ),
                Section(
                    heading="Look for paste artifacts",
                    paragraphs=[
                        "When the name is replaced with an image rather than text, "
                        "the paste leaves measurable traces.",
                    ],
                    bullets=[
                        "<strong>Resolution break</strong>: a 96 dpi block sitting on "
                        "a 300 dpi page.",
                        "<strong>White that is not the same white</strong>: a "
                        "document background is rarely pure white, a pasted "
                        "rectangle often is.",
                        "<strong>Inconsistent local noise</strong>: a scan has "
                        "uniform noise across the page; a re-pasted region is "
                        "abnormally smooth or abnormally noisy.",
                        "<strong>JPEG compression artifacts</strong> concentrated "
                        "around one block of text.",
                    ],
                ),
                Section(
                    heading="The scanned document: knowing when to say \"I don't know\"",
                    paragraphs=[
                        "A document scanned and flattened into an image has no text "
                        "layer, no fonts and no revision history left. Nearly every "
                        "check described here becomes inapplicable.",
                        "This matters methodologically: the absence of anomalies "
                        "means nothing when there was nothing to examine. TrustyFile "
                        "therefore reports such documents as "
                        "<strong>unverifiable</strong> rather than giving them a good "
                        "score by default. Many tools do the opposite, which is "
                        "misleading.",
                        "Asking for the digital original instead of a scan or a "
                        "photo turns an impossible verification into a possible one. "
                        "It is the single best move when you are in a position to "
                        "insist.",
                    ],
                ),
                Section(
                    heading="Non-technical consistency checks",
                    bullets=[
                        "Is the bill date inside the required window, often three "
                        "months?",
                        "Is the consumption period earlier than the issue date? A "
                        "bill charging for the future is an anomaly.",
                        "Does the customer number match the supplier's format?",
                        "Is the amount plausible for the property and the season?",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-tell-if-a-pdf-was-modified",
            counterpart="savoir-si-un-pdf-a-ete-modifie",
            title="How to tell if a PDF was modified",
            description=(
                "A PDF keeps a record of its successive saves: incremental "
                "revisions, ghost objects, two sets of metadata. Where to look to "
                "find out whether a file was edited."
            ),
            heading="How to tell whether a PDF has been modified",
            intro=(
                "A PDF is not a flat image: it is a documented format that retains "
                "far more about its own history than the visible page suggests. "
                "Here is what it keeps, and what can be inferred from it."
            ),
            sections=[
                Section(
                    heading="Incremental revisions",
                    paragraphs=[
                        "When software saves a change it can rewrite the whole file, "
                        "or append the changed objects to the end and leave "
                        "everything before untouched. The second mode — the "
                        "incremental update — is the default in several editors, and "
                        "it is a gift for analysis: every revision is a preserved "
                        "snapshot.",
                        "A file with three revisions was saved three times. Truncate "
                        "it at the end of the first to recover the original state, "
                        "then compare.",
                    ],
                    bullets=[
                        "A single revision: the file was written once.",
                        "Several revisions <strong>with</strong> an electronic "
                        "signature: normal, signing adds one.",
                        "Several revisions <strong>without</strong> a signature on a "
                        "document supposedly produced by an automated pipeline: not "
                        "normal.",
                    ],
                ),
                Section(
                    heading="Ghost objects",
                    paragraphs=[
                        "Deleting an element in a PDF editor removes it from the "
                        "display, not necessarily from the file. The object becomes "
                        "orphaned: no longer referenced, still present. Finding an "
                        "old amount, an old address or a logo that no longer appears "
                        "is strong evidence.",
                    ],
                ),
                Section(
                    heading="The two sets of metadata",
                    paragraphs=[
                        "A PDF can carry metadata in two places: the legacy "
                        "<em>Info</em> dictionary, and an <em>XMP</em> stream in "
                        "XML. Many tools update one and forget the other.",
                        "A divergence between them — different dates, different "
                        "producers — reveals processing by a tool that did not clean "
                        "up after itself. XMP sometimes keeps a <em>DocumentID</em> "
                        "and a version history more talkative than the Producer "
                        "field.",
                    ],
                ),
                Section(
                    heading="What metadata does not prove",
                    paragraphs=[
                        "Metadata fields are free text: anyone can write whatever "
                        "producer they like. A well-informed forger restores the "
                        "expected value after editing.",
                        "This is why <em>structural</em> evidence matters more: "
                        "object ordering, compression style, how the cross-reference "
                        "table is written. Those details are emitted by the library "
                        "that wrote the file, and a forger does not reproduce them "
                        "by editing a string. A file announcing one producer while "
                        "its structure matches another is an interesting case.",
                    ],
                ),
                Section(
                    heading="Modified is not the same as fraudulent",
                    paragraphs=[
                        "This deserves saying plainly: most modified PDFs were "
                        "modified for entirely mundane reasons — compression before "
                        "sending, adding a signature, merging documents, flattening "
                        "a form, passing through an archiving tool.",
                        "Detecting a modification therefore only moves the question "
                        "along: <em>what</em> was modified, and does it touch data "
                        "that matters? Several signals converging on the same region "
                        "of the document make a case; one signal on its own does "
                        "not.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="2d-doc-verifying-official-documents",
            counterpart="2d-doc-verifier-un-document-officiel",
            title="2D-Doc: verifying French official documents",
            description=(
                "The 2D-Doc barcode on tax notices, pay slips and official "
                "certificates: what it contains, what it actually proves, and why "
                "it is different from every other check."
            ),
            heading="2D-Doc: the only check that proves something",
            intro=(
                "Every check described in the other guides is circumstantial — it "
                "makes fraud more or less likely, never certain. 2D-Doc is "
                "different: it rests on cryptography, and allows a clear-cut "
                "answer."
            ),
            sections=[
                Section(
                    heading="What a 2D-Doc is",
                    paragraphs=[
                        "2D-Doc is a two-dimensional DataMatrix barcode specified by "
                        "the French ANTS (national agency for secure documents). It "
                        "appears on tax notices, pay slips, social security "
                        "certificates, proof-of-address bills from some suppliers, "
                        "and assorted administrative documents.",
                        "It holds two things: the document's key data in clear text "
                        "(name, amount, date, reference), and a "
                        "<strong>cryptographic signature</strong> of that data, "
                        "produced with the issuer's private key.",
                    ],
                ),
                Section(
                    heading="Why it beats everything else",
                    paragraphs=[
                        "Metadata can be rewritten, fonts can be imitated, a scan "
                        "erases the traces. A cryptographic signature cannot: "
                        "without the issuer's private key you cannot produce a valid "
                        "signature over modified data.",
                        "That yields two distinct checks, and the difference between "
                        "them matters:",
                    ],
                    bullets=[
                        "<strong>Is the code readable and consistent?</strong> Does "
                        "the encoded data match the text printed on the document? If "
                        "the PDF shows one amount and the code holds another, "
                        "somebody edited the visual without being able to touch the "
                        "code.",
                        "<strong>Is the signature valid?</strong> This requires the "
                        "issuer's certificate and an ECDSA verification. It is the "
                        "only check that can assert a document is genuine, rather "
                        "than merely \"no anomaly detected\".",
                    ],
                ),
                Section(
                    heading="Where TrustyFile stands",
                    paragraphs=[
                        "TrustyFile currently decodes the 2D-Doc and compares the "
                        "encoded data against the visible text. That is the first "
                        "check, and it already catches visual edits.",
                        "Signature verification itself is not implemented yet. We "
                        "would rather write that here than imply a guarantee that "
                        "does not exist: today, no TrustyFile result proves a "
                        "document is genuine. The tool reports anomalies; it does "
                        "not issue certificates.",
                    ],
                ),
                Section(
                    heading="Check with the issuer in parallel",
                    paragraphs=[
                        "For a French tax notice, the tax administration runs an "
                        "online verification service (SVAIR) where you enter the tax "
                        "number and the notice reference. The answer comes from the "
                        "source itself, which beats any file analysis.",
                        "General rule: when the issuer offers a way to verify "
                        "directly with them, that is always the route to take "
                        "first.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-verify-a-pay-slip",
            counterpart="verifier-un-bulletin-de-salaire",
            title="How to verify a pay slip",
            description=(
                "Fake pay slips in rental files and loan applications: payroll "
                "software fingerprints, arithmetic checks and the French 2D-Doc "
                "barcode that expose an edited document."
            ),
            heading="How to verify a pay slip",
            intro=(
                "The forged pay slip has become a staple of rental files and "
                "loan applications: an inflated salary on one's own slip, or a "
                "swapped name on somebody else's. Like every edit it leaves "
                "traces — and the French pay slip has one feature that makes it "
                "more verifiable than most documents."
            ),
            sections=[
                Section(
                    heading="Look for the 2D-Doc code first",
                    paragraphs=[
                        "Many French payroll systems print a 2D-Doc code on the "
                        "slip: a small DataMatrix square, usually near the top or "
                        "bottom of the page. It holds the key data — name, "
                        "period, net amount — cryptographically signed by the "
                        "issuer.",
                        "Its presence changes everything: if the salary printed "
                        "on the page does not match the one encoded in the "
                        "square, somebody edited the visual without being able "
                        "to touch the code. It is one of the few checks close to "
                        "proof. The details are in "
                        "<a href=\"/en/guides/2d-doc-verifying-official-documents\">"
                        "our 2D-Doc guide</a>.",
                        "Its absence, on the other hand, proves nothing: not "
                        "every employer uses software that generates one.",
                    ],
                ),
                Section(
                    heading="Identify the software that produced the slip",
                    paragraphs=[
                        "Pay slips come out of industrial payroll pipelines — "
                        "Silae, PayFit, ADP, Sage, Cegid — whose technical "
                        "fingerprint is very stable: same PDF producer, same "
                        "embedded fonts, same structure month after month.",
                        "A slip whose metadata announces a word processor, an "
                        "online converter or a screenshot tool contradicts what "
                        "it claims to be. Employers do not type pay slips in "
                        "Word.",
                    ],
                ),
                Section(
                    heading="Redo the arithmetic",
                    paragraphs=[
                        "A pay slip is a public addition. A forger who inflates "
                        "the net pay almost always forgets to fix everything "
                        "that depends on it:",
                    ],
                    bullets=[
                        "<strong>Gross minus contributions = net</strong>: each "
                        "contribution line applies a rate to a base; a net "
                        "edited without recomputing the lines breaks the sum.",
                        "<strong>The taxable and social net amounts</strong> "
                        "must stay consistent with the net pay.",
                        "<strong>Year-to-date totals</strong>: a cumulative "
                        "figure lower than the month's amount, or one that does "
                        "not grow from one slip to the next, betrays an isolated "
                        "edit.",
                        "<strong>Salary versus job title</strong>: an amount out "
                        "of proportion with the position and grade deserves a "
                        "question.",
                    ],
                ),
                Section(
                    heading="Examine the name and amount areas",
                    paragraphs=[
                        "As with proof of address, the common fraud does not "
                        "build a slip from scratch: it swaps a name or an amount "
                        "on a genuine one. The evidence therefore concentrates "
                        "on those areas: a font change mid-line, a white "
                        "rectangle covering the old value, a pasted image block "
                        "whose resolution does not match the rest of the page.",
                    ],
                ),
                Section(
                    heading="Verify the employer",
                    paragraphs=[
                        "A French pay slip must state the employer's company "
                        "registration number (SIRET), activity code and "
                        "collective agreement. All of it can be checked from "
                        "outside:",
                    ],
                    bullets=[
                        "Does the <strong>SIRET</strong> exist in the public "
                        "registry, and does it match the displayed name?",
                        "Is the <strong>activity code</strong> plausible for the "
                        "job held?",
                        "Is the <strong>collective agreement</strong> the one "
                        "for that sector?",
                        "Was the company still active during the slip's period?",
                    ],
                ),
                Section(
                    heading="Scans, once again: unverifiable",
                    paragraphs=[
                        "A scanned or photographed slip keeps no text layer, no "
                        "fonts, no payroll metadata. Most of the checks above "
                        "become inapplicable, and the absence of anomalies then "
                        "means nothing. When you are in a position to insist, "
                        "ask for the original PDF — the one the employee "
                        "downloads from the HR portal — rather than a flattened "
                        "copy.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-spot-a-fake-iban",
            counterpart="detecter-un-faux-rib",
            title="How to spot a fake IBAN on an invoice",
            description=(
                "Payment redirection fraud swaps the bank details on a genuine "
                "invoice. IBAN checksum, bank-code consistency and PDF tampering "
                "traces: how to check details received by email."
            ),
            heading="How to spot fake bank details",
            intro=(
                "Payment redirection fraud is among the most expensive there "
                "is: a supplier announces \"new bank details\", the transfer "
                "goes to the fraudster's account, and nobody notices until the "
                "real supplier chases the unpaid invoice weeks later. The "
                "attached document is often genuine in every detail but one: "
                "the IBAN."
            ),
            sections=[
                Section(
                    heading="Understand the scenario",
                    paragraphs=[
                        "The attack takes three main forms: an email imitating "
                        "the supplier with a fraudulent bank-details document "
                        "attached; a genuinely compromised mailbox, from which a "
                        "perfectly credible message is sent; or an authentic "
                        "invoice intercepted and re-sent with only the IBAN "
                        "changed.",
                        "In all three cases the attachment is the control point: "
                        "it is often a real document with a single edited field, "
                        "and that localised edit is detectable.",
                    ],
                ),
                Section(
                    heading="Verify the IBAN checksum",
                    paragraphs=[
                        "An IBAN carries its own checksum: the two digits after "
                        "the country code are computed from the rest of the "
                        "number (modulo 97), offline, with no database needed. A "
                        "hand-edited IBAN — one digit changed to resemble the "
                        "original — almost always fails this test.",
                        "Mind what the test means: a <em>wrong</em> checksum "
                        "condemns the document, but a <em>valid</em> one does "
                        "not clear it. Fraudsters open real accounts, with real, "
                        "perfectly valid IBANs.",
                    ],
                ),
                Section(
                    heading="Match the IBAN against the displayed bank",
                    paragraphs=[
                        "The digits at the start of the account portion identify "
                        "the bank (in a French IBAN, the five digits after the "
                        "check digits). A document showing one bank's logo while "
                        "the bank code belongs to another — often an online "
                        "payment institution — contradicts itself. The BIC, when "
                        "printed, must name the same institution.",
                        "A foreign IBAN for a domestic supplier is not proof of "
                        "fraud, but it is a signal that alone justifies a phone "
                        "check.",
                    ],
                ),
                Section(
                    heading="Look for the edit in the file",
                    paragraphs=[
                        "When the fraudulent document is a genuine invoice or "
                        "bank-details sheet that was modified, the evidence is "
                        "the same as in "
                        "<a href=\"/en/guides/how-to-spot-a-tampered-invoice\">"
                        "the tampered-invoice guide</a> — concentrated on the "
                        "IBAN line:",
                    ],
                    bullets=[
                        "A <strong>font change</strong> in the middle of the "
                        "bank-details line.",
                        "A <strong>white rectangle</strong> covering the old "
                        "IBAN with the new one typed on top — the old number is "
                        "often still present underneath.",
                        "A <strong>PDF revision</strong> in which the IBAN "
                        "differs from one version to the next: one of the few "
                        "pieces of evidence close to proof.",
                        "<strong>PDF-editor metadata</strong> on a document "
                        "supposedly produced by an automated billing pipeline.",
                    ],
                ),
                Section(
                    heading="The phone call remains the rule",
                    paragraphs=[
                        "Since late 2025, European banks must check that the "
                        "beneficiary name matches the IBAN at transfer time "
                        "(verification of payee). That is an extra net, not a "
                        "reason to relax: the alert comes when the payment is "
                        "made, not when the decision is taken.",
                        "The golden rule does not change: any change of bank "
                        "details is confirmed by phone, on a number you already "
                        "had — never the one in the message announcing the "
                        "change, nor the one printed on the suspicious "
                        "document.",
                    ],
                ),
            ],
        ),
    ],
}


# --- Rendering ---------------------------------------------------------------

STYLESHEET = """\
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 760px; margin: 0 auto; padding: 2rem 1.25rem 4rem;
    color: #333; line-height: 1.65; background: #fff;
}
nav.top { font-size: 0.85rem; color: #888; margin-bottom: 2rem; }
nav.top a { color: #2563eb; text-decoration: none; }
nav.top a:hover { text-decoration: underline; }
nav.top .sep { margin: 0 0.4rem; color: #ccc; }
nav.top .lang { float: right; }
h1 { color: #1a1a2e; font-size: 1.9rem; line-height: 1.25; margin: 0 0 0.5rem; }
h2 { color: #2a2a3e; font-size: 1.25rem; margin: 2.5rem 0 0.75rem; }
h3 { color: #2a2a3e; font-size: 1.05rem; margin: 1.75rem 0 0.5rem; }
p.intro { font-size: 1.08rem; color: #444; }
p.updated { font-size: 0.8rem; color: #999; margin: 0 0 2rem; }
ul { padding-left: 1.25rem; }
li { margin: 0.4rem 0; }
ol.toc { background: #f8f9fa; border-radius: 8px; padding: 1rem 1rem 1rem 2.25rem; }
ol.toc li { margin: 0.25rem 0; }
a { color: #2563eb; }
.cta {
    display: block; margin: 3rem 0 1rem; padding: 1.25rem; text-align: center;
    background: #1a1a2e; color: #fff; border-radius: 10px;
    text-decoration: none; font-weight: 600;
}
.cta:hover { background: #2a2a4e; }
.card {
    display: block; padding: 1.1rem 1.25rem; margin: 0.85rem 0;
    background: #f8f9fa; border-radius: 10px; text-decoration: none; color: inherit;
    border: 1px solid #eceef1;
}
.card:hover { border-color: #2563eb; }
.card h2 { margin: 0 0 0.35rem; font-size: 1.1rem; color: #1a1a2e; }
.card h3 { margin: 0 0 0.35rem; font-size: 1.05rem; color: #1a1a2e; }
.card p { margin: 0; color: #555; font-size: 0.92rem; }
.card .more { color: #2563eb; font-size: 0.88rem; }
footer {
    margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid #eee;
    text-align: center; color: #888; font-size: 0.85rem;
}
/* Guides index FAQ */
dl.faq dt { font-weight: 600; color: #1a1a2e; margin-top: 1.25rem; }
dl.faq dd { margin: 0.3rem 0 0; color: #555; }
/* Crawler landing page — mirrors the Streamlit landing, element for element */
header.brand { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 2.5rem; }
header.brand .name { font-weight: 700; color: #1a1a2e; text-decoration: none; font-size: 1.05rem; }
header.brand .subtitle { color: #888; font-size: 0.85rem; }
p.tagline { font-size: 1.15rem; color: #444; margin: 0 0 1rem; }
p.note { font-size: 0.85rem; color: #777; }
ol.steps { list-style: none; padding: 0; display: flex; gap: 1rem; flex-wrap: wrap; margin: 2rem 0; }
ol.steps li { flex: 1 1 180px; background: #f8f9fa; border-radius: 10px; padding: 1rem 1.1rem; }
ol.steps .n {
    display: inline-flex; width: 26px; height: 26px; border-radius: 50%;
    background: #1a1a2e; color: #fff; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; margin-bottom: 0.5rem;
}
ol.steps strong { display: block; color: #1a1a2e; margin-bottom: 0.2rem; }
ol.steps .d { color: #666; font-size: 0.9rem; }
@media (max-width: 480px) {
    body { padding: 1.25rem 1rem 3rem; }
    h1 { font-size: 1.55rem; }
}
"""


def _slugify_heading(heading: str) -> str:
    """Turn a section heading into a stable #anchor.

    Anchors let the table of contents link into the page, and let Google show
    "jump to" links in the results. Accents and punctuation are dropped rather
    than percent-encoded, which keeps the URLs readable.

    Example:
        >>> _slugify_heading("1. Regarder qui a produit le fichier")
        'regarder-qui-a-produit-le-fichier'
    """
    import unicodedata

    # NFKD splits "é" into "e" + combining accent; the ASCII encode then drops
    # the accent rather than the whole letter.
    normalised = unicodedata.normalize("NFKD", heading)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    kept = [c if c.isalnum() else "-" for c in ascii_only]
    slug = "".join(kept)
    # Collapse runs of separators left by punctuation ("2d-doc :" -> "2d-doc-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _head(
    *,
    lang: str,
    title: str,
    description: str,
    canonical: str,
    alternates: dict[str, str],
    json_ld: list[dict],
    og_type: str = "article",
    extra_head: str = "",
) -> str:
    """Build the <head> shared by every generated page.

    Args:
        lang: "fr" or "en".
        title: <title> text, already plain (escaped here).
        description: meta description text.
        canonical: absolute URL of this page.
        alternates: {hreflang: absolute URL}, including x-default.
        json_ld: structured-data objects, one <script> each.
        og_type: Open Graph type — "article" for guides, "website" for the
            landing pages.
        extra_head: extra lines inserted before </head> (already escaped).
    """
    chrome = CHROME[lang]
    esc = html.escape
    alt_links = "\n".join(
        f'    <link rel="alternate" hreflang="{code}" href="{url}">'
        for code, url in alternates.items()
    )
    ld_blocks = "\n".join(
        '    <script type="application/ld+json">\n'
        + json.dumps(block, ensure_ascii=False, indent=8)
        + "\n    </script>"
        for block in json_ld
    )
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)}</title>
    <meta name="description" content="{esc(description)}">
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">
    <meta name="theme-color" content="#1a1a2e">
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <link rel="icon" href="/favicon-48x48.png" type="image/png" sizes="48x48">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    <link rel="canonical" href="{canonical}">
{alt_links}
    <meta property="og:type" content="{og_type}">
    <meta property="og:title" content="{esc(title)}">
    <meta property="og:description" content="{esc(description)}">
    <meta property="og:url" content="{canonical}">
    <meta property="og:image" content="{SITE}/og-image.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:site_name" content="TrustyFile">
    <meta property="og:locale" content="{chrome['locale']}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{esc(title)}">
    <meta name="twitter:description" content="{esc(description)}">
    <meta name="twitter:image" content="{SITE}/og-image.png">
    <style>
{STYLESHEET}    </style>
{ld_blocks}
{extra_head}</head>
<body>"""


def _nav(lang: str, other_url: str) -> str:
    """Breadcrumb-ish top nav plus the language switch."""
    chrome = CHROME[lang]
    return f"""    <nav class="top">
        <a href="{chrome['app_url']}">{chrome['nav_home']}</a>
        <span class="sep">/</span>
        <a href="{chrome['index_path']}">{chrome['nav_guides']}</a>
        <a class="lang" href="{other_url}">{chrome['other_lang_label']}</a>
    </nav>
"""


def _footer(lang: str) -> str:
    chrome = CHROME[lang]
    return f"""    <footer>
        <p><a href="{chrome['app_url']}">TrustyFile</a>
        &middot; <a href="{chrome['index_path']}">{chrome['nav_guides']}</a>
        &middot; <a href="mailto:contact@trustyfile.com">{chrome['footer_contact']}</a></p>
    </footer>
</body>
</html>
"""


def render_guide(lang: str, guide: Guide) -> str:
    """Render one guide page to a complete HTML document."""
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    canonical = f"{SITE}{chrome['guides_path']}/{guide.slug}"

    # hreflang set. A guide without a translation still declares itself, which
    # is required: hreflang annotations must be reciprocal and self-referential.
    alternates = {lang: canonical}
    if guide.counterpart:
        other_canonical = (
            f"{SITE}{CHROME[other_lang]['guides_path']}/{guide.counterpart}"
        )
        alternates[other_lang] = other_canonical
        # French is the site's default language, hence x-default.
        alternates["x-default"] = canonical if lang == "fr" else other_canonical
        other_url = f"{CHROME[other_lang]['guides_path']}/{guide.counterpart}"
    else:
        other_url = CHROME[other_lang]["index_path"]

    article_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": guide.title,
        "description": guide.description,
        "inLanguage": lang,
        "datePublished": LAST_MODIFIED,
        "dateModified": LAST_MODIFIED,
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
        "image": f"{SITE}/og-image.png",
        "author": {"@type": "Organization", "name": "TrustyFile", "url": SITE},
        "publisher": {
            "@type": "Organization",
            "name": "TrustyFile",
            "url": SITE,
            "logo": {
                "@type": "ImageObject",
                "url": f"{SITE}/favicon-192x192.png",
            },
        },
    }
    breadcrumb_ld = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": 1,
                "name": "TrustyFile",
                "item": chrome["app_url"],
            },
            {
                "@type": "ListItem",
                "position": 2,
                "name": chrome["nav_guides"],
                "item": f"{SITE}{chrome['index_path']}",
            },
            {"@type": "ListItem", "position": 3, "name": guide.title},
        ],
    }

    parts = [
        _head(
            lang=lang,
            title=guide.title,
            description=guide.description,
            canonical=canonical,
            alternates=alternates,
            json_ld=[article_ld, breadcrumb_ld],
        ),
        _nav(lang, other_url),
        f"    <h1>{html.escape(guide.heading)}</h1>",
        f'    <p class="updated">{chrome["updated"]}</p>',
        f'    <p class="intro">{guide.intro}</p>',
    ]

    # Table of contents. Only worth rendering when there are enough sections
    # for a reader to actually want to skip around.
    if len(guide.sections) >= 4:
        parts.append(f"    <h2>{chrome['toc']}</h2>")
        parts.append('    <ol class="toc">')
        for section in guide.sections:
            anchor = _slugify_heading(section.heading)
            parts.append(
                f'        <li><a href="#{anchor}">{html.escape(section.heading)}</a></li>'
            )
        parts.append("    </ol>")

    for section in guide.sections:
        anchor = _slugify_heading(section.heading)
        parts.append(f'    <h2 id="{anchor}">{html.escape(section.heading)}</h2>')
        for paragraph in section.paragraphs:
            parts.append(f"    <p>{paragraph}</p>")
        if section.bullets:
            parts.append("    <ul>")
            for bullet in section.bullets:
                parts.append(f"        <li>{bullet}</li>")
            parts.append("    </ul>")

    # Related guides: every sibling in the same language. Internal links are
    # the main way crawlers discover pages and how link signal flows between
    # them; the sitemap alone is a much weaker channel. h3 (not h2) so the
    # cards do not pollute the article's own outline.
    siblings = [g for g in GUIDES[lang] if g.slug != guide.slug]
    if siblings:
        parts.append(f"    <h2>{chrome['related']}</h2>")
        for other in siblings:
            url = f"{chrome['guides_path']}/{other.slug}"
            parts.append(f'    <a class="card" href="{url}">')
            parts.append(f"        <h3>{html.escape(other.title)}</h3>")
            parts.append(f"        <p>{html.escape(other.description)}</p>")
            parts.append("    </a>")

    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(_footer(lang))
    return "\n".join(parts)


def _faq_ld(lang: str) -> dict:
    """FAQPage structured data for the FAQ rendered on the guides index.

    Only valid because the same questions and answers are visible on the page
    to every visitor: structured data describing content that is not shown is
    a Google Search spam-policy violation in its own right.
    """
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {"@type": "Answer", "text": answer},
            }
            for question, answer in CHROME[lang]["faq"]
        ],
    }


def _faq_html(lang: str) -> list[str]:
    """The visible FAQ: a heading and a definition list."""
    chrome = CHROME[lang]
    parts = [f"    <h2>{html.escape(chrome['faq_heading'])}</h2>", '    <dl class="faq">']
    for question, answer in chrome["faq"]:
        parts.append(f"        <dt>{html.escape(question)}</dt>")
        parts.append(f"        <dd>{html.escape(answer)}</dd>")
    parts.append("    </dl>")
    return parts


def render_index(lang: str) -> str:
    """Render the guides index: the hub that links to every guide, plus the FAQ."""
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    canonical = f"{SITE}{chrome['index_path']}"
    other_canonical = f"{SITE}{CHROME[other_lang]['index_path']}"

    alternates = {
        lang: canonical,
        other_lang: other_canonical,
        "x-default": canonical if lang == "fr" else other_canonical,
    }

    collection_ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": chrome["index_heading"],
        "description": chrome["index_description"],
        "url": canonical,
        "inLanguage": lang,
        "hasPart": [
            {
                "@type": "Article",
                "headline": guide.title,
                "description": guide.description,
                "url": f"{SITE}{chrome['guides_path']}/{guide.slug}",
            }
            for guide in GUIDES[lang]
        ],
    }

    parts = [
        _head(
            lang=lang,
            title=chrome["index_title"],
            description=chrome["index_description"],
            canonical=canonical,
            alternates=alternates,
            json_ld=[collection_ld, _faq_ld(lang)],
        ),
        _nav(lang, CHROME[other_lang]["index_path"]),
        f"    <h1>{html.escape(chrome['index_heading'])}</h1>",
        f'    <p class="intro">{chrome["index_intro"]}</p>',
    ]
    for guide in GUIDES[lang]:
        url = f"{chrome['guides_path']}/{guide.slug}"
        parts.append(f'    <a class="card" href="{url}">')
        parts.append(f"        <h2>{html.escape(guide.title)}</h2>")
        parts.append(f"        <p>{html.escape(guide.description)}</p>")
        parts.append(f'        <p class="more">{chrome["read_more"]} &rarr;</p>')
        parts.append("    </a>")
    parts.extend(_faq_html(lang))
    parts.append(f"    <p>{chrome['index_outro']}</p>")
    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(_footer(lang))
    return "\n".join(parts)


def render_landing(lang: str) -> str:
    """Render the crawler-facing landing page (seo-fr.html / seo-en.html).

    Caddy serves this file to search-engine and link-preview bots instead of
    the Streamlit shell, whose landing content only exists after JavaScript
    runs. To stay clear of cloaking, the *visible* body is limited to what
    app.py renders when no file is uploaded — headline, tagline, the upload
    disclaimer, the three steps, the API disclaimer and the footer links — and
    every one of those strings is read from src/translations/<lang>.json, the
    same file app.py's t() reads. Head metadata (title, description, JSON-LD)
    comes from CHROME[lang]; it describes the app rather than adding content.

    Example:
        >>> "Vous avez un doute" in render_landing("fr")
        True
    """
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    t = _translations(lang)
    esc = html.escape

    fr_home = f"{SITE}{CHROME['fr']['landing_path']}"
    en_home = f"{SITE}{CHROME['en']['landing_path']}"
    canonical = fr_home if lang == "fr" else en_home
    alternates = {"fr": fr_home, "en": en_home, "x-default": fr_home}
    id_suffix = "" if lang == "fr" else f"-{lang}"

    graph_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "@id": f"{SITE}/#organization",
                "name": "TrustyFile",
                "url": SITE,
                "logo": f"{SITE}/favicon-192x192.png",
                "email": "contact@trustyfile.com",
            },
            {
                "@type": "WebSite",
                "@id": f"{SITE}/#website{id_suffix}",
                "name": "TrustyFile",
                "url": canonical,
                "inLanguage": lang,
                "publisher": {"@id": f"{SITE}/#organization"},
            },
            {
                "@type": "WebApplication",
                "@id": f"{SITE}/#webapp{id_suffix}",
                "name": "TrustyFile",
                "url": canonical,
                "description": chrome["landing_webapp_description"],
                "applicationCategory": "SecurityApplication",
                "browserRequirements": "Requires JavaScript",
                "operatingSystem": "Any",
                "inLanguage": lang,
                "featureList": chrome["landing_feature_list"],
                "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
                "image": f"{SITE}/og-image.png",
                "publisher": {"@id": f"{SITE}/#organization"},
            },
        ],
    }

    extra_head = "\n".join(
        [
            '    <link rel="icon" href="/favicon-192x192.png" type="image/png" sizes="192x192">',
            '    <link rel="manifest" href="/site.webmanifest">',
            f'    <meta property="og:locale:alternate" content="{CHROME[other_lang]["locale"]}">',
            f'    <meta property="og:image:alt" content="{esc(chrome["landing_og_image_alt"])}">',
            "",
        ]
    )

    steps = "\n".join(
        f'        <li><span class="n">{i}</span>'
        f'<strong>{esc(t[f"step{i}_title"])}</strong>'
        f'<span class="d">{esc(t[f"step{i}_desc"])}</span></li>'
        for i in (1, 2, 3)
    )
    other_home = CHROME[other_lang]["landing_path"]

    parts = [
        _head(
            lang=lang,
            title=chrome["landing_title"],
            description=chrome["landing_description"],
            canonical=canonical,
            alternates=alternates,
            json_ld=[graph_ld],
            og_type="website",
            extra_head=extra_head,
        ),
        '    <header class="brand">',
        f'        <a class="name" href="{chrome["landing_path"]}">TrustyFile</a>',
        f'        <span class="subtitle">{esc(t["subtitle"])}</span>',
        "    </header>",
        f"    <h1>{esc(t['landing_headline'])}</h1>",
        f'    <p class="tagline">{esc(t["landing_tagline"])}</p>',
        f'    <p class="note">{esc(t["disclaimer_upload"])}</p>',
        '    <ol class="steps">',
        steps,
        "    </ol>",
        f'    <p class="note">{esc(t["disclaimer_api"])}</p>',
        "    <footer>",
        f'        <p><a href="{chrome["index_path"]}">{esc(t["footer_guides"])}</a>',
        f'        &middot; {esc(t["footer_contact"])} '
        '<a href="mailto:contact@trustyfile.com">contact@trustyfile.com</a>',
        f'        &middot; <a class="lang" href="{other_home}">{chrome["lang_switch"]}</a></p>',
        "    </footer>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


@dataclass
class SitemapEntry:
    """One <url> in sitemap.xml, with its hreflang alternates.

    Attributes:
        loc: the URL itself.
        fr: French equivalent, for the xhtml:link annotation.
        en: English equivalent.
        priority: relative importance within this site only. Google largely
            ignores it; kept because other engines still read it.
    """

    loc: str
    fr: str
    en: str
    priority: str


def sitemap_entries() -> list[SitemapEntry]:
    """Every indexable URL: the two app landing pages plus all guides.

    hreflang annotations must be reciprocal — each URL lists both languages,
    including itself — or Google discards the whole cluster.
    """
    fr_home, en_home = SITE + "/", SITE + "/en"
    entries = [
        SitemapEntry(fr_home, fr_home, en_home, "1.0"),
        SitemapEntry(en_home, fr_home, en_home, "0.9"),
    ]
    for lang in ("fr", "en"):
        other = "en" if lang == "fr" else "fr"
        index = f"{SITE}{CHROME[lang]['index_path']}"
        entries.append(
            SitemapEntry(
                index,
                f"{SITE}{CHROME['fr']['index_path']}",
                f"{SITE}{CHROME['en']['index_path']}",
                "0.8",
            )
        )
        for guide in GUIDES[lang]:
            loc = f"{SITE}{CHROME[lang]['guides_path']}/{guide.slug}"
            if guide.counterpart:
                other_loc = f"{SITE}{CHROME[other]['guides_path']}/{guide.counterpart}"
                fr_url, en_url = (loc, other_loc) if lang == "fr" else (other_loc, loc)
            else:
                # No translation yet: the page is its own only alternate.
                fr_url = en_url = loc
            entries.append(SitemapEntry(loc, fr_url, en_url, "0.7"))
    return entries


def render_sitemap() -> str:
    """Render sitemap.xml.

    Generated rather than hand-edited: a guide added to GUIDES but forgotten in
    the sitemap is invisible to search engines, and that mistake is silent.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!-- Generated by scripts/build_guides.py - do not edit by hand. -->',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ]
    for entry in sitemap_entries():
        lines += [
            "  <url>",
            f"    <loc>{entry.loc}</loc>",
            f'    <xhtml:link rel="alternate" hreflang="fr" href="{entry.fr}" />',
            f'    <xhtml:link rel="alternate" hreflang="en" href="{entry.en}" />',
            f'    <xhtml:link rel="alternate" hreflang="x-default" href="{entry.fr}" />',
            f"    <lastmod>{LAST_MODIFIED}</lastmod>",
            "    <changefreq>monthly</changefreq>",
            f"    <priority>{entry.priority}</priority>",
            "  </url>",
        ]
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def build() -> dict[Path, str]:
    """Every generated file. Returns {absolute path: content}, without writing."""
    pages: dict[Path, str] = {}
    for lang in ("fr", "en"):
        pages[OUTPUT_DIR / lang / "index.html"] = render_index(lang)
        for guide in GUIDES[lang]:
            pages[OUTPUT_DIR / lang / f"{guide.slug}.html"] = render_guide(lang, guide)
        pages[ROOT / f"seo-{lang}.html"] = render_landing(lang)
    pages[ROOT / "sitemap.xml"] = render_sitemap()
    return pages


def main() -> None:
    pages = build()
    written = 0
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Skip identical writes so rebuilding does not churn file mtimes.
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        path.write_text(content, encoding="utf-8")
        written += 1
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"{written} file(s) updated, {len(pages)} total")


if __name__ == "__main__":
    main()
