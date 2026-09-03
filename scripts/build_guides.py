"""Render the static pages: guides (/guides, /en/guides), the crawler landing
pages (seo-*.html), the checker tools (/outils, /en/tools), sitemap.xml
and llms.txt.

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


# Default `published` / `modified` date for the guides written before pages
# carried their own dates. Bumped by hand when a guide is substantially
# rewritten — never auto-set to "today" on every build: a lastmod that moves
# without the content moving is the kind of freshness signal Google discounts.
LAST_MODIFIED = "2026-08-13"

# The day the checker tools first went live; their sitemap <lastmod>.
TOOLS_LAUNCHED = "2026-09-03"


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
        published: ISO date the page first went live (Article datePublished).
        modified: ISO date of the last substantive rewrite (dateModified, the
            visible "updated" line and the sitemap <lastmod>). Bump by hand.
        tools: slugs of the checker tools (TOOLS[lang]) worth offering at the
            end of this guide, e.g. the IBAN checker under the fake-RIB guide.
            Slugs rather than objects so a typo fails the tests, not the reader.
        audience: which section of the hub lists this guide — "individuals"
            (rental files, personal documents), "business" (suppliers, company
            paperwork) or "method" (how PDFs and 2D-Doc work, for everyone).
    """

    slug: str
    title: str
    description: str
    heading: str
    intro: str
    sections: list[Section]
    counterpart: str | None = None
    published: str = LAST_MODIFIED
    modified: str = LAST_MODIFIED
    tools: list[str] = field(default_factory=list)
    audience: str = ""


AUDIENCES = ("individuals", "business", "method")  # hub section order


@dataclass
class Tool:
    """A standalone checker page (IBAN / SIRET / SIREN), in a single language.

    Same head/nav/footer as a guide, plus a form whose validation runs in the
    visitor's browser: nothing is sent anywhere. FR pages live at
    /outils/<slug>, EN at /en/tools/<slug>.

    Attributes:
        kind: "iban" | "siret" | "siren" | "vat" — selects the JS validator and the
            result messages (CHROME[lang]["tool_i18n"]).
        placeholder: example value shown in the empty input.
        sections: the explanatory body under the form: what the check proves,
            what it does not, how the algorithm works.
        modified: ISO date of the last substantive change (sitemap <lastmod>).
        guides: slugs of the guides (GUIDES[lang]) to offer under the tool —
            the IBAN checker points at the fake-RIB and supplier-fraud guides.
    """

    slug: str
    title: str
    description: str
    heading: str
    intro: str
    kind: str
    placeholder: str
    sections: list[Section]
    counterpart: str | None = None
    modified: str = TOOLS_LAUNCHED
    guides: list[str] = field(default_factory=list)


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
            "Guides pratiques pour vérifier un document : facture, RIB, Kbis, "
            "attestation Urssaf, justificatif de domicile, bulletin de salaire, avis "
            "d'imposition, PDF modifié."
        ),
        "audience_headings": {
            "individuals": "Particuliers et dossiers de location",
            "business": "Entreprises et fournisseurs",
            "method": "Méthode et technique",
        },
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
        "updated": "Mis à jour le {date}",
        # Crawler landing page (seo-fr.html). Metadata only — the visible body
        # comes from src/translations/fr.json, see render_landing().
        "landing_path": "/",
        "landing_title": "TrustyFile - Vérificateur de documents PDF gratuit",
        "landing_description": (
            "Vérificateur de documents PDF gratuit en ligne. Déposez un PDF, obtenez "
            "un score de confiance instantané : falsifications, polices, dates, texte "
            "masqué et plus encore."
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
        # Free checker tools (/outils/<slug>)
        "tools_path": "/outils",
        "tools_heading": "Outils gratuits",
        "tools_intro": (
            "Un numéro à vérifier plutôt qu'un document entier ? Ces outils "
            "contrôlent la clé d'un IBAN, d'un SIRET, d'un SIREN ou d'un numéro de "
            "TVA directement dans votre navigateur."
        ),
        "tool_button": "Vérifier",
        "tool_privacy": (
            "Le calcul se fait dans votre navigateur : le numéro saisi n'est envoyé "
            "à aucun serveur."
        ),
        "related_tools": "Autres outils",
        "open_tool": "Ouvrir l'outil",
        "tool_i18n": {
            "format_iban": "Format inattendu : un IBAN commence par deux lettres (pays) et deux chiffres (clé), suivis de 11 à 30 caractères.",
            "valid_iban": "Clé de contrôle valide (mod 97)",
            "invalid_iban": "Clé de contrôle invalide : ce numéro n'est pas un IBAN correctement formé — chiffre erroné, caractère manquant ou numéro inventé.",
            "country": "Pays : {country}",
            "length_ok": "Longueur conforme pour {country} ({n} caractères)",
            "length_ko": "Longueur inhabituelle pour {country} : {n} caractères au lieu de {expected}",
            "length_unknown": "Longueur non vérifiée : pays hors de la table de référence",
            "rib_ok": "Clé RIB valide (IBAN français)",
            "rib_ko": "Clé RIB invalide : les deux derniers chiffres ne correspondent pas au compte",
            "format_siret": "Un SIRET compte exactement 14 chiffres",
            "valid_siret": "Clé de contrôle valide",
            "invalid_siret": "Clé de contrôle invalide : ce numéro n'est pas un SIRET correctement formé",
            "siren_part": "SIREN {siren} · NIC {nic}",
            "la_poste": "Établissement La Poste : règle spécifique appliquée (somme des chiffres multiple de 5)",
            "format_siren": "Un SIREN compte exactement 9 chiffres",
            "valid_siren": "Clé de contrôle valide",
            "invalid_siren": "Clé de contrôle invalide : ce numéro n'est pas un SIREN correctement formé",
            "vat_hint": "Numéro de TVA intracommunautaire correspondant : FR{key} {siren}",
            "format_vat": "Format inattendu : un numéro de TVA français s'écrit FR, deux chiffres de clé, puis les 9 chiffres du SIREN. Les clés comportant des lettres (rares) ne sont pas vérifiées par cet outil.",
            "valid_vat": "Clé de contrôle valide : la clé correspond au SIREN",
            "invalid_vat": "Clé de contrôle invalide : ce numéro de TVA n'est pas cohérent",
            "vat_parts": "Clé {key} · SIREN {siren}",
            "vat_siren_ko": "Le SIREN contenu dans le numéro échoue lui-même à la formule de Luhn : le numéro est mal recopié ou inventé",
            "vat_key_expected": "Le SIREN est bien formé mais sa clé de TVA devrait être {expected}",
            "reminder": "Une clé valide prouve seulement que le numéro est bien formé — pas que le compte ou l'entreprise existe, ni qu'il appartient à l'émetteur du document.",
        },
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
            "Practical guides to verifying a document: invoices, bank details, Kbis, "
            "URSSAF certificates, proof of address, pay slips, French tax notices, "
            "modified PDFs. Method and limits."
        ),
        "audience_headings": {
            "individuals": "Individuals and rental files",
            "business": "Businesses and suppliers",
            "method": "Method and technique",
        },
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
        "updated": "Updated {date}",
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
        "tools_path": "/en/tools",
        "tools_heading": "Free tools",
        "tools_intro": (
            "A single number to check rather than a whole document? These tools "
            "validate the check digits of an IBAN, a SIRET, a SIREN or a French VAT "
            "number right in your browser."
        ),
        "tool_button": "Check",
        "tool_privacy": (
            "The calculation runs in your browser: the number you type is not sent "
            "to any server."
        ),
        "related_tools": "Other tools",
        "open_tool": "Open the tool",
        "tool_i18n": {
            "format_iban": "Unexpected format: an IBAN starts with two letters (country) and two digits (check), followed by 11 to 30 characters.",
            "valid_iban": "Check digits valid (mod 97)",
            "invalid_iban": "Check digits invalid: this is not a well-formed IBAN — a wrong digit, a missing character or a made-up number.",
            "country": "Country: {country}",
            "length_ok": "Length matches {country} ({n} characters)",
            "length_ko": "Unusual length for {country}: {n} characters instead of {expected}",
            "length_unknown": "Length not checked: country outside the reference table",
            "rib_ok": "RIB key valid (French IBAN)",
            "rib_ko": "RIB key invalid: the last two digits do not match the account",
            "format_siret": "A SIRET has exactly 14 digits",
            "valid_siret": "Check digit valid",
            "invalid_siret": "Check digit invalid: this is not a well-formed SIRET",
            "siren_part": "SIREN {siren} · NIC {nic}",
            "la_poste": "La Poste establishment: specific rule applied (digit sum is a multiple of 5)",
            "format_siren": "A SIREN has exactly 9 digits",
            "valid_siren": "Check digit valid",
            "invalid_siren": "Check digit invalid: this is not a well-formed SIREN",
            "vat_hint": "Matching intra-EU VAT number: FR{key} {siren}",
            "format_vat": "Unexpected format: a French VAT number is FR, a two-digit key, then the 9 digits of the SIREN. Keys containing letters (rare) are not checked by this tool.",
            "valid_vat": "Check key valid: the key matches the SIREN",
            "invalid_vat": "Check key invalid: this VAT number is not self-consistent",
            "vat_parts": "Key {key} · SIREN {siren}",
            "vat_siren_ko": "The SIREN inside the number fails the Luhn formula itself: the number was miscopied or made up",
            "vat_key_expected": "The SIREN is well-formed but its VAT key should be {expected}",
            "reminder": "A valid check digit only proves the number is well-formed — not that the account or company exists, nor that it belongs to whoever issued the document.",
        },
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
            audience="business",
            counterpart="how-to-spot-a-tampered-invoice",
            tools=["verifier-un-siret", "verifier-un-numero-de-tva", "verifier-un-siren"],
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
            audience="individuals",
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
            audience="method",
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
            audience="method",
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
            audience="individuals",
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
            audience="business",
            counterpart="how-to-spot-a-fake-iban",
            tools=["verifier-un-iban"],
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
                        "Depuis le 9 octobre 2025, les banques de la zone euro doivent "
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
        Guide(
            slug="fraude-au-faux-fournisseur-changement-de-rib",
            audience="business",
            counterpart="supplier-bank-detail-change-fraud",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["verifier-un-iban", "verifier-un-siret"],
            title="Fraude au faux fournisseur : vérifier un changement de RIB",
            description=(
                "Un fournisseur annonce de nouvelles coordonnées bancaires ? Le "
                "scénario de la fraude, les signaux d'alerte dans le message et le "
                "RIB, et la procédure à suivre avant de payer."
            ),
            heading="Fraude au faux fournisseur : que faire face à un changement de RIB",
            intro=(
                "La fraude au faux fournisseur consiste à se faire passer pour un "
                "fournisseur habituel afin de détourner ses paiements : un message "
                "annonce un « changement de coordonnées bancaires », la comptabilité "
                "met la fiche à jour, et le règlement suivant part sur le compte de "
                "l'escroc. Le vrai fournisseur ne s'en aperçoit qu'en relançant sa "
                "facture impayée, souvent des semaines plus tard. Ce guide s'adresse à "
                "ceux qui traitent les paiements : reconnaître l'attaque, contrôler le "
                "document reçu, et surtout appliquer la procédure qui la neutralise."
            ),
            sections=[
                Section(
                    heading="Comment se déroule l'attaque",
                    paragraphs=[
                        "Le fraudeur commence par se renseigner : qui sont vos "
                        "fournisseurs, qui traite les factures chez vous, quand tombent "
                        "les échéances. Sites web, appels d'offres, réseaux "
                        "professionnels et, de plus en plus souvent, une boîte mail "
                        "compromise chez le fournisseur lui-même fournissent tout cela.",
                        "Vient ensuite l'usurpation. Dans la version la plus simple, le "
                        "message part d'un domaine sosie — une lettre doublée, un "
                        "<em>.co</em> à la place d'un <em>.com</em>, un tiret ajouté. "
                        "Dans la version la plus dangereuse, il part de la vraie boîte "
                        "mail du fournisseur, piratée, en réponse à un échange réel : "
                        "l'expéditeur, le fil, la signature, tout est authentique. Seul "
                        "le RIB ne l'est pas.",
                        "Le déclencheur est un prétexte banal : changement de banque, "
                        "fusion, réorganisation comptable, « suite à un contrôle ». Un "
                        "RIB est joint, parfois une facture. Puis vient la pression : "
                        "un rappel téléphonique, une échéance proche, une remise si le "
                        "paiement est rapide. La <a href=\"/guides/detecter-un-faux-rib\">"
                        "fraude au faux RIB</a> est le volet documentaire de ce scénario ; "
                        "la « fraude au président », où l'ordre de virement urgent vient "
                        "d'un prétendu dirigeant, en est la cousine.",
                    ],
                ),
                Section(
                    heading="Les signaux d'alerte dans le message",
                    paragraphs=[
                        "Aucun de ces indices ne prouve la fraude ; un seul suffit à "
                        "déclencher la vérification décrite plus bas.",
                    ],
                    bullets=[
                        "L'<strong>adresse d'expédition</strong> diffère d'une lettre du "
                        "domaine habituel, ou l'adresse de réponse (<em>Reply-To</em>) "
                        "n'est pas celle de l'expéditeur affiché.",
                        "Un <strong>nouvel interlocuteur</strong> apparaît, ou l'on vous "
                        "demande de ne plus passer par le contact habituel, « en congé » "
                        "ou « parti ».",
                        "<strong>Urgence et confidentialité</strong> : régler avant telle "
                        "date, ne pas en parler, traiter « en direct ».",
                        "La nouvelle banque est dans un <strong>autre pays</strong> que "
                        "le fournisseur, ou c'est un établissement de paiement en ligne "
                        "pour une entreprise qui, jusque-là, était dans une banque de "
                        "réseau.",
                        "Le fournisseur <strong>évite le téléphone</strong> ou ne rappelle "
                        "que depuis un numéro inconnu.",
                        "Le message arrive <strong>juste après l'émission d'une facture "
                        "importante</strong> ou juste avant son échéance.",
                    ],
                ),
                Section(
                    heading="Ce que révèle le RIB joint",
                    paragraphs=[
                        "Le document est le seul élément tangible que vous tenez. Trois "
                        "contrôles se font en quelques minutes :",
                    ],
                    bullets=[
                        "La <strong>clé de l'IBAN</strong> (et la clé RIB pour un compte "
                        "français) : une clé fausse condamne le document. Une clé valide "
                        "ne l'innocente pas — les fraudeurs ouvrent de vrais comptes.",
                        "La <strong>cohérence bancaire</strong> : le code établissement "
                        "dans l'IBAN, le BIC et le nom de banque imprimé doivent désigner "
                        "le même établissement ; un logo de grande banque sur un IBAN "
                        "d'établissement de paiement se contredit.",
                        "Les <strong>traces de retouche</strong> : changement de police "
                        "sur la ligne de l'IBAN, rectangle blanc recouvrant l'ancien "
                        "numéro, révision du PDF postérieure à sa création, métadonnées "
                        "d'un éditeur PDF sur un document censé sortir de la banque. Le "
                        "guide <a href=\"/guides/verifier-une-facture-falsifiee\">sur "
                        "les factures falsifiées</a> détaille chacun de ces indices.",
                    ],
                ),
                Section(
                    heading="La procédure qui protège vraiment : le rappel",
                    paragraphs=[
                        "Aucun contrôle du document ne peut établir à qui appartient le "
                        "compte. Une seule chose le peut : demander au fournisseur, par "
                        "un canal que le fraudeur ne contrôle pas.",
                        "Rappelez à un <strong>numéro que vous possédiez déjà</strong> — "
                        "contrat, ancienne facture, fiche fournisseur, site officiel — "
                        "jamais celui du message ni celui imprimé sur le RIB reçu. "
                        "Parlez à une personne que vous connaissez, ou à défaut au "
                        "service comptable, et faites-lui lire l'IBAN. Si la boîte mail "
                        "du fournisseur est compromise, c'est aussi ainsi qu'il "
                        "l'apprendra.",
                        "Inscrivez ce rappel dans une <strong>procédure</strong> : tout "
                        "changement de coordonnées bancaires est validé par deux "
                        "personnes, après confirmation téléphonique tracée, et le premier "
                        "paiement vers un nouveau compte attend ce feu vert quelle que "
                        "soit l'urgence invoquée. L'urgence est l'outil du fraudeur ; la "
                        "procédure est le vôtre.",
                        "Depuis le 9 octobre 2025, les banques de la zone euro vérifient la "
                        "concordance entre le nom du bénéficiaire saisi et le titulaire "
                        "de l'IBAN avant d'exécuter un virement. C'est un filet "
                        "supplémentaire, pas une raison de sauter le rappel : le compte "
                        "frauduleux est souvent ouvert sous un nom proche de celui du "
                        "fournisseur, ou au nom d'une société écran.",
                    ],
                ),
                Section(
                    heading="Si le virement est déjà parti",
                    paragraphs=[
                        "Chaque heure compte : les fonds sont généralement retirés ou "
                        "renvoyés à l'étranger en quelques heures ou quelques jours.",
                    ],
                    bullets=[
                        "<strong>Appelez votre banque</strong> immédiatement et demandez "
                        "un rappel de fonds (<em>recall</em>) ; les chances baissent "
                        "vite mais ne sont pas nulles tant que le compte destinataire "
                        "n'a pas été vidé.",
                        "<strong>Prévenez le vrai fournisseur</strong> : si sa messagerie "
                        "est compromise, il doit la sécuriser et alerter ses autres "
                        "clients.",
                        "<strong>Déposez plainte</strong> et conservez tout : e-mails "
                        "avec leurs en-têtes complets, RIB reçu, ordres de virement, "
                        "relevés d'appels.",
                        "<strong>Déclarez le sinistre</strong> à votre assureur si vous "
                        "avez une garantie fraude ou cyber, dans les délais du contrat.",
                        "<strong>Signalez</strong> l'attaque sur cybermalveillance.gouv.fr "
                        "et revoyez la procédure interne qui a laissé passer le "
                        "changement.",
                    ],
                ),
                Section(
                    heading="Ce qu'un contrôle automatique peut et ne peut pas faire",
                    paragraphs=[
                        "TrustyFile vérifie le document : clé de l'IBAN, cohérence des "
                        "identifiants d'entreprise (SIRET, TVA), traces de retouche, "
                        "révisions, texte recouvert. Il repère le RIB fabriqué ou modifié, "
                        "cas le plus fréquent. Il ne peut pas savoir à qui appartient un "
                        "compte parfaitement réel : ce dernier maillon reste le coup de "
                        "téléphone — et il n'est pas remplaçable.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="verifier-un-extrait-kbis",
            audience="business",
            counterpart="how-to-verify-a-kbis-extract",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["verifier-un-siret", "verifier-un-siren", "verifier-un-numero-de-tva"],
            title="Vérifier un extrait Kbis : authenticité et cohérence",
            description=(
                "Confronter un Kbis au registre (SIREN, dirigeants, siège, état de "
                "la société), repérer les indices d'un faux dans le PDF, et savoir ce "
                "qu'un Kbis authentique ne prouve pas."
            ),
            heading="Comment vérifier un extrait Kbis",
            intro=(
                "L'extrait Kbis est la « carte d'identité » d'une société "
                "commerciale : il atteste de son immatriculation au registre du "
                "commerce et des sociétés (RCS) et résume ses informations légales. "
                "On l'exige d'un nouveau fournisseur, d'un sous-traitant, d'un "
                "locataire professionnel, d'un client en compte. C'est aussi l'un des "
                "documents les plus faciles à falsifier : un PDF, un modèle que tout "
                "le monde connaît, et une lecture souvent rapide. La bonne nouvelle : "
                "tout ce qu'il contient se vérifie dans des registres publics, "
                "gratuitement."
            ),
            sections=[
                Section(
                    heading="Ce que contient un Kbis, et ce qu'il n'est pas",
                    paragraphs=[
                        "Un extrait Kbis mentionne la dénomination, la forme juridique, "
                        "le capital, l'adresse du siège, le numéro d'identification "
                        "(SIREN) et le greffe d'immatriculation (« RCS » suivi de la "
                        "ville), le code APE, la date d'immatriculation, la durée de la "
                        "société, son activité, l'identité et la date de naissance des "
                        "dirigeants, et le cas échéant les procédures collectives en "
                        "cours (sauvegarde, redressement, liquidation). Il est délivré "
                        "par le greffe du tribunal de commerce, en ligne via Infogreffe, "
                        "et daté du jour de sa délivrance : c'est une photographie, "
                        "d'où l'usage d'exiger un extrait de moins de trois mois.",
                        "Deux confusions fréquentes. L'« extrait K » est l'équivalent "
                        "pour un commerçant personne physique. Et un micro-entrepreneur, "
                        "un artisan ou un professionnel libéral n'a <em>pas</em> de "
                        "Kbis : il justifie de son existence par un avis de situation au "
                        "répertoire SIRENE (INSEE) ou un extrait d'immatriculation au "
                        "Registre national des entreprises (INPI). Un « Kbis » présenté "
                        "par un auto-entrepreneur est, à lui seul, un signal d'alerte.",
                    ],
                ),
                Section(
                    heading="Confronter le document au registre",
                    paragraphs=[
                        "C'est le contrôle qui compte, et il est gratuit. L'<a "
                        "href=\"https://annuaire-entreprises.data.gouv.fr\">Annuaire des "
                        "entreprises</a> (service public) agrège les données de l'INSEE, "
                        "de l'INPI et des greffes : état administratif (active ou "
                        "cessée), siège, dirigeants, établissements, date de création. "
                        "Infogreffe donne la fiche du RCS et délivre l'extrait officiel ; "
                        "le BODACC publie les annonces légales — immatriculation, "
                        "modifications, procédures collectives, radiation.",
                        "Reprenez le Kbis ligne à ligne et comparez : même SIREN, même "
                        "dénomination, même forme, même siège, mêmes dirigeants, même "
                        "date d'immatriculation, même greffe. Un dirigeant absent du "
                        "registre, une adresse différente, une société « active » sur le "
                        "papier mais radiée ou en liquidation dans le registre : chacun "
                        "de ces écarts vaut refus jusqu'à explication. Une modification "
                        "toute récente peut ne pas être encore publiée — demandez alors "
                        "le récépissé du greffe, pas un nouveau Kbis.",
                    ],
                ),
                Section(
                    heading="Les indices d'un faux dans le document lui-même",
                    paragraphs=[
                        "Avant même d'ouvrir le registre, le fichier peut se trahir :",
                    ],
                    bullets=[
                        "Un <strong>SIREN dont la clé est fausse</strong> : les neuf "
                        "chiffres obéissent à la formule de Luhn ; un numéro inventé ou "
                        "modifié échoue presque toujours à ce test.",
                        "Un <strong>greffe incohérent avec le siège</strong> : une "
                        "société immatriculée « RCS Paris » dont le siège est à Lyon. "
                        "Les transferts de siège existent, mais le décalage doit "
                        "s'expliquer.",
                        "Des <strong>dates impossibles</strong> : immatriculation "
                        "postérieure à la date de délivrance, dirigeant né après la "
                        "création, extrait « de moins de trois mois » daté de l'an "
                        "dernier.",
                        "Une <strong>mise en page approximative</strong> : police qui "
                        "change au milieu d'une ligne, alignements irréguliers, "
                        "rectangle blanc sous un champ. Le guide "
                        "<a href=\"/guides/savoir-si-un-pdf-a-ete-modifie\">sur les PDF "
                        "modifiés</a> explique comment les révisions et les objets "
                        "fantômes trahissent la retouche.",
                        "Des <strong>métadonnées d'éditeur PDF</strong> ou de "
                        "convertisseur en ligne sur un document censé sortir de la chaîne "
                        "du greffe.",
                    ],
                ),
                Section(
                    heading="Ce qu'un Kbis authentique ne prouve pas",
                    paragraphs=[
                        "Un Kbis authentique prouve qu'une société existe. Il ne prouve "
                        "pas que la personne qui vous l'envoie la représente. Les "
                        "usurpations d'identité de sociétés réelles, les sociétés "
                        "dormantes rachetées pour servir de façade et les « sociétés "
                        "éphémères » créées pour quelques mois d'escroquerie présentent "
                        "toutes un Kbis parfaitement valide.",
                        "Vérifiez donc aussi que votre interlocuteur figure parmi les "
                        "dirigeants — ou dispose d'un mandat —, que le domaine de son "
                        "adresse e-mail est bien celui de l'entreprise, et que la "
                        "société n'est ni radiée ni en procédure collective. Une "
                        "immatriculation de quelques mois, un capital symbolique et une "
                        "première commande importante forment un ensemble qui justifie "
                        "des garanties supplémentaires.",
                    ],
                ),
                Section(
                    heading="Vérifier l'ensemble du dossier",
                    paragraphs=[
                        "Le Kbis n'arrive jamais seul. Le RIB doit être au nom exact de "
                        "la dénomination ; la facture doit porter un SIRET dont les neuf "
                        "premiers chiffres sont le SIREN du Kbis et un numéro de TVA "
                        "intracommunautaire cohérent avec lui (FR, une clé à deux "
                        "chiffres, puis le SIREN) ; pour la sous-traitance, "
                        "l'attestation de vigilance URSSAF complète l'obligation de "
                        "vigilance. Toute incohérence entre ces pièces pèse plus lourd "
                        "que la qualité de chacune prise isolément.",
                        "TrustyFile fait ces recoupements sur les PDF que vous déposez : "
                        "clés SIREN, SIRET et TVA, existence de l'entreprise auprès de "
                        "l'INSEE, et traces de retouche dans le fichier. La confrontation "
                        "au registre, elle, reste un geste que rien ne remplace.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="verifier-un-avis-d-imposition",
            audience="individuals",
            counterpart="how-to-verify-a-french-tax-notice",
            published="2026-09-03",
            modified="2026-09-03",
            title="Vérifier un avis d'imposition : SVAIR, 2D-Doc et cohérence",
            description=(
                "Propriétaire ou agence : vérifier l'avis d'imposition d'un candidat "
                "avec le service officiel des impôts (SVAIR), le code 2D-Doc signé et "
                "les autres pièces du dossier."
            ),
            heading="Comment vérifier un avis d'imposition",
            intro=(
                "L'avis d'impôt sur le revenu est la pièce maîtresse d'un dossier de "
                "location et de nombreuses demandes d'aide ou de crédit : il fixe le "
                "revenu fiscal de référence du foyer. C'est aussi le justificatif le "
                "plus retouché des dossiers locatifs, parce qu'un revenu gonflé ouvre "
                "des portes que le vrai revenu ferme. Il se vérifie pourtant mieux que "
                "presque tout autre document : l'administration fiscale met à "
                "disposition un service de contrôle en ligne, et l'avis porte un code "
                "2D-Doc signé."
            ),
            sections=[
                Section(
                    heading="Ce que dit un avis, et ce qui se falsifie",
                    paragraphs=[
                        "L'avis indique le numéro fiscal du déclarant (13 chiffres), la "
                        "référence de l'avis (13 chiffres), le revenu fiscal de "
                        "référence, le nombre de parts, le montant de l'impôt, "
                        "l'adresse et la situation de famille. Il porte deux années : "
                        "celle des revenus déclarés et celle de l'émission, qui la suit. "
                        "Juste après la déclaration, l'« avis de situation déclarative » "
                        "(ASDIR) en tient lieu, avec les mêmes identifiants.",
                        "Les fraudes suivent quatre schémas : un revenu fiscal de "
                        "référence gonflé sur un avis authentique ; un vieil avis dont on "
                        "a changé l'année ; l'avis d'une autre personne dont on a "
                        "remplacé le nom et l'adresse ; un document entièrement fabriqué "
                        "à partir d'un modèle. Les deux premiers sont les plus courants, "
                        "et les deux contrôles ci-dessous, pris ensemble, les attrapent "
                        "tous : le premier établit que l'avis existe et qu'il est le "
                        "dernier émis, le second que ses chiffres n'ont pas été touchés.",
                    ],
                ),
                Section(
                    heading="Le service de vérification des avis (SVAIR)",
                    paragraphs=[
                        "La Direction générale des Finances publiques propose un "
                        "<a href=\"https://www.impots.gouv.fr/verifavis2-api/front\">"
                        "service de vérification des avis</a> (SVAIR), accessible sans "
                        "compte depuis impots.gouv.fr, rubrique « Vérifier un avis "
                        "d'impôt », et destiné précisément aux tiers à qui l'avis est "
                        "présenté : bailleurs, agences, administrations. Saisissez les "
                        "deux identifiants imprimés sur le document — le numéro fiscal "
                        "(13 chiffres) et la référence de l'avis (13 chiffres) — et le "
                        "service répond d'un mot : <em>conforme</em> si ce couple "
                        "correspond au dernier avis connu de l'administration pour cet "
                        "usager, <em>non conforme</em> sinon. Il n'affiche aucun "
                        "montant : la confidentialité des données fiscales l'interdit.",
                        "Lisez ce verdict pour ce qu'il est. « Non conforme » signifie "
                        "que le couple est inconnu — document fabriqué, identifiant "
                        "retouché ou mal saisi — ou qu'un avis plus récent a été émis "
                        "depuis : le candidat montre un ancien avis, peut-être remplacé "
                        "par une déclaration rectificative. « Conforme » prouve qu'un "
                        "avis authentique porte bien ces identifiants et qu'il est le "
                        "dernier en date ; il ne prouve pas que les chiffres imprimés "
                        "sont ceux d'origine, puisque le service ne les compare pas. Un "
                        "revenu gonflé sur un avis authentique et à jour passe ce "
                        "contrôle : c'est le code 2D-Doc qui l'attrape.",
                    ],
                ),
                Section(
                    heading="Le code 2D-Doc",
                    paragraphs=[
                        "Les avis portent un code 2D-Doc : un DataMatrix signé "
                        "électroniquement par l'administration fiscale, qui reprend les "
                        "données clés du document (revenu fiscal de référence, nombre de "
                        "parts, référence de l'avis, numéro fiscal, année). Un lecteur "
                        "2D-Doc le décode et vérifie la signature ; TrustyFile le fait "
                        "pour tout PDF déposé, hors ligne, avec les certificats publiés "
                        "par l'autorité.",
                        "Si les données décodées diffèrent de celles imprimées, le "
                        "texte visible a été retouché et le code, lui, dit la vérité. Si "
                        "la signature est invalide, le code a été recomposé — ce qu'un "
                        "faussaire ne peut pas faire correctement sans la clé privée de "
                        "l'administration. Un code illisible (scan flou, photo) n'est en "
                        "revanche pas une preuve : il vous laisse avec le contrôle SVAIR "
                        "— qui atteste l'existence de l'avis, pas ses chiffres — et les "
                        "recoupements ci-dessous. Le guide <a href=\"/guides/2d-doc-verifier-un-"
                        "document-officiel\">sur le 2D-Doc</a> détaille le mécanisme.",
                    ],
                ),
                Section(
                    heading="Les recoupements avec le reste du dossier",
                    paragraphs=[
                        "Un avis cohérent avec lui-même peut ne pas l'être avec le "
                        "dossier :",
                    ],
                    bullets=[
                        "<strong>Bulletins de salaire</strong> : pour un salarié sans "
                        "autre revenu, le revenu fiscal de référence est proche du cumul "
                        "annuel « net imposable » des bulletins, diminué de l'abattement "
                        "de 10 % pour frais professionnels ; certains revenus exonérés y "
                        "sont réintégrés et peuvent creuser l'écart. Un écart de "
                        "plusieurs milliers d'euros "
                        "demande une explication (autres revenus, changement de "
                        "situation). Voir le guide <a href=\"/guides/verifier-un-"
                        "bulletin-de-salaire\">sur les bulletins de salaire</a>.",
                        "<strong>Adresse</strong> : celle de l'avis doit correspondre au "
                        "<a href=\"/guides/verifier-un-justificatif-de-domicile\">"
                        "justificatif de domicile</a> et à l'historique déclaré.",
                        "<strong>Nombre de parts</strong> : cohérent avec la situation "
                        "de famille annoncée (une part pour une personne seule, deux pour "
                        "un couple, puis les demi-parts pour les enfants à charge).",
                        "<strong>Années</strong> : un avis émis l'année N porte sur les "
                        "revenus de N-1 ; un « dernier avis » de deux ans n'est pas le "
                        "dernier.",
                        "<strong>Orthographe des noms</strong> et civilité identiques sur "
                        "toutes les pièces.",
                    ],
                ),
                Section(
                    heading="Les indices dans le fichier PDF",
                    paragraphs=[
                        "Un avis téléchargé depuis l'espace particulier d'impots.gouv.fr "
                        "est un PDF numérique natif, produit d'un bloc par la chaîne de "
                        "l'administration. Un avis modifié dans un éditeur PDF garde des "
                        "traces : une révision ajoutée après la création, une police "
                        "légèrement différente sur le montant retouché, l'ancien chiffre "
                        "encore présent sous un rectangle blanc, des métadonnées "
                        "d'éditeur ou de convertisseur en ligne. Le guide "
                        "<a href=\"/guides/savoir-si-un-pdf-a-ete-modifie\">sur les PDF "
                        "modifiés</a> passe ces indices en revue.",
                        "Un avis scanné ou photographié perd tout cela : ni structure "
                        "PDF exploitable, ni code 2D-Doc souvent lisible. Ce n'est pas "
                        "un signe de fraude — beaucoup de gens n'ont qu'une version "
                        "papier — mais c'est un document non vérifiable par lui-même. "
                        "Demandez le PDF d'origine : lui seul permet de lire le code "
                        "2D-Doc et de contrôler les montants.",
                    ],
                ),
                Section(
                    heading="Cadre légal et bons réflexes",
                    paragraphs=[
                        "Un bailleur peut demander à un candidat son dernier ou "
                        "avant-dernier avis d'imposition ou de non-imposition : la "
                        "pièce figure dans la liste limitative fixée par le décret du 5 "
                        "novembre 2015, qui interdit d'exiger d'autres documents que ceux "
                        "qu'il énumère. Présenter un avis falsifié relève du faux et de "
                        "l'usage de faux (article 441-1 du code pénal, jusqu'à trois ans "
                        "d'emprisonnement et 45 000 euros d'amende).",
                        "Face à un avis douteux : ne concluez pas, conservez le document "
                        "et les échanges, et gardez à l'esprit que les pièces d'un "
                        "candidat non retenu ne doivent pas être conservées au-delà du "
                        "nécessaire. TrustyFile n'archive rien : le fichier est analysé, "
                        "puis supprimé.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="verifier-une-attestation-de-vigilance-urssaf",
            audience="business",
            counterpart="how-to-verify-an-urssaf-vigilance-certificate",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["verifier-un-siret", "verifier-un-siren"],
            title="Vérifier une attestation de vigilance Urssaf",
            description=(
                "Vérifier l'attestation de vigilance d'un sous-traitant : code de "
                "sécurité à 15 caractères sur urssaf.fr, validité de six mois, et ce "
                "que la loi impose au donneur d'ordre."
            ),
            heading="Comment vérifier une attestation de vigilance Urssaf",
            intro=(
                "L'attestation de vigilance est le document par lequel l'Urssaf "
                "certifie qu'une entreprise déclare et paie ses cotisations sociales. "
                "Tout donneur d'ordre doit l'exiger de ses sous-traitants au-delà d'un "
                "certain montant, et la faire renouveler : c'est sa protection contre "
                "la solidarité financière en cas de travail dissimulé. C'est aussi, "
                "pour cette raison même, un document que l'on falsifie — et l'un des "
                "plus simples à vérifier, grâce au code de sécurité que l'Urssaf "
                "imprime dessus."
            ),
            sections=[
                Section(
                    heading="Ce que l'attestation certifie, et ce qu'elle contient",
                    paragraphs=[
                        "L'Urssaf délivre l'attestation de vigilance à une entreprise ou "
                        "à un travailleur indépendant qui a payé ses cotisations et "
                        "contributions aux échéances, ou qui a souscrit un plan "
                        "d'apurement qu'il respecte, ou qui en conteste le montant dans "
                        "les formes. Elle se télécharge depuis l'espace en ligne de "
                        "l'entreprise — urssaf.fr, ou autoentrepreneur.urssaf.fr pour "
                        "un micro-entrepreneur — et vaut six mois.",
                        "Le document identifie l'entreprise (dénomination, SIRET, "
                        "adresse), indique sa date de délivrance et sa période de "
                        "validité et, pour un employeur, le nombre de salariés et le "
                        "montant total des rémunérations déclarées. Il porte enfin un "
                        "code de sécurité de quinze caractères : c'est lui qui rend la "
                        "vérification possible.",
                    ],
                ),
                Section(
                    heading="Ce que la loi impose au donneur d'ordre",
                    paragraphs=[
                        "Pour tout contrat d'au moins 5 000 € hors taxes, le donneur "
                        "d'ordre doit se faire remettre l'attestation de vigilance à la "
                        "signature, puis tous les six mois jusqu'à la fin de l'exécution "
                        "(code du travail, articles L8222-1 et suivants, D8222-5). S'y "
                        "ajoute un justificatif d'immatriculation : extrait K ou Kbis, "
                        "justificatif d'inscription au Registre national des "
                        "entreprises, ou un devis, un document publicitaire ou un "
                        "courrier professionnel portant le nom, l'adresse et le numéro "
                        "d'immatriculation.",
                        "L'enjeu n'est pas administratif. Si le sous-traitant est "
                        "condamné pour travail dissimulé et que ces vérifications n'ont "
                        "pas été faites, le donneur d'ordre est tenu solidairement au "
                        "paiement de ses impôts, taxes, cotisations et rémunérations "
                        "dues. Une attestation fausse acceptée sans contrôle ne protège "
                        "de rien.",
                    ],
                ),
                Section(
                    heading="Le code de sécurité : le contrôle qui compte",
                    paragraphs=[
                        "L'Urssaf met à disposition un outil public de <a href=\""
                        "https://www.urssaf.fr/accueil/outils-documentation/outils/"
                        "verification-attestation.html\">vérification des "
                        "attestations</a>. Saisissez le code de sécurité tel qu'il "
                        "figure sur le document — quinze caractères, sans espace. Si "
                        "l'Urssaf ne reconnaît pas le code, elle répond qu'il ne "
                        "correspond à aucune attestation en cours de validité : le code "
                        "est mal recopié, l'attestation est périmée, ou elle a été "
                        "modifiée ou fabriquée.",
                        "Si le code est reconnu, l'Urssaf affiche les éléments de "
                        "l'attestation qu'elle a réellement émise : le nom de l'entreprise "
                        "(« identification du diffuseur »), le SIRET de chaque "
                        "établissement, la date de certification du document et la "
                        "dernière période analysée. Comparez-les ligne à ligne au document "
                        "reçu : un fraudeur peut coller le code authentique d'une autre "
                        "entreprise sur une attestation fabriquée à son nom — le code "
                        "passe, la comparaison ne passe pas.",
                    ],
                ),
                Section(
                    heading="Les indices d'un faux dans le document lui-même",
                    bullets=[
                        "Une <strong>période de validité</strong> qui n'est pas de six "
                        "mois, une date de délivrance postérieure à celle du courrier qui "
                        "l'accompagne, ou une attestation « à jour » vieille de huit "
                        "mois.",
                        "Un <strong>effectif incohérent</strong> : le sous-traitant "
                        "annonce trente personnes sur le chantier, l'attestation déclare "
                        "deux salariés.",
                        "Un <strong>SIRET dont la clé est fausse</strong>, ou qui ne "
                        "correspond pas au Kbis et aux factures.",
                        "Un <strong>code de sécurité</strong> qui n'a pas quinze "
                        "caractères, contient des espaces, ou se retrouve à l'identique "
                        "sur deux attestations de dates différentes.",
                        "Des <strong>traces de retouche PDF</strong> : police différente "
                        "sur une date ou un chiffre, rectangle blanc, révision ajoutée "
                        "après la création, métadonnées d'éditeur — voir <a href=\""
                        "/guides/savoir-si-un-pdf-a-ete-modifie\">le guide sur les PDF "
                        "modifiés</a>.",
                    ],
                ),
                Section(
                    heading="Recouper avec le reste du dossier",
                    paragraphs=[
                        "L'attestation n'arrive jamais seule. Le SIRET doit être celui "
                        "de <a href=\"/guides/verifier-un-extrait-kbis\">l'extrait "
                        "Kbis</a> et des factures ; le RIB doit être au nom de la même "
                        "dénomination — la <a href=\"/guides/fraude-au-faux-fournisseur-"
                        "changement-de-rib\">fraude au faux fournisseur</a> commence "
                        "souvent par un dossier de référencement impeccable. Et ne "
                        "confondez pas l'attestation de vigilance avec l'attestation de "
                        "régularité fiscale, délivrée par l'administration fiscale : "
                        "deux documents, deux émetteurs, deux vérifications.",
                    ],
                ),
                Section(
                    heading="Ce qu'une attestation valide ne prouve pas",
                    paragraphs=[
                        "Qu'une entreprise soit à jour de ses cotisations aujourd'hui ne "
                        "dit rien de sa solvabilité demain, ni de l'identité de la "
                        "personne qui vous écrit en son nom. L'attestation se périme en "
                        "six mois : inscrivez le renouvellement dans le suivi du contrat "
                        "plutôt que de compter sur le sous-traitant pour y penser. "
                        "TrustyFile examine le fichier — révisions, retouches, clés "
                        "SIRET et TVA ; le code de sécurité, lui, se vérifie sur "
                        "urssaf.fr, et rien ne remplace ce geste.",
                    ],
                ),
            ],
        ),
    ],
    "en": [
        Guide(
            slug="how-to-spot-a-tampered-invoice",
            audience="business",
            counterpart="verifier-une-facture-falsifiee",
            tools=["siret-checker", "vat-number-checker", "siren-checker"],
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
            audience="individuals",
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
            audience="method",
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
            audience="method",
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
            audience="individuals",
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
            audience="business",
            counterpart="detecter-un-faux-rib",
            tools=["iban-checker"],
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
                        "Since 9 October 2025, euro-area banks must check that the "
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
        Guide(
            slug="supplier-bank-detail-change-fraud",
            audience="business",
            counterpart="fraude-au-faux-fournisseur-changement-de-rib",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["iban-checker", "siret-checker"],
            title="Supplier bank-detail change fraud: verifying a new IBAN",
            description=(
                "A supplier announces new bank details? How the fraud works, the "
                "warning signs in the message and in the attached document, and the "
                "checks to run before you pay."
            ),
            heading="Supplier bank-detail change fraud: what to do with a new IBAN",
            intro=(
                "Supplier impersonation fraud — also called invoice redirection or "
                "payment diversion fraud — consists of posing as a regular supplier "
                "to hijack its payments: a message announces a \"change of bank "
                "details\", accounts payable updates the vendor record, and the "
                "next payment lands in the fraudster's account. The real supplier "
                "only notices when it chases the unpaid invoice, often weeks later. "
                "This guide is for whoever handles payments: recognise the attack, "
                "check the document you received, and above all follow the one "
                "procedure that defeats it."
            ),
            sections=[
                Section(
                    heading="How the attack unfolds",
                    paragraphs=[
                        "The fraudster starts with reconnaissance: who your suppliers "
                        "are, who processes invoices on your side, when payments fall "
                        "due. Websites, tender notices, professional networks and, "
                        "increasingly, a compromised mailbox at the supplier itself "
                        "provide all of it.",
                        "Then comes the impersonation. In the simplest version the "
                        "message comes from a look-alike domain — a doubled letter, "
                        "<em>.co</em> instead of <em>.com</em>, an added hyphen. In the "
                        "most dangerous version it comes from the supplier's real, "
                        "hacked mailbox, as a reply inside a genuine thread: sender, "
                        "history and signature are all authentic. Only the bank details "
                        "are not.",
                        "The trigger is a mundane pretext: a change of bank, a merger, "
                        "an accounting reorganisation, \"following an audit\". A bank "
                        "details document is attached, sometimes an invoice. Then the "
                        "pressure: a follow-up call, a close due date, a discount for "
                        "fast payment. <a href=\"/en/guides/how-to-spot-a-fake-iban\">"
                        "Fake IBAN fraud</a> is the document side of this scenario; "
                        "\"CEO fraud\", where the urgent transfer order comes from a "
                        "supposed executive, is its cousin.",
                    ],
                ),
                Section(
                    heading="Warning signs in the message",
                    paragraphs=[
                        "None of these proves fraud; any one of them is enough to "
                        "trigger the verification described below.",
                    ],
                    bullets=[
                        "The <strong>sender address</strong> differs from the usual "
                        "domain by one letter, or the reply address (<em>Reply-To</em>) "
                        "is not the displayed sender.",
                        "A <strong>new contact</strong> appears, or you are asked to stop "
                        "going through the usual person, who is \"on leave\" or \"has "
                        "left\".",
                        "<strong>Urgency and secrecy</strong>: pay before a given date, "
                        "keep it quiet, deal \"directly\".",
                        "The new bank is in a <strong>different country</strong> from the "
                        "supplier, or it is an online payment institution for a company "
                        "that had always banked with a high-street bank.",
                        "The supplier <strong>avoids the phone</strong> or only calls "
                        "back from an unknown number.",
                        "The message arrives <strong>right after a large invoice was "
                        "issued</strong> or just before it falls due.",
                    ],
                ),
                Section(
                    heading="What the attached bank-detail document reveals",
                    paragraphs=[
                        "The document is the only tangible thing you hold. Three checks "
                        "take a few minutes:",
                    ],
                    bullets=[
                        "The <strong>IBAN check digits</strong> (and the RIB key for a "
                        "French account): a wrong key condemns the document. A valid "
                        "key does not clear it — fraudsters open real accounts.",
                        "<strong>Bank consistency</strong>: the bank code inside the "
                        "IBAN, the BIC and the printed bank name must all point to the "
                        "same institution; a major bank's logo over a payment "
                        "institution's IBAN contradicts itself.",
                        "<strong>Traces of editing</strong>: a font change on the IBAN "
                        "line, a white rectangle covering the old number, a PDF "
                        "revision later than its creation, PDF-editor metadata on a "
                        "document supposedly issued by a bank. The guide "
                        "<a href=\"/en/guides/how-to-spot-a-tampered-invoice\">on "
                        "tampered invoices</a> details each of these signals.",
                    ],
                ),
                Section(
                    heading="The procedure that actually protects you: the call-back",
                    paragraphs=[
                        "No check of the document can establish who owns the account. "
                        "Only one thing can: asking the supplier, through a channel the "
                        "fraudster does not control.",
                        "Call a <strong>number you already had</strong> — the contract, "
                        "an old invoice, the vendor record, the official website — never "
                        "the one in the message or the one printed on the document you "
                        "just received. Speak to someone you know, or failing that to "
                        "the accounts department, and have them read the IBAN back. If "
                        "the supplier's mailbox is compromised, this is also how they "
                        "will find out.",
                        "Write the call-back into a <strong>procedure</strong>: every "
                        "change of bank details is approved by two people after a logged "
                        "phone confirmation, and the first payment to a new account "
                        "waits for that approval whatever urgency is claimed. Urgency is "
                        "the fraudster's tool; the procedure is yours.",
                        "Since 9 October 2025, banks in the euro area check that the payee "
                        "name entered matches the IBAN holder before executing a "
                        "transfer. That is an extra net, not a reason to skip the call: "
                        "the fraudulent account is often opened under a name close to "
                        "the supplier's, or in the name of a shell company.",
                    ],
                ),
                Section(
                    heading="If the payment has already gone out",
                    paragraphs=[
                        "Every hour counts: the funds are usually withdrawn or moved "
                        "abroad within hours or days.",
                    ],
                    bullets=[
                        "<strong>Call your bank</strong> immediately and request a "
                        "recall of the transfer; the odds fall quickly but are not zero "
                        "while the receiving account has not been emptied.",
                        "<strong>Warn the real supplier</strong>: if its mailbox is "
                        "compromised it must secure it and alert its other customers.",
                        "<strong>File a police report</strong> and keep everything: "
                        "e-mails with full headers, the bank-detail document, payment "
                        "orders, call logs.",
                        "<strong>Notify your insurer</strong> if you hold fraud or cyber "
                        "cover, within the policy's deadlines.",
                        "<strong>Review the internal process</strong> that let the "
                        "change through, and report the attack to your national "
                        "cybercrime service.",
                    ],
                ),
                Section(
                    heading="What an automated check can and cannot do",
                    paragraphs=[
                        "TrustyFile checks the document: IBAN check digits, consistency "
                        "of company identifiers (SIRET, VAT), traces of editing, "
                        "revisions, covered text. It catches the fabricated or altered "
                        "bank-detail document, which is the most common case. It cannot "
                        "know who owns a perfectly real account: that last link remains "
                        "the phone call — and nothing replaces it.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-verify-a-kbis-extract",
            audience="business",
            counterpart="verifier-un-extrait-kbis",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["siret-checker", "siren-checker", "vat-number-checker"],
            title="How to verify a Kbis extract (French company registration)",
            description=(
                "Check a Kbis against the public registers (SIREN, directors, "
                "registered office, company status), spot a forged PDF, and know "
                "what a genuine Kbis does not prove."
            ),
            heading="How to verify a Kbis extract",
            intro=(
                "The Kbis extract is the \"identity card\" of a French commercial "
                "company: it certifies its registration in the trade and companies "
                "register (RCS) and summarises its legal particulars. It is asked of "
                "any new supplier, subcontractor, business tenant or trade-credit "
                "customer. It is also one of the easiest documents to forge: a PDF, "
                "a layout everyone knows, and a reading that is often quick. The good "
                "news: everything it contains can be checked in public registers, "
                "for free."
            ),
            sections=[
                Section(
                    heading="What a Kbis contains, and what it is not",
                    paragraphs=[
                        "A Kbis states the company name, legal form, share capital, "
                        "registered office address, identification number (SIREN) and "
                        "registering court (\"RCS\" followed by a city), APE activity "
                        "code, registration date, duration, activity, the identity and "
                        "date of birth of its directors, and any ongoing insolvency "
                        "proceedings (safeguard, receivership, liquidation). It is "
                        "issued by the clerk of the commercial court, online through "
                        "Infogreffe, and dated the day it is issued: it is a snapshot, "
                        "hence the custom of demanding one less than three months old.",
                        "Two frequent confusions. The \"extrait K\" is the equivalent for "
                        "a sole trader. And a micro-entrepreneur, a craftsperson or a "
                        "liberal professional has <em>no</em> Kbis: they prove their "
                        "existence with a SIRENE status notice (INSEE) or an extract "
                        "from the National Business Register (INPI). A \"Kbis\" "
                        "presented by a micro-entrepreneur is, by itself, a warning "
                        "sign.",
                    ],
                ),
                Section(
                    heading="Check the document against the register",
                    paragraphs=[
                        "This is the check that matters, and it is free. The official "
                        "<a href=\"https://annuaire-entreprises.data.gouv.fr\">Annuaire "
                        "des entreprises</a> aggregates data from INSEE, INPI and the "
                        "court clerks: administrative status (active or ceased), "
                        "registered office, directors, establishments, creation date. "
                        "Infogreffe shows the RCS record and issues the official "
                        "extract; the BODACC publishes legal notices — registration, "
                        "amendments, insolvency proceedings, deregistration.",
                        "Go through the Kbis line by line and compare: same SIREN, same "
                        "name, same legal form, same office, same directors, same "
                        "registration date, same court. A director missing from the "
                        "register, a different address, a company \"active\" on paper "
                        "but struck off or in liquidation in the register: each of these "
                        "gaps means refusal until explained. A very recent amendment may "
                        "not be published yet — then ask for the court's filing receipt, "
                        "not for another Kbis.",
                    ],
                ),
                Section(
                    heading="Signs of a forgery in the document itself",
                    paragraphs=[
                        "Before you even open the register, the file can give itself "
                        "away:",
                    ],
                    bullets=[
                        "A <strong>SIREN with a wrong check digit</strong>: the nine "
                        "digits follow the Luhn formula; an invented or altered number "
                        "almost always fails it.",
                        "A <strong>court inconsistent with the office</strong>: a company "
                        "registered \"RCS Paris\" whose office is in Lyon. Office "
                        "transfers happen, but the gap must be explained.",
                        "<strong>Impossible dates</strong>: registration later than the "
                        "issue date, a director born after the company was created, an "
                        "extract \"less than three months old\" dated last year.",
                        "<strong>Sloppy layout</strong>: a font that changes mid-line, "
                        "uneven alignment, a white rectangle under a field. The guide "
                        "<a href=\"/en/guides/how-to-tell-if-a-pdf-was-modified\">on "
                        "modified PDFs</a> explains how revisions and ghost objects "
                        "betray editing.",
                        "<strong>PDF-editor or online-converter metadata</strong> on a "
                        "document supposedly produced by the court clerk's system.",
                    ],
                ),
                Section(
                    heading="What a genuine Kbis does not prove",
                    paragraphs=[
                        "A genuine Kbis proves that a company exists. It does not prove "
                        "that the person sending it represents that company. "
                        "Impersonation of real companies, dormant companies bought as a "
                        "front, and short-lived companies created for a few months of "
                        "fraud all come with a perfectly valid Kbis.",
                        "So also check that your contact is among the directors — or "
                        "holds a mandate —, that their e-mail domain is the company's, "
                        "and that the company is neither struck off nor in insolvency "
                        "proceedings. A registration a few months old, a token share "
                        "capital and a large first order together justify additional "
                        "guarantees.",
                    ],
                ),
                Section(
                    heading="Check the whole file",
                    paragraphs=[
                        "A Kbis never comes alone. The bank-detail document must be in "
                        "the exact company name; the invoice must carry a SIRET whose "
                        "first nine digits are the Kbis SIREN and an intra-EU VAT number "
                        "consistent with it (FR, a two-digit key, then the SIREN); for "
                        "subcontracting, the URSSAF compliance certificate completes the "
                        "statutory due-diligence file. Any inconsistency between these "
                        "documents weighs more than the quality of each taken alone.",
                        "TrustyFile runs these cross-checks on the PDFs you upload: "
                        "SIREN, SIRET and VAT check digits, the company's existence in "
                        "the INSEE register, and traces of editing in the file. "
                        "Comparing with the register itself remains a step nothing "
                        "replaces.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-verify-a-french-tax-notice",
            audience="individuals",
            counterpart="verifier-un-avis-d-imposition",
            published="2026-09-03",
            modified="2026-09-03",
            title="How to verify a French tax notice (avis d'imposition)",
            description=(
                "Landlord or agency: verify a tenant's French income-tax notice with "
                "the tax office's online service (SVAIR), the signed 2D-Doc code and "
                "the other documents in the file."
            ),
            heading="How to verify a French tax notice",
            intro=(
                "The income-tax notice (<em>avis d'imposition</em>) is the "
                "cornerstone of a French rental application and of many credit or "
                "benefit requests: it sets the household's reference taxable income "
                "(<em>revenu fiscal de référence</em>). It is also the most tampered "
                "document in rental files, because an inflated income opens doors the "
                "real one keeps shut. Yet it is easier to verify than almost any "
                "other document: the tax administration offers an online check, and "
                "the notice carries a signed 2D-Doc code."
            ),
            sections=[
                Section(
                    heading="What a notice says, and what gets forged",
                    paragraphs=[
                        "The notice shows the taxpayer's tax number (13 digits), the "
                        "notice reference (13 digits), the reference taxable "
                        "income, the number of household shares (<em>parts</em>), the "
                        "tax due, the address and the family situation. It carries two "
                        "years: the year of the income declared and the following year, "
                        "when it is issued. Right after filing, the declarative status "
                        "notice (ASDIR) stands in for it, with the same identifiers.",
                        "Forgeries follow four patterns: an inflated reference income "
                        "on a genuine notice; an old notice with the year changed; "
                        "someone else's notice with the name and address replaced; a "
                        "document built from scratch on a template. The first two are "
                        "the most common, and the two checks below, taken together, "
                        "catch all four: the first establishes that the notice exists "
                        "and is the latest issued, the second that its figures were not "
                        "touched.",
                    ],
                ),
                Section(
                    heading="The tax office's verification service (SVAIR)",
                    paragraphs=[
                        "The French tax administration runs a "
                        "<a href=\"https://www.impots.gouv.fr/verifavis2-api/front\">"
                        "notice verification service</a> (SVAIR), open without an "
                        "account from impots.gouv.fr under \"Vérifier un avis d'impôt\" "
                        "and meant precisely for the third parties a notice is shown to: "
                        "landlords, agencies, public bodies. Enter the two identifiers "
                        "printed on the document — the tax number (13 digits) and the "
                        "notice reference (13 digits) — and the service answers with one "
                        "word: <em>conforme</em> if that pair matches the latest notice "
                        "the administration holds for that taxpayer, <em>non "
                        "conforme</em> otherwise. It displays no amounts: tax "
                        "confidentiality forbids it.",
                        "Read that verdict for what it is. \"Non conforme\" means the "
                        "pair is unknown — a fabricated document, an altered or mistyped "
                        "identifier — or that a more recent notice has since been "
                        "issued: the applicant is showing you an old notice, perhaps "
                        "superseded by a corrective return. \"Conforme\" proves that a "
                        "genuine notice carries these identifiers and that it is the "
                        "latest one; it does not prove that the printed figures are the "
                        "original ones, since the service does not compare them. An "
                        "inflated income on a genuine, current notice passes this check: "
                        "the 2D-Doc code is what catches it.",
                    ],
                ),
                Section(
                    heading="The 2D-Doc code",
                    paragraphs=[
                        "Notices carry a 2D-Doc code: a DataMatrix electronically signed "
                        "by the tax administration that repeats the document's key data "
                        "(reference income, number of shares, notice reference, tax "
                        "number, year). A 2D-Doc reader decodes it and verifies the "
                        "signature; TrustyFile does so for every uploaded PDF, offline, "
                        "with the certificates the authority publishes.",
                        "If the decoded data differs from what is printed, the visible "
                        "text was edited and the code tells the truth. If the signature "
                        "is invalid, the code was recomposed — something a forger cannot "
                        "do correctly without the administration's private key. An "
                        "unreadable code (blurry scan, photo) is not proof of anything, "
                        "though: it leaves you with the SVAIR check — which attests that "
                        "the notice exists, not its figures — and the cross-checks "
                        "below. The guide "
                        "<a href=\"/en/guides/2d-doc-verifying-official-documents\">on "
                        "2D-Doc</a> explains the mechanism.",
                    ],
                ),
                Section(
                    heading="Cross-checks with the rest of the file",
                    paragraphs=[
                        "A notice consistent with itself may not be consistent with the "
                        "file:",
                    ],
                    bullets=[
                        "<strong>Pay slips</strong>: for an employee with no other "
                        "income, the reference taxable income is close to the annual "
                        "cumulative \"net imposable\" on the pay slips, less the 10% flat "
                        "deduction for professional expenses; some tax-exempt income is "
                        "added back and can widen the gap. A gap of several thousand "
                        "euros needs an "
                        "explanation (other income, a change of situation). See the "
                        "guide <a href=\"/en/guides/how-to-verify-a-pay-slip\">on pay "
                        "slips</a>.",
                        "<strong>Address</strong>: the one on the notice must match the "
                        "<a href=\"/en/guides/how-to-verify-a-proof-of-address\">proof "
                        "of address</a> and the declared history.",
                        "<strong>Number of shares</strong>: consistent with the stated "
                        "family situation (one share for a single person, two for a "
                        "couple, then half-shares for dependent children).",
                        "<strong>Years</strong>: a notice issued in year N covers income "
                        "of N-1; a two-year-old \"latest notice\" is not the latest.",
                        "<strong>Spelling of names</strong> and title identical across "
                        "every document.",
                    ],
                ),
                Section(
                    heading="Signs inside the PDF file",
                    paragraphs=[
                        "A notice downloaded from the taxpayer's personal space on "
                        "impots.gouv.fr is a born-digital PDF, produced in one go by the "
                        "administration's system. A notice altered in a PDF editor keeps "
                        "traces: a revision added after creation, a slightly different "
                        "font on the edited amount, the old figure still present under a "
                        "white rectangle, editor or online-converter metadata. The guide "
                        "<a href=\"/en/guides/how-to-tell-if-a-pdf-was-modified\">on "
                        "modified PDFs</a> reviews these signals.",
                        "A scanned or photographed notice loses all of this: no usable "
                        "PDF structure, and often no readable 2D-Doc. That is not a sign "
                        "of fraud — many people only have a paper copy — but it is a "
                        "document that cannot be verified on its own. Ask for the "
                        "original PDF: it alone lets you read the 2D-Doc code and check "
                        "the amounts.",
                    ],
                ),
                Section(
                    heading="Legal framework and good habits",
                    paragraphs=[
                        "A landlord may ask an applicant for their latest or "
                        "previous-year tax notice: it is on the exhaustive list of "
                        "documents set by the decree of 5 November 2015, which forbids "
                        "demanding anything not listed. Presenting a forged notice is "
                        "forgery and use of forged documents (article 441-1 of the "
                        "French criminal code, up to three years' imprisonment and a "
                        "€45,000 fine).",
                        "Faced with a doubtful notice: do not sign, keep the document "
                        "and the exchanges, and remember that an unsuccessful "
                        "applicant's documents must not be kept longer than necessary. "
                        "TrustyFile archives nothing: the file is analysed, then "
                        "deleted.",
                    ],
                ),
            ],
        ),
        Guide(
            slug="how-to-verify-an-urssaf-vigilance-certificate",
            audience="business",
            counterpart="verifier-une-attestation-de-vigilance-urssaf",
            published="2026-09-03",
            modified="2026-09-03",
            tools=["siret-checker", "siren-checker"],
            title="How to verify an URSSAF vigilance certificate (France)",
            description=(
                "Checking a French subcontractor's attestation de vigilance: the "
                "15-character security code on urssaf.fr, the six-month validity, and "
                "what the law requires of the client."
            ),
            heading="How to verify an URSSAF vigilance certificate",
            intro=(
                "The <em>attestation de vigilance</em> is the document by which URSSAF, "
                "the French social-security collector, certifies that a business "
                "declares and pays its social contributions. Any client must demand it "
                "from subcontractors above a certain contract value, and have it "
                "renewed: it is the client's protection against joint liability if "
                "the subcontractor turns out to use undeclared labour. For that very "
                "reason it gets forged — and it is one of the easiest documents to "
                "verify, thanks to the security code URSSAF prints on it."
            ),
            sections=[
                Section(
                    heading="What the certificate certifies, and what it contains",
                    paragraphs=[
                        "URSSAF issues the certificate to a business or self-employed "
                        "person that has paid its contributions when due, or holds and "
                        "honours a payment plan, or formally disputes the amount. It is "
                        "downloaded from the business's online account — urssaf.fr, or "
                        "autoentrepreneur.urssaf.fr for a micro-entrepreneur — and is "
                        "valid for six months.",
                        "The document identifies the business (name, SIRET, address), "
                        "states its issue date and validity period and, for an employer, "
                        "the number of employees and the total remuneration declared. It "
                        "carries a fifteen-character security code: that code is what "
                        "makes verification possible.",
                    ],
                ),
                Section(
                    heading="What the law requires of the client",
                    paragraphs=[
                        "For any contract worth at least €5,000 excluding VAT, the client "
                        "must obtain the vigilance certificate at signature, then every "
                        "six months until the work is complete (French labour code, "
                        "articles L8222-1 and following, D8222-5). A proof of "
                        "registration comes with it: a K or Kbis extract, proof of entry "
                        "in the National Business Register, or a quote, advertising "
                        "document or business letter showing the name, address and "
                        "registration number.",
                        "The stakes are not administrative. If the subcontractor is "
                        "convicted of undeclared labour and these checks were not made, "
                        "the client is jointly liable for its unpaid taxes, "
                        "contributions and wages. A forged certificate accepted without "
                        "checking protects against nothing.",
                    ],
                ),
                Section(
                    heading="The security code: the check that matters",
                    paragraphs=[
                        "URSSAF runs a public <a href=\"https://www.urssaf.fr/accueil/"
                        "outils-documentation/outils/verification-attestation.html\">"
                        "certificate verification tool</a>. Enter the security code "
                        "exactly as printed — fifteen characters, no spaces. If URSSAF "
                        "does not recognise it, it answers that the code matches no "
                        "certificate currently in force: the code was miscopied, the "
                        "certificate has expired, or it was altered or fabricated.",
                        "If the code is recognised, URSSAF displays the details of the "
                        "certificate it actually issued: the company name (\"identification "
                        "du diffuseur\"), the SIRET of each establishment, the document's "
                        "certification date and the last period analysed. Compare them line "
                        "by line with the document you received: a fraudster can paste "
                        "another company's genuine code onto a certificate fabricated in "
                        "their own name — the code passes, the comparison does not.",
                    ],
                ),
                Section(
                    heading="Signs of a forgery in the document itself",
                    bullets=[
                        "A <strong>validity period</strong> that is not six months, an "
                        "issue date later than the letter it came with, or an "
                        "\"up to date\" certificate that is eight months old.",
                        "An <strong>inconsistent headcount</strong>: the subcontractor "
                        "promises thirty people on site, the certificate declares two "
                        "employees.",
                        "A <strong>SIRET with a wrong check digit</strong>, or one that "
                        "does not match the Kbis and the invoices.",
                        "A <strong>security code</strong> that is not fifteen characters, "
                        "contains spaces, or appears identically on two certificates "
                        "with different dates.",
                        "<strong>Traces of PDF editing</strong>: a different font on a "
                        "date or figure, a white rectangle, a revision added after "
                        "creation, editor metadata — see <a href=\"/en/guides/"
                        "how-to-tell-if-a-pdf-was-modified\">the guide on modified "
                        "PDFs</a>.",
                    ],
                ),
                Section(
                    heading="Cross-check with the rest of the file",
                    paragraphs=[
                        "The certificate never comes alone. The SIRET must be the one on "
                        "the <a href=\"/en/guides/how-to-verify-a-kbis-extract\">Kbis "
                        "extract</a> and the invoices; the bank details must be in the "
                        "same company name — <a href=\"/en/guides/"
                        "supplier-bank-detail-change-fraud\">supplier impersonation "
                        "fraud</a> often starts with a flawless onboarding file. And do "
                        "not confuse the vigilance certificate with the tax-compliance "
                        "certificate issued by the tax administration: two documents, "
                        "two issuers, two checks.",
                    ],
                ),
                Section(
                    heading="What a valid certificate does not prove",
                    paragraphs=[
                        "That a business is up to date with its contributions today says "
                        "nothing about its solvency tomorrow, nor about who is actually "
                        "writing to you in its name. The certificate expires in six "
                        "months: put the renewal in the contract's follow-up rather than "
                        "relying on the subcontractor to remember. TrustyFile examines "
                        "the file — revisions, edits, SIRET and VAT check digits; the "
                        "security code is verified on urssaf.fr, and nothing replaces "
                        "that step.",
                    ],
                ),
            ],
        ),
    ],
}


# --- Rendering ---------------------------------------------------------------

# --- Checker tools -----------------------------------------------------------

TOOLS: dict[str, list[Tool]] = {
    "fr": [
        Tool(
            slug="verifier-un-iban",
            guides=["detecter-un-faux-rib", "fraude-au-faux-fournisseur-changement-de-rib"],
            kind="iban",
            counterpart="iban-checker",
            title="Vérifier un IBAN : clé de contrôle et clé RIB",
            description=(
                "Vérifiez gratuitement la clé de contrôle d'un IBAN (mod 97), sa "
                "longueur selon le pays et la clé RIB d'un IBAN français. Le calcul "
                "reste dans votre navigateur."
            ),
            heading="Vérifier un IBAN",
            intro=(
                "Collez un IBAN pour contrôler sa clé (mod 97), sa longueur pour le "
                "pays et, pour un IBAN français, la clé RIB. Tout se calcule dans "
                "votre navigateur : le numéro n'est envoyé nulle part."
            ),
            placeholder="FR76 3000 6000 0112 3456 7890 189",
            sections=[
                Section(
                    "Ce que ce contrôle prouve",
                    paragraphs=[
                        "Un IBAN porte deux chiffres de contrôle (positions 3 et 4) "
                        "calculés à partir de tout le reste. Si la clé est valide, le "
                        "numéro est bien formé : une faute de frappe, deux chiffres "
                        "inversés ou un numéro inventé au hasard échouent dans la "
                        "quasi-totalité des cas.",
                        "Pour un IBAN français, l'outil vérifie aussi la clé RIB (les "
                        "deux derniers chiffres), un second contrôle indépendant hérité "
                        "du relevé d'identité bancaire.",
                    ],
                ),
                Section(
                    "Ce qu'il ne prouve pas",
                    paragraphs=[
                        "Que le compte existe, qu'il est ouvert, ou qu'il appartient à "
                        "l'entreprise dont le nom figure sur la facture. La fraude au "
                        "faux fournisseur — un e-mail annonçant un « changement de "
                        "coordonnées bancaires » — utilise des IBAN parfaitement "
                        "valides : le compte est simplement celui du fraudeur.",
                        "La seule vérification qui vaille avant de payer sur un nouvel "
                        "IBAN : rappeler le fournisseur à un numéro que vous connaissiez "
                        "déjà, jamais celui indiqué dans le message.",
                    ],
                    bullets=[
                        '<a href="/guides/detecter-un-faux-rib">Détecter un faux RIB ou '
                        "un IBAN falsifié</a> : les indices à chercher dans le document "
                        "lui-même.",
                    ],
                ),
                Section(
                    "Comment lire le résultat",
                    bullets=[
                        "Clé valide, longueur conforme : le numéro est bien formé — rien "
                        "de plus.",
                        "Clé invalide : recopiez le numéro depuis le document original ; "
                        "si l'erreur persiste, le RIB a été mal saisi ou trafiqué.",
                        "Longueur inhabituelle : chaque pays fixe une longueur (27 "
                        "caractères en France, 22 en Allemagne, 16 en Belgique) ; un "
                        "écart signale un caractère perdu ou ajouté.",
                    ],
                ),
                Section(
                    "L'algorithme, en quatre étapes (ISO 13616)",
                    bullets=[
                        "Déplacer les quatre premiers caractères (pays et clé) à la fin.",
                        "Remplacer chaque lettre par un nombre : A = 10, B = 11 … Z = 35.",
                        "Calculer le reste de la division du grand nombre obtenu par 97.",
                        "Le reste doit valoir 1. Pour un IBAN français, la clé RIB doit "
                        "valoir 97 moins le reste de « banque, guichet, compte » × 100 "
                        "divisé par 97.",
                    ],
                ),
                Section(
                    "Vérifier toute la facture, pas seulement l'IBAN",
                    paragraphs=[
                        "TrustyFile fait ce contrôle sur les IBAN trouvés dans un PDF, "
                        "puis cherche ce qu'un IBAN valide ne dira jamais : un montant "
                        "recouvert, une révision ajoutée après coup, un texte masqué, un "
                        "identifiant d'entreprise inconnu du registre.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="verifier-un-siret",
            guides=["verifier-une-facture-falsifiee", "verifier-un-extrait-kbis", "verifier-une-attestation-de-vigilance-urssaf"],
            kind="siret",
            counterpart="siret-checker",
            title="Vérifier un SIRET : clé de Luhn et exception La Poste",
            description=(
                "Vérifiez la clé de contrôle d'un numéro SIRET (14 chiffres, formule "
                "de Luhn), y compris la règle spécifique aux établissements de La "
                "Poste. Calcul local, sans envoi."
            ),
            heading="Vérifier un SIRET",
            intro=(
                "Collez un SIRET : l'outil contrôle ses 14 chiffres avec la formule de "
                "Luhn et applique la règle particulière des établissements de La "
                "Poste. Le numéro reste dans votre navigateur."
            ),
            placeholder="552 081 317 66522",
            sections=[
                Section(
                    "SIRET, SIREN, NIC",
                    paragraphs=[
                        "Le SIRET identifie un établissement : les 9 chiffres du SIREN "
                        "(l'entreprise) suivis des 5 chiffres du NIC (l'établissement). "
                        "Une même entreprise a autant de SIRET que de sites.",
                    ],
                ),
                Section(
                    "Ce que ce contrôle prouve",
                    paragraphs=[
                        "Que le numéro est bien formé : le dernier chiffre est une clé "
                        "de Luhn calculée sur les treize autres. Une faute de frappe ou "
                        "un numéro inventé échouent.",
                    ],
                ),
                Section(
                    "Ce qu'il ne prouve pas",
                    paragraphs=[
                        "Que l'entreprise existe, qu'elle est active, ou qu'elle porte "
                        "le nom indiqué sur le document. Pour cela il faut interroger le "
                        'registre : l\'<a href="https://annuaire-entreprises.data.gouv.fr/">'
                        "Annuaire des entreprises</a> (gratuit, officiel) donne la raison "
                        "sociale, l'adresse et l'état de l'établissement. TrustyFile "
                        "effectue cette interrogation automatiquement lors de l'analyse "
                        "d'un document.",
                    ],
                ),
                Section(
                    "L'exception La Poste",
                    paragraphs=[
                        "La Poste (SIREN 356 000 000) compte tant d'établissements que "
                        "l'INSEE lui a attribué sa propre règle : seul le siège "
                        "(356 000 000 00048) suit la formule de Luhn ; pour tout autre "
                        "SIRET de La Poste, la somme des 14 chiffres doit être un "
                        "multiple de 5. Un vérificateur qui l'ignore déclare fausses "
                        "toutes les factures de La Poste.",
                    ],
                ),
                Section(
                    "La formule de Luhn",
                    bullets=[
                        "Doubler un chiffre sur deux, en commençant par le premier.",
                        "Si un chiffre doublé dépasse 9, lui retirer 9.",
                        "Additionner les 14 chiffres obtenus.",
                        "Le total doit être un multiple de 10.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="verifier-un-siren",
            guides=["verifier-un-extrait-kbis", "verifier-une-facture-falsifiee"],
            kind="siren",
            counterpart="siren-checker",
            title="Vérifier un SIREN et retrouver le numéro de TVA",
            description=(
                "Vérifiez la clé de Luhn d'un numéro SIREN (9 chiffres) et obtenez le "
                "numéro de TVA intracommunautaire qui lui correspond. Calcul dans "
                "votre navigateur, sans envoi."
            ),
            heading="Vérifier un SIREN",
            intro=(
                "Collez un SIREN : l'outil contrôle ses 9 chiffres avec la formule de "
                "Luhn et calcule le numéro de TVA intracommunautaire attendu (FR, clé, "
                "SIREN). Le numéro reste dans votre navigateur."
            ),
            placeholder="552 081 317",
            sections=[
                Section(
                    "Ce que ce contrôle prouve",
                    paragraphs=[
                        "Que le numéro est bien formé : le neuvième chiffre est une clé "
                        "de Luhn calculée sur les huit autres. Une faute de frappe ou un "
                        "numéro inventé échouent.",
                    ],
                ),
                Section(
                    "Ce qu'il ne prouve pas",
                    paragraphs=[
                        "Que l'entreprise existe ou qu'elle porte le nom indiqué. "
                        'L\'<a href="https://annuaire-entreprises.data.gouv.fr/">Annuaire '
                        "des entreprises</a> répond à cette question gratuitement ; "
                        "TrustyFile l'interroge automatiquement pendant l'analyse d'un "
                        "document.",
                    ],
                ),
                Section(
                    "Du SIREN au numéro de TVA",
                    paragraphs=[
                        "En France, le numéro de TVA intracommunautaire dérive du "
                        "SIREN : FR, une clé à deux chiffres, puis le SIREN. La clé vaut "
                        "(12 + 3 × (SIREN modulo 97)) modulo 97. Un numéro de TVA dont "
                        "la clé ne correspond pas au SIREN imprimé à côté est un signal "
                        "d'alerte fort — l'une des incohérences que TrustyFile recherche "
                        "dans une facture.",
                    ],
                ),
                Section(
                    "La formule de Luhn",
                    bullets=[
                        "Doubler un chiffre sur deux, en commençant par le deuxième.",
                        "Si un chiffre doublé dépasse 9, lui retirer 9.",
                        "Additionner les 9 chiffres obtenus.",
                        "Le total doit être un multiple de 10.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="verifier-un-numero-de-tva",
            guides=["verifier-une-facture-falsifiee", "verifier-un-extrait-kbis"],
            kind="vat",
            counterpart="vat-number-checker",
            title="Vérifier un numéro de TVA intracommunautaire français",
            description=(
                "Vérifiez la clé d'un numéro de TVA français (FR + clé + SIREN) et sa "
                "cohérence avec le SIREN. Calcul dans votre navigateur, sans envoi ; "
                "pour l'existence du numéro, VIES."
            ),
            heading="Vérifier un numéro de TVA intracommunautaire",
            intro=(
                "Collez un numéro de TVA français : l'outil vérifie que la clé à deux "
                "chiffres correspond bien au SIREN qui la suit, et que ce SIREN est "
                "lui-même bien formé. Tout se calcule dans votre navigateur : le numéro "
                "n'est envoyé nulle part."
            ),
            placeholder="FR03 552 081 317",
            sections=[
                Section(
                    "Ce que ce contrôle prouve",
                    paragraphs=[
                        "Qu'un numéro de TVA français est cohérent avec lui-même : la clé "
                        "se calcule à partir du SIREN — (12 + 3 × (SIREN modulo 97)) "
                        "modulo 97 — et le SIREN porte sa propre clé de Luhn. Un numéro "
                        "recopié avec une faute, ou inventé pour habiller une fausse "
                        "facture, échoue presque toujours.",
                        "Sur une facture, le numéro de TVA doit correspondre au SIRET "
                        "imprimé à côté : les neuf premiers chiffres du SIRET sont le "
                        "SIREN qui suit la clé. Un écart entre les deux est l'une des "
                        "incohérences les plus parlantes, et l'une de celles que "
                        "TrustyFile recherche dans un document.",
                    ],
                ),
                Section(
                    "Ce qu'il ne prouve pas",
                    paragraphs=[
                        "Que le numéro est attribué, actif et enregistré pour les "
                        "échanges intracommunautaires. Seule la base "
                        '<a href="https://ec.europa.eu/taxation_customs/vies/">VIES</a> '
                        "de la Commission européenne répond à cette question, en "
                        "interrogeant l'administration fiscale du pays. Une entreprise "
                        "française en franchise en base de TVA peut n'avoir aucun numéro "
                        "intracommunautaire tout en étant parfaitement réelle.",
                        "Certains numéros français ont une clé qui contient des lettres, "
                        "calculée selon un autre algorithme : cet outil les signale comme "
                        "hors de son périmètre plutôt que de les valider ou de les "
                        "rejeter à tort.",
                    ],
                ),
                Section(
                    "Comment lire le résultat",
                    bullets=[
                        "Clé valide : le numéro est cohérent. Comparez-le au SIRET de la "
                        "facture et, pour un nouveau fournisseur, vérifiez-le dans VIES.",
                        "Clé invalide, SIREN bien formé : la clé attendue est affichée. "
                        "Une seule faute de frappe suffit à l'expliquer — un numéro "
                        "fabriqué aussi.",
                        "SIREN invalide : le cœur du numéro est faux ; recopiez-le depuis "
                        "le document original avant de conclure.",
                    ],
                ),
                Section(
                    "Vérifier toute la facture, pas seulement le numéro de TVA",
                    paragraphs=[
                        "TrustyFile fait ce contrôle sur les numéros trouvés dans un PDF, "
                        "confronte la TVA au SIRET, interroge le registre des entreprises, "
                        "puis cherche ce qu'un numéro cohérent ne dira jamais : un montant "
                        "recouvert, une révision ajoutée après coup, un texte masqué.",
                    ],
                ),
            ],
        ),
    ],
    "en": [
        Tool(
            slug="iban-checker",
            guides=["how-to-spot-a-fake-iban", "supplier-bank-detail-change-fraud"],
            kind="iban",
            counterpart="verifier-un-iban",
            title="IBAN checker: validate the check digits and the length",
            description=(
                "Free IBAN checker: validates the mod-97 check digits, the expected "
                "length for the country and, for French IBANs, the RIB key. "
                "Everything runs in your browser."
            ),
            heading="Check an IBAN",
            intro=(
                "Paste an IBAN to validate its check digits (mod 97), its length for "
                "the country and, for a French IBAN, the RIB key. Everything is "
                "computed in your browser: the number is sent nowhere."
            ),
            placeholder="GB82 WEST 1234 5698 7654 32",
            sections=[
                Section(
                    "What this check proves",
                    paragraphs=[
                        "An IBAN carries two check digits (positions 3 and 4) computed "
                        "from everything else. If they validate, the number is "
                        "well-formed: a typo, two swapped digits or a number made up at "
                        "random fail almost every time.",
                        "For a French IBAN the tool also verifies the RIB key (the last "
                        "two digits), a second, independent check inherited from the "
                        "French bank account statement.",
                    ],
                ),
                Section(
                    "What it does not prove",
                    paragraphs=[
                        "That the account exists, is open, or belongs to the company "
                        "named on the invoice. Vendor-impersonation fraud — an e-mail "
                        "announcing “new bank details” — uses perfectly valid IBANs: "
                        "the account simply belongs to the fraudster.",
                        "The only check that matters before paying a new IBAN: call the "
                        "supplier on a number you already had, never the one in the "
                        "message.",
                    ],
                    bullets=[
                        '<a href="/en/guides/how-to-spot-a-fake-iban">How to spot a fake '
                        "IBAN on an invoice</a>: the signals to look for in the document "
                        "itself.",
                    ],
                ),
                Section(
                    "How to read the result",
                    bullets=[
                        "Valid check digits, expected length: the number is well-formed "
                        "— nothing more.",
                        "Invalid check digits: copy the number again from the original "
                        "document; if it still fails, the IBAN was mistyped or tampered "
                        "with.",
                        "Unusual length: every country fixes one (27 characters in "
                        "France, 22 in Germany and the UK, 16 in Belgium); a mismatch "
                        "means a character was lost or added.",
                    ],
                ),
                Section(
                    "The algorithm in four steps (ISO 13616)",
                    bullets=[
                        "Move the first four characters (country and check digits) to the end.",
                        "Replace every letter with a number: A = 10, B = 11 … Z = 35.",
                        "Take the remainder of the resulting large number divided by 97.",
                        "The remainder must be 1. For a French IBAN, the RIB key must "
                        "equal 97 minus the remainder of “bank, branch, account” × 100 "
                        "divided by 97.",
                    ],
                ),
                Section(
                    "Check the whole invoice, not only the IBAN",
                    paragraphs=[
                        "TrustyFile runs this check on the IBANs found in a PDF, then "
                        "looks for what a valid IBAN will never tell you: a covered "
                        "amount, a revision added afterwards, hidden text, a company "
                        "identifier the registry does not know.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="siret-checker",
            guides=["how-to-spot-a-tampered-invoice", "how-to-verify-a-kbis-extract", "how-to-verify-an-urssaf-vigilance-certificate"],
            kind="siret",
            counterpart="verifier-un-siret",
            title="SIRET checker: Luhn check digit and the La Poste rule",
            description=(
                "Free SIRET checker: validates the 14-digit French establishment "
                "number with the Luhn formula, including the special rule for La "
                "Poste establishments. Runs locally."
            ),
            heading="Check a SIRET",
            intro=(
                "Paste a SIRET: the tool validates its 14 digits with the Luhn formula "
                "and applies the special rule for La Poste establishments. The number "
                "stays in your browser."
            ),
            placeholder="552 081 317 66522",
            sections=[
                Section(
                    "SIRET, SIREN, NIC",
                    paragraphs=[
                        "A SIRET identifies an establishment: the 9 digits of the SIREN "
                        "(the company) followed by the 5 digits of the NIC (the site). "
                        "One company has as many SIRETs as it has sites.",
                    ],
                ),
                Section(
                    "What this check proves",
                    paragraphs=[
                        "That the number is well-formed: the last digit is a Luhn check "
                        "digit computed from the other thirteen. A typo or a made-up "
                        "number fails.",
                    ],
                ),
                Section(
                    "What it does not prove",
                    paragraphs=[
                        "That the company exists, is active, or bears the name printed "
                        "on the document. For that you need the registry: the official, "
                        'free <a href="https://annuaire-entreprises.data.gouv.fr/">Annuaire '
                        "des entreprises</a> returns the legal name, address and status "
                        "of the establishment. TrustyFile queries it automatically while "
                        "analysing a document.",
                    ],
                ),
                Section(
                    "The La Poste exception",
                    paragraphs=[
                        "La Poste (SIREN 356 000 000) has so many establishments that "
                        "INSEE gave it a rule of its own: only the head office "
                        "(356 000 000 00048) follows Luhn; for any other La Poste SIRET "
                        "the sum of the 14 digits must be a multiple of 5. A checker "
                        "that ignores this calls every La Poste invoice a forgery.",
                    ],
                ),
                Section(
                    "The Luhn formula",
                    bullets=[
                        "Double every other digit, starting with the first.",
                        "If a doubled digit exceeds 9, subtract 9.",
                        "Add up the 14 resulting digits.",
                        "The total must be a multiple of 10.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="siren-checker",
            guides=["how-to-verify-a-kbis-extract", "how-to-spot-a-tampered-invoice"],
            kind="siren",
            counterpart="verifier-un-siren",
            title="SIREN checker: Luhn check and matching French VAT number",
            description=(
                "Free SIREN checker: validates the 9-digit French company number with "
                "the Luhn formula and derives the intra-EU VAT number it implies. "
                "Runs in your browser, nothing is sent."
            ),
            heading="Check a SIREN",
            intro=(
                "Paste a SIREN: the tool validates its 9 digits with the Luhn formula "
                "and derives the intra-EU VAT number to expect (FR, key, SIREN). The "
                "number stays in your browser."
            ),
            placeholder="552 081 317",
            sections=[
                Section(
                    "What this check proves",
                    paragraphs=[
                        "That the number is well-formed: the ninth digit is a Luhn check "
                        "digit computed from the other eight. A typo or a made-up number "
                        "fails.",
                    ],
                ),
                Section(
                    "What it does not prove",
                    paragraphs=[
                        "That the company exists or bears the name shown. The "
                        '<a href="https://annuaire-entreprises.data.gouv.fr/">Annuaire des '
                        "entreprises</a> answers that for free; TrustyFile queries it "
                        "automatically while analysing a document.",
                    ],
                ),
                Section(
                    "From SIREN to VAT number",
                    paragraphs=[
                        "In France the intra-EU VAT number derives from the SIREN: FR, "
                        "a two-digit key, then the SIREN. The key is (12 + 3 × (SIREN "
                        "mod 97)) mod 97. A VAT number whose key does not match the "
                        "SIREN printed next to it is a strong warning sign — one of the "
                        "inconsistencies TrustyFile looks for in an invoice.",
                    ],
                ),
                Section(
                    "The Luhn formula",
                    bullets=[
                        "Double every other digit, starting with the second.",
                        "If a doubled digit exceeds 9, subtract 9.",
                        "Add up the 9 resulting digits.",
                        "The total must be a multiple of 10.",
                    ],
                ),
            ],
        ),
        Tool(
            slug="vat-number-checker",
            guides=["how-to-spot-a-tampered-invoice", "how-to-verify-a-kbis-extract"],
            kind="vat",
            counterpart="verifier-un-numero-de-tva",
            title="French VAT number checker: key and SIREN consistency",
            description=(
                "Free French VAT number checker: validates the two-digit key against "
                "the SIREN it precedes and the SIREN's own Luhn digit. Runs in your "
                "browser; use VIES for registration."
            ),
            heading="Check a French VAT number",
            intro=(
                "Paste a French intra-EU VAT number: the tool checks that its two-digit "
                "key matches the SIREN that follows, and that the SIREN is itself "
                "well-formed. Everything runs in your browser: the number is not sent "
                "anywhere."
            ),
            placeholder="FR03 552 081 317",
            sections=[
                Section(
                    "What this check proves",
                    paragraphs=[
                        "That a French VAT number is self-consistent: the key is computed "
                        "from the SIREN — (12 + 3 × (SIREN mod 97)) mod 97 — and the "
                        "SIREN carries its own Luhn digit. A number copied with a typo, "
                        "or invented to dress up a fake invoice, almost always fails.",
                        "On an invoice the VAT number must match the SIRET printed next "
                        "to it: the first nine digits of the SIRET are the SIREN that "
                        "follows the key. A gap between the two is one of the most "
                        "telling inconsistencies, and one TrustyFile looks for in a "
                        "document.",
                    ],
                ),
                Section(
                    "What it does not prove",
                    paragraphs=[
                        "That the number is assigned, active and registered for intra-EU "
                        "trade. Only the European Commission's "
                        '<a href="https://ec.europa.eu/taxation_customs/vies/">VIES</a> '
                        "database answers that, by querying the country's tax "
                        "administration. A French business under the VAT exemption "
                        "threshold may have no intra-EU number at all and still be "
                        "perfectly real.",
                        "Some French numbers have a key containing letters, computed with "
                        "a different algorithm: this tool reports them as out of scope "
                        "rather than validating or rejecting them wrongly.",
                    ],
                ),
                Section(
                    "How to read the result",
                    bullets=[
                        "Key valid: the number is consistent. Compare it with the SIRET "
                        "on the invoice and, for a new supplier, look it up in VIES.",
                        "Key invalid, SIREN well-formed: the expected key is shown. A "
                        "single typo explains it — so does a fabricated number.",
                        "SIREN invalid: the core of the number is wrong; copy it again "
                        "from the original document before concluding.",
                    ],
                ),
                Section(
                    "Check the whole invoice, not just the VAT number",
                    paragraphs=[
                        "TrustyFile runs this check on the numbers found in a PDF, matches "
                        "the VAT number against the SIRET, queries the company register, "
                        "then looks for what a consistent number will never reveal: a "
                        "covered amount, a revision added afterwards, hidden text.",
                    ],
                ),
            ],
        ),
    ],
}

# The validators the tool pages run in the browser. This is a JavaScript
# reimplementation of src/modules/content.py's is_valid_iban / is_valid_siret /
# is_valid_siren — same normalisation, same rules — and tests/test_checksum_vectors.py
# runs both over tests/fixtures/checksum_vectors.json so they cannot drift.
# It is inlined into every tool page and also written to tools/checksums.js
# for the Node side of that test.
CHECKSUM_JS = r"""// TrustyFile checksum validators. Generated by scripts/build_guides.py —
// mirrors src/modules/content.py; both are held to tests/fixtures/checksum_vectors.json.
(function (root) {
  'use strict';
  var LA_POSTE_SIREN = '356000000';
  var IBAN_LENGTHS = { AT: 20, BE: 16, CH: 21, DE: 22, ES: 24, FR: 27, GB: 22, IE: 22,
                       IT: 27, LU: 20, MC: 27, NL: 18, PL: 28, PT: 25 };

  function stripSeparators(text) { return String(text == null ? '' : text).replace(/[\s.\-]/g, ''); }

  // Luhn as content.py applies it: for 14 digits the doubled positions are the
  // even indexes from the left (0, 2, ... 12); for 9 digits the odd ones (1, 3, 5, 7).
  function luhnOk(digits, doubleEvenIndexes) {
    var total = 0;
    for (var i = 0; i < digits.length; i++) {
      var d = digits.charCodeAt(i) - 48;
      if ((i % 2 === 0) === doubleEvenIndexes) { d *= 2; if (d > 9) { d -= 9; } }
      total += d;
    }
    return total % 10 === 0;
  }
  function digitSum(digits) {
    var s = 0;
    for (var i = 0; i < digits.length; i++) { s += digits.charCodeAt(i) - 48; }
    return s;
  }
  // Remainder of a long decimal string divided by 97, digit by digit (no BigInt needed).
  function mod97(digits) {
    var m = 0;
    for (var i = 0; i < digits.length; i++) { m = (m * 10 + (digits.charCodeAt(i) - 48)) % 97; }
    return m;
  }

  function sirenCheck(text) {
    var siren = stripSeparators(text);
    if (!/^\d{9}$/.test(siren)) { return { ok: false, reason: 'format', value: siren }; }
    var ok = luhnOk(siren, false);
    // French VAT key, as content.py's validate_french_vat computes it.
    var key = (12 + 3 * mod97(siren)) % 97;
    return { ok: ok, reason: ok ? null : 'checksum', value: siren, vatKey: (key < 10 ? '0' : '') + key };
  }

  function siretCheck(text) {
    var siret = stripSeparators(text);
    if (!/^\d{14}$/.test(siret)) { return { ok: false, reason: 'format', value: siret }; }
    var luhn = luhnOk(siret, true);
    var laPoste = siret.indexOf(LA_POSTE_SIREN) === 0;
    var ok = luhn || (laPoste && digitSum(siret) % 5 === 0);
    return { ok: ok, reason: ok ? null : 'checksum', value: siret,
             siren: siret.slice(0, 9), nic: siret.slice(9), laPoste: laPoste && ok && !luhn };
  }

  // French RIB key: bank (5) branch (5) account (11) key (2). Letters in the
  // account map to digits (A,J=1 B,K,S=2 ... I,R,Z=9); key = 97 - (BGC x 100 mod 97).
  function ribKeyOk(bban) {
    var account = bban.slice(10, 21).toUpperCase().replace(/[A-Z]/g, function (ch) {
      var n = ch.charCodeAt(0) - 64;
      return String(n < 10 ? n : (n < 19 ? n - 9 : n - 17));
    });
    var remainder = (mod97(bban.slice(0, 10) + account) * 100) % 97;
    return (97 - remainder) === parseInt(bban.slice(21, 23), 10);
  }

  function ibanCheck(text) {
    var iban = stripSeparators(text).toUpperCase();
    if (!/^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(iban)) { return { ok: false, reason: 'format', value: iban }; }
    var rearranged = iban.slice(4) + iban.slice(0, 4);
    var numeric = '';
    for (var i = 0; i < rearranged.length; i++) {
      var c = rearranged.charCodeAt(i);
      numeric += c >= 65 ? String(c - 55) : rearranged.charAt(i);   // A=10 ... Z=35
    }
    var ok = mod97(numeric) === 1;
    var country = iban.slice(0, 2);
    var expected = IBAN_LENGTHS[country] || null;
    return { ok: ok, reason: ok ? null : 'checksum', value: iban, country: country,
             expectedLength: expected, lengthOk: expected ? iban.length === expected : null,
             formatted: iban.replace(/(.{4})/g, '$1 ').trim(),
             ribKeyOk: (country === 'FR' && iban.length === 27) ? ribKeyOk(iban.slice(4)) : null };
  }

  // French VAT number: FR + two-digit key + SIREN; key = (12 + 3 x (SIREN mod 97)) mod 97,
  // as content.py's validate_french_vat computes it. Keys with letters (another
  // algorithm) are a format failure here, never a guess.
  function vatCheck(text) {
    var vat = stripSeparators(text).toUpperCase();
    if (!/^FR\d{11}$/.test(vat)) { return { ok: false, reason: 'format', value: vat }; }
    var key = vat.slice(2, 4), siren = vat.slice(4);
    var sirenOk = luhnOk(siren, false);
    var expected = (12 + 3 * mod97(siren)) % 97;
    var expectedKey = (expected < 10 ? '0' : '') + expected;
    var ok = sirenOk && key === expectedKey;
    return { ok: ok, reason: ok ? null : 'checksum', value: vat, key: key, siren: siren,
             sirenOk: sirenOk, expectedKey: expectedKey };
  }

  var api = { ibanCheck: ibanCheck, siretCheck: siretCheck, sirenCheck: sirenCheck, vatCheck: vatCheck };
  root.TFChecks = api;
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; }
})(typeof window !== 'undefined' ? window : this);
"""

# Page glue: reads the per-language messages from the #tf-i18n JSON block,
# runs the right validator and renders the result. Every string goes through
# esc() — the input is the visitor's own, but the habit is cheap.
TOOL_GLUE_JS = r"""(function () {
  'use strict';
  var M = JSON.parse(document.getElementById('tf-i18n').textContent);
  var kind = M.kind;
  var form = document.getElementById('tf-form');
  var input = document.getElementById('tf-input');
  var out = document.getElementById('tf-result');
  var check = { iban: TFChecks.ibanCheck, siret: TFChecks.siretCheck, siren: TFChecks.sirenCheck,
                vat: TFChecks.vatCheck }[kind];
  var minLength = { iban: 15, siret: 14, siren: 9, vat: 13 }[kind];

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function fmt(template, values) {
    return template.replace(/\{(\w+)\}/g, function (_, k) { return values[k] == null ? '' : values[k]; });
  }
  function render(r) {
    if (!r.value) { out.className = 'result'; out.innerHTML = ''; return; }
    var cls, title, lines = [];
    if (r.reason === 'format') { cls = 'ko'; title = M['format_' + kind]; }
    else if (r.ok) { cls = 'ok'; title = M['valid_' + kind]; }
    else { cls = 'ko'; title = M['invalid_' + kind]; }
    if (kind === 'iban' && r.reason !== 'format') {
      lines.push(r.formatted);
      lines.push(fmt(M.country, { country: r.country }));
      if (r.expectedLength) {
        lines.push(fmt(r.lengthOk ? M.length_ok : M.length_ko,
                       { country: r.country, n: r.value.length, expected: r.expectedLength }));
        if (r.ok && !r.lengthOk) { cls = 'warn'; }
      } else {
        lines.push(M.length_unknown);
      }
      if (r.ribKeyOk !== null) { lines.push(r.ribKeyOk ? M.rib_ok : M.rib_ko); if (r.ok && !r.ribKeyOk) { cls = 'warn'; } }
    }
    if (kind === 'siret' && r.reason !== 'format') {
      lines.push(fmt(M.siren_part, { siren: r.siren, nic: r.nic }));
      if (r.laPoste) { lines.push(M.la_poste); }
    }
    if (kind === 'siren' && r.ok) { lines.push(fmt(M.vat_hint, { key: r.vatKey, siren: r.value })); }
    if (kind === 'vat' && r.reason !== 'format') {
      lines.push(fmt(M.vat_parts, { key: r.key, siren: r.siren }));
      if (!r.sirenOk) { lines.push(M.vat_siren_ko); }
      else if (!r.ok) { lines.push(fmt(M.vat_key_expected, { expected: r.expectedKey })); }
    }
    var items = lines.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('');
    out.className = 'result show ' + cls;
    out.innerHTML = '<p class="verdict">' + esc(title) + '</p>' + (items ? '<ul>' + items + '</ul>' : '') +
                    '<p class="fine">' + esc(M.reminder) + '</p>';
  }
  function run() { render(check(input.value)); }
  form.addEventListener('submit', function (e) { e.preventDefault(); run(); });
  input.addEventListener('input', function () {
    var compact = input.value.replace(/[\s.\-]/g, '');
    if (compact.length >= minLength) { run(); }
    else { out.className = 'result'; out.innerHTML = ''; }
  });
})();
"""


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
/* Checker tools */
form.tool { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.5rem 0 0.75rem; }
form.tool input {
    flex: 1 1 260px; font-size: 1.05rem; padding: 0.7rem 0.9rem;
    border: 1px solid #c8ccd8; border-radius: 8px;
    font-family: ui-monospace, Menlo, Consolas, monospace; letter-spacing: 0.04em;
}
form.tool input:focus { outline: 2px solid #2563eb; outline-offset: 1px; border-color: #2563eb; }
form.tool button {
    padding: 0.7rem 1.2rem; border: 0; border-radius: 8px;
    background: #1a1a2e; color: #fff; font-weight: 600; cursor: pointer;
}
form.tool button:hover { background: #2a2a4e; }
.result { display: none; border-radius: 10px; padding: 1rem 1.1rem; margin: 0 0 0.75rem; }
.result.show { display: block; }
.result.ok { background: #e7f6ec; border: 1px solid #9fd7b1; }
.result.ko { background: #fdecec; border: 1px solid #f1a9a9; }
.result.warn { background: #fff6e0; border: 1px solid #f3d27a; }
.result .verdict { margin: 0; font-weight: 600; color: #1a1a2e; }
.result ul { margin: 0.5rem 0 0; }
.result .fine { color: #666; font-size: 0.9rem; margin: 0.7rem 0 0; }
p.privacy { color: #666; font-size: 0.88rem; margin: 0 0 1rem; }
@media (max-width: 480px) {
    body { padding: 1.25rem 1rem 3rem; }
    h1 { font-size: 1.55rem; }
}
"""


_MONTHS = {
    "fr": [
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre",
    ],
    "en": [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ],
}


def _format_date(lang: str, iso_date: str) -> str:
    """Spell out an ISO date the way a reader of that language expects.

    French writes the day first with no leading zero, and "1er" for the
    first; English here follows the same day-month-year order, which is what
    the previous hand-written labels used.

    Example:
        >>> _format_date("fr", "2026-09-01")
        '1er septembre 2026'
        >>> _format_date("en", "2026-08-13")
        '13 August 2026'
    """
    year, month, day = (int(part) for part in iso_date.split("-"))
    day_text = "1er" if lang == "fr" and day == 1 else str(day)
    return f"{day_text} {_MONTHS[lang][month - 1]} {year}"


def _updated_line(lang: str, iso_date: str) -> str:
    """The visible "updated on" line, with a machine-readable <time>.

    Example:
        >>> _updated_line("fr", "2026-08-13")
        '<p class="updated"><time datetime="2026-08-13">Mis à jour le 13 août 2026</time></p>'
    """
    label = CHROME[lang]["updated"].format(date=_format_date(lang, iso_date))
    return f'<p class="updated"><time datetime="{iso_date}">{label}</time></p>'


def _guide_cards(lang: str, guides: list["Guide"]) -> list[str]:
    """Link cards for guides — shared by the hub and the tool pages."""
    chrome = CHROME[lang]
    parts: list[str] = []
    for guide in guides:
        parts.append(f'    <a class="card" href="{chrome["guides_path"]}/{guide.slug}">')
        parts.append(f"        <h3>{html.escape(guide.title)}</h3>")
        parts.append(f"        <p>{html.escape(guide.description)}</p>")
        parts.append(f'        <p class="more">{chrome["read_more"]} &rarr;</p>')
        parts.append("    </a>")
    return parts


def _tool_cards(lang: str, tools: list["Tool"]) -> list[str]:
    """Link cards for checker tools — shared by the hub, the guides and the tools."""
    chrome = CHROME[lang]
    parts: list[str] = []
    for tool in tools:
        parts.append(f'    <a class="card" href="{chrome["tools_path"]}/{tool.slug}">')
        parts.append(f"        <h3>{html.escape(tool.title)}</h3>")
        parts.append(f"        <p>{html.escape(tool.description)}</p>")
        parts.append(f'        <p class="more">{chrome["open_tool"]} &rarr;</p>')
        parts.append("    </a>")
    return parts


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
    og_description: str | None = None,
    twitter_description: str | None = None,
) -> str:
    """Build the <head> shared by every generated page.

    Args:
        lang: "fr" or "en".
        title: <title> text, already plain (escaped here).
        description: meta description text; also the Open Graph and Twitter
            description unless the two overrides below are given.
        og_description: shorter copy for link previews (Facebook, LinkedIn,
            Slack...), which truncate harder than a results page.
        twitter_description: same for Twitter/X cards.
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
    <meta property="og:description" content="{esc(og_description or description)}">
    <meta property="og:url" content="{canonical}">
    <meta property="og:image" content="{SITE}/og-image.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:site_name" content="TrustyFile">
    <meta property="og:locale" content="{chrome['locale']}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{esc(title)}">
    <meta name="twitter:description" content="{esc(twitter_description or description)}">
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
        "datePublished": guide.published,
        "dateModified": guide.modified,
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
        f"    {_updated_line(lang, guide.modified)}",
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

    # Checker tools the reader is likely to want next (the fake-RIB guide
    # offers the IBAN checker, the invoice guide the SIRET one). A guide is the
    # page a searcher lands on; the tool is what they came to do. Also the
    # only inbound links the tools get besides the hub and the landing.
    if guide.tools:
        by_slug = {tool.slug: tool for tool in TOOLS[lang]}
        parts.append(f"    <h2>{html.escape(chrome['tools_heading'])}</h2>")
        parts.extend(_tool_cards(lang, [by_slug[slug] for slug in guide.tools]))

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
    # One section per audience: with ten guides a flat list makes a landlord
    # scan past supplier fraud and an accounts-payable clerk past pay slips.
    for audience in AUDIENCES:
        section = [g for g in GUIDES[lang] if g.audience == audience]
        if not section:
            continue
        parts.append(f'    <h2 id="{audience}">{html.escape(chrome["audience_headings"][audience])}</h2>')
        parts.extend(_guide_cards(lang, section))
    # Free checker tools: the hub is the one place that links every tool.
    parts.append(f'    <h2 id="outils">{html.escape(chrome["tools_heading"])}</h2>')
    parts.append(f'    <p>{html.escape(chrome["tools_intro"])}</p>')
    parts.extend(_tool_cards(lang, TOOLS[lang]))
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

    guide_links = " &middot; ".join(
        f'<a href="{chrome["guides_path"]}/{g.slug}">{esc(g.title)}</a>' for g in GUIDES[lang]
    )
    tool_links = " &middot; ".join(
        f'<a href="{chrome["tools_path"]}/{tool.slug}">{esc(tool.title)}</a>'
        for tool in TOOLS[lang]
    )
    colon = " :" if lang == "fr" else ":"
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
            og_description=chrome["landing_og_description"],
            twitter_description=chrome["landing_twitter_description"],
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
        f'    <p class="note">{esc(t["footer_guides"])}{colon} {guide_links}</p>',
        f'    <p class="note">{esc(t["footer_tools"])}{colon} {tool_links}</p>',
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


def render_tool(lang: str, tool: Tool) -> str:
    """Render one checker page: guide chrome around a client-side form.

    The validator runs in the visitor's browser (CHECKSUM_JS, inlined), so the
    page is fully static and the number typed never reaches a server — which
    is also what the page says, so keep that true.
    """
    chrome = CHROME[lang]
    other_lang = "en" if lang == "fr" else "fr"
    esc = html.escape
    canonical = f"{SITE}{chrome['tools_path']}/{tool.slug}"

    alternates = {lang: canonical}
    if tool.counterpart:
        other_canonical = f"{SITE}{CHROME[other_lang]['tools_path']}/{tool.counterpart}"
        alternates[other_lang] = other_canonical
        alternates["x-default"] = canonical if lang == "fr" else other_canonical
        other_url = f"{CHROME[other_lang]['tools_path']}/{tool.counterpart}"
    else:
        other_url = CHROME[other_lang]["index_path"]

    app_ld = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": tool.title,
        "url": canonical,
        "description": tool.description,
        "applicationCategory": "UtilityApplication",
        "operatingSystem": "Any",
        "browserRequirements": "Requires JavaScript",
        "isAccessibleForFree": True,
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
        "inLanguage": lang,
        "publisher": {"@type": "Organization", "name": "TrustyFile", "url": SITE},
    }
    breadcrumb_ld = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "TrustyFile", "item": chrome["app_url"]},
            {"@type": "ListItem", "position": 2, "name": chrome["tools_heading"],
             "item": f"{SITE}{chrome['index_path']}#outils"},
            {"@type": "ListItem", "position": 3, "name": tool.title},
        ],
    }

    # Messages for the page script. "</" must not appear inside a <script>
    # block, even a JSON one; json.dumps does not escape it, so do it here.
    i18n = dict(chrome["tool_i18n"], kind=tool.kind)
    i18n_json = json.dumps(i18n, ensure_ascii=False).replace("</", "<\\/")

    parts = [
        _head(
            lang=lang,
            title=tool.title,
            description=tool.description,
            canonical=canonical,
            alternates=alternates,
            json_ld=[app_ld, breadcrumb_ld],
            og_type="website",
        ),
        _nav(lang, other_url),
        f"    <h1>{esc(tool.heading)}</h1>",
        f'    <p class="intro">{tool.intro}</p>',
        '    <form class="tool" id="tf-form" autocomplete="off">',
        f'        <input id="tf-input" type="text" spellcheck="false" placeholder="{esc(tool.placeholder)}" aria-label="{esc(tool.heading)}">',
        f'        <button type="submit">{esc(chrome["tool_button"])}</button>',
        "    </form>",
        '    <div id="tf-result" class="result" role="status" aria-live="polite"></div>',
        f'    <p class="privacy">{esc(chrome["tool_privacy"])}</p>',
    ]
    for section in tool.sections:
        anchor = _slugify_heading(section.heading)
        parts.append(f'    <h2 id="{anchor}">{html.escape(section.heading)}</h2>')
        for paragraph in section.paragraphs:
            parts.append(f"    <p>{paragraph}</p>")
        if section.bullets:
            parts.append("    <ul>")
            for bullet in section.bullets:
                parts.append(f"        <li>{bullet}</li>")
            parts.append("    </ul>")

    siblings = [t for t in TOOLS[lang] if t.slug != tool.slug]
    if siblings:
        parts.append(f"    <h2>{chrome['related_tools']}</h2>")
        parts.extend(_tool_cards(lang, siblings))

    # The guides this tool serves (the IBAN checker -> fake RIB, supplier
    # fraud). A visitor who came for one number often has the whole document
    # in front of them; and these are the tools' only links back to the guides.
    if tool.guides:
        by_slug = {guide.slug: guide for guide in GUIDES[lang]}
        parts.append(f"    <h2>{chrome['related']}</h2>")
        parts.extend(_guide_cards(lang, [by_slug[slug] for slug in tool.guides]))

    parts.append(f'    <a class="cta" href="{chrome["app_url"]}">{chrome["cta"]}</a>')
    parts.append(f'    <script type="application/json" id="tf-i18n">{i18n_json}</script>')
    parts.append("    <script>")
    parts.append(CHECKSUM_JS.rstrip("\n"))
    parts.append(TOOL_GLUE_JS.rstrip("\n"))
    parts.append("    </script>")
    parts.append(_footer(lang))
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
        lastmod: ISO date of the last substantive change to that URL. Google
            does use this one — provided it is truthful, so it comes from the
            page's own date, not from the build time.
    """

    loc: str
    fr: str
    en: str
    priority: str
    lastmod: str


def sitemap_entries() -> list[SitemapEntry]:
    """Every indexable URL: the two app landing pages plus all guides.

    hreflang annotations must be reciprocal — each URL lists both languages,
    including itself — or Google discards the whole cluster.
    """
    fr_home, en_home = SITE + "/", SITE + "/en"
    # The landing pages and the hubs list every guide and tool, so they last
    # changed when the newest of those did.
    newest = latest_content_date()
    entries = [
        SitemapEntry(fr_home, fr_home, en_home, "1.0", newest),
        SitemapEntry(en_home, fr_home, en_home, "0.9", newest),
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
                newest,
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
            entries.append(SitemapEntry(loc, fr_url, en_url, "0.7", guide.modified))
        for tool in TOOLS[lang]:
            loc = f"{SITE}{CHROME[lang]['tools_path']}/{tool.slug}"
            other_loc = f"{SITE}{CHROME[other]['tools_path']}/{tool.counterpart}"
            fr_url, en_url = (loc, other_loc) if lang == "fr" else (other_loc, loc)
            entries.append(SitemapEntry(loc, fr_url, en_url, "0.7", tool.modified))
    return entries


def latest_content_date() -> str:
    """The most recent `modified` date across every guide and tool.

    ISO dates sort as strings, so max() is enough.
    """
    dates = [g.modified for guides in GUIDES.values() for g in guides]
    dates += [t.modified for tools in TOOLS.values() for t in tools]
    return max(dates)


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
            f"    <lastmod>{entry.lastmod}</lastmod>",
            "    <changefreq>monthly</changefreq>",
            f"    <priority>{entry.priority}</priority>",
            "  </url>",
        ]
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def render_llms_txt() -> str:
    """Render /llms.txt: a Markdown map of the site for AI answer engines.

    The llms.txt convention (llmstxt.org) is a plain Markdown file at the site
    root: an h1, a one-line summary in a blockquote, then h2 sections of
    "- [title](url): description" links. Assistants that browse (ChatGPT,
    Perplexity, Claude — all already in the Caddyfile's crawler list) read it
    to find the pages worth citing, and citations are a growing referral
    source. Everything listed is a public page served identically to every
    visitor, so this adds no content the site does not already show.
    """
    lines = [
        "# TrustyFile",
        "",
        "> Vérificateur de documents PDF gratuit : déposez un PDF et obtenez un "
        "score de confiance fondé sur des indices techniques (métadonnées, "
        "révisions, polices, texte masqué, 2D-Doc, identifiants d'entreprise). "
        "Free online PDF document fraud detector: upload a PDF and get a trust "
        "score built from technical evidence.",
        "",
        "TrustyFile est un outil d'aide à la décision, pas une autorité de "
        "certification : un score élevé signale peu de traces de manipulation, "
        "il ne prouve pas l'authenticité. Les documents ne sont pas conservés. "
        "Decision support, not certification: a high score means few signs of "
        "tampering, not proof of authenticity. Documents are not retained.",
        "",
        "## Application",
        "",
        f"- [TrustyFile (français)]({SITE}/): analyse gratuite d'un PDF, "
        "sans inscription, résultat en quelques secondes.",
        f"- [TrustyFile (English)]({SITE}/en): free PDF analysis, no sign-up, "
        "results in seconds.",
        "",
    ]
    headings = {
        "fr": ("Guides (français)", "Outils gratuits (français)"),
        "en": ("Guides (English)", "Free tools (English)"),
    }
    for lang in ("fr", "en"):
        chrome = CHROME[lang]
        guides_heading, tools_heading = headings[lang]
        lines.append(f"## {guides_heading}")
        lines.append("")
        lines.append(f"- [{chrome['index_heading']}]({SITE}{chrome['index_path']})")
        for guide in GUIDES[lang]:
            url = f"{SITE}{chrome['guides_path']}/{guide.slug}"
            lines.append(f"- [{guide.title}]({url}): {guide.description}")
        lines.append("")
        lines.append(f"## {tools_heading}")
        lines.append("")
        for tool in TOOLS[lang]:
            url = f"{SITE}{chrome['tools_path']}/{tool.slug}"
            lines.append(f"- [{tool.title}]({url}): {tool.description}")
        lines.append("")
    lines.append("## Optional")
    lines.append("")
    lines.append(f"- [Sitemap]({SITE}/sitemap.xml)")
    lines.append("- [Contact](mailto:contact@trustyfile.com)")
    return "\n".join(lines) + "\n"


def build() -> dict[Path, str]:
    """Every generated file. Returns {absolute path: content}, without writing."""
    pages: dict[Path, str] = {}
    for lang in ("fr", "en"):
        pages[OUTPUT_DIR / lang / "index.html"] = render_index(lang)
        for guide in GUIDES[lang]:
            pages[OUTPUT_DIR / lang / f"{guide.slug}.html"] = render_guide(lang, guide)
        pages[ROOT / f"seo-{lang}.html"] = render_landing(lang)
        for tool in TOOLS[lang]:
            pages[ROOT / "tools" / lang / f"{tool.slug}.html"] = render_tool(lang, tool)
    # The same validators the pages inline, as a Node module for the parity
    # test (tests/test_checksum_vectors.py). Not served by Caddy.
    pages[ROOT / "tools" / "checksums.js"] = CHECKSUM_JS
    pages[ROOT / "sitemap.xml"] = render_sitemap()
    pages[ROOT / "llms.txt"] = render_llms_txt()
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
