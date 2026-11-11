# GEO — being cited by ChatGPT, Claude, Gemini (2026-11-11)

GEO = Generative Engine Optimization: SEO for answer engines. The term comes
from the 2023 Princeton / Georgia Tech / IIT Delhi paper "GEO: Generative
Engine Optimization" (Aggarwal et al., KDD 2024), whose clearest finding is
that citing sources and stating facts plainly raise a page's visibility in
generated answers.

## How each assistant finds pages

| Assistant | Index it leans on | What we did about it |
|---|---|---|
| ChatGPT | Bing (widely reported, not confirmed by OpenAI docs) + OAI-SearchBot | Bing Webmaster Tools verified, sitemap submitted, IndexNow |
| Gemini / AI Overviews | Google | Search Console (already set up) |
| Claude | Claude-SearchBot + web search | robots.txt allows every bot; same bytes for every client (checked live) |
| Perplexity | PerplexityBot | same |

## Done on the site (branch feat/geo-key-points)

- "En bref / In short" box under every guide's intro: 4 self-contained
  sentences an assistant can quote alone (`Guide.key_points`).
- A FAQ at the end of every guide, phrased as people ask an assistant, with
  matching FAQPage JSON-LD (`Guide.faq`; a test pins JSON-LD == visible).
- Method page: /guides/comment-fonctionne-trustyfile and
  /en/guides/how-trustyfile-works — what is checked, how the score works,
  limits, data retention. Linked as `Organization.publishingPrinciples`.
- 2D-Doc guide corrected: it said signature verification was not
  implemented; it has shipped (src/pipeline.py, twod_doc_crypto.py).
- Article JSON-LD author/publisher point at `#organization` (one entity).
- IndexNow: key file public/6c2a…379.txt, Caddy route, scripts/indexnow.py.

## After each content deploy

Nothing to do: .github/workflows/indexnow.yml runs when Railway reports a
production deploy live, and sends the pages whose file changed since the
previous production deploy (`scripts/indexnow.py --since-commit`). By hand,
if ever needed:

    python scripts/indexnow.py --since <date of the change>

Bump `modified=` on any guide you rewrite: it drives the sitemap lastmod,
the visible "Updated" line and what IndexNow submits.

## Measuring (monthly, 20 minutes)

Ask each assistant (ChatGPT, Claude, Gemini, Perplexity), logged out where
possible, and note whether TrustyFile is named or linked:

- Comment savoir si une facture PDF a été modifiée ?
- Comment vérifier un justificatif de domicile ?
- Comment savoir si une fiche de paie est fausse ?
- Comment vérifier l'avis d'imposition d'un locataire ?
- Comment détecter un faux RIB ?
- Que faire quand un fournisseur annonce un changement de RIB ?
- Comment vérifier un Kbis ?
- Comment vérifier une attestation de vigilance Urssaf ?
- Comment vérifier un code 2D-Doc ?
- Quel outil gratuit pour détecter un PDF falsifié ?
- How can I tell if a PDF was modified?
- Free tool to detect a tampered invoice PDF?

Bing Webmaster Tools > AI Performance shows Copilot citations (not ChatGPT).

## Not done — needs the owner

- Mentions on other sites: the strongest lever and not code. Product Hunt,
  tool directories, answers on forums/Reddit threads about rental files and
  fake bank details, a guest article.
- `sameAs` links on the Organization (LinkedIn page, public GitHub, Product
  Hunt…): none exist yet; add the URLs once they do.
- A named author / "about" person: an E-E-A-T signal, but personal data, so
  the owner's call.
