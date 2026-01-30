"""
Module G: External Verification

This module verifies document information against external databases.
It requires internet access and may have rate limits.

What we verify:
1. SIRET/SIREN against INSEE API (French company registry)
2. VAT number against VIES API (EU VAT validation)
3. Company name matches the registered name

IMPORTANT: This module makes network requests and should be optional.
Users may want to skip it for privacy or speed reasons.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData
from src.modules.content import extract_siret, extract_french_vat, extract_siren, validate_siren_checksum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CompanyInfo:
    """
    Information about a company from official registries.

    Attributes:
        siren: 9-digit SIREN number
        siret: 14-digit SIRET number (if available)
        name: Official company name
        trade_name: Commercial name (nom commercial)
        address: Registered address
        postal_code: Postal code
        city: City
        status: Company status (active, closed, etc.)
        legal_form: Legal form (SARL, SAS, SA, etc.)
        creation_date: Company creation date
        closure_date: Closure date (if closed)
    """
    siren: str
    siret: Optional[str] = None
    name: Optional[str] = None
    trade_name: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    city: Optional[str] = None
    status: Optional[str] = None  # "active", "closed"
    legal_form: Optional[str] = None
    creation_date: Optional[str] = None
    closure_date: Optional[str] = None


# =============================================================================
# POTENTIAL SIREN EXTRACTION (XXX XXX XXX patterns)
# =============================================================================

def extract_potential_sirens(text: str) -> list[tuple[str, str]]:
    """
    Extract potential SIREN numbers from text based on pattern matching.

    Looks for 9-digit patterns in groups of 3 (XXX XXX XXX) that might be
    SIRENs even without explicit "SIREN" labels. We'll verify them via API.

    Args:
        text: Full document text

    Returns:
        List of (siren, context) tuples
    """
    results = []
    seen = set()

    # Pattern: 3 groups of 3 digits separated by spaces
    # e.g., "383 960 135"
    pattern = r"\b(\d{3})\s+(\d{3})\s+(\d{3})\b"

    for match in re.finditer(pattern, text):
        siren = match.group(1) + match.group(2) + match.group(3)

        if siren in seen:
            continue
        seen.add(siren)

        # Check Luhn checksum first - if invalid, probably not a SIREN
        if not validate_siren_checksum(siren):
            continue

        # Get context
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end].strip()

        results.append((siren, context))

    return results


# =============================================================================
# ANNUAIRE DES ENTREPRISES API (French Company Registry - FREE, NO AUTH)
# =============================================================================

# This API is free and doesn't require authentication!
# Documentation: https://recherche-entreprises.api.gouv.fr/docs/
# Source: https://github.com/annuaire-entreprises-data-gouv-fr/search-api

ANNUAIRE_API_BASE = "https://recherche-entreprises.api.gouv.fr"


def verify_siret_annuaire(siret: str) -> tuple[CompanyInfo | None, str | None]:
    """
    Verify a SIRET number against the Annuaire des Entreprises API.

    This API is FREE and doesn't require authentication!

    Args:
        siret: 14-digit SIRET number

    Returns:
        Tuple of (CompanyInfo, error_message)
        - If successful: (CompanyInfo, None)
        - If not found: (None, "SIRET not found")
        - If error: (None, error_message)

    Example:
        >>> info, error = verify_siret_annuaire("55208131766522")
        >>> print(info.name)
        "ELECTRICITE DE FRANCE"
    """
    try:
        import requests
    except ImportError:
        return None, "requests library not installed"

    try:
        # Search by SIRET - use the number directly, not with prefix
        # The API searches across all fields including SIRET
        url = f"{ANNUAIRE_API_BASE}/search"
        params = {"q": siret}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return None, f"SIRET {siret} not found in registry"

            # Search through results to find exact SIRET match
            matching_company = None
            matching_etablissement = None

            for company in results:
                # Check siege
                siege = company.get("siege", {})
                if siege.get("siret") == siret:
                    matching_company = company
                    matching_etablissement = siege
                    break

                # Check matching_etablissements
                for etab in company.get("matching_etablissements", []):
                    if etab.get("siret") == siret:
                        matching_company = company
                        matching_etablissement = etab
                        break

                if matching_etablissement:
                    break

                # Check if SIREN matches (SIRET = SIREN + NIC)
                if company.get("siren") == siret[:9]:
                    matching_company = company
                    matching_etablissement = siege
                    # Don't break - keep looking for exact SIRET match

            if not matching_company:
                return None, f"SIRET {siret} not found in registry"

            company = matching_company
            if not matching_etablissement:
                matching_etablissement = company.get("siege", {})

            # Build CompanyInfo
            info = CompanyInfo(
                siren=company.get("siren"),
                siret=siret,
                name=company.get("nom_complet"),
                trade_name=company.get("nom_raison_sociale"),
                address=matching_etablissement.get("adresse"),
                postal_code=matching_etablissement.get("code_postal"),
                city=matching_etablissement.get("libelle_commune"),
                status="active" if matching_etablissement.get("etat_administratif") == "A" else "closed",
                legal_form=company.get("nature_juridique"),
                creation_date=company.get("date_creation"),
            )
            return info, None

        elif response.status_code == 404:
            return None, f"SIRET {siret} not found in registry"
        elif response.status_code == 429:
            return None, "API rate limit exceeded (try again later)"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.Timeout:
        return None, "API timeout (server too slow)"
    except requests.RequestException as e:
        return None, f"API request failed: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def verify_siren_annuaire(siren: str) -> tuple[CompanyInfo | None, str | None]:
    """
    Verify a SIREN number against the Annuaire des Entreprises API.

    Args:
        siren: 9-digit SIREN number

    Returns:
        Tuple of (CompanyInfo, error_message)
    """
    try:
        import requests
    except ImportError:
        return None, "requests library not installed"

    try:
        # Search by SIREN - use the number directly
        url = f"{ANNUAIRE_API_BASE}/search"
        params = {"q": siren}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return None, f"SIREN {siren} not found in registry"

            company = results[0]
            siege = company.get("siege", {})

            info = CompanyInfo(
                siren=company.get("siren"),
                siret=siege.get("siret"),
                name=company.get("nom_complet"),
                trade_name=company.get("nom_raison_sociale"),
                address=siege.get("adresse"),
                postal_code=siege.get("code_postal"),
                city=siege.get("libelle_commune"),
                status="active" if siege.get("etat_administratif") == "A" else "closed",
                legal_form=company.get("nature_juridique"),
                creation_date=company.get("date_creation"),
            )
            return info, None

        elif response.status_code == 404:
            return None, f"SIREN {siren} not found in registry"
        else:
            return None, f"API error: HTTP {response.status_code}"

    except requests.Timeout:
        return None, "API timeout"
    except requests.RequestException as e:
        return None, f"API request failed: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


# =============================================================================
# VIES API (EU VAT Validation)
# =============================================================================

# VIES (VAT Information Exchange System) is free and doesn't require authentication
VIES_WSDL = "https://ec.europa.eu/taxation_customs/vies/checkVatService.wsdl"


def verify_vat_vies(vat_number: str) -> tuple[dict | None, str | None]:
    """
    Verify a VAT number against the EU VIES system.

    VIES is free and doesn't require authentication.

    Args:
        vat_number: VAT number with country prefix (e.g., "FR03552081317")

    Returns:
        Tuple of (result_dict, error_message)
        - If valid: ({"valid": True, "name": "...", "address": "..."}, None)
        - If invalid: ({"valid": False}, None)
        - If error: (None, error_message)
    """
    # Extract country code and number
    if len(vat_number) < 3:
        return None, "VAT number too short"

    country_code = vat_number[:2].upper()
    vat_num = vat_number[2:]

    try:
        import requests
    except ImportError:
        return None, "requests library not installed"

    # VIES provides a REST-like endpoint (simpler than SOAP)
    # We'll use the web form endpoint which returns JSON-like data
    try:
        url = "https://ec.europa.eu/taxation_customs/vies/rest-api/ms/{}/vat/{}".format(
            country_code, vat_num
        )
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return {
                "valid": data.get("isValid", False),
                "name": data.get("name"),
                "address": data.get("address"),
                "country_code": country_code,
                "vat_number": vat_num,
            }, None
        else:
            # Try the alternative check
            return {"valid": False, "reason": "VIES lookup failed"}, None

    except requests.Timeout:
        return None, "VIES API timeout"
    except requests.RequestException as e:
        return None, f"VIES API request failed: {e}"
    except Exception as e:
        return None, f"VIES verification error: {e}"


# =============================================================================
# COMPANY NAME MATCHING
# =============================================================================

def normalize_company_name(name: str) -> str:
    """
    Normalize a company name for comparison.

    Removes common variations like:
    - Legal form suffixes (SA, SAS, SARL, etc.)
    - Punctuation and extra spaces
    - Case differences

    Args:
        name: Company name

    Returns:
        Normalized name for comparison
    """
    if not name:
        return ""

    name = name.upper()

    # Remove legal form suffixes
    legal_forms = [
        r"\bSA\b", r"\bSAS\b", r"\bSARL\b", r"\bEURL\b",
        r"\bSNC\b", r"\bSCI\b", r"\bSCOP\b", r"\bSEL\b",
        r"\bGIE\b", r"\bSE\b", r"\bSCA\b",
    ]
    for form in legal_forms:
        name = re.sub(form, "", name)

    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize whitespace
    name = " ".join(name.split())

    return name.strip()


def company_names_match(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    Check if two company names match, allowing for minor variations.

    Uses a simple similarity check based on common words.

    Args:
        name1: First company name
        name2: Second company name
        threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        True if names are similar enough
    """
    n1 = normalize_company_name(name1)
    n2 = normalize_company_name(name2)

    if not n1 or not n2:
        return False

    # Exact match after normalization
    if n1 == n2:
        return True

    # Check word overlap
    words1 = set(n1.split())
    words2 = set(n2.split())

    if not words1 or not words2:
        return False

    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    similarity = intersection / union if union > 0 else 0

    return similarity >= threshold


# =============================================================================
# SEVERITY POINTS
# =============================================================================

SEVERITY_POINTS = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "critical": 50,
}


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_external(
    pdf_data: PDFData,
    verify_vat: bool = True,
    verify_siret: bool = True,
    extracted_company_name: str | None = None,
) -> ModuleResult:
    """
    Verify document information against external databases.

    This module requires internet access but NO authentication.
    Uses the free Annuaire des Entreprises API for SIRET verification.

    Args:
        pdf_data: Extracted PDF data
        verify_vat: Whether to verify VAT numbers via VIES
        verify_siret: Whether to verify SIRET via Annuaire des Entreprises
        extracted_company_name: Optional company name from document to compare

    Returns:
        ModuleResult with score, flags, and confidence
    """
    all_flags = []
    verifications_attempted = 0
    verifications_successful = 0

    # Combine all pages
    full_text = "\n".join(pdf_data.text_by_page)

    # Extract legal mentions
    sirets = extract_siret(full_text)
    sirens = extract_siren(full_text)
    vats = extract_french_vat(full_text)
    potential_sirens = extract_potential_sirens(full_text)

    # Verify SIRET numbers via Annuaire des Entreprises (FREE, NO AUTH!)
    if verify_siret:
        for siret, is_checksum_valid, context in sirets:
            if not is_checksum_valid:
                continue  # Already flagged by content module

            verifications_attempted += 1
            company_info, error = verify_siret_annuaire(siret)

            if error:
                logger.warning(f"SIRET verification failed: {error}")
                # Only flag as low severity if it's a network error
                if "not found" in error.lower():
                    all_flags.append(Flag(
                        severity="critical",
                        code="EXTERNAL_SIRET_NOT_FOUND",
                        message=f"SIRET {siret} not found in official registry",
                        details={"siret": siret, "error": error}
                    ))
                else:
                    all_flags.append(Flag(
                        severity="low",
                        code="EXTERNAL_SIRET_VERIFICATION_FAILED",
                        message=f"Could not verify SIRET {siret}: {error}",
                        details={"siret": siret, "error": error}
                    ))
            elif company_info:
                verifications_successful += 1

                # Check if company is closed
                if company_info.status == "closed":
                    all_flags.append(Flag(
                        severity="high",
                        code="EXTERNAL_COMPANY_CLOSED",
                        message=f"Company with SIRET {siret} is closed/inactive",
                        details={
                            "siret": siret,
                            "company_name": company_info.name,
                            "status": company_info.status,
                        }
                    ))

                # Check if company name matches (if provided)
                if extracted_company_name and company_info.name:
                    if not company_names_match(extracted_company_name, company_info.name):
                        all_flags.append(Flag(
                            severity="high",
                            code="EXTERNAL_COMPANY_NAME_MISMATCH",
                            message=f"Company name doesn't match SIRET registry",
                            details={
                                "siret": siret,
                                "name_in_document": extracted_company_name,
                                "name_in_registry": company_info.name,
                            }
                        ))

        # Also verify SIREN numbers (9 digits, e.g., "383 960 135 RCS Créteil")
        # Avoid duplicates: skip SIRENs that are already part of a verified SIRET
        verified_sirens = {siret[:9] for siret, _, _ in sirets}

        for siren, is_checksum_valid, context in sirens:
            if siren in verified_sirens:
                continue  # Already verified via SIRET

            if not is_checksum_valid:
                continue  # Already flagged by content module

            verifications_attempted += 1
            company_info, error = verify_siren_annuaire(siren)

            if error:
                logger.warning(f"SIREN verification failed: {error}")
                if "not found" in error.lower():
                    all_flags.append(Flag(
                        severity="critical",
                        code="EXTERNAL_SIREN_NOT_FOUND",
                        message=f"SIREN {siren} not found in official registry",
                        details={"siren": siren, "error": error}
                    ))
                else:
                    all_flags.append(Flag(
                        severity="low",
                        code="EXTERNAL_SIREN_VERIFICATION_FAILED",
                        message=f"Could not verify SIREN {siren}: {error}",
                        details={"siren": siren, "error": error}
                    ))
            elif company_info:
                verifications_successful += 1

                # Check if company is closed
                if company_info.status == "closed":
                    all_flags.append(Flag(
                        severity="high",
                        code="EXTERNAL_COMPANY_CLOSED",
                        message=f"Company with SIREN {siren} is closed/inactive",
                        details={
                            "siren": siren,
                            "company_name": company_info.name,
                            "status": company_info.status,
                        }
                    ))

        # Also verify potential SIRENs (XXX XXX XXX patterns with valid Luhn checksum)
        # These are 9-digit patterns that passed checksum but don't have explicit labels
        already_verified = verified_sirens | {s for s, _, _ in sirens}

        for potential_siren, context in potential_sirens:
            if potential_siren in already_verified:
                continue  # Already verified

            verifications_attempted += 1
            company_info, error = verify_siren_annuaire(potential_siren)

            if error and "not found" in error.lower():
                # Pattern looked like SIREN but not in registry - might be something else
                # Don't flag as critical since it wasn't explicitly labeled as SIREN
                logger.debug(f"Potential SIREN {potential_siren} not found: {error}")
            elif company_info:
                verifications_successful += 1
                logger.info(f"Found valid SIREN via pattern matching: {potential_siren} ({company_info.name})")

                if company_info.status == "closed":
                    all_flags.append(Flag(
                        severity="high",
                        code="EXTERNAL_COMPANY_CLOSED",
                        message=f"Company with SIREN {potential_siren} is closed/inactive",
                        details={
                            "siren": potential_siren,
                            "company_name": company_info.name,
                            "status": company_info.status,
                            "detection_method": "pattern_matching",
                        }
                    ))

    # Verify VAT numbers via VIES
    if verify_vat:
        for vat, is_checksum_valid, context in vats:
            if not is_checksum_valid:
                continue  # Already flagged by content module

            verifications_attempted += 1
            result, error = verify_vat_vies(vat)

            if error:
                logger.warning(f"VAT verification failed: {error}")
                all_flags.append(Flag(
                    severity="low",
                    code="EXTERNAL_VAT_VERIFICATION_FAILED",
                    message=f"Could not verify VAT {vat}: {error}",
                    details={"vat": vat, "error": error}
                ))
            elif result:
                if result.get("valid"):
                    verifications_successful += 1
                    # VAT is valid - could also check company name match here
                else:
                    all_flags.append(Flag(
                        severity="critical",
                        code="EXTERNAL_VAT_INVALID",
                        message=f"VAT number {vat} is not valid according to VIES",
                        details={"vat": vat, "vies_response": result}
                    ))

    # Calculate score
    score = 100
    for flag in all_flags:
        score -= SEVERITY_POINTS[flag.severity]
    score = max(0, score)

    # Calculate confidence based on verifications performed
    if verifications_attempted == 0:
        confidence = 0.1  # No verifications possible
    elif verifications_successful == verifications_attempted:
        confidence = 1.0  # All verifications successful
    else:
        confidence = 0.5 + (0.5 * verifications_successful / verifications_attempted)

    return ModuleResult(
        module="external",
        flags=all_flags,
        score=score,
        confidence=confidence,
    )
