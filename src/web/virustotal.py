"""VirusTotal lookup: is this file hash known to antivirus engines?"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_virustotal(sha256: str, api_key: str | None) -> dict | None:
    """
    Look up a file hash on VirusTotal (v3 API).

    Only the hash is sent, never the document. One network call, 10 s timeout.

    Args:
        sha256: The SHA-256 of the file.
        api_key: VirusTotal API key; empty or None skips the lookup.

    Returns:
        dict with keys:
            - "status": "clean" | "malicious" | "unknown" (never scanned)
            - "malicious": number of engines that flagged the file
            - "total": total number of engines
            - "link": URL of the VirusTotal report
        or None when there is no key or the call failed.
    """
    if not api_key:
        return None
    try:
        import requests

        resp = requests.get(
            f"https://www.virustotal.com/api/v3/files/{sha256}",
            headers={"x-apikey": api_key},
            timeout=10,
        )
        link = f"https://www.virustotal.com/gui/file/{sha256}"

        if resp.status_code == 404:
            # File was never scanned on VirusTotal
            return {"status": "unknown", "malicious": 0, "total": 0, "link": link}
        if resp.status_code != 200:
            return None

        stats = resp.json()["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0) + stats.get("suspicious", 0)
        total = sum(stats.values())
        if malicious > 0:
            return {"status": "malicious", "malicious": malicious, "total": total, "link": link}
        return {"status": "clean", "malicious": 0, "total": total, "link": link}
    except Exception:
        logger.warning("VirusTotal lookup failed", exc_info=True)
        return None
