"""Install one exact Caddy release, verified against its published SHA-512.

Called by start.sh at boot. It is written in Python because Python is the one
tool the running container is sure to have: the site itself runs on it. The
shell version of this step needed curl, which was missing from the runtime
image, so the container exited on every start (staging, 2026-11-11).

Usage:
    python3 scripts/install_caddy.py 2.11.9 <sha512> /usr/local/bin/caddy-2.11.9
"""

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

import requests

RELEASE_URL = (
    "https://github.com/caddyserver/caddy/releases/download/"
    "v{version}/caddy_{version}_linux_amd64.tar.gz"
)
DOWNLOAD_TIMEOUT_SECONDS = 120


def install_from_archive(archive: bytes, expected_sha512: str, destination: Path) -> None:
    """Check a release archive, then write its `caddy` binary to destination.

    The binary is written under a temporary name and renamed at the end, so an
    interrupted install never leaves a half-written executable at destination.

    Args:
        archive: The bytes of the release's .tar.gz.
        expected_sha512: The checksum published with the release, in hex.
        destination: Where the executable goes, e.g. /usr/local/bin/caddy-2.11.9.

    Raises:
        ValueError: The archive is not byte for byte the expected release.
        KeyError: The archive holds no file named `caddy`.
    """
    actual_sha512 = hashlib.sha512(archive).hexdigest()
    if actual_sha512 != expected_sha512:
        raise ValueError(f"checksum mismatch: got {actual_sha512}")

    temporary = destination.with_name(destination.name + ".tmp")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as release:
        # Read the one file we want by name rather than extracting the whole
        # archive, so no path inside it decides where anything is written.
        binary = release.extractfile("caddy")
        if binary is None:
            raise KeyError("caddy is not a regular file in the archive")
        temporary.write_bytes(binary.read())

    temporary.chmod(0o755)
    os.replace(temporary, destination)


def main(version: str, expected_sha512: str, destination: str) -> None:
    """Download the release and install it.

    Args:
        version: The Caddy version, e.g. "2.11.9".
        expected_sha512: The checksum published with the release, in hex.
        destination: Where the executable goes.

    Raises:
        requests.RequestException: The download failed.
    """
    url = RELEASE_URL.format(version=version)
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    install_from_archive(response.content, expected_sha512, Path(destination))


if __name__ == "__main__":
    main(*sys.argv[1:])
