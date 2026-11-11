"""Tests for scripts/install_caddy.py, the boot-time Caddy installer."""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from scripts.install_caddy import install_from_archive


def _release_archive(name: str, content: bytes) -> bytes:
    """Build an in-memory .tar.gz holding one file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


def test_installs_an_executable_binary(tmp_path: Path) -> None:
    archive = _release_archive("caddy", b"binary")
    destination = tmp_path / "caddy-1.0"

    install_from_archive(archive, hashlib.sha512(archive).hexdigest(), destination)

    assert destination.read_bytes() == b"binary"
    assert os.access(destination, os.X_OK)


def test_refuses_an_archive_with_the_wrong_checksum(tmp_path: Path) -> None:
    archive = _release_archive("caddy", b"tampered")
    destination = tmp_path / "caddy-1.0"

    with pytest.raises(ValueError, match="checksum mismatch"):
        install_from_archive(archive, "0" * 128, destination)

    assert list(tmp_path.iterdir()) == []


def test_refuses_an_archive_without_caddy(tmp_path: Path) -> None:
    archive = _release_archive("README.md", b"not the binary")
    destination = tmp_path / "caddy-1.0"

    with pytest.raises(KeyError):
        install_from_archive(archive, hashlib.sha512(archive).hexdigest(), destination)

    assert not destination.exists()
