"""Tests for Python <3.12 tarfile.extractall fallback in pm/store.py.

On Python <3.12, tarfile.extractall() does not accept a `filter` argument.
The code in pm/store.py._tar_filter implements the same logic manually, but
the call site must catch TypeError and fall back to a manual getmembers loop
with commonpath containment checks.

These tests verify that fallback path by monkeypatching tarfile to simulate
Python <3.12 behavior.
"""

import io
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_tar(members):
    """Create a tar.gz archive in memory with the given members.

    members: list of (name, content_bytes, is_symlink, linkname)
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for member in members:
            name = member[0]
            content = member[1] if len(member) > 1 and member[1] is not None else b""
            is_symlink = member[2] if len(member) > 2 else False
            linkname = member[3] if len(member) > 3 else None

            info = tarfile.TarInfo(name=name)
            if is_symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = linkname
                info.size = 0
                tf.addfile(info)
            else:
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf


def test_extract_tar_uses_data_filter_on_python_312_plus():
    """On Python >=3.12, extract_tar should use tarfile.data_filter directly."""
    from pm.store import extract_tar

    # Create a simple tar with one file
    archive = _make_tar([("hello.txt", b"world")])
    dest = Path("/tmp/test_extract_312")
    dest.mkdir(parents=True, exist_ok=True)

    # This should work without error on Python >=3.12
    extract_tar(archive, dest)
    assert (dest / "hello.txt").read_text() == "world"


def test_extract_tar_fallback_on_python_below_312(monkeypatch):
    """On Python <3.12, extract_tar should fall back to manual extraction."""
    from pm.store import extract_tar

    # Create a tar with a regular file and a symlink
    archive = _make_tar([
        ("python/share/terminfo/a/adm1178", b"terminfo data"),
        ("python/share/terminfo/1/1178", None, True, "../a/adm1178"),
    ])
    dest = Path("/tmp/test_extract_fallback")
    dest.mkdir(parents=True, exist_ok=True)

    # Monkeypatch tarfile.open to simulate Python <3.12 behavior
    # where extractall doesn't accept filter parameter
    original_open = tarfile.open

    def patched_open(*args, **kwargs):
        tf = original_open(*args, **kwargs)
        original_extractall = tf.extractall

        def extractall(path=".", members=None, *, filter=None):
            # Simulate Python <3.12: ignore filter, use old behavior
            if filter is not None:
                raise TypeError("extractall() does not accept a 'filter' argument")
            return original_extractall(path, members)

        tf.extractall = extractall
        return tf

    monkeypatch.setattr(tarfile, "open", patched_open)

    # This should fall back to manual extraction without error
    extract_tar(archive, dest)

    # Verify the files were extracted correctly
    assert (dest / "python/share/terminfo/a/adm1178").read_text() == "terminfo data"
    link = dest / "python/share/terminfo/1/1178"
    assert link.is_symlink()
    assert os.readlink(link) == "../a/adm1178"


def test_extract_tar_fallback_rejects_escaping_symlinks(monkeypatch):
    """The fallback path should still reject symlinks that escape the destination."""
    from pm.store import extract_tar

    # Create a tar with an escaping symlink
    archive = _make_tar([
        ("python/bin/evil", None, True, "../../../etc/passwd"),
    ])
    dest = Path("/tmp/test_extract_fallback_safe")
    dest.mkdir(parents=True, exist_ok=True)

    # Monkeypatch tarfile.open to simulate Python <3.12
    original_open = tarfile.open

    def patched_open(*args, **kwargs):
        tf = original_open(*args, **kwargs)
        original_extractall = tf.extractall

        def extractall(path=".", members=None, *, filter=None):
            if filter is not None:
                raise TypeError("extractall() does not accept a 'filter' argument")
            return original_extractall(path, members)

        tf.extractall = extractall
        return tf

    monkeypatch.setattr(tarfile, "open", patched_open)

    # This should raise FilterError even in fallback mode
    with pytest.raises(tarfile.FilterError):
        extract_tar(archive, dest)

    # Verify the evil symlink was not created
    assert not (dest / "python/bin/evil").exists()


def test_extract_tar_fallback_rejects_absolute_symlinks(monkeypatch):
    """The fallback path should reject absolute symlinks."""
    from pm.store import extract_tar

    # Create a tar with an absolute symlink
    archive = _make_tar([
        ("python/bin/evil", None, True, "/etc/passwd"),
    ])
    dest = Path("/tmp/test_extract_fallback_abs")
    dest.mkdir(parents=True, exist_ok=True)

    # Monkeypatch tarfile.open to simulate Python <3.12
    original_open = tarfile.open

    def patched_open(*args, **kwargs):
        tf = original_open(*args, **kwargs)
        original_extractall = tf.extractall

        def extractall(path=".", members=None, *, filter=None):
            if filter is not None:
                raise TypeError("extractall() does not accept a 'filter' argument")
            return original_extractall(path, members)

        tf.extractall = extractall
        return tf

    monkeypatch.setattr(tarfile, "open", patched_open)

    # This should raise FilterError even in fallback mode
    with pytest.raises(tarfile.FilterError):
        extract_tar(archive, dest)

    # Verify the evil symlink was not created
    assert not (dest / "python/bin/evil").exists()


def test_tar_filter_resolves_symlink_targets_from_own_directory():
    """_tar_filter should resolve relative symlink targets from the link's directory."""
    from pm.store import _tar_filter

    # Simulate a tar member with a relative symlink
    member = tarfile.TarInfo(name="python/share/terminfo/1/1178")
    member.type = tarfile.SYMTYPE
    member.linkname = "../a/adm1178"

    dest = "/tmp/test_tar_filter"
    os.makedirs(dest, exist_ok=True)

    # This should not raise and should return the member
    result = _tar_filter(member, dest)
    assert result is not None
    assert result.linkname == "../a/adm1178"


def test_tar_filter_rejects_absolute_symlinks():
    """_tar_filter should reject absolute symlinks."""
    from pm.store import _tar_filter

    member = tarfile.TarInfo(name="python/bin/evil")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"

    dest = "/tmp/test_tar_filter_abs"
    os.makedirs(dest, exist_ok=True)

    with pytest.raises(tarfile.AbsoluteLinkError):
        _tar_filter(member, dest)


def test_tar_filter_rejects_symlinks_escaping_destination():
    """_tar_filter should reject symlinks that escape the destination."""
    from pm.store import _tar_filter

    member = tarfile.TarInfo(name="python/bin/evil")
    member.type = tarfile.SYMTYPE
    member.linkname = "../../../etc/passwd"

    dest = "/tmp/test_tar_filter_escape"
    os.makedirs(dest, exist_ok=True)

    with pytest.raises(tarfile.LinkOutsideDestinationError):
        _tar_filter(member, dest)
