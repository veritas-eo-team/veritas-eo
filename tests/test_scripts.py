"""Tests for scripts/anon_check.py (offline, temporary files)."""
from __future__ import annotations

import importlib.util
import json
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def anon():
    return load_script("anon_check")


def kinds(hits) -> list[str]:
    return [h[2] for h in hits]


def test_scan_text_finds_terms_emails_and_paths(anon):
    banned = anon.banned_patterns(["Alice", "Kr"])
    text = "\n".join([
        "written by alice",                    # banned term, case-insensitive
        "Kraken is fine",                      # whole words only: no hit
        "contact: someone@school.tn",         # email
        "safe: bot@users.noreply.github.com",  # allowed
        "path C:\\Users\\bob\\data",           # personal path
        "linux /home/carol/x",
    ])
    hits = anon.scan_text("f.txt", text, banned, [])
    found = kinds(hits)
    assert found.count("banned term 'Alice'") == 1
    assert "banned term 'Kr'" not in found
    assert found.count("email address") == 1
    assert found.count("personal path") == 2
    assert {h[1] for h in hits} == {1, 3, 5, 6}


def test_allow_list(anon):
    hits = anon.scan_text("f", "team: alias@team-mail.org", [], ["alias@team-mail.org"])
    assert hits == []


def test_notebook_outputs(anon):
    dirty = json.dumps({"cells": [{"cell_type": "code", "outputs": [{"text": "x"}], "execution_count": 3}]})
    clean = json.dumps({"cells": [{"cell_type": "code", "outputs": [], "execution_count": None}]})
    assert kinds(anon.check_notebook("a.ipynb", dirty)) == ["notebook outputs"]
    assert anon.check_notebook("b.ipynb", clean) == []


def _png_with_text(key: bytes, value: bytes) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"tEXt", key + b"\x00" + value) + chunk(b"IEND", b"")


def test_binary_metadata(anon, tmp_path):
    png = tmp_path / "chart.png"
    png.write_bytes(_png_with_text(b"Author", b"Somebody"))
    assert "PNG author metadata" in kinds(anon.check_binary_metadata("chart.png", png, [], []))
    plain = tmp_path / "plain.png"
    plain.write_bytes(_png_with_text(b"Software", b"Matplotlib"))
    assert anon.check_binary_metadata("plain.png", plain, [], []) == []

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4\n<< /Author (Somebody) /Creator (tool) >>\n%%EOF")
    assert "PDF author metadata" in kinds(anon.check_binary_metadata("r.pdf", pdf, [], []))

    docx = tmp_path / "r.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("docProps/core.xml", "<cp:coreProperties><dc:creator>Somebody</dc:creator>"
                                         "<cp:lastModifiedBy></cp:lastModifiedBy></cp:coreProperties>")
    assert kinds(anon.check_binary_metadata("r.docx", docx, [], [])) == [".docx dc:creator metadata"]


def test_load_banned_skips_comments(anon, tmp_path):
    f = tmp_path / "banned.txt"
    f.write_text("# names\nAlice\n\n  Bob  \n", encoding="utf-8")
    assert anon.load_banned(f) == ["Alice", "Bob"]
    assert anon.load_banned(tmp_path / "missing.txt") == []