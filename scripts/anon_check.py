"""Anonymity check for the Phase 1 submission.

python scripts/anon_check.py [--banned .anon_banned.txt] [--allow TEXT ...] [--no-history]

Scans every tracked and untracked-but-not-ignored file, and the full git history, for:
  - terms from a banned-terms file (names, school, branch, city ...; one per line,
    "#" comments; whole-word, case-insensitive). Keep that file OUT of the repo
    (it is listed in .gitignore).
  - email addresses (except example.com/org, GitHub noreply, and --allow values)
  - personal paths: /home/<name>, /Users/<name>, C:\\Users\\<name>
  - notebook outputs (.ipynb cells with outputs or execution counts)
  - author metadata in PNG (tEXt/iTXt "Author"), PDF (/Author, dc:creator) and
    DOCX/XLSX/PPTX (dc:creator, cp:lastModifiedBy)
It also lists every git author/committer identity for a manual look (a banned term in an
identity counts as a hit). Exit code 1 on any hit, printing the file and line.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANNED = ROOT / ".anon_banned.txt"
ALLOWED_EMAIL_PARTS = ("example.com", "example.org", "users.noreply.github.com")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
USER_PATHS = (
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+", re.IGNORECASE),
)
BINARY_SUFFIXES = {".parquet", ".joblib", ".pkl", ".zip", ".gz", ".jpg", ".jpeg", ".gif", ".ico", ".npy"}
OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}
MAX_TEXT_BYTES = 20 * 2 ** 20

Hit = tuple[str, int, str, str]  # (where, line, kind, snippet)


def load_banned(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def banned_patterns(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    return [(t, re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", re.IGNORECASE)) for t in terms]


def _email_allowed(email: str, allow: list[str]) -> bool:
    low = email.lower()
    return any(part in low for part in ALLOWED_EMAIL_PARTS) or any(a.lower() in low for a in allow)


def scan_text(where: str, text: str, banned: list[tuple[str, re.Pattern]], allow: list[str]) -> list[Hit]:
    """Banned terms, email addresses and personal paths, line by line."""
    hits: list[Hit] = []
    for no, line in enumerate(text.splitlines(), start=1):
        snippet = line.strip()[:100]
        for term, pattern in banned:
            if pattern.search(line):
                hits.append((where, no, f"banned term '{term}'", snippet))
        for email in EMAIL.findall(line):
            if not _email_allowed(email, allow):
                hits.append((where, no, "email address", snippet))
        for pattern in USER_PATHS:
            if pattern.search(line) and not any(a.lower() in line.lower() for a in allow):
                hits.append((where, no, "personal path", snippet))
    return hits


def check_notebook(where: str, text: str) -> list[Hit]:
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("outputs") or cell.get("execution_count"):
            return [(where, 0, "notebook outputs", f"cell {i} has outputs; clear them before committing")]
    return []


def png_text_chunks(data: bytes) -> dict[str, str]:
    """tEXt / iTXt key -> value from a PNG file (no CRC check needed for this purpose)."""
    out: dict[str, str] = {}
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return out
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"tEXt" and b"\x00" in body:
            key, value = body.split(b"\x00", 1)
            out[key.decode("latin-1")] = value.decode("latin-1", "replace")
        elif kind == b"iTXt" and b"\x00" in body:
            key, rest = body.split(b"\x00", 1)
            out[key.decode("latin-1")] = rest[2:].split(b"\x00", 2)[-1].decode("utf-8", "replace")
        elif kind == b"IEND":
            break
        pos += 12 + length
    return out


def check_binary_metadata(where: str, path: Path, banned, allow) -> list[Hit]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    hits: list[Hit] = []
    if suffix == ".png":
        for key, value in png_text_chunks(data).items():
            if key.lower() == "author" and value.strip():
                hits.append((where, 0, "PNG author metadata", f"{key}={value[:60]}"))
            hits += scan_text(f"{where} [PNG {key}]", value, banned, allow)
    elif suffix == ".pdf":
        for m in re.finditer(rb"/Author\s*\(([^)]*)\)", data):
            if m.group(1).strip():
                hits.append((where, 0, "PDF author metadata", m.group(1)[:60].decode("latin-1")))
        for m in re.finditer(rb"<dc:creator>(.*?)</dc:creator>", data, re.DOTALL):
            if re.sub(rb"<[^>]+>|\s", b"", m.group(1)):
                hits.append((where, 0, "PDF creator metadata", m.group(1)[:60].decode("utf-8", "replace")))
    elif suffix in OFFICE_SUFFIXES:
        try:
            with zipfile.ZipFile(path) as zf:
                core = zf.read("docProps/core.xml").decode("utf-8", "replace")
        except (KeyError, zipfile.BadZipFile):
            return hits
        for tag in ("dc:creator", "cp:lastModifiedBy"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", core, re.DOTALL)
            if m and m.group(1).strip():
                hits.append((where, 0, f"{path.suffix} {tag} metadata", m.group(1).strip()[:60]))
    return hits


def repo_files() -> list[str]:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT,
                               capture_output=True, text=True).stdout.splitlines()
    return sorted(set(tracked) | set(untracked))


def scan_files(files: list[str], banned, allow, banned_file: Path | None) -> list[Hit]:
    hits: list[Hit] = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        if banned_file is not None and path.resolve() == banned_file.resolve():
            hits.append((rel, 0, "banned-terms file is in the repo", "remove it and keep it in .gitignore"))
            continue
        suffix = path.suffix.lower()
        if suffix in BINARY_SUFFIXES or path.stat().st_size > MAX_TEXT_BYTES:
            continue
        if suffix in {".png", ".pdf"} | OFFICE_SUFFIXES:
            hits += check_binary_metadata(rel, path, banned, allow)
            continue
        raw = path.read_bytes()
        if b"\x00" in raw[:8000]:
            continue
        text = raw.decode("utf-8", "replace")
        hits += scan_text(rel, text, banned, allow)
        if suffix == ".ipynb":
            hits += check_notebook(rel, text)
    return hits


def scan_history(banned, allow) -> tuple[list[Hit], list[str]]:
    """Banned terms, emails and paths in all commits (messages + diffs); identities listed."""
    idents = subprocess.run(["git", "log", "--all", "--format=%an <%ae> | %cn <%ce>"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    identities = sorted(set(idents))
    hits: list[Hit] = []
    for ident in identities:
        for term, pattern in banned:
            if pattern.search(ident):
                hits.append(("git identity", 0, f"banned term '{term}'", ident))
    log = subprocess.run(["git", "log", "--all", "-p", "--no-color", "--no-ext-diff"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    commit = "?"
    content: list[str] = []
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1][:10]
            continue
        if line.startswith(("Author:", "Date:", "Merge:")):
            continue
        content.append(f"{commit}\x00{line}")
    for item in content:
        sha, line = item.split("\x00", 1)
        for _, _, kind, snippet in scan_text("h", line, banned, allow):
            hits.append((f"git history (commit {sha})", 0, kind, snippet))
    return hits, identities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 anonymity check.")
    parser.add_argument("--banned", type=Path, default=DEFAULT_BANNED, help="file with banned terms, one per line")
    parser.add_argument("--allow", action="append", default=[], help="text that may appear (e.g. the team alias email)")
    parser.add_argument("--no-history", action="store_true", help="skip the git history scan")
    args = parser.parse_args(argv)

    terms = load_banned(args.banned)
    if not terms:
        print(f"note: no banned terms loaded (looked for {args.banned.name}); names will not be checked")
    banned = banned_patterns(terms)
    files = repo_files()
    hits = scan_files(files, banned, args.allow, args.banned if args.banned.exists() else None)
    identities: list[str] = []
    if not args.no_history:
        history_hits, identities = scan_history(banned, args.allow)
        hits += history_hits

    for where, line, kind, snippet in hits:
        loc = f"{where}:{line}" if line else where
        print(f"{loc}: {kind}: {snippet}")
    if identities:
        print("\ngit identities in history (check by eye):")
        for ident in identities:
            print(f"  {ident}")
    print(f"\nscanned {len(files)} files{'' if args.no_history else ' and the git history'}, "
          f"{len(terms)} banned terms: {len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())