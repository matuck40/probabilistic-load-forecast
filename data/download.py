"""Download ONS hourly load-curve CSVs and record their provenance.

The raw files are large and are never committed. This script fetches them into
``data/raw/`` and writes ``data/manifest.json``, which *is* committed, so that a
reader can obtain byte-identical inputs and verify them.

The manifest separates two timestamps that are easy to conflate. ``retrieved_at``
is when these bytes were downloaded and changes only on a real download;
``verified_at`` is when the file on disk was last re-hashed and changes on every
run. Collapsing them into one field would let a routine re-run overwrite the
provenance the manifest exists to preserve.

Source: ONS Dados Abertos, "Curva de Carga Horaria"
        https://dados.ons.org.br/dataset/curva-carga
License: CC BY 4.0 -- attribution to ONS (Operador Nacional do Sistema Eletrico)
        is required by any work that uses this data.

Standard library only, on purpose: obtaining the data must not depend on the
modelling environment being installed correctly.

Usage:
    python data/download.py                    # 2021..2026
    python data/download.py --years 2021 2022
    python data/download.py --force            # re-download and re-hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_PAGE = "https://dados.ons.org.br/dataset/curva-carga"
BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho"
DICTIONARY_JSON = f"{BASE_URL}/DicionarioDados_CurvaCarga.json"
DICTIONARY_PDF = f"{BASE_URL}/DicionarioDados_CurvaCarga.pdf"
LICENSE = "CC BY 4.0"
ATTRIBUTION = "Operador Nacional do Sistema Eletrico (ONS), Dados Abertos"

DEFAULT_YEARS = tuple(range(2021, 2027))

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"

CHUNK_BYTES = 1 << 20  # 1 MiB


def year_url(year: int) -> str:
    return f"{BASE_URL}/CURVA_CARGA_{year}.csv"


def sha256_of(path: Path) -> str:
    """Hash a file in chunks so that memory use does not scale with file size."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


CERT_HELP = r"""
TLS certificate verification failed. This is an environment problem, not a bad
download, and it is common on macOS with a python.org framework build, which
ships without a CA bundle. Fix it in the environment rather than by disabling
verification:

    pip install certifi          # this script picks it up automatically
    # or, for a python.org build:
    /Applications/Python\ 3.x/Install\ Certificates.command
"""


def ssl_context() -> ssl.SSLContext:
    """Return a verifying context, preferring certifi's CA bundle when present.

    Verification is never disabled. An unverified download would defeat the
    point of recording a SHA-256: the hash would attest to whatever an attacker
    served, not to what ONS published.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, destination: Path, context: ssl.SSLContext) -> None:
    """Fetch ``url`` to ``destination`` atomically.

    The bytes land in a temporary file first and are moved into place only on a
    complete read, so an interrupted run cannot leave a truncated file that a
    later run would happily hash and treat as valid.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "probabilistic-load-forecast/0.1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=180, context=context) as response:
                shutil.copyfileobj(response, tmp, length=CHUNK_BYTES)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(destination)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "dataset": "ONS - Curva de Carga Horaria",
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "data_dictionary": {"json": DICTIONARY_JSON, "pdf": DICTIONARY_PDF},
        "columns": {
            "id_subsistema": "subsystem code, 3 characters",
            "nom_subsistema": "subsystem name; its content changed in dictionary v1.2 (2026-04-06), so join on id_subsistema instead",
            "din_instante": "reference timestamp, YYYY-MM-DD HH:MM:SS",
            "val_cargaenergiahomwmed": "load in MWmed (mean power over the hour)",
        },
        "files": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS),
                        help=f"years to download (default: {DEFAULT_YEARS[0]}..{DEFAULT_YEARS[-1]})")
    parser.add_argument("--force", action="store_true", help="re-download even if the file is already present")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    files = manifest.setdefault("files", {})
    context = ssl_context()

    for year in args.years:
        name = f"CURVA_CARGA_{year}.csv"
        destination = RAW_DIR / name
        url = year_url(year)

        if destination.exists() and not args.force:
            print(f"{name}: already present, verifying")
            downloaded = False
        else:
            print(f"{name}: downloading")
            downloaded = True
            try:
                download(url, destination, context)
            except urllib.error.HTTPError as exc:
                print(f"{name}: HTTP {exc.code} at {url} -- skipped", file=sys.stderr)
                continue
            except urllib.error.URLError as exc:
                print(f"{name}: network error {exc.reason} -- skipped", file=sys.stderr)
                if isinstance(exc.reason, ssl.SSLCertVerificationError):
                    print(CERT_HELP, file=sys.stderr)
                    return 2
                continue

        digest = sha256_of(destination)
        recorded = files.get(name, {})
        previous = recorded.get("sha256")
        if previous and previous != digest:
            # ONS revises published data; a changed hash is information, not a failure.
            print(f"{name}: contents changed since the last recorded download")
            downloaded = True  # the bytes on disk are new, whoever put them there

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if downloaded:
            retrieved_at = now
        else:
            # The file was already on disk and only re-hashed. Overwriting
            # retrieved_at here would silently turn "when ONS served these bytes"
            # into "when I last ran the script", which is the whole point of the
            # manifest. A file present but absent from the manifest has no known
            # retrieval time, and null says so rather than inventing one.
            retrieved_at = recorded.get("retrieved_at")
            if retrieved_at is None:
                print(f"{name}: on disk but not in the manifest; retrieval time unknown, use --force")

        files[name] = {
            "url": url,
            "sha256": digest,
            "bytes": destination.stat().st_size,
            "retrieved_at": retrieved_at,
            "verified_at": now,
        }
        print(f"{name}: {destination.stat().st_size:,} bytes  sha256={digest[:16]}...")

    manifest["files"] = dict(sorted(files.items()))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nmanifest written to {MANIFEST_PATH.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
