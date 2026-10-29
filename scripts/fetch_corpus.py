"""Download the public documents listed in data/corpus_manifest.yaml into data/raw/.

Usage: python scripts/fetch_corpus.py [--manifest data/corpus_manifest.yaml] [--out data/raw]

Skips files that already exist. With an empty manifest it does nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "corpus_manifest.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    entries = (yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}).get("documents") or []
    if not entries:
        print(f"{args.manifest} lists no documents; nothing to fetch.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for entry in entries:
        url = entry["url"]
        name = entry.get("filename") or Path(urllib.parse.urlparse(url).path).name
        if not name or Path(name).name != name:
            print(f"skip {url}: unusable filename {name!r}", file=sys.stderr)
            failures += 1
            continue
        target = args.out / name
        if target.exists():
            print(f"exists  {target}")
            continue
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - manifest is operator-controlled
            data = response.read()
        expected = entry.get("sha256")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            print(f"FAILED  {url}: sha256 mismatch", file=sys.stderr)
            failures += 1
            continue
        target.write_bytes(data)
        print(f"fetched {target} ({len(data)} bytes)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
