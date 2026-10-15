#!/usr/bin/env python3
"""Download and SHA-256 verify the selected PAST, JOME, and ARISE checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry: dict, output: Path) -> None:
    destination = output / Path(entry["checkpoint"]).name
    expected = entry["sha256"]
    if destination.is_file():
        if sha256(destination) != expected:
            raise RuntimeError(f"Existing checkpoint SHA-256 mismatch: {destination}")
        print(f"Verified existing checkpoint: {destination}", flush=True)
        return
    url = entry["release_asset_url"]
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Downloaded checkpoint SHA-256 mismatch: {url}")
        os.replace(temporary, destination)
        print(f"Downloaded and verified checkpoint: {destination}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "model/MODEL_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=REPO / "checkpoints")
    parser.add_argument("--model", choices=["past", "jome", "arise", "all"], default="all",
                        help="Add-ons also download their required PAST anchor")
    args = parser.parse_args()
    needed = {"past", "jome", "arise"} if args.model == "all" else {"past", args.model}
    entries = {entry["model"]: entry for entry in json.loads(args.manifest.read_text())["models"]}
    for name in ("past", "jome", "arise"):
        if name in needed:
            download(entries[name], args.output)


if __name__ == "__main__":
    main()
