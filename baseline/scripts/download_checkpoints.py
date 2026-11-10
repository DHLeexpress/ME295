"""Mirror openpi checkpoints from the public bucket gs://openpi-assets into openpi's cache.

openpi's own downloader (openpi.shared.download.maybe_download) fetches one object at a time; this script fetches
the objects of a checkpoint in parallel over HTTPS (anonymous, stdlib only) into the layout maybe_download expects:

    $OPENPI_DATA_HOME (default ~/.cache/openpi) / openpi-assets / checkpoints / <name> / ...

so openpi later finds them in its cache and does not download again.

    python baseline/scripts/download_checkpoints.py pi05_droid pi05_base
    python baseline/scripts/download_checkpoints.py --list pi05_base       # objects and sizes only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "openpi-assets"
API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}&pageToken={token}"
MEDIA = "https://storage.googleapis.com/{bucket}/{name}"


def cache_dir() -> Path:
    return Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()


def list_objects(prefix: str) -> list[dict]:
    items, token = [], ""
    while True:
        url = API.format(bucket=BUCKET, prefix=urllib.parse.quote(prefix, safe=""), token=token)
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        items += page.get("items", [])
        token = page.get("nextPageToken", "")
        if not token:
            return items


def fetch(obj: dict, root: Path, retries: int = 4) -> int:
    dst = root / BUCKET / obj["name"]
    size = int(obj["size"])
    if obj["name"].endswith("/"):  # folder placeholder object
        dst.mkdir(parents=True, exist_ok=True)
        return 0
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    url = MEDIA.format(bucket=BUCKET, name=urllib.parse.quote(obj["name"]))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=8 << 20)
            if tmp.stat().st_size != size:
                raise OSError(f"size mismatch {tmp.stat().st_size} != {size}")
            tmp.replace(dst)
            return size
        except OSError as e:  # urllib errors are OSErrors
            if attempt == retries - 1:
                raise
            print(f"  retry {obj['name']}: {e}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+", help="checkpoint names, e.g. pi05_droid pi05_base")
    ap.add_argument("--list", action="store_true", help="only list objects and sizes")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = cache_dir()
    for name in args.names:
        objs = list_objects(f"checkpoints/{name}/")
        total = sum(int(o["size"]) for o in objs)
        print(f"{name}: {len(objs)} objects, {total / 1e9:.2f} GB -> {root / BUCKET / 'checkpoints' / name}")
        if args.list:
            for o in objs:
                print(f"  {int(o['size']) / 1e6:10.1f} MB  {o['name']}")
            continue
        free = shutil.disk_usage(root.parent if not root.exists() else root).free
        if free < 1.2 * total:
            sys.exit(f"not enough disk: {free / 1e9:.1f} GB free, need ~{1.2 * total / 1e9:.1f} GB")
        done = 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for n in ex.map(lambda o: fetch(o, root), objs):
                done += n
        print(f"{name}: done ({done / 1e9:.2f} GB fetched now)")


if __name__ == "__main__":
    main()
