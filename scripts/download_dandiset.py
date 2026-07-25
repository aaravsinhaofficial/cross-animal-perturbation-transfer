"""Generic DANDI asset downloader with a checksummed manifest.

Usage
-----
    python scripts/download_dandiset.py 000011 --out data/raw/dandi000011 \
        --include ogen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.dandiarchive.org/api"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=180) as fh:
        return json.load(fh)


def list_assets(dandiset: str, version: str = "draft") -> list[dict]:
    assets: list[dict] = []
    url = f"{API}/dandisets/{dandiset}/versions/{version}/assets/?page_size=400"
    while url:
        page = fetch_json(url)
        assets.extend(page["results"])
        url = page.get("next")
    return assets


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_one(asset: dict, out_root: Path) -> tuple[str, str]:
    dest = out_root / asset["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == asset["size"]:
        return asset["path"], "cached"
    url = f"{API}/assets/{asset['asset_id']}/download/"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=3600) as src, tmp.open("wb") as fh:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            if tmp.stat().st_size == asset["size"]:
                tmp.rename(dest)
                return asset["path"], "downloaded"
        except Exception:  # pragma: no cover - network retry
            if attempt == 3:
                raise
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed {asset['path']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dandiset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include", default=None, help="substring filter on asset path")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    assets = list_assets(args.dandiset)
    if args.include:
        assets = [a for a in assets if args.include in a["path"]]
    print(f"{len(assets)} assets, {sum(a['size'] for a in assets)/1e9:.2f} GB", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, a, args.out): a for a in assets}
        for fut in as_completed(futs):
            path, status = fut.result()
            done += 1
            print(f"[{done}/{len(assets)}] {status:10s} {path}", flush=True)

    manifest = [
        {"path": a["path"], "asset_id": a["asset_id"], "size": a["size"],
         "sha256": sha256(args.out / a["path"])}
        for a in sorted(assets, key=lambda x: x["path"])
    ]
    mpath = args.out.parent / f"dandi{args.dandiset}_manifest.json"
    mpath.write_text(json.dumps({"dandiset": args.dandiset, "assets": manifest}, indent=1))
    print(f"wrote manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
