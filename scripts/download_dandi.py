"""Download the DANDI:001868 NWB assets used by this project.

Dataset: "Chronic electrophysiology and two-photon calcium imaging of mouse
primary somatosensory cortex during intracortical microstimulation learning"
(DANDI:001868, CC-BY-4.0).

Only the six task-trained mice that carry simultaneous behaviour + ecephys are
required for the cross-animal transfer benchmark; the earlier anaesthetised
cohort (ICMS43/45/48/54/56/57) has no behavioural readout and is skipped by
default.

Usage
-----
    python scripts/download_dandi.py --out data/raw/dandi001868
    python scripts/download_dandi.py --out data/raw/dandi001868 --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DANDISET = "001868"
VERSION = "draft"
API = "https://api.dandiarchive.org/api"

# Mice with simultaneous behaviour, electrophysiology and (mostly) imaging.
BEHAVIOUR_SUBJECTS = (
    "sub-ICMS83",
    "sub-ICMS92",
    "sub-ICMS93",
    "sub-ICMS98",
    "sub-ICMS100",
    "sub-ICMS101",
)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as fh:
        return json.load(fh)


def list_assets() -> list[dict]:
    assets: list[dict] = []
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=200"
    while url:
        page = fetch_json(url)
        assets.extend(page["results"])
        url = page.get("next")
    return assets


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_one(asset: dict, out_root: Path) -> tuple[str, str]:
    dest = out_root / asset["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == asset["size"]:
        return asset["path"], "cached"
    url = f"{API}/assets/{asset['asset_id']}/download/"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=1800) as src, tmp.open("wb") as fh:
        while True:
            block = src.read(1 << 22)
            if not block:
                break
            fh.write(block)
    if tmp.stat().st_size != asset["size"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {asset['path']}")
    tmp.rename(dest)
    return asset["path"], "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/dandi001868", type=Path)
    ap.add_argument("--all", action="store_true", help="include the anaesthetised cohort")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--manifest", default="data/raw/dandi001868_manifest.json", type=Path)
    args = ap.parse_args()

    assets = list_assets()
    if not args.all:
        assets = [
            a
            for a in assets
            if a["path"].split("/")[0] in BEHAVIOUR_SUBJECTS and "behavior" in a["path"]
        ]
    total_gb = sum(a["size"] for a in assets) / 1e9
    print(f"{len(assets)} assets, {total_gb:.2f} GB", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, a, args.out): a for a in assets}
        for fut in as_completed(futures):
            path, status = fut.result()
            done += 1
            print(f"[{done}/{len(assets)}] {status:10s} {path}", flush=True)

    manifest = []
    for a in sorted(assets, key=lambda x: x["path"]):
        dest = args.out / a["path"]
        manifest.append(
            {
                "path": a["path"],
                "asset_id": a["asset_id"],
                "size": a["size"],
                "sha256": sha256(dest),
            }
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"dandiset": DANDISET, "assets": manifest}, indent=1))
    print(f"wrote manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
