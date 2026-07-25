"""Rebuild a run's result file from its per-fold log lines.

The training loop prints one line per held-out animal and only afterwards computes the
animal-level statistics. When those statistics were the thing that broke, the expensive
part is already done and sitting in the log, so recover it from there instead of
spending another hour on the folds.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cadence import metrics as M

LINE = re.compile(
    r"^\s*\[([^\]]+)\].*operator=([+-][\d.]+) group=([+-][\d.]+) blend=([+-][\d.]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    animals, vals = [], {"operator": [], "group": [], "blend": []}
    for line in args.log.read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        animals.append(m.group(1))
        vals["operator"].append(float(m.group(2)))
        vals["group"].append(float(m.group(3)))
        vals["blend"].append(float(m.group(4)))
    if not animals:
        print("no fold lines found")
        return 1

    res = {}
    print(f"{len(animals)} animals recovered from {args.log}")
    print(f"\n{'method':12s} {'dR2':>7s} {'95% CI':>18s} {'animals>0':>10s} {'p':>10s}")
    print("-" * 62)
    for k in ("group", "operator", "blend"):
        rep = M.animal_level_report(vals[k], animals)
        res[k] = rep
        print(f"{k:12s} {rep['animal_mean']:+7.3f} "
              f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}]".rjust(19) +
              f" {rep['sign_test']['n_positive']:>4d}/{rep['sign_test']['n']:<4d} "
              f"{rep['permutation']['p']:10.2e}")
    for name in ("operator", "blend"):
        ks = [k for k in res[name]["per_animal"] if k in res["group"]["per_animal"]]
        t = M.animal_permutation_test([res[name]["per_animal"][k] for k in ks],
                                      [res["group"]["per_animal"][k] for k in ks])
        res[f"test_{name}_vs_group"] = t
        print(f"{name:9s} vs group average: diff={t['mean_diff']:+.3f} "
              f"p={t['p']:.2e} (n={t['n']}, "
              f"{'exact' if t['exact'] else str(t['n_draws']) + ' draws'})")
    res["recovered_from_log"] = str(args.log)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
