"""
aggregate_lowshot.py — Collect k-shot results into summary tables.

Reads test_results.json files produced by eval_lowshot.py,
aggregates over label seeds x training seeds, and prints
mean +/- std in a format ready for LaTeX tables.

Usage:
  python scripts/aggregate_lowshot.py --input_dir output/lowshot
  python scripts/aggregate_lowshot.py --input_dir output/lowshot --csv results.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description="Aggregate k-shot results into summary tables"
    )
    p.add_argument("--input_dir", type=str, required=True,
                   help="Base output directory (e.g. output/lowshot)")
    p.add_argument("--metric", type=str, default="test_acc",
                   help="Metric to aggregate (default: test_acc)")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional: save full results to CSV")
    args = p.parse_args()

    base = Path(args.input_dir)
    if not base.exists():
        print(f"ERROR: Directory not found: {base}")
        return

    # Collect all results
    all_results = []
    for json_path in sorted(base.rglob("test_results.json")):
        with open(json_path) as f:
            r = json.load(f)
        all_results.append(r)

    if not all_results:
        print(f"No test_results.json files found in {base}")
        return

    print(f"Found {len(all_results)} result files\n")

    # Group by (method, dataset, mode, k_shot)
    groups = defaultdict(list)
    for r in all_results:
        key = (
            r.get("method_tag", "unknown"),
            r.get("dataset", "unknown"),
            r.get("mode", "unknown"),
            r.get("k_shot", 0),
        )
        groups[key].append(r)

    # ── Print tables per (mode, k_shot) ───────────────────────────────
    # Organize: mode → k_shot → dataset → method → values
    tables = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for (method, dataset, mode, k_shot), results in sorted(groups.items()):
        vals = [r[args.metric] for r in results if r.get(args.metric) is not None]
        if not vals:
            continue
        arr = np.array(vals)
        # Convert to percentage if metric is accuracy-like and values are in [0, 1]
        if arr.max() <= 1.0 and "acc" in args.metric:
            arr = arr * 100
        tables[mode][k_shot][dataset][method] = (arr.mean(), arr.std(), len(arr))

    # Collect all methods and datasets
    all_methods = sorted(set(r.get("method_tag", "unknown") for r in all_results))
    all_datasets = sorted(set(r.get("dataset", "unknown") for r in all_results))

    for mode in sorted(tables.keys()):
        for k_shot in sorted(tables[mode].keys()):
            print(f"{'=' * 72}")
            print(f"  Mode: {mode}   |   {k_shot}-shot   |   Metric: {args.metric}")
            print(f"{'=' * 72}")

            # Header
            header = f"{'Method':<15}"
            for ds in all_datasets:
                header += f"  {ds:>14}"
            print(header)
            print("-" * len(header))

            for method in all_methods:
                row = f"{method:<15}"
                for ds in all_datasets:
                    entry = tables[mode][k_shot].get(ds, {}).get(method)
                    if entry is None:
                        row += f"  {'--':>14}"
                    else:
                        mean, std, n = entry
                        row += f"  {mean:5.1f}±{std:4.1f} ({n:d})"
                print(row)

            print()

    # ── LaTeX snippet ────────────────────────────────────────────────
    print("=" * 72)
    print("LaTeX-ready values (mean ± std):")
    print("=" * 72)
    for mode in sorted(tables.keys()):
        for k_shot in sorted(tables[mode].keys()):
            print(f"\n% {mode}, {k_shot}-shot")
            for method in all_methods:
                parts = []
                for ds in all_datasets:
                    entry = tables[mode][k_shot].get(ds, {}).get(method)
                    if entry is None:
                        parts.append("\\NA")
                    else:
                        mean, std, _ = entry
                        parts.append(f"{mean:.1f}{{\\scriptsize$\\pm${std:.1f}}}")
                print(f"% {method}: " + " & ".join(parts))

    # ── Optional CSV ─────────────────────────────────────────────────
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "method_tag", "dataset", "mode", "k_shot",
                "label_seed", "train_seed",
                "test_acc",
                "n_train_lowshot", "n_classes",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        print(f"\nFull results saved to: {args.csv}")


if __name__ == "__main__":
    main()
