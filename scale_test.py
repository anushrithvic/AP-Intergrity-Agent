"""
scale_test.py
Section D1 of the v2 roadmap: "before you say the word 'scalable', time it."

This does NOT change any detection logic. It generates batches of
increasing size (using generate_data.generate()'s n_fillers parameter,
which only adds more random clean filler payments -- the seeded
duplicate/anomaly/trap scenarios are identical at every size) and times:

  1. detect_duplicates() alone -- the specific O(n^2) fuzzy-matching pass
     called out in the roadmap (itertools.combinations() over every
     remaining payment, each pair compared with SequenceMatcher).
  2. The full build_combined_output() pipeline (both detectors + merge).

The point is to find out, empirically, whether the current
"itertools.combinations() over every remaining payment" fuzzy pass is
actually a problem at realistic batch sizes, before claiming "scalable"
in a pitch -- not to assume it is or isn't.

Usage:
    python3 scale_test.py                       # default sizes: 79..5000
    python3 scale_test.py --sizes 100 1000 5000  # custom sizes
"""
import argparse
import time

from generate_data import generate
from detectors.duplicate_detector import detect_duplicates
from report import build_combined_output

DEFAULT_SIZES = [79, 250, 500, 1000, 2000, 3000, 5000]


def time_one_size(n_fillers, seed=1):
    rows, _ = generate(seed=seed, n_fillers=n_fillers)
    payments = [r for r in rows if r["transaction_type"] == "payment"]
    n = len(rows)
    n_payments = len(payments)

    # naive pair count actually walked by itertools.combinations() in
    # pass 2 of detect_duplicates() (an upper bound: pass 1's exact-key
    # matches remove some payments from pass 2's pool first)
    naive_pairs = n_payments * (n_payments - 1) // 2

    t0 = time.perf_counter()
    dup_flags = detect_duplicates(payments)
    t_dup = time.perf_counter() - t0

    t0 = time.perf_counter()
    combined, summary, unresolved = build_combined_output(rows)
    t_full = time.perf_counter() - t0

    return {
        "n_fillers": n_fillers,
        "batch_size": n,
        "n_payments": n_payments,
        "naive_pairs": naive_pairs,
        "dup_flagged": len(dup_flags),
        "t_dup_ms": t_dup * 1000,
        "t_full_ms": t_full * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Time the detection pipeline at increasing batch sizes.")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                         help="n_fillers values to test (batch size is roughly this + ~45 seeded rows)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sizes = args.sizes if args.sizes else DEFAULT_SIZES

    print("Timing detect_duplicates() (the O(n^2) fuzzy pass) and the full pipeline")
    print("at increasing batch sizes. Seeded duplicate/anomaly/trap scenarios are")
    print("identical at every size -- only the random clean filler count changes.")
    print("-" * 88)
    print(f"{'n_fillers':<10} {'batch':<8} {'payments':<9} {'naive_pairs':<12} "
          f"{'dup_flagged':<12} {'dup_ms':<10} {'full_ms':<10}")

    results = []
    for size in sizes:
        r = time_one_size(size, seed=args.seed)
        results.append(r)
        print(f"{r['n_fillers']:<10} {r['batch_size']:<8} {r['n_payments']:<9} "
              f"{r['naive_pairs']:<12,} {r['dup_flagged']:<12} "
              f"{r['t_dup_ms']:<10.1f} {r['t_full_ms']:<10.1f}")

    print("-" * 88)
    biggest = results[-1]
    print(f"At {biggest['batch_size']} transactions ({biggest['n_payments']} payments, "
          f"~{biggest['naive_pairs']:,} naive pairs): "
          f"detect_duplicates() took {biggest['t_dup_ms']:.1f}ms, "
          f"full pipeline took {biggest['t_full_ms']:.1f}ms.")
    print()
    print("This is a timing measurement only -- it does not change or fix the")
    print("detector. See the v2 roadmap's D1 for the record-linkage blocking fix,")
    print("which should only be prioritized if the numbers above are actually slow")
    print("relative to what you plan to claim in the pitch.")


if __name__ == "__main__":
    main()
