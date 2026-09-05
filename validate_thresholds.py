"""
validate_thresholds.py
Runs the full detection pipeline across many INDEPENDENTLY generated batches
(different seeds -> different vendors, amounts, dates, and scenario
placements for every duplicate/anomaly/trap case -- see generate_data.py)
and reports precision/recall per seed plus the aggregate.

This is the concrete answer to "did you just tune this to your one demo
batch?" -- it's evidence, not an assertion. If precision/recall stayed
high and stable across seeds never used during development, the fuzzy-match
cutoff, amount tolerance, date window, and z-score threshold aren't
overfit to one specific dataset.

Usage:
    python3 validate_thresholds.py                # seeds 1-20
    python3 validate_thresholds.py --n 50          # seeds 1-50
    python3 validate_thresholds.py --start 100 --n 10   # seeds 100-109
"""
import argparse
import statistics

from generate_data import generate
from report import build_combined_output
from scoring import score


def run_seed(seed):
    rows, answer_rows = generate(seed)
    answer_key = {r["transaction_id"]: r["true_label"] for r in answer_rows}
    combined, summary, unresolved = build_combined_output(rows)  # starting_balances=None -> sample defaults
    tag_by_id = {c["transaction_id"]: c["tag"] for c in combined}
    result = score(tag_by_id, answer_key)
    return {
        "seed": seed,
        "precision": result["precision"],
        "recall": result["recall"],
        "n_flagged": result["n_flagged"],
        "n_true_exceptions": result["n_true_exceptions"],
        "false_positives": result["false_positives"],
        "false_negatives": result["false_negatives"],
        "batch_size": summary["batch_size"],
        "unresolved": len(unresolved),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate detector thresholds across many independently seeded batches.")
    parser.add_argument("--start", type=int, default=1, help="First seed (default 1)")
    parser.add_argument("--n", type=int, default=20, help="How many seeds to run (default 20)")
    args = parser.parse_args()

    seeds = list(range(args.start, args.start + args.n))
    results = [run_seed(s) for s in seeds]

    print(f"Ran the full detection pipeline on {len(seeds)} independently generated batches "
          f"(seeds {seeds[0]}-{seeds[-1]}), none of which were used while tuning the thresholds.")
    print("-" * 78)
    print(f"{'seed':<6} {'batch':<7} {'flagged':<9} {'true_exc':<9} {'precision':<11} {'recall':<9} {'FP':<4} {'FN':<4}")
    for r in results:
        print(f"{r['seed']:<6} {r['batch_size']:<7} {r['n_flagged']:<9} {r['n_true_exceptions']:<9} "
              f"{r['precision']*100:>8.1f}%  {r['recall']*100:>6.1f}%  {len(r['false_positives']):<4} {len(r['false_negatives']):<4}")

    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    total_fp = sum(len(r["false_positives"]) for r in results)
    total_fn = sum(len(r["false_negatives"]) for r in results)
    total_flagged = sum(r["n_flagged"] for r in results)
    total_true = sum(r["n_true_exceptions"] for r in results)

    print("-" * 78)
    print(f"Mean precision: {statistics.mean(precisions)*100:.1f}%  "
          f"(min {min(precisions)*100:.1f}%, max {max(precisions)*100:.1f}%, stdev {statistics.pstdev(precisions)*100:.1f}pp)")
    print(f"Mean recall:    {statistics.mean(recalls)*100:.1f}%  "
          f"(min {min(recalls)*100:.1f}%, max {max(recalls)*100:.1f}%, stdev {statistics.pstdev(recalls)*100:.1f}pp)")
    print(f"Totals across all {len(seeds)} batches: {total_flagged} flagged, {total_true} true exceptions, "
          f"{total_fp} false positives, {total_fn} false negatives")

    seeds_with_fp = [r["seed"] for r in results if r["false_positives"]]
    seeds_with_fn = [r["seed"] for r in results if r["false_negatives"]]
    if seeds_with_fp:
        print(f"Seeds with at least one false positive: {seeds_with_fp}")
    if seeds_with_fn:
        print(f"Seeds with at least one false negative: {seeds_with_fn}")


if __name__ == "__main__":
    main()
