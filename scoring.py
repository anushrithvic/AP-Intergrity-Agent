"""
scoring.py
Section 5.3: scores the combined detector output against the hidden
answer key. Reports precision, recall, and false positives/negatives
separately (not one blended "accuracy" number), since accuracy alone is
misleading on an imbalanced batch where most rows are clean.
"""

def score(combined_flags, answer_key):
    """
    combined_flags: dict {transaction_id: tag} where tag is 'duplicate_payment' or 'fee_anomaly'
    answer_key: dict {transaction_id: true_label}
    Returns a dict with precision, recall, tp/fp/fn lists and per-class breakdowns.
    """
    flagged_ids = set(combined_flags.keys())
    true_exception_ids = {tid for tid, label in answer_key.items() if label != "clean"}

    true_positives = flagged_ids & true_exception_ids
    false_positives = flagged_ids - true_exception_ids
    false_negatives = true_exception_ids - flagged_ids

    # Only count a TP as a "correct label" TP if the tag matches the true label too
    correct_label_tps = {
        tid for tid in true_positives
        if combined_flags[tid] == answer_key[tid]
    }
    mislabeled_tps = true_positives - correct_label_tps

    precision = len(true_positives) / len(flagged_ids) if flagged_ids else 0.0
    recall = len(true_positives) / len(true_exception_ids) if true_exception_ids else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "correct_label_tps": sorted(correct_label_tps),
        "mislabeled_tps": sorted(mislabeled_tps),
        "n_flagged": len(flagged_ids),
        "n_true_exceptions": len(true_exception_ids),
    }
