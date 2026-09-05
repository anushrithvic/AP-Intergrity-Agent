"""
report.py
Builds the single unified AP Integrity Report (Section 6): one batch, two
detectors, one combined exception table, one measured accuracy block.
"""
import csv
from detectors.duplicate_detector import detect_duplicates
from detectors.fee_anomaly_detector import detect_fee_anomalies
from scoring import score


def load_transactions(path="data/transactions.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_answer_key(path="data/answer_key.csv"):
    with open(path, newline="") as f:
        return {row["transaction_id"]: row["true_label"] for row in csv.DictReader(f)}


def load_answer_key_notes(path="data/answer_key.csv"):
    with open(path, newline="") as f:
        return {row["transaction_id"]: row["notes"] for row in csv.DictReader(f)}


def build_combined_output(txns=None, starting_balances=None):
    """
    Runs both detectors and merges results into one combined exception table.
    Returns (combined_rows, summary_counts, unresolved_rows) where combined_rows
    is a list of dicts: transaction_id, tag, confidence, reason,
    matched_against/fee_type/amount extras. unresolved_rows are items the
    fee detector could not conclusively evaluate (see fee_anomaly_detector.py).

    starting_balances: passed through to the fee anomaly detector's overdraft
    check. If None, the sample batch's known balances are used (see
    detectors/fee_anomaly_detector.py). For uploaded data, pass an explicit
    dict (possibly {}) so unknown accounts are marked unresolved rather than
    checked against a fabricated default balance.
    """
    if txns is None:
        txns = load_transactions()

    payments = [t for t in txns if t["transaction_type"] == "payment"]
    fees = [t for t in txns if t["transaction_type"] == "fee"]

    dup_flags = detect_duplicates(payments)
    fee_flags, fee_unresolved = detect_fee_anomalies(fees, txns, starting_balances)

    combined = []
    for d in dup_flags:
        combined.append({
            "transaction_id": d["transaction_id"],
            "tag": "duplicate_payment",
            "display_tag": "[Duplicate]",
            "confidence": d["confidence"],
            "reason": d["reason"],
            "matched_against": d["matched_against"],
        })
    for f in fee_flags:
        # Fee anomalies don't have the payment "confidence" tiering; map tags
        # to a comparable confidence level for the unified table.
        if "logically_invalid" in f["tags"] or "novel_type" in f["tags"]:
            conf = "high"
        else:
            conf = "medium"
        combined.append({
            "transaction_id": f["transaction_id"],
            "tag": "fee_anomaly",
            "display_tag": "[Fee Anomaly]",
            "confidence": conf,
            "reason": f["reason"],
            "fee_type": f["fee_type"],
            "amount": f["amount"],
            "baseline_avg": f["baseline_avg"],
        })

    summary = {
        "batch_size": len(txns),
        "n_payments": len(payments),
        "n_fees": len(fees),
        "dup_total": len(dup_flags),
        "dup_high": sum(1 for d in dup_flags if d["confidence"] == "high"),
        "dup_medium": sum(1 for d in dup_flags if d["confidence"] == "medium"),
        "dup_low": sum(1 for d in dup_flags if d["confidence"] == "low"),
        "fee_total": len(fee_flags),
        "fee_stat_outlier": sum(1 for f in fee_flags if "statistical_outlier" in f["tags"]),
        "fee_novel": sum(1 for f in fee_flags if "novel_type" in f["tags"]),
        "fee_invalid": sum(1 for f in fee_flags if "logically_invalid" in f["tags"]),
        "fee_unresolved": len(fee_unresolved),
    }

    return combined, summary, fee_unresolved


def measure_trap_cases(answer_key_notes, combined):
    """
    Explicitly verifies the false-positive guard (spec Section 4.2/10): the
    deliberately-seeded 'legitimate repeat business' trap cases must NOT be
    flagged. Returns a dict so this can be shown in the report and the UI,
    rather than left as an implicit, unverified claim.

    answer_key_notes: dict {transaction_id: notes} from the raw answer_key.csv rows
    Returns {passed: bool, trap_ids: [...], wrongly_flagged: [...]}
    """
    trap_ids = [tid for tid, note in answer_key_notes.items() if "Trap case" in note]
    flagged_ids = {c["transaction_id"] for c in combined}
    wrongly_flagged = [tid for tid in trap_ids if tid in flagged_ids]
    return {
        "trap_ids": trap_ids,
        "wrongly_flagged": wrongly_flagged,
        "passed": len(wrongly_flagged) == 0 and len(trap_ids) > 0,
    }


def render_report(combined, summary, answer_key, elapsed_seconds=None, answer_key_notes=None, unresolved=None):
    tag_by_id = {c["transaction_id"]: c["tag"] for c in combined}
    result = score(tag_by_id, answer_key)

    lines = []
    lines.append("AP INTEGRITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Batch size: {summary['batch_size']} transactions "
                  f"({summary['n_payments']} payments, {summary['n_fees']} fees)")
    if elapsed_seconds is not None:
        rate = summary["batch_size"] / elapsed_seconds if elapsed_seconds > 0 else float("inf")
        lines.append(f"Throughput: processed in {elapsed_seconds*1000:.1f}ms "
                      f"(~{rate:,.0f} transactions/sec)")
    lines.append("")

    if answer_key_notes:
        trap = measure_trap_cases(answer_key_notes, combined)
        lines.append("FALSE-POSITIVE GUARD CHECK (legitimate repeat-business trap cases):")
        if trap["trap_ids"]:
            status = "PASS" if trap["passed"] else "FAIL"
            lines.append(f"  - {status}: {len(trap['trap_ids'])} trap transactions seeded "
                          f"({', '.join(trap['trap_ids'])}), "
                          f"{len(trap['wrongly_flagged'])} incorrectly flagged")
        lines.append("")
    lines.append(f"DUPLICATE PAYMENTS FLAGGED: {summary['dup_total']}")
    lines.append(f"  - High confidence:   {summary['dup_high']}")
    lines.append(f"  - Medium confidence: {summary['dup_medium']}")
    lines.append(f"  - Low confidence:    {summary['dup_low']}")
    lines.append("")
    lines.append(f"FEE ANOMALIES FLAGGED: {summary['fee_total']}")
    lines.append(f"  - Statistical outliers (z > 2): {summary['fee_stat_outlier']}")
    lines.append(f"  - Novel fee types:               {summary['fee_novel']}")
    lines.append(f"  - Logically invalid:             {summary['fee_invalid']}")
    lines.append("")
    lines.append("MEASURED ACCURACY (vs. hidden answer key):")
    lines.append(f"  - Precision: {result['precision']*100:.1f}%  "
                  f"({len(result['correct_label_tps'])} correctly labeled of {result['n_flagged']} flagged)")
    lines.append(f"  - Recall:    {result['recall']*100:.1f}%  "
                  f"({len(result['true_positives'])} of {result['n_true_exceptions']} true exceptions caught)")
    lines.append(f"  - False positives: {len(result['false_positives'])}"
                  + (f" -> {', '.join(result['false_positives'])}" if result['false_positives'] else ""))
    lines.append(f"  - False negatives (missed): {len(result['false_negatives'])}"
                  + (f" -> {', '.join(result['false_negatives'])}" if result['false_negatives'] else ""))
    if result["mislabeled_tps"]:
        lines.append(f"  - Flagged correctly but with wrong tag: {', '.join(result['mislabeled_tps'])}")
    lines.append("")

    low_conf = [c for c in combined if c["confidence"] == "low"]
    lines.append("EXCEPTION LIST (could not confidently resolve):")
    if low_conf:
        for c in low_conf:
            lines.append(f"  - {c['transaction_id']} {c['display_tag']} (low confidence): {c['reason']}")
    else:
        lines.append("  - None. All flagged items met medium or high confidence.")
    if result["false_negatives"]:
        lines.append("  Missed true exceptions (present in answer key but not caught):")
        for tid in result["false_negatives"]:
            lines.append(f"  - {tid} (true label: {answer_key[tid]})")
    if unresolved:
        lines.append(f"  {len(unresolved)} item(s) could not be verified at all (not flagged, not cleared):")
        for u in unresolved:
            lines.append(f"  - {u['transaction_id']} [{u['fee_type']}, ${u['amount']:.2f}]: {u['reason']}")
    lines.append("")

    lines.append("COMBINED EXCEPTION TABLE")
    lines.append("-" * 60)
    lines.append(f"{'transaction_id':<14} {'tag':<15} {'confidence':<10} reason")
    for c in sorted(combined, key=lambda r: r["transaction_id"]):
        lines.append(f"{c['transaction_id']:<14} {c['display_tag']:<15} {c['confidence']:<10} {c['reason']}")

    return "\n".join(lines), result


if __name__ == "__main__":
    txns = load_transactions()
    answer_key = load_answer_key()
    combined, summary, unresolved = build_combined_output(txns)
    text, result = render_report(combined, summary, answer_key, unresolved=unresolved)
    print(text)
