"""
app.py
A small local web dashboard on top of the existing, tested pipeline
(report.py / detectors / qa_agent.py) -- no detection or Q&A logic is
duplicated here, the Flask layer only renders what those modules already
compute and scored.

Supports uploading your own transactions.csv (and optionally an
answer_key.csv, if you have one, e.g. for testing) so you're not limited to
the seeded sample batch.

Run:
    pip install flask
    python3 generate_data.py   # if data/ doesn't exist yet
    python3 app.py
    -> open http://127.0.0.1:5000
"""
import csv
import io
import random
import re
import time
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, session

from report import (
    load_transactions, load_answer_key, load_answer_key_notes,
    build_combined_output, measure_trap_cases,
)
from scoring import score
from qa_agent_llm import LLMBackedQA
import generate_data

app = Flask(__name__)
app.secret_key = "dev-only-not-for-production"  # only needed to sign the session cookie in this local prototype

REQUIRED_COLUMNS = {
    "transaction_id", "date", "vendor_name", "transaction_type",
    "fee_type", "amount", "invoice_number", "account_id",
}

# Server-side state keyed by a per-browser session id (stored in a signed
# cookie via flask.session). Each visitor gets their own batch/upload/QA
# state instead of one global dict shared by everyone hitting the server --
# without this, one person's upload would silently replace what everyone
# else sees. Still in-memory (lost on restart, not shared across processes)
# which is fine for a local single-process prototype; a real deployment
# would move this to a proper session store (Redis, a database, etc.).
_sessions = {}


def compute_brief_checks(summary, result, rate, elapsed_ms, has_answer_key):
    """Section A2: a small, honest checklist of how this run maps to the
    judging brief, computed from numbers already produced above -- no new
    detection logic, just making the "are we on brief" claim inspectable
    in the product instead of only in the pitch."""
    n_exceptions = summary["dup_total"] + summary["fee_total"]
    checks = [
        {
            "label": "50+ transaction records",
            "detail": f"{summary['batch_size']} in this batch",
            "passed": summary["batch_size"] >= 50,
        },
        {
            "label": "Exceptions detected",
            "detail": f"{n_exceptions} flagged ({summary['dup_total']} duplicate, {summary['fee_total']} fee anomaly)",
            "passed": n_exceptions > 0,
        },
    ]
    if has_answer_key and result is not None:
        checks.append({
            "label": "Match rate measured against a hidden answer key",
            "detail": f"{result['precision']*100:.1f}% precision / {result['recall']*100:.1f}% recall",
            "passed": True,
        })
    else:
        checks.append({
            "label": "Match rate measured against a hidden answer key",
            "detail": "no answer key uploaded for this batch",
            "passed": False,
        })
    checks.append({
        "label": "Throughput reported",
        "detail": f"~{rate:,.0f} txns/sec ({elapsed_ms:.0f}ms total)",
        "passed": True,
    })
    return checks


def compute_analytics(combined, txns):
    """Section C1: the numbers behind the visual analytics panel -- a
    duplicate-vs-fee-anomaly dollar split, a vendor risk leaderboard, and
    the single '$ at risk caught' headline number. Pure aggregation over
    data already produced by the detectors; no new detection logic."""
    txns_by_id = {t["transaction_id"]: t for t in txns}

    dup_dollars = 0.0
    fee_dollars = 0.0
    vendor_totals = {}  # vendor_name -> {"amount": float, "count": int}

    for c in combined:
        txn = txns_by_id.get(c["transaction_id"], {})
        amount = float(c.get("amount", txn.get("amount", 0)) or 0)

        if c["tag"] == "duplicate_payment":
            dup_dollars += amount
            vendor = (txn.get("vendor_name") or "Unknown vendor").strip() or "Unknown vendor"
            bucket = vendor_totals.setdefault(vendor, {"amount": 0.0, "count": 0})
            bucket["amount"] += amount
            bucket["count"] += 1
        elif c["tag"] == "fee_anomaly":
            fee_dollars += amount

    top_vendors = sorted(vendor_totals.items(), key=lambda kv: kv[1]["amount"], reverse=True)[:8]

    return {
        "total_at_risk": dup_dollars + fee_dollars,
        "dup_dollars": dup_dollars,
        "fee_dollars": fee_dollars,
        "split_labels": ["Duplicate Payments", "Fee Anomalies"],
        "split_counts": [
            sum(1 for c in combined if c["tag"] == "duplicate_payment"),
            sum(1 for c in combined if c["tag"] == "fee_anomaly"),
        ],
        "split_dollars": [round(dup_dollars, 2), round(fee_dollars, 2)],
        "vendor_labels": [v for v, _ in top_vendors],
        "vendor_amounts": [round(vals["amount"], 2) for _, vals in top_vendors],
        "vendor_counts": [vals["count"] for _, vals in top_vendors],
    }


def compute_state(txns, answer_key=None, answer_key_notes=None, source_label="sample batch", starting_balances=None):
    start = time.perf_counter()
    combined, summary, unresolved = build_combined_output(txns, starting_balances=starting_balances)
    elapsed = time.perf_counter() - start

    result = None
    trap = None
    false_negatives = []
    if answer_key:
        tag_by_id = {c["transaction_id"]: c["tag"] for c in combined}
        result = score(tag_by_id, answer_key)
        false_negatives = [
            {"id": tid, "label": answer_key.get(tid, "?")}
            for tid in result["false_negatives"]
        ]
    if answer_key_notes:
        trap = measure_trap_cases(answer_key_notes, combined)

    qa = LLMBackedQA(combined, summary, txns, answer_key)
    rate = summary["batch_size"] / elapsed if elapsed > 0 else float("inf")
    elapsed_ms = elapsed * 1000
    brief_checks = compute_brief_checks(summary, result, rate, elapsed_ms, bool(answer_key))
    analytics = compute_analytics(combined, txns)

    return {
        "txns": txns,
        "combined": combined,
        "summary": summary,
        "score": result,
        "trap": trap,
        "false_negatives": false_negatives,
        "unresolved": unresolved,
        "qa": qa,
        "using_llm": qa.using_llm,
        "elapsed_ms": elapsed_ms,
        "rate": rate,
        "source_label": source_label,
        "has_answer_key": bool(answer_key),
        "brief_checks": brief_checks,
        "analytics": analytics,
    }


def load_sample_state():
    txns = load_transactions()
    answer_key = load_answer_key()
    answer_key_notes = load_answer_key_notes()
    # starting_balances=None -> falls back to the sample batch's known
    # balances inside the fee detector (see SAMPLE_STARTING_BALANCE).
    return compute_state(txns, answer_key, answer_key_notes, source_label="sample batch (data/transactions.csv)")


def parse_transactions_csv(file_storage):
    """Parses + validates an uploaded transactions CSV. Returns (rows, warnings).
    Raises ValueError with a user-facing message on unrecoverable problems."""
    raw = file_storage.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None:
        raise ValueError("The file appears to be empty.")

    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected columns: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )

    rows = []
    warnings = []
    seen_ids = set()
    for i, row in enumerate(reader, start=2):  # start=2: row 1 is the header
        tid = (row.get("transaction_id") or "").strip()
        if not tid:
            warnings.append(f"Row {i}: skipped, missing transaction_id.")
            continue
        if tid in seen_ids:
            warnings.append(f"Row {i}: skipped, duplicate transaction_id '{tid}'.")
            continue
        seen_ids.add(tid)

        ttype = (row.get("transaction_type") or "").strip().lower()
        if ttype not in ("payment", "fee"):
            warnings.append(f"Row {i} ({tid}): skipped, transaction_type must be 'payment' or 'fee', got '{ttype}'.")
            continue

        try:
            amount = float(row.get("amount") or 0)
        except ValueError:
            warnings.append(f"Row {i} ({tid}): skipped, amount '{row.get('amount')}' isn't a number.")
            continue

        date_val = (row.get("date") or "").strip()
        if len(date_val) != 10 or date_val[4] != "-" or date_val[7] != "-":
            warnings.append(f"Row {i} ({tid}): date '{date_val}' isn't in YYYY-MM-DD format, kept as-is but may confuse date-window matching.")

        rows.append({
            "transaction_id": tid,
            "date": date_val,
            "vendor_name": (row.get("vendor_name") or "").strip(),
            "transaction_type": ttype,
            "fee_type": (row.get("fee_type") or "").strip(),
            "amount": amount,
            "invoice_number": (row.get("invoice_number") or "").strip(),
            "account_id": (row.get("account_id") or "UNSPECIFIED").strip() or "UNSPECIFIED",
        })

    if not rows:
        raise ValueError("No valid transaction rows found after validation.")

    return rows, warnings


def parse_answer_key_csv(file_storage):
    raw = file_storage.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None or "transaction_id" not in reader.fieldnames or "true_label" not in reader.fieldnames:
        raise ValueError("Answer key CSV needs at least 'transaction_id' and 'true_label' columns.")
    answer_key = {}
    notes = {}
    for row in reader:
        tid = (row.get("transaction_id") or "").strip()
        if not tid:
            continue
        answer_key[tid] = (row.get("true_label") or "clean").strip()
        notes[tid] = row.get("notes") or ""
    return answer_key, notes


def parse_starting_balances(text):
    """Parses lines like 'ACC-001: 15000' or 'ACC-001, 15000' into a dict.
    Malformed lines are skipped and returned as warnings, not fatal errors --
    consistent with how CSV row validation degrades gracefully elsewhere."""
    balances = {}
    warnings = []
    if not text:
        return balances, warnings
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(",", ":").split(":", 1)]
        if len(parts) != 2:
            warnings.append(f"Balance line {i}: couldn't parse '{line}' (expected 'ACCOUNT_ID: AMOUNT').")
            continue
        acct, amt_str = parts
        try:
            balances[acct] = float(amt_str.replace("$", "").replace(",", ""))
        except ValueError:
            warnings.append(f"Balance line {i}: '{amt_str}' isn't a number for account {acct}.")
    return balances, warnings


def get_session_state():
    """Returns this browser's state, creating a fresh sample-batch state on
    first visit. Each session id maps to its own independent batch/upload/
    Q&A state -- see the module docstring note above."""
    sid = session.get("sid")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _sessions[sid] = load_sample_state()
    return sid, _sessions[sid]


def set_session_state(new_state):
    sid = session.get("sid") or str(uuid.uuid4())
    session["sid"] = sid
    _sessions[sid] = new_state


@app.route("/")
def index():
    _, state = get_session_state()
    rows = sorted(state["combined"], key=lambda r: r["transaction_id"])
    return render_template("index.html", rows=rows, **{
        k: v for k, v in state.items() if k not in ("combined", "txns", "qa")
    })


@app.route("/upload", methods=["POST"])
def upload():
    txn_file = request.files.get("transactions_file")
    if not txn_file or txn_file.filename == "":
        flash("Please choose a transactions CSV file to upload.", "error")
        return redirect(url_for("index"))

    try:
        rows, warnings = parse_transactions_csv(txn_file)
    except ValueError as e:
        flash(f"Upload failed: {e}", "error")
        return redirect(url_for("index"))

    answer_key, answer_key_notes = None, None
    ak_file = request.files.get("answer_key_file")
    if ak_file and ak_file.filename:
        try:
            answer_key, answer_key_notes = parse_answer_key_csv(ak_file)
        except ValueError as e:
            flash(f"Answer key upload ignored: {e}", "error")

    balances_text = (request.form.get("starting_balances") or "").strip()
    starting_balances, balance_warnings = parse_starting_balances(balances_text)
    # Explicitly {} (not None) when nothing was provided, so uploaded data
    # never silently falls back to the sample batch's ACC-001/002/003
    # balances -- see fee_anomaly_detector.py's honesty note.

    new_state = compute_state(
        rows, answer_key, answer_key_notes,
        source_label=f"uploaded: {txn_file.filename}",
        starting_balances=starting_balances,
    )
    set_session_state(new_state)

    msg = f"Loaded {len(rows)} transactions from {txn_file.filename}."
    if warnings:
        msg += f" {len(warnings)} row(s) skipped or flagged during validation."
    flash(msg, "success")
    for w in warnings[:10]:
        flash(w, "warning")
    if len(warnings) > 10:
        flash(f"...and {len(warnings) - 10} more warnings not shown.", "warning")
    for w in balance_warnings:
        flash(w, "warning")
    if not answer_key:
        flash("No answer key provided, so precision/recall/false-positive-guard scoring is not shown for this batch.", "info")
    if not starting_balances and new_state["summary"]["fee_unresolved"] > 0:
        flash(
            f"{new_state['summary']['fee_unresolved']} overdraft fee(s) could not be verified \u2014 "
            f"no starting balance was provided for their account(s). Add balances above and re-upload to check them.",
            "info",
        )

    return redirect(url_for("index"))


@app.route("/reset")
def reset():
    set_session_state(load_sample_state())
    flash("Reset to the seeded sample batch.", "success")
    return redirect(url_for("index"))


@app.route("/regenerate", methods=["POST"])
def regenerate():
    """Section A1: regenerates a brand-new, never-before-seen batch with a
    fresh random seed and reruns the full pipeline live, in the browser --
    the concrete answer to "one cherry-picked match proves nothing." Reuses
    generate_data.generate() and compute_state() as-is; no new detection or
    scoring logic."""
    seed = random.randint(1, 1_000_000)
    rows, answer_rows = generate_data.generate(seed)
    answer_key = {r["transaction_id"]: r["true_label"] for r in answer_rows}
    answer_key_notes = {r["transaction_id"]: r["notes"] for r in answer_rows}
    # starting_balances=None -> falls back to the same known sample balances
    # generate_data.generate() writes its scenarios against (see
    # detectors/fee_anomaly_detector.py's SAMPLE_STARTING_BALANCE), same as
    # validate_thresholds.py does for every seed it runs.
    new_state = compute_state(
        rows, answer_key, answer_key_notes,
        source_label=f"regenerated live \u2014 seed {seed}",
        starting_balances=None,
    )
    set_session_state(new_state)
    flash(
        f"Regenerated a brand-new, never-before-seen batch (seed {seed}, {len(rows)} transactions) "
        f"and reran the full detection + scoring pipeline just now.",
        "success",
    )
    return redirect(url_for("index"))


def _safe_filename_fragment(text):
    frag = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return frag[:40] or "batch"


@app.route("/export.csv")
def export_csv():
    """Section A3: exports the combined exception ledger as a CSV a
    controller could actually action -- transaction id, tag, confidence,
    reason, amount, plus a little extra context (date, vendor, what it
    matched against) pulled from the already-loaded batch, not recomputed."""
    _, state = get_session_state()
    combined = state["combined"]
    txns_by_id = {t["transaction_id"]: t for t in state["txns"]}

    buf = io.StringIO()
    fieldnames = ["transaction_id", "tag", "confidence", "reason", "amount",
                  "date", "vendor_name", "matched_against", "fee_type"]
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for c in sorted(combined, key=lambda r: r["transaction_id"]):
        txn = txns_by_id.get(c["transaction_id"], {})
        amount = c.get("amount", txn.get("amount", ""))
        w.writerow({
            "transaction_id": c["transaction_id"],
            "tag": c["tag"],
            "confidence": c["confidence"],
            "reason": c["reason"],
            "amount": amount,
            "date": txn.get("date", ""),
            "vendor_name": txn.get("vendor_name", ""),
            "matched_against": c.get("matched_against", ""),
            "fee_type": c.get("fee_type", txn.get("fee_type", "")),
        })

    filename = f"ap_exceptions_{_safe_filename_fragment(state['source_label'])}.csv"
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/sample-template.csv")
def sample_template():
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(REQUIRED_COLUMNS))
    w.writeheader()
    w.writerow({
        "transaction_id": "TXN-0001", "date": "2026-01-05", "vendor_name": "Acme Corp",
        "transaction_type": "payment", "fee_type": "", "amount": "1200.00",
        "invoice_number": "INV-1001", "account_id": "ACC-001",
    })
    w.writerow({
        "transaction_id": "TXN-0002", "date": "2026-01-12", "vendor_name": "",
        "transaction_type": "fee", "fee_type": "wire_fee", "amount": "25.00",
        "invoice_number": "", "account_id": "ACC-001",
    })
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions_template.csv"},
    )


@app.route("/ask", methods=["POST"])
def ask():
    _, state = get_session_state()
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Ask a question about a transaction ID, a count, a total, or a vendor."})
    return jsonify({"answer": state["qa"].answer(question)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
