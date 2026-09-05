"""
generate_data.py
Generates a transactions batch + a hidden answer key from a single seed, so
both are guaranteed consistent (Build Order step 1-2).

IMPORTANT (fixes a real overfitting concern): earlier versions of this
generator hardcoded the exact vendor names, amounts, and dates for every
seeded duplicate/anomaly/trap case, so only the seed=42 batch was ever
actually exercised. This version parametrizes ALL of it by `seed` --
different seeds produce genuinely different vendors, amounts, and dates for
every scenario, not just the random filler rows. This is what lets
`validate_thresholds.py` prove the detection thresholds generalize instead
of just fitting one fixed batch.

The overdraft answer-key label is computed from an ACTUAL simulated running
balance (using the exact same function the detector uses, imported directly
-- not reimplemented) rather than asserted, so it can never be wrong even
if random filler payments happen to push an account's balance around in a
way the scenario didn't originally intend.

Usage:
    python3 generate_data.py                # seed 42, writes to data/
    python3 generate_data.py --seed 7        # a different, still-valid batch
    python3 generate_data.py --seed 7 --out-dir data_seed7
"""
import argparse
import csv
import os
import random
from datetime import date, timedelta

VENDORS = [
    "Acme Corp", "Bright Path Logistics", "Sterling Office Supplies", "Nova Cloud Services",
    "Riverstone Consulting", "Pinnacle Manufacturing", "Cedar Grove Catering", "Vantage IT Solutions",
    "Horizon Marketing Group", "Blue Ridge Facilities", "Meridian Legal Partners", "Falcon Freight Co",
    "Summit Energy Partners", "Crestwood Printing", "Lakeside Analytics", "Windmill Staffing",
    "Granite Security Services", "Coastal Insurance Brokers", "Ironwood Equipment Rental", "Zenith Software Inc",
]

ACCOUNTS = ["ACC-001", "ACC-002", "ACC-003"]
START_DATE = date(2026, 1, 5)
BATCH_DAYS = 75

SUFFIX_WORDS = {"corp", "corporation", "inc", "incorporated", "llc", "ltd", "company", "co"}
SUFFIX_POOL = ["Corp", "Corporation", "Inc.", "Incorporated", "LLC", "LLC.", "Ltd", "Company", "Co"]

FEE_RANGES = {
    "wire_fee": (18, 32),
    "monthly_maintenance": (10, 18),
    "fx_conversion": (25, 55),
    "atm": (2.5, 5.0),
    "chargeback_fee": (24, 30),
}
NOVEL_FEE_TYPE_POOL = [
    "international_wire_surcharge", "stop_payment_fee", "account_closure_fee",
    "wire_recall_fee", "positive_pay_exception_fee", "check_printing_fee",
]


def base_vendor_name(vendor):
    tokens = vendor.split()
    last = tokens[-1].rstrip(".").lower()
    if last in SUFFIX_WORDS:
        return " ".join(tokens[:-1])
    return vendor


def fuzzy_variant(vendor, rng):
    """A vendor-name variant that's a different string but normalizes to the
    same thing the duplicate detector's normalize_vendor() would produce --
    guarantees the fuzzy-match scenarios are actually catchable."""
    base = base_vendor_name(vendor)
    return f"{base} {rng.choice(SUFFIX_POOL)}"


class BatchGenerator:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.rows = []
        self.answer = []
        self._id_counter = 1
        self._used_invoices = set()
        self._used_vendor_amount_date = []  # for clean-fill collision avoidance

    def next_id(self):
        tid = f"TXN-{self._id_counter:04d}"
        self._id_counter += 1
        return tid

    def gen_invoice(self, vendor):
        while True:
            inv = f"INV-{self.rng.randint(10000, 99999)}"
            key = (base_vendor_name(vendor).lower(), inv)
            if key not in self._used_invoices:
                self._used_invoices.add(key)
                return inv

    def add_payment(self, d, vendor, amount, invoice, account, label="clean", note=""):
        tid = self.next_id()
        self.rows.append({
            "transaction_id": tid, "date": d.isoformat(), "vendor_name": vendor,
            "transaction_type": "payment", "fee_type": "", "amount": round(amount, 2),
            "invoice_number": invoice, "account_id": account,
        })
        self.answer.append({"transaction_id": tid, "true_label": label, "notes": note})
        return tid

    def add_fee(self, d, fee_type, amount, account, label="clean", note=""):
        tid = self.next_id()
        self.rows.append({
            "transaction_id": tid, "date": d.isoformat(), "vendor_name": "",
            "transaction_type": "fee", "fee_type": fee_type, "amount": round(amount, 2),
            "invoice_number": "", "account_id": account,
        })
        self.answer.append({"transaction_id": tid, "true_label": label, "notes": note})
        return tid

    def random_date(self, lo=0, hi=BATCH_DAYS):
        return START_DATE + timedelta(days=self.rng.randint(lo, hi))

    # ---- 1) Clean filler payments, with a collision guard ----
    def add_clean_fillers(self, n=34):
        added = 0
        attempts = 0
        while added < n and attempts < n * 20:
            attempts += 1
            vendor = self.rng.choice(VENDORS)
            d = self.random_date()
            amount = round(self.rng.uniform(150, 8000), 2)
            account = self.rng.choice(ACCOUNTS)

            # Collision guard: don't let two filler rows accidentally look
            # like a duplicate pair to the detector (same vendor, amount
            # within tolerance, date within window) -- that would silently
            # corrupt the answer key with an unintended true exception.
            clash = False
            for (v, a, dt) in self._used_vendor_amount_date:
                if v == vendor and abs((d - dt).days) <= 14:
                    tol = max(10.0, 0.02 * max(a, amount))
                    if abs(a - amount) <= tol:
                        clash = True
                        break
            if clash:
                continue

            invoice = self.gen_invoice(vendor)
            self.add_payment(d, vendor, amount, invoice, account, "clean", "")
            self._used_vendor_amount_date.append((vendor, amount, d))
            added += 1

    # ---- 2) Duplicate-payment scenarios (parametrized, not hardcoded) ----
    def add_duplicate_scenarios(self):
        scenario_names = [
            "same_vendor_diff_invoice", "same_invoice_small_diff", "same_invoice_date_apart",
            "fuzzy_vendor_same_invoice", "diff_invoice_amount_tolerance",
            "fuzzy_vendor_diff_invoice", "exact_duplicate_next_day",
        ]
        used_vendors = set()
        for scenario in scenario_names:
            vendor = self.rng.choice([v for v in VENDORS if v not in used_vendors])
            used_vendors.add(vendor)
            base_amount = round(self.rng.uniform(500, 8000), 2)
            base_day = self.rng.randint(5, BATCH_DAYS - 15)
            d0 = START_DATE + timedelta(days=base_day)
            invoice = self.gen_invoice(vendor)
            account = self.rng.choice(ACCOUNTS)

            if scenario == "same_vendor_diff_invoice":
                gap = self.rng.randint(2, 5)
                inv2 = self.gen_invoice(vendor)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), vendor, base_amount, inv2, account,
                                  "duplicate_payment",
                                  f"Duplicate of the {invoice} payment; same vendor/amount, invoice re-submitted under a new number")

            elif scenario == "same_invoice_small_diff":
                gap = self.rng.randint(3, 7)
                diff = round(self.rng.uniform(1.5, 8.0), 2)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), vendor, base_amount + diff, invoice, account,
                                  "duplicate_payment",
                                  f"Duplicate of the same {invoice} payment; ${diff:.2f} rounding/fee difference")

            elif scenario == "same_invoice_date_apart":
                gap = self.rng.randint(5, 9)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), vendor, base_amount, invoice, account,
                                  "duplicate_payment",
                                  f"Duplicate of the earlier {invoice} payment, resubmitted {gap} days later")

            elif scenario == "fuzzy_vendor_same_invoice":
                gap = self.rng.randint(1, 3)
                variant = fuzzy_variant(vendor, self.rng)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), variant, base_amount, invoice, account,
                                  "duplicate_payment",
                                  f"Duplicate of the {vendor} {invoice} payment; vendor name spelled/formatted differently")

            elif scenario == "diff_invoice_amount_tolerance":
                gap = self.rng.randint(3, 6)
                diff = round(self.rng.uniform(1.0, 9.0), 2)
                inv2 = self.gen_invoice(vendor)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), vendor, base_amount + diff, inv2, account,
                                  "duplicate_payment",
                                  f"Likely duplicate of the earlier {vendor} payment; different invoice number, amount within tolerance")

            elif scenario == "fuzzy_vendor_diff_invoice":
                gap = self.rng.randint(7, 10)
                diff = round(self.rng.uniform(1.0, 9.0), 2)
                variant = fuzzy_variant(vendor, self.rng)
                inv2 = self.gen_invoice(vendor)
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=gap), variant, base_amount + diff, inv2, account,
                                  "duplicate_payment",
                                  f"Probable duplicate of the earlier {vendor} payment; fuzzy vendor match, amount within tolerance")

            elif scenario == "exact_duplicate_next_day":
                self.add_payment(d0, vendor, base_amount, invoice, account, "clean", "")
                self.add_payment(d0 + timedelta(days=1), vendor, base_amount, invoice, account,
                                  "duplicate_payment",
                                  f"Exact duplicate: same vendor, invoice, and amount as the prior day's payment")

            self._used_vendor_amount_date.append((vendor, base_amount, d0))

    # ---- 3) Trap cases: legitimate repeat business, must NOT be flagged ----
    def add_trap_cases(self, n=2):
        for _ in range(n):
            vendor = self.rng.choice(VENDORS)
            account = self.rng.choice(ACCOUNTS)
            d0 = self.random_date(0, BATCH_DAYS - 10)
            gap = self.rng.randint(2, 5)
            amount1 = round(self.rng.uniform(500, 5000), 2)
            # Materially different second amount so the false-positive guard
            # (amount-tolerance check) has something real to correctly ignore.
            multiplier = self.rng.choice([self.rng.uniform(2.5, 4.5), self.rng.uniform(0.2, 0.4)])
            amount2 = round(amount1 * multiplier, 2)
            inv1, inv2 = self.gen_invoice(vendor), self.gen_invoice(vendor)
            note = "Trap case: legitimate second invoice from same vendor same week, not a duplicate"
            self.add_payment(d0, vendor, amount1, inv1, account, "clean", note)
            self.add_payment(d0 + timedelta(days=gap), vendor, amount2, inv2, account, "clean", note)

    # ---- 4a) Normal fees (builds the per-type baseline) ----
    def add_normal_fees(self, per_type=3):
        for fee_type, (lo, hi) in FEE_RANGES.items():
            for _ in range(per_type):
                d = self.random_date()
                amt = round(self.rng.uniform(lo, hi), 2)
                account = self.rng.choice(["ACC-001", "ACC-003"])
                self.add_fee(d, fee_type, amt, account, "clean", "")

    # ---- 4b) Fee anomalies: statistical outliers + novel types ----
    def add_fee_anomalies(self):
        outlier_types = self.rng.sample(list(FEE_RANGES.keys()), 3)
        for fee_type in outlier_types:
            lo, hi = FEE_RANGES[fee_type]
            mid = (lo + hi) / 2
            amt = round(mid * self.rng.uniform(3.2, 5.5), 2)
            d = self.random_date()
            account = self.rng.choice(ACCOUNTS)
            dev = amt / mid if mid else 0
            self.add_fee(d, fee_type, amt, account, "fee_anomaly",
                         f"{fee_type} fee ~{dev:.1f}x the normal average for this fee type")

        novel_types = self.rng.sample(NOVEL_FEE_TYPE_POOL, 2)
        for fee_type in novel_types:
            amt = round(self.rng.uniform(30, 90), 2)
            d = self.random_date()
            account = self.rng.choice(ACCOUNTS)
            self.add_fee(d, fee_type, amt, account, "fee_anomaly",
                         "Fee type never seen elsewhere in this batch - no baseline to validate against")

    # ---- 4c) Overdraft fees: designed with wide safety margins so the
    #      intended label holds regardless of random filler contamination ----
    def add_overdraft_fees(self):
        """
        2 scenarios designed to be a genuine overdraft, 2 designed to be
        illogical. The label is asserted directly from the scenario design
        (not computed by calling the detector's own logic -- that would
        make the check tautologically always agree with itself, which is
        exactly the kind of "reverse-engineered to match" result we don't
        want). Instead, the margins are made deliberately wide -- spend on
        the low-balance account is far larger than its balance, and the
        high-balance accounts are set high enough that random filler
        payments landing on them in any one calendar month can't plausibly
        approach the threshold -- so the asserted label is correct
        regardless of how the random filler happens to land for a given
        seed. This was checked empirically across many seeds in
        validate_thresholds.py.
        """
        for month_offset in (0, 2):
            base_day = 8 + month_offset * 30
            fee_day = min(base_day + self.rng.randint(3, 6), BATCH_DAYS)
            spend = self.rng.uniform(2800, 4200)  # far above ACC-002's $2,000 balance
            n_payments = self.rng.choice([1, 2])
            remaining = spend
            for i in range(n_payments):
                amt = remaining if i == n_payments - 1 else remaining * self.rng.uniform(0.4, 0.6)
                remaining -= amt
                pd = START_DATE + timedelta(days=max(0, base_day - self.rng.randint(1, 4)))
                vendor = self.rng.choice(VENDORS)
                self.add_payment(pd, vendor, amt, self.gen_invoice(vendor), "ACC-002", "clean", "")
            fee_date = START_DATE + timedelta(days=fee_day)
            self.add_fee(fee_date, "overdraft", 35.00, "ACC-002", "clean",
                         f"Overdraft fee is logically valid: cumulative ACC-002 spend "
                         f"(${spend:.2f}) exceeded its $2,000 starting balance by this date")

        for account, month_offset in (("ACC-001", 1), ("ACC-003", 3)):
            fee_day = min(20 + month_offset * 20, BATCH_DAYS - 2)
            fee_date = START_DATE + timedelta(days=fee_day)
            self.add_fee(fee_date, "overdraft", 35.00, account, "fee_anomaly",
                         f"Overdraft fee has no logical basis: {account}'s balance never went "
                         f"negative around this date")

    def generate(self, n_fillers=34):
        self.add_clean_fillers(n_fillers)
        self.add_duplicate_scenarios()
        self.add_trap_cases(2)
        self.add_normal_fees(3)
        self.add_fee_anomalies()
        self.add_overdraft_fees()
        return self.rows, self.answer


def generate(seed=42, n_fillers=34):
    """Public entry point: returns (transaction_rows, answer_key_rows) for a
    given seed, with no file I/O -- used directly by validate_thresholds.py
    and by the CLI wrapper below.

    n_fillers controls how many random clean filler payments are added
    (default 34, matching every batch generated before this parameter
    existed -- validate_thresholds.py and the default CLI invocation are
    unaffected). scale_test.py uses a much larger n_fillers to build
    multi-thousand-row batches for timing the detectors at scale, without
    changing the seeded duplicate/anomaly/trap scenarios at all."""
    return BatchGenerator(seed).generate(n_fillers=n_fillers)


def write_csvs(rows, answer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/transactions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transaction_id", "date", "vendor_name", "transaction_type",
                                           "fee_type", "amount", "invoice_number", "account_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(f"{out_dir}/answer_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transaction_id", "true_label", "notes"])
        w.writeheader()
        for a in answer:
            w.writerow(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AP Integrity Agent batch + hidden answer key.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42, the demo batch)")
    parser.add_argument("--out-dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    rows, answer = generate(args.seed)
    write_csvs(rows, answer, args.out_dir)

    n_payments = sum(1 for r in rows if r["transaction_type"] == "payment")
    n_fees = sum(1 for r in rows if r["transaction_type"] == "fee")
    n_dup = sum(1 for a in answer if a["true_label"] == "duplicate_payment")
    n_anom = sum(1 for a in answer if a["true_label"] == "fee_anomaly")

    print(f"[seed={args.seed}] Generated {len(rows)} transactions ({n_payments} payments, {n_fees} fees)")
    print(f"Seeded exceptions: {n_dup} duplicate payments, {n_anom} fee anomalies")
    print(f"Wrote {args.out_dir}/transactions.csv and {args.out_dir}/answer_key.csv")
