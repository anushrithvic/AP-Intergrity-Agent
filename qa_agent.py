"""
qa_agent.py
Section 7 (MANDATORY): a grounded Q&A layer that sits on top of the FINISHED
combined output (report.py's `combined` list), not raw transaction data.

Design: the agent has a small set of callable tools (lookup / filter /
aggregate) and a lightweight intent router that maps a natural-language
question to one of those tools. Every answer is built directly from a tool
call's return value and cites the transaction_id(s) it used. Nothing is
answered from free-text reasoning over the question alone, so nothing can be
hallucinated beyond what the data actually contains.

Note on "agentic" design: this router is deliberately simple/deterministic
(regex + keyword intent matching) rather than calling an external LLM,
so the whole prototype runs offline with no API key. The tool-use contract
(lookup_transaction / filter_transactions / aggregate, with every answer
required to cite transaction_ids) is the same shape you'd use if you swapped
in an LLM-based router later -- see README "what to extend first".

Extended capabilities (v2):
  - Follow-up memory: "why?" or "explain that" after a lookup resolves to
    the last transaction discussed, without repeating the ID.
  - List/filter questions: "list high confidence duplicates", "show fee
    anomalies", "which transactions were flagged for Acme Corp".
  - Amount-threshold filters: "duplicate payments over $3000".
  - Confidence-count aggregation: "how many high confidence flags".
  - A help/capabilities intent so the agent can describe what it can do
    instead of just declining.
"""
import re

MONEY_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)")
ABOVE_WORDS = ("over", "above", "more than", "greater than", "exceeding", "at least")
BELOW_WORDS = ("under", "below", "less than", "at most")
TAG_PLURAL = {
    "duplicate_payment": "duplicate payments",
    "fee_anomaly": "fee anomalies",
    None: "flagged items",
}


class APIntegrityQA:
    def __init__(self, combined, summary, all_txns, answer_key=None):
        self.combined = combined  # list of flagged dicts (see report.py)
        self.summary = summary
        self.by_id = {c["transaction_id"]: c for c in combined}
        self.all_txns_by_id = {t["transaction_id"]: t for t in all_txns}
        self.answer_key = answer_key or {}
        self.known_vendors = sorted({
            t["vendor_name"] for t in all_txns if t.get("vendor_name")
        }, key=len, reverse=True)  # longest first, so "Acme Corp" beats a shorter false match
        self.last_transaction_id = None  # conversational memory for follow-ups
        # Case-preserved lookup so ID matching works for ANY id scheme the
        # uploaded batch uses (not just the sample's "TXN-0001" format).
        self._id_by_upper = {tid.upper(): tid for tid in self.all_txns_by_id}

    # ---------------- Tools (Section 7.4) ----------------

    def lookup_transaction(self, transaction_id):
        """Returns the flagged record for a transaction_id, or None if it
        wasn't flagged. Also checks whether the ID exists in the batch at
        all, to distinguish 'clean' from 'unknown'."""
        exists = transaction_id in self.all_txns_by_id
        flagged = self.by_id.get(transaction_id)
        return {"exists": exists, "flagged": flagged}

    def filter_transactions(self, tag=None, confidence=None, vendor=None,
                             amount_min=None, amount_max=None):
        results = []
        for c in self.combined:
            if tag and c["tag"] != tag:
                continue
            if confidence and c["confidence"] != confidence:
                continue
            if vendor:
                txn = self.all_txns_by_id.get(c["transaction_id"], {})
                if vendor.lower() not in txn.get("vendor_name", "").lower():
                    continue
            if amount_min is not None or amount_max is not None:
                amt = self._amount_of(c)
                if amount_min is not None and amt < amount_min:
                    continue
                if amount_max is not None and amt > amount_max:
                    continue
            results.append(c)
        return results

    def aggregate(self, field, group_by=None, tag=None, confidence=None):
        rows = self.filter_transactions(tag=tag, confidence=confidence) if (tag or confidence) else self.combined
        if group_by == "vendor":
            counts = {}
            for c in rows:
                if c["tag"] != "duplicate_payment":
                    continue
                txn = self.all_txns_by_id.get(c["transaction_id"], {})
                v = txn.get("vendor_name", "unknown")
                counts.setdefault(v, []).append(c["transaction_id"])
            return counts
        if field == "count":
            return len(rows)
        if field == "total_amount":
            total = sum(self._amount_of(c) for c in rows)
            ids = [c["transaction_id"] for c in rows]
            return total, ids
        return None

    def _amount_of(self, c):
        """Resolves a flagged record's dollar amount regardless of whether
        it's a duplicate payment (amount lives on the underlying txn) or a
        fee anomaly (amount is already on the flag record)."""
        if c["tag"] == "fee_anomaly":
            return float(c.get("amount", 0))
        txn = self.all_txns_by_id.get(c["transaction_id"], {})
        return float(txn.get("amount", 0))

    # ---------------- Vendor / amount extraction helpers ----------------

    def _extract_transaction_id(self, q):
        """Matches any token in the question against a known transaction_id,
        case-insensitively -- works for any ID naming scheme an uploaded
        batch might use (TXN-0001, PAY-A002, INV-99, ...), not just the
        sample batch's fixed format. If no known ID matches but a token is
        still ID-shaped (letters-hyphen-alphanumerics), it's returned as-is
        so the caller can correctly report "that ID doesn't exist" instead
        of silently falling through to a generic decline."""
        tokens = re.findall(r"[A-Za-z0-9_\-]+", q)
        for tok in tokens:
            canonical = self._id_by_upper.get(tok.upper())
            if canonical:
                return canonical
        for tok in tokens:
            if re.match(r"^[A-Za-z]{2,}-[A-Za-z0-9]+$", tok):
                return tok.upper()
        return None

    def _extract_vendor(self, q):
        ql = q.lower()
        for v in self.known_vendors:
            if v.lower() in ql:
                return v
            first_word = v.split()[0].lower()
            if len(first_word) > 3 and re.search(rf"\b{re.escape(first_word)}\b", ql):
                return v
        return None

    def _extract_amount_filter(self, ql):
        """Returns (amount_min, amount_max) or (None, None)."""
        m = MONEY_RE.search(ql)
        if not m:
            return None, None
        amount = float(m.group(1).replace(",", ""))
        if any(w in ql for w in ABOVE_WORDS):
            return amount, None
        if any(w in ql for w in BELOW_WORDS):
            return None, amount
        return None, None

    # ---------------- Intent router ----------------

    def answer(self, question):
        q = question.strip()
        ql = q.lower()

        if not q:
            return "Ask me about a transaction ID, a count or total, why something was flagged, or a specific vendor."

        if any(p in ql for p in ("what can you", "what can i ask", "help", "capabilities", "how do you work")):
            return self._answer_help()

        tid = self._extract_transaction_id(q)

        is_followup_phrase = bool(re.match(
            r"^(why\??|why though\??|why not\??|explain( that| it)?\??|and\??|"
            r"what about it\??|is it flagged\??|what'?s (its |the )?status\??)$",
            ql
        )) or (not tid and any(w in ql for w in ("why", "explain")) and len(ql.split()) <= 6)
        if not tid and self.last_transaction_id and is_followup_phrase:
            tid = self.last_transaction_id

        if tid and ("why" in ql or "explain" in ql):
            return self._answer_explanation(tid)

        if tid and any(w in ql for w in ("status", "look up", "lookup", "flagged", "about", "what")):
            return self._answer_lookup(tid)

        list_trigger = any(w in ql for w in ("list", "show", "which transactions", "give me", "all "))
        mentions_category = any(w in ql for w in ("duplicate", "fee anomal", "flagged", "exception", "confidence"))
        if list_trigger and mentions_category:
            return self._answer_list(ql)

        vendor = self._extract_vendor(ql)
        if vendor and any(w in ql for w in ("flag", "duplicate", "fee", "exception")):
            return self._answer_list(ql, forced_vendor=vendor)

        if "vendor" in ql and ("most" in ql or "top" in ql or "ranking" in ql or "which vendor" in ql):
            return self._answer_vendor_ranking()

        amt_min, amt_max = self._extract_amount_filter(ql)
        if amt_min is not None or amt_max is not None:
            return self._answer_amount_filter(ql, amt_min, amt_max)

        if "confidence" in ql and ("how many" in ql or "count" in ql):
            return self._answer_confidence_count(ql)

        if "how many" in ql or "count" in ql or "total" in ql or "sum" in ql:
            return self._answer_aggregation(ql)

        if tid:
            return self._answer_lookup(tid)

        return (
            "I can't answer that from this batch's data. Try asking about a specific transaction ID, "
            "a count or total of flagged items, why something was flagged, a vendor by name, or say "
            "'help' to see everything I can do."
        )

    # ---------------- Answer builders (each cites transaction_ids) ----------------

    def _answer_help(self):
        return (
            "I can answer questions grounded in this batch's flagged output:\n"
            "- Lookup: \"What's the status of TXN-0032?\"\n"
            "- Explanation: \"Why was TXN-0045 flagged?\" (and follow-ups like \"why?\" after that)\n"
            "- Aggregation: \"How many fee anomalies were found?\" / \"Total amount of flagged duplicates?\"\n"
            "- Filters: \"List high confidence duplicates\" / \"Duplicate payments over $3000\"\n"
            "- Vendor: \"Any flags for Acme Corp?\"\n"
            "- Ranking: \"Which vendor has the most flagged transactions?\"\n"
            "I'll say so if a question falls outside this batch rather than guessing."
        )

    def _answer_lookup(self, tid):
        result = self.lookup_transaction(tid)
        if not result["exists"]:
            return f"I don't have a transaction {tid} in this batch \u2014 that ID doesn't exist in the data, so I can't report a status for it."
        self.last_transaction_id = tid
        if result["flagged"]:
            c = result["flagged"]
            return f"{tid} was flagged as {c['display_tag']} ({c['confidence']} confidence). Reason: {c['reason']} [source: {tid}]"
        return f"{tid} exists in the batch and was NOT flagged \u2014 it's clean. [source: {tid}]"

    def _answer_explanation(self, tid):
        result = self.lookup_transaction(tid)
        if not result["exists"]:
            return f"I don't have a transaction {tid} in this batch, so I can't explain a flag that doesn't exist."
        self.last_transaction_id = tid
        if not result["flagged"]:
            return f"{tid} was not flagged \u2014 there's nothing to explain; it was treated as a clean transaction. [source: {tid}]"
        c = result["flagged"]
        cite = tid
        if c.get("matched_against"):
            cite += f", {c['matched_against']}"
        return f"{tid} was flagged as {c['display_tag']}. {c['reason']} [source: {cite}]"

    def _answer_vendor_ranking(self):
        counts = self.aggregate(None, group_by="vendor")
        if not counts:
            return "No vendors had flagged duplicate payments in this batch."
        ranked = sorted(counts.items(), key=lambda kv: -len(kv[1]))
        top_vendor, top_ids = ranked[0]
        lines = [f"{top_vendor} has the most flagged transactions: {len(top_ids)} duplicate payment(s). [source: {', '.join(top_ids)}]"]
        if len(ranked) > 1:
            lines.append("Full ranking: " + "; ".join(f"{v} ({len(ids)})" for v, ids in ranked))
        return "\n".join(lines)

    def _answer_aggregation(self, ql):
        if "fee anomal" in ql:
            n = self.aggregate("count", tag="fee_anomaly")
            ids = [c["transaction_id"] for c in self.filter_transactions(tag="fee_anomaly")]
            if n == 0:
                return "0 fee anomalies were found in this batch."
            return f"{n} fee anomalies were found. [source: {', '.join(ids)}]"
        if "duplicate" in ql and ("total" in ql or "sum" in ql or "dollar" in ql or "amount" in ql):
            total, ids = self.aggregate("total_amount", tag="duplicate_payment")
            if not ids:
                return "No duplicate payments were flagged, so there's no total to report."
            return f"The total dollar amount of flagged duplicate payments is ${total:,.2f}. [source: {', '.join(ids)}]"
        if "duplicate" in ql:
            n = self.aggregate("count", tag="duplicate_payment")
            ids = [c["transaction_id"] for c in self.filter_transactions(tag="duplicate_payment")]
            if n == 0:
                return "0 duplicate payments were found in this batch."
            return f"{n} duplicate payments were found. [source: {', '.join(ids)}]"
        if "fee" in ql and ("total" in ql or "sum" in ql or "dollar" in ql or "amount" in ql):
            total, ids = self.aggregate("total_amount", tag="fee_anomaly")
            if not ids:
                return "No fee anomalies were flagged, so there's no total to report."
            return f"The total dollar amount of flagged fee anomalies is ${total:,.2f}. [source: {', '.join(ids)}]"
        n = len(self.combined)
        ids = [c["transaction_id"] for c in self.combined]
        if n == 0:
            return "0 exceptions were flagged in this batch."
        return f"{n} total exceptions were flagged in this batch. [source: {', '.join(ids)}]"

    def _answer_confidence_count(self, ql):
        conf = None
        for c in ("high", "medium", "low"):
            if c in ql:
                conf = c
                break
        tag = "duplicate_payment" if "duplicate" in ql else ("fee_anomaly" if "fee" in ql else None)
        rows = self.filter_transactions(tag=tag, confidence=conf)
        ids = [c["transaction_id"] for c in rows]
        desc = " ".join(filter(None, [
            f"{conf} confidence" if conf else None,
            TAG_PLURAL[tag] if tag else "flags",
        ])).strip()
        if not ids:
            return f"0 flagged items match '{desc or 'that confidence level'}'."
        return f"{len(ids)} flagged item(s) match '{desc}'. [source: {', '.join(ids)}]"

    def _answer_amount_filter(self, ql, amt_min, amt_max):
        tag = "duplicate_payment" if "duplicate" in ql else ("fee_anomaly" if "fee" in ql else None)
        rows = self.filter_transactions(tag=tag, amount_min=amt_min, amount_max=amt_max)
        ids = [c["transaction_id"] for c in rows]
        bound = f"over ${amt_min:,.2f}" if amt_min is not None else f"under ${amt_max:,.2f}"
        label = TAG_PLURAL[tag]
        if not ids:
            return f"No {label} were found {bound}."
        lines = [f"{len(ids)} {label} {bound}:"]
        for c in rows:
            lines.append(f"  - {c['transaction_id']}: ${self._amount_of(c):,.2f} \u2014 {c['reason']}")
        lines.append(f"[source: {', '.join(ids)}]")
        return "\n".join(lines)

    def _answer_list(self, ql, forced_vendor=None):
        tag = None
        if "duplicate" in ql:
            tag = "duplicate_payment"
        elif "fee" in ql:
            tag = "fee_anomaly"
        confidence = None
        for c in ("high", "medium", "low"):
            if c in ql:
                confidence = c
                break
        vendor = forced_vendor or self._extract_vendor(ql)

        rows = self.filter_transactions(tag=tag, confidence=confidence, vendor=vendor)
        ids = [c["transaction_id"] for c in rows]

        filters_desc = " ".join(filter(None, [
            f"{confidence} confidence" if confidence else None,
            TAG_PLURAL[tag] if tag else "flagged items",
            f"for {vendor}" if vendor else "",
        ])).strip()

        if not ids:
            return f"No {filters_desc} found in this batch."

        lines = [f"{len(ids)} {filters_desc}:"]
        for c in rows:
            lines.append(f"  - {c['transaction_id']} ({c['confidence']}): {c['reason']}")
        lines.append(f"[source: {', '.join(ids)}]")
        self.last_transaction_id = rows[0]["transaction_id"] if len(rows) == 1 else self.last_transaction_id
        return "\n".join(lines)
