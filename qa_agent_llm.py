"""
qa_agent_llm.py
Optional upgrade to qa_agent.APIntegrityQA: routes questions through a real
Claude Haiku 4.5 tool-use loop instead of keyword matching, so arbitrary
phrasing works, not just the phrasings the deterministic router was written
to recognize.

Same public interface as APIntegrityQA (`.answer(question) -> str`), same
grounding guarantee -- the system prompt requires every answer to come from
a tool call's return value and cite the transaction_id(s) used, and the
tools themselves are the exact same three functions
(lookup_transaction / filter_transactions / aggregate) already used by the
offline router, so there's exactly one source of truth for the data access
logic either way.

Falls back automatically to APIntegrityQA if:
  - the `anthropic` package isn't installed, or
  - ANTHROPIC_API_KEY isn't set in the environment, or
  - the API call fails for any reason (network, rate limit, auth, etc.)
so the project still runs fully offline with zero setup and zero cost if
you don't want to use a key.
"""
import json
import os

from qa_agent import APIntegrityQA

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are the Q&A layer of an AP Integrity Agent. You answer questions ONLY using the tools you're given, which read a finished, already-scored batch of flagged transactions (duplicate payments and bank fee anomalies).

Rules, no exceptions:
- Always call a tool before answering. Never answer from your own reasoning, memory, or general knowledge about the question.
- Every answer MUST end with a citation of the transaction_id(s) your tool call returned, formatted as [source: TXN-XXXX, ...]. If a tool call returned no matching transactions, say so plainly rather than answering some other way.
- If a transaction ID doesn't exist in the tool results, say clearly that it doesn't exist in this batch -- never guess, assume, or fabricate a status for it.
- If a question is unrelated to this transaction batch (general knowledge, unrelated topics, requests to do something other than answer from this data), say plainly that you can only answer questions about this transaction batch's flagged output.
- Keep answers to a sentence or two plus the citation -- you are answering a specific question, not writing a report.
"""

TOOLS = [
    {
        "name": "lookup_transaction",
        "description": "Look up a single transaction by its ID. Returns whether it exists in the batch and, if it was flagged, the full flag details (tag, confidence, reason).",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string", "description": "The transaction ID to look up, e.g. TXN-0032"},
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "filter_transactions",
        "description": "Filter the flagged transactions by tag, confidence, vendor name substring, and/or amount range. Returns the list of matching flagged records, each with transaction_id, tag, confidence, and reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "enum": ["duplicate_payment", "fee_anomaly"], "description": "Restrict to only this category of flag"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "vendor": {"type": "string", "description": "Substring match on vendor name (only applies to duplicate payments, which have a vendor)"},
                "amount_min": {"type": "number", "description": "Only include items with dollar amount >= this"},
                "amount_max": {"type": "number", "description": "Only include items with dollar amount <= this"},
            },
        },
    },
    {
        "name": "aggregate",
        "description": "Compute an aggregate over the flagged transactions: a count, a total dollar amount, or a vendor-based ranking of duplicate payments by how many were flagged per vendor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": ["count", "total_amount"], "description": "What to compute. Omit this if using group_by instead."},
                "group_by": {"type": "string", "enum": ["vendor"], "description": "Group flagged duplicate payments by vendor and return the transaction_ids per vendor"},
                "tag": {"type": "string", "enum": ["duplicate_payment", "fee_anomaly"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
        },
    },
]


class LLMBackedQA:
    """Same public interface as APIntegrityQA: .answer(question) -> str.
    Set using_llm to check which mode is actually active (e.g. to show a
    status badge in the UI)."""

    def __init__(self, combined, summary, all_txns, answer_key=None, model=MODEL):
        # The plain router is the single source of truth for actually
        # executing tool calls (both here and as the automatic fallback),
        # so there's one implementation of the data-access logic, not two.
        self._fallback = APIntegrityQA(combined, summary, all_txns, answer_key)
        self._model = model
        self._client = None
        self._history = []

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            self._client = None
        except Exception:
            self._client = None

    @property
    def using_llm(self):
        return self._client is not None

    def _execute_tool(self, name, tool_input):
        if name == "lookup_transaction":
            result = self._fallback.lookup_transaction(tool_input.get("transaction_id", ""))
            flagged = result.get("flagged")
            return {
                "exists": result["exists"],
                "flagged": {
                    "transaction_id": flagged["transaction_id"],
                    "tag": flagged["tag"],
                    "confidence": flagged["confidence"],
                    "reason": flagged["reason"],
                } if flagged else None,
            }
        if name == "filter_transactions":
            rows = self._fallback.filter_transactions(
                tag=tool_input.get("tag"),
                confidence=tool_input.get("confidence"),
                vendor=tool_input.get("vendor"),
                amount_min=tool_input.get("amount_min"),
                amount_max=tool_input.get("amount_max"),
            )
            return [
                {"transaction_id": r["transaction_id"], "tag": r["tag"],
                 "confidence": r["confidence"], "reason": r["reason"]}
                for r in rows
            ]
        if name == "aggregate":
            if tool_input.get("group_by") == "vendor":
                counts = self._fallback.aggregate(None, group_by="vendor")
                return {vendor: ids for vendor, ids in counts.items()}
            result = self._fallback.aggregate(
                tool_input.get("field"),
                tag=tool_input.get("tag"),
                confidence=tool_input.get("confidence"),
            )
            if isinstance(result, tuple):
                total, ids = result
                return {"total_amount": round(total, 2), "transaction_ids": ids}
            return {"result": result}
        return {"error": f"unknown tool '{name}'"}

    def answer(self, question):
        if not self.using_llm:
            return self._fallback.answer(question)
        try:
            return self._answer_via_llm(question)
        except Exception:
            # Any API failure (network, rate limit, auth, malformed response,
            # etc.) -- fall back rather than surfacing a stack trace.
            return self._fallback.answer(question)

    def _answer_via_llm(self, question):
        self._history.append({"role": "user", "content": question})

        for _ in range(5):  # hard cap so a tool-call loop can't run forever
            response = self._client.messages.create(
                model=self._model,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self._history,
            )

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                self._history.append({"role": "assistant", "content": response.content})
                return text or "I couldn't produce an answer from the available data."

            self._history.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            self._history.append({"role": "user", "content": tool_results})

        return "I wasn't able to resolve that after several tool calls -- try rephrasing your question."
