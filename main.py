"""
main.py
Orchestrates the full AP Integrity Agent pipeline (Build Order, Section 8):
  1-2. (already run once via generate_data.py to produce data/*.csv)
  3-6. run both detectors, merge into combined report, score vs answer key
  7-9. run the mandatory Q&A layer against all 4 required question types
       plus an out-of-scope question, to prove grounding (Section 7.3)

Usage:
    python3 main.py            # full report + scripted Q&A demo
    python3 main.py --interactive   # also drop into a live Q&A prompt
"""
import sys
import time
from report import (
    load_transactions, load_answer_key, load_answer_key_notes,
    build_combined_output, render_report,
)
from qa_agent_llm import LLMBackedQA


def run_qa_demo(qa):
    print("\n" + "=" * 60)
    print("Q&A LAYER DEMO \u2014 all 4 required question types + out-of-scope case")
    print("=" * 60)

    demo_questions = [
        ("LOOKUP", "What's the status of TXN-0038?"),
        ("AGGREGATION", "How many fee anomalies were found?"),
        ("AGGREGATION", "What's the total dollar amount of flagged duplicate payments?"),
        ("EXPLANATION", "Why was TXN-0036 flagged as a duplicate?"),
        ("SUMMARY/RANKING", "Which vendor has the most flagged transactions?"),
        ("OUT-OF-SCOPE", "What's the status of TXN-9999?"),
        ("OUT-OF-SCOPE", "What was the weather like on the invoice date?"),
    ]

    for qtype, q in demo_questions:
        print(f"\n[{qtype}] Q: {q}")
        print(f"A: {qa.answer(q)}")


def main():
    txns = load_transactions()
    answer_key = load_answer_key()
    answer_key_notes = load_answer_key_notes()

    start = time.perf_counter()
    combined, summary, unresolved = build_combined_output(txns)
    elapsed = time.perf_counter() - start

    report_text, score_result = render_report(
        combined, summary, answer_key, elapsed_seconds=elapsed,
        answer_key_notes=answer_key_notes, unresolved=unresolved,
    )

    print(report_text)

    qa = LLMBackedQA(combined, summary, txns, answer_key)
    mode = "Claude Haiku 4.5 (real tool-use)" if qa.using_llm else "offline keyword router (set ANTHROPIC_API_KEY to use Claude)"
    print(f"\nQ&A mode: {mode}")
    run_qa_demo(qa)

    if "--interactive" in sys.argv:
        print("\n" + "=" * 60)
        print("Interactive mode \u2014 ask your own questions (type 'quit' to exit)")
        print("=" * 60)
        while True:
            try:
                q = input("\nYour question: ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            print(f"A: {qa.answer(q)}")


if __name__ == "__main__":
    main()
