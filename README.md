# Inbox Triage & Spend Analysis — LangGraph Workflow

A LangGraph pipeline that reads a batch of emails, classifies each into
**Work**, **Online Orders**, **Subscriptions**, or **Miscellaneous**, extracts
every recorded payment, gates risky or low-confidence items through a
human-in-the-loop interrupt, deduplicates transactions, and produces a
month-to-date spend summary answering:

> **"How much did I spend this month, and where?"**

Use any model api example (OpenAI-compatible API) for structured classification and
extraction, `python-dotenv` for configuration, and LangGraph's `interrupt()` +
`MemorySaver` checkpointer so a run can pause at a review gate and resume on
the same `thread_id`.

---
                  ┌──────────────────┐
                  │      START       │
                  └────────┬─────────┘
                           │  emails: list[Email]
                           ▼
                  ┌──────────────────┐
                  │     ingest       │  deterministic, no LLM
                  └────────┬─────────┘
                           │  state["emails"]
                           ▼
                  ┌──────────────────┐
                  │    classify      │  llm × 17
                  │  (per-email loop)│  → Classification{category, confidence, reason}
                  └────────┬─────────┘
                           │  classifications  (operator.add)
                           ▼
                     ╱────────────╲
                    ╱  conditional  ╲
                   ╱     edge        ╲
                  ╱  route_maybe_    ╲
                 ╱   extract()        ╲
                ╱     → "extract"      ╲
                ╲                       ╱
                 ╲─────────────────────╱
                           │
                           ▼
                  ┌──────────────────┐
                  │     extract      │  LLM × 12 (regex pre-filter)
                  │  (per-email loop)│  → TransactionList.transactions
                  └────────┬─────────┘
                           │  transactions  (operator.add)
                           ▼
                  ┌──────────────────┐
                  │   review_gate    │  collect needs[]:
                  │                  │    • confidence < 0.70
                  │                  │    • amount  > ₹5,000
                  │                  │  interrupt({review_queue})
                  └────────┬─────────┘
                           │  review_queue  (operator.add)
                           │  review_decisions (dict-merge)
                           ▼
                  ╔══════════════════╗
                  ║  ⏸  PAUSED       ║  ← LangGraph checkpointer
                  ║  (MemorySaver)   ║     writes this state
                  ╚════════┬═════════╝
                           │  run.py: Command(resume=decisions)
                           │  same thread_id → resumes here
                           ▼
                  ┌──────────────────┐
                  │   apply_review   │  merge reviewer decisions:
                  │                  │    • category override
                  │                  │    • amount/merchant override
                  │                  │    • approved=False → drop txn
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    aggregate     │  dedupe → per-category,
                  │                  │  per-merchant, month-to-date
                  └────────┬─────────┘
                           │  summary
                           ▼
                     ┌──────────┐
                     │   END    │
                     └──────────┘



---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY

# 3. Run
python run.py