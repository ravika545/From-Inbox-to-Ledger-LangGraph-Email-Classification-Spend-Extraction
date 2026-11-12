from __future__ import annotations
import re, logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import (
    GraphState, Email, Classification, Transaction, TransactionList,
)
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    CONFIDENCE_THRESHOLD, AMOUNT_THRESHOLD_INR,
)

log = logging.getLogger(__name__)

# ---------- DeepSeek via OpenAI-compatible client ----------
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0,
    max_retries=2,
    timeout=60,
)


# ---------- 1. INGEST ----------
def ingest(state: GraphState) -> GraphState:
    return {"emails": state["emails"]}


# ---------- 2. CLASSIFY ----------
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You triage an inbox into exactly one category: Work, Online Orders, "
     "Subscriptions, or Miscellaneous. Return a confidence 0-1 and a one-line reason. "
     "Rules: a newsletter/promo is Subscriptions if it is a recurring service or a "
     "marketing digest; a promo from a store you buy from is still Miscellaneous/"
     "Subscriptions unless it is an order confirmation. A vendor invoice sent to you "
     "for work is Work. Subscription renewals that charge a card are Subscriptions."),
    ("human",
     "From: {from_}\nSubject: {subject}\nDate: {date}\nBody: {body}"),
])


def classify_one(email: Email) -> dict:
    chain = CLASSIFY_PROMPT | llm.with_structured_output(Classification)
    payload = {"from_": email.from_, "subject": email.subject,
               "date": email.date, "body": email.body}
    try:
        result: Classification = chain.invoke(payload)
    except Exception as e:
        log.warning("classify failed for %s: %s — retrying", email.id, e)
        try:
            result = chain.invoke(payload)
        except Exception as e2:
            log.error("classify failed again for %s: %s — fallback", email.id, e2)
            return {"email_id": email.id, "category": "Miscellaneous",
                    "confidence": 0.0, "reason": f"LLM failure fallback: {e2}"}
    return {"email_id": email.id, **result.model_dump()}


def classify(state: GraphState) -> GraphState:
    return {"classifications": [classify_one(e) for e in state["emails"]]}


# ---------- 3. ROUTE ----------
PAYMENT_SIGNAL = re.compile(
    r"(paid|charged|invoice|total|debit|debited|fare|receipt|renew(ed|al)?|"
    r"payment received|amount|rs\.?|\$|₹|usd|inr)",
    re.I,
)


def route_maybe_extract(state: GraphState) -> str:
    return "extract"


# ---------- 4. EXTRACT SPEND ----------
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract a transaction from this email. Return an object with a single "
     "field `transactions` whose value is a list. If there is NO real payment "
     "recorded (e.g. a promo, a newsletter, a work email without an invoice "
     "amount), return {{\"transactions\": []}}. Never guess an amount. "
     "Currency: default INR if 'Rs.' appears, else the symbol's currency. "
     "Date: use the date in the email; if absent use the email date. "
     "Merchant: the payee. Category: Shopping, Food, Subscriptions, Software, "
     "Transport, Bills, or Other."),
    ("human",
     "Email date: {date}\nSubject: {subject}\nBody: {body}"),
])


def extract_for(email: Email) -> list[dict]:
    if not PAYMENT_SIGNAL.search(f"{email.subject} {email.body}"):
        return []
    chain = EXTRACT_PROMPT | llm.with_structured_output(TransactionList)
    try:
        result: TransactionList = chain.invoke(
            {"date": email.date, "subject": email.subject, "body": email.body}
        )
    except Exception as e:
        log.error("extract failed for %s: %s", email.id, e)
        return []
    out = []
    for t in result.transactions:
        d = t.model_dump()
        d["source_email_id"] = email.id
        out.append(d)
    return out


def extract(state: GraphState) -> GraphState:
    txns: list[dict] = []
    for e in state["emails"]:
        txns.extend(extract_for(e))
    return {"transactions": txns}


# ---------- 5. REVIEW GATE (HITL) ----------
def review_gate(state: GraphState) -> GraphState:
    needs: list[dict] = []
    for c in state["classifications"]:
        if c["confidence"] < CONFIDENCE_THRESHOLD:
            needs.append({"email_id": c["email_id"],
                          "kind": "low_confidence",
                          "classification": c})
    for t in state["transactions"]:
        if t["amount"] > AMOUNT_THRESHOLD_INR:
            needs.append({"email_id": t["source_email_id"],
                          "kind": "high_amount",
                          "transaction": t})

    if not needs:
        return {"review_queue": [], "review_decisions": {}}

    decisions = interrupt({"review_queue": needs})
    return {"review_queue": needs, "review_decisions": decisions or {}}


def apply_review(state: GraphState) -> GraphState:
    dec = state.get("review_decisions") or {}
    if not dec:
        return {}

    new_classes, new_txns = [], []
    for c in state["classifications"]:
        d = dec.get(c["email_id"])
        if d and "category" in d:
            c = {**c, "category": d["category"],
                 "reason": c["reason"] + " | reviewer override"}
        new_classes.append(c)

    for t in state["transactions"]:
        d = dec.get(t["source_email_id"])
        if d:
            if d.get("approved") is False:
                continue
            t = {**t, **{k: d[k] for k in ("amount", "merchant") if k in d}}
        new_txns.append(t)

    return {"classifications": new_classes, "transactions": new_txns}


# ---------- 6. AGGREGATE ----------
def aggregate(state: GraphState) -> GraphState:
    seen, uniq = set(), []
    for t in state["transactions"]:
        key = (round(t["amount"], 2),
               t["merchant"].lower().strip(),
               t["date"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    per_cat, per_merchant, total = {}, {}, 0.0
    for t in uniq:
        per_cat[t["category"]] = per_cat.get(t["category"], 0) + t["amount"]
        per_merchant[t["merchant"]] = per_merchant.get(t["merchant"], 0) + t["amount"]
        total += t["amount"]

    email_counts: dict[str, int] = {}
    for c in state["classifications"]:
        email_counts[c["category"]] = email_counts.get(c["category"], 0) + 1

    return {"transactions": uniq, "summary": {
        "month_to_date_total": round(total, 2),
        "currency": "INR (normalized — see notes)",
        "spend_per_category": {k: round(v, 2) for k, v in per_cat.items()},
        "spend_per_merchant": {k: round(v, 2) for k, v in per_merchant.items()},
        "email_count_per_category": email_counts,
        "transactions_after_dedup": len(uniq),
    }}


# ---------- BUILD GRAPH ----------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("ingest", ingest)
    g.add_node("classify", classify)
    g.add_node("extract", extract)
    g.add_node("review_gate", review_gate)
    g.add_node("apply_review", apply_review)
    g.add_node("aggregate", aggregate)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_conditional_edges("classify", route_maybe_extract,
                            {"extract": "extract"})
    g.add_edge("extract", "review_gate")
    g.add_edge("review_gate", "apply_review")
    g.add_edge("apply_review", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile(checkpointer=MemorySaver())