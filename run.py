import json
from graph import build_graph
from state import Email
from langgraph.types import Command


def load_emails(path="emails.json"):
    with open(path) as f:
        return [Email(**e) for e in json.load(f)]


def _get_interrupt_payload(app, config):
    
    snap = app.get_state(config)
    if not snap or not snap.next:
        return None
    for task in (getattr(snap, "tasks", None) or []):
        for intr in (getattr(task, "interrupts", None) or []):
            val = getattr(intr, "value", intr)
            if isinstance(val, dict) and "review_queue" in val:
                return val
    return None


def run():
    app = build_graph()
    config = {"configurable": {"thread_id": "inbox-run-001"}}

    # ── First invocation: ingest → classify → extract → review_gate ──
    out = app.invoke({"emails": load_emails()}, config=config)

    # ── Detect pause via snapshot.tasks (authoritative on 0.2.76) ──
    payload = _get_interrupt_payload(app, config)

    # ── Fallback: some versions also surface out['__interrupt__'] ──
    if payload is None and isinstance(out, dict) and "__interrupt__" in out:
        raw = out["__interrupt__"]
        candidate = raw[0].value if hasattr(raw[0], "value") else raw[0]
        if isinstance(candidate, dict) and "review_queue" in candidate:
            payload = candidate

    if payload:
        
        print(json.dumps(payload, indent=2, default=str))

        decisions = {}
        for item in payload.get("review_queue", []):
            eid = item["email_id"]
            if item["kind"] == "high_amount":
                decisions[eid] = {"approved": True,
                                  "amount": item["transaction"]["amount"]}
            else:
                decisions[eid] = {"approved": True,
                                  "category": item["classification"]["category"]}

        # Resume on the same thread
        out = app.invoke(Command(resume=decisions), config=config)

    # ── Print final summary, if aggregate ran ──
    summary = out.get("summary") if isinstance(out, dict) else None
    if not summary:
        print("\n=== NO SUMMARY PRODUCED ===")
        print("State keys:", list(out.keys()) if isinstance(out, dict) else "n/a")
        print("Full state:", json.dumps(out, indent=2, default=str)
              if isinstance(out, dict) else str(out))
        return

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    run()