# Instamart Refund Support Agent - offline simulation only

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import csv
import json
import re
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
LEDGER = OUT / "refund_ledger.jsonl"
RUN_LOG = OUT / "agent_runs.jsonl"

sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import mock_llm
import policy_qa

DOCS = policy_qa.load_policy_documents()
CHUNKS = policy_qa.chunk_documents(DOCS, overlap=10)
RETRIEVER = policy_qa.PolicyRetriever(CHUNKS)

# Batch demonstrations use an explicitly synthetic request time two days
# after estimated delivery. Interactive use requires the request timestamp.
SIMULATED_REQUEST_DELAY_DAYS = 2


@contextmanager
def _isolated_demo_ledger(label):
    """Use a temporary ledger so demonstrations never alter audit history."""
    global LEDGER
    original_ledger = LEDGER
    demo_ledger = STATE / f"{label}-{uuid.uuid4().hex}.jsonl"
    STATE.mkdir(parents=True, exist_ok=True)
    LEDGER = demo_ledger
    try:
        yield demo_ledger
    finally:
        LEDGER = original_ledger
        if demo_ledger.exists():
            demo_ledger.unlink()


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in SQLite.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant policy chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a full refund for delivery later than 45 minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible simulated refund of INR 1000 or less.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
                "request_datetime": {"type": "string"},
            },
            "required": ["order_id", "refund_inr", "request_datetime"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human review.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
                "request_datetime": {"type": "string"},
            },
            "required": ["order_id", "refund_inr", "request_datetime"],
        },
    },
]
TOOLS = [{"type": "function", "function": spec} for spec in TOOLS_SPEC]


# ------------------------------------------------------------
# Date handling and order lookup
# ------------------------------------------------------------

def _parse_local_datetime(value, field_name):
    """Parse an ISO local timestamp; this project stores naive local times."""
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required.")
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must use ISO format YYYY-MM-DD HH:MM:SS."
        ) from exc
    if parsed.tzinfo is not None:
        raise ValueError(
            f"{field_name} must be a local timestamp without a UTC offset."
        )
    return parsed


def get_order(order_id):
    """Retrieve an order and estimate delivery time from order time + duration."""
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes, order_ts
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    order = {key: row[key] for key in (
        "order_id", "status", "order_amount", "delivery_minutes", "order_ts"
    )}
    order["estimated_delivery_at"] = None

    if order["order_ts"] is not None and order["delivery_minutes"] is not None:
        try:
            ordered_at = _parse_local_datetime(order["order_ts"], "order_ts")
            estimated_delivery = ordered_at + timedelta(
                minutes=float(order["delivery_minutes"])
            )
            order["estimated_delivery_at"] = estimated_delivery.isoformat(sep=" ")
        except (TypeError, ValueError):
            # A missing/invalid timestamp is handled safely by the eligibility check.
            order["estimated_delivery_at"] = None

    return order


def simulated_request_datetime(order_id, delay_days=SIMULATED_REQUEST_DELAY_DAYS):
    """Create a clearly labelled deterministic request-date fixture for batch demos.

    Estimated delivery = order_ts + delivery_minutes. The synthetic request is
    placed delay_days after that estimate. It is test metadata, not a real event.
    """
    order = get_order(order_id)
    if "error" in order or not order.get("estimated_delivery_at"):
        return None
    delivered_at = _parse_local_datetime(
        order["estimated_delivery_at"], "estimated_delivery_at"
    )
    return (delivered_at + timedelta(days=int(delay_days))).isoformat(sep=" ")


def check_seven_day_window(order, request_datetime):
    """Return None when eligible, otherwise a clear error requiring denial/review."""
    estimated_delivery_at = order.get("estimated_delivery_at")
    if not estimated_delivery_at:
        return {
            "error": (
                "Delivery timestamp is unavailable; human review is required "
                "to verify the seven-day refund window."
            ),
            "review_required": True,
        }

    try:
        delivery_time = _parse_local_datetime(
            estimated_delivery_at, "estimated_delivery_at"
        )
        request_time = _parse_local_datetime(
            request_datetime, "request_datetime"
        )
    except ValueError as exc:
        return {"error": str(exc), "review_required": True}

    elapsed = request_time - delivery_time
    if elapsed.total_seconds() < 0:
        return {
            "error": "Refund request timestamp is earlier than delivery; human review is required.",
            "review_required": True,
        }

    if elapsed > timedelta(days=7):
        return {
            "error": "Refund request was made more than 7 days after delivery; refund is outside the policy window.",
            "review_required": False,
        }

    return None


# ------------------------------------------------------------
# Other tool functions
# ------------------------------------------------------------

def search_policy(query):
    """Retrieve relevant policy document chunks."""
    return {"results": RETRIEVER.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate the simulated late-delivery refund amount."""
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger(tool_name, order_id, amount):
    """Append a simulated refund or escalation record."""
    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(amount), 2),
    }
    with LEDGER.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _has_previous_refund(order_id):
    """Check whether a refund or escalation was already recorded for an order."""
    if not LEDGER.exists():
        return False
    recorded_actions = {"issue_refund", "escalate_to_human"}
    with LEDGER.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("tool_name") in recorded_actions
                and entry.get("order_id") == order_id
            ):
                return True
    return False


def issue_refund(order_id, refund_inr, request_datetime=None):
    """Issue an eligible simulated refund of INR 1000 or less."""
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}
    if _has_previous_refund(order_id):
        return {"error": "A refund has already been issued for this order."}
    if str(order["status"]).lower() != "completed":
        return {"error": "Only completed orders qualify for refunds."}

    minutes = order["delivery_minutes"]
    if minutes is None or float(minutes) <= 45:
        return {"error": "Order does not qualify for a late-delivery refund."}

    window_error = check_seven_day_window(order, request_datetime)
    if window_error:
        return window_error

    actual = round(float(order["order_amount"]), 2)
    if actual > 1000:
        return {"error": "Refund above INR 1000 requires human approval."}
    if abs(float(refund_inr) - actual) > 0.01:
        return {"error": "Refund amount does not match the order amount."}

    _ledger("issue_refund", order_id, actual)
    return {"refund_inr": actual}


def escalate_to_human(order_id, refund_inr, request_datetime=None):
    """Record an eligible refund above INR 1000 for human review."""
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}
    if _has_previous_refund(order_id):
        return {
            "error": "A refund or escalation has already been recorded for this order."
        }
    if str(order["status"]).lower() != "completed":
        return {"error": "Only completed orders qualify for refunds."}

    minutes = order["delivery_minutes"]
    if minutes is None or float(minutes) <= 45:
        return {"error": "Order does not qualify for a late-delivery refund."}

    window_error = check_seven_day_window(order, request_datetime)
    if window_error:
        return window_error

    actual = round(float(order["order_amount"]), 2)
    if actual <= 1000:
        return {"error": "Refund is not eligible for escalation."}
    if abs(float(refund_inr) - actual) > 0.01:
        return {"error": "Refund amount does not match the order amount."}

    _ledger("escalate_to_human", order_id, actual)
    return {"refund_inr": actual}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def run_tool(name, arguments):
    """Validate arguments and execute a registered tool."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}
    return TOOL_MAP[name](**arguments)


def field(obj, name, default=None):
    """Read a field from either a dictionary or an object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def safe_tool_call(call):
    """Normalize mock-model tool calls."""
    fn = field(call, "function")
    name = field(call, "name") or field(fn, "name")
    args = field(call, "arguments")
    if args is None:
        args = field(fn, "arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {
        "id": field(call, "id", "tool_call"),
        "name": name,
        "arguments": args or {},
    }


def status_for(answer):
    """Extract the status prefix from a model answer."""
    return answer.split(":", 1)[0].strip().upper() if answer else "UNKNOWN"


# ------------------------------------------------------------
# Human handoff
# ------------------------------------------------------------

def fallback(status, steps):
    """Create a human-handoff message containing completed work."""
    summaries = {
        "FALLBACK_MAX_ITERATIONS": "The agent reached its model-call limit.",
        "FALLBACK_TOKEN_LIMIT": "The token budget was exceeded before the requested tool ran.",
        "FALLBACK_TIMEOUT": "The agent reached its time limit.",
    }
    completed_work = []
    for step in steps[-4:]:
        tool_name = step.get("tool_name", "unknown_tool")
        observation = step.get("observation", {})
        if not isinstance(observation, dict):
            detail = "Tool completed."
        elif "error" in observation:
            detail = str(observation["error"])
        elif tool_name == "get_order":
            detail = (
                f"Order status: {observation.get('status', 'unknown')}; "
                f"amount: INR {observation.get('order_amount', 'unknown')}"
            )
        elif tool_name == "search_policy":
            sources = sorted({
                item.get("source", "")
                for item in observation.get("results", [])
                if item.get("source")
            })
            detail = "Policy sources: " + ", ".join(sources) if sources else "Policy search completed."
        elif "refund_inr" in observation:
            detail = f"Calculated refund: INR {observation['refund_inr']}"
        else:
            detail = "Tool completed."
        completed_work.append(f"{tool_name}: {detail}")

    handoff_summary = (
        "Completed work: " + " | ".join(completed_work) + "."
        if completed_work
        else "No tools were completed before handoff."
    )
    reason = summaries.get(status, "The agent could not complete the request.")
    return (
        f"{status}: {reason} {handoff_summary} "
        "Please hand this case to a human support agent."
    )


class SimulatedCrash(RuntimeError):
    """Raised by the resume demonstration after a saved tool observation."""


# ------------------------------------------------------------
# Checkpoint and agent execution
# ------------------------------------------------------------

def checkpoint_path(run_id):
    return STATE / f"{run_id}.json"


def save_checkpoint(state):
    STATE.mkdir(parents=True, exist_ok=True)
    with checkpoint_path(state["run_id"]).open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def finish(state, status, answer):
    """Save final state and return the public result."""
    state["status"] = status
    state["final_answer"] = answer
    save_checkpoint(state)
    return {
        key: state[key]
        for key in (
            "run_id", "scenario_id", "status", "final_answer",
            "steps", "llm_calls", "tokens_used", "tools_executed",
        )
    }


def continue_agent(state, crash_after_tool_executions=None):
    """Continue an agent run, saving every tool observation."""
    started = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return finish(
                state, "FALLBACK_TIMEOUT",
                fallback("FALLBACK_TIMEOUT", state["steps"]),
            )

        response = mock_llm.chat(state["messages"], tools=TOOLS)
        state["llm_calls"] += 1
        usage = field(response, "usage")
        state["tokens_used"] += int(field(usage, "total_tokens", 0) or 0)
        msg = response.choices[0].message
        calls = field(msg, "tool_calls", []) or []

        if state["tokens_used"] > state["token_limit"]:
            return finish(
                state, "FALLBACK_TOKEN_LIMIT",
                fallback("FALLBACK_TOKEN_LIMIT", state["steps"]),
            )

        if not calls:
            answer = field(msg, "content", "") or ""
            return finish(state, status_for(answer), answer)

        calls = [safe_tool_call(call) for call in calls]
        state["messages"].append({
            "role": "assistant", "content": None, "tool_calls": calls,
        })

        for call in calls:
            arguments = dict(call["arguments"])
            # The mock model emits only the business arguments. Inject the
            # explicit request timestamp from the run context at execution.
            if call["name"] in {"issue_refund", "escalate_to_human"}:
                arguments["request_datetime"] = state.get("request_datetime")

            try:
                result = run_tool(call["name"], arguments)
            except (
                TypeError, ValueError, KeyError, sqlite3.Error,
                json.JSONDecodeError,
            ) as error:
                result = {"error": str(error)}

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": json.dumps(result, ensure_ascii=False),
            })
            state["steps"].append({
                "step": len(state["steps"]) + 1,
                "tool_name": call["name"],
                "arguments": arguments,
                "observation": result,
            })
            state["tools_executed"].append(call["name"])
            state["step_number"] = len(state["steps"])
            save_checkpoint(state)

            if crash_after_tool_executions == len(state["tools_executed"]):
                raise SimulatedCrash(
                    "Simulated crash; checkpoint saved after tool execution."
                )

            if isinstance(result, dict) and "error" in result:
                # The supplied mock LLM has explicit handling for a missing
                # get_order result; pass that observation back so NOT_FOUND
                # is produced by the next model call (assignment call count).
                if call["name"] == "get_order":
                    continue

                error_message = str(result["error"])
                if result.get("review_required") or "human review is required" in error_message.lower():
                    status = "NEEDS_HUMAN_REVIEW"
                else:
                    status = "DENIED"
                return finish(state, status, f"{status}: {error_message}")

    return finish(
        state,
        "FALLBACK_MAX_ITERATIONS",
        fallback("FALLBACK_MAX_ITERATIONS", state["steps"]),
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_executions=None,
    request_datetime=None,
):
    """Start a new agent run with an explicit request timestamp."""
    run_id = run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}"
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "request_datetime": request_datetime,
        "max_iterations": int(max_iterations),
        "token_limit": int(token_limit),
        "time_limit_seconds": float(time_limit_seconds),
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT\n"
                    "You are an Instamart refund support agent. "
                    "Use tools to check the order and policy. "
                    "Only completed orders qualify. A late-delivery "
                    "refund requires delivery over 45 minutes. "
                    "Refunds above INR 1000 require human approval. "
                    "Never claim real money was transferred. "
                    "The backend verifies the seven-day request window "
                    "using the supplied request timestamp and estimated delivery time."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "step_number": 0,
        "tools_executed": [],
        "status": "RUNNING",
        "final_answer": "",
    }
    save_checkpoint(state)
    return continue_agent(
        state,
        crash_after_tool_executions=crash_after_tool_executions,
    )


def resume_agent(run_id):
    """Resume an unfinished agent run from its checkpoint."""
    path = checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    if state.get("status") != "RUNNING":
        return {
            key: state[key]
            for key in (
                "run_id", "scenario_id", "status", "final_answer",
                "steps", "llm_calls", "tokens_used", "tools_executed",
            )
        }
    return continue_agent(state)


# ------------------------------------------------------------
# Batch runs and demonstrations
# ------------------------------------------------------------

def append_run(result):
    """Append one completed scenario result to the run log."""
    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        key: result[key]
        for key in (
            "scenario_id", "status", "final_answer", "llm_calls",
            "tokens_used", "steps",
        )
    }
    with RUN_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scenario_rows():
    path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def run_all_scenarios():
    """Run the seven supplied scenarios using a temporary demo ledger."""
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_rows()
    RUN_LOG.write_text("", encoding="utf-8")
    results = []
    with _isolated_demo_ledger("batch-demo-ledger"):
        for row in scenarios:
            mock_llm.reset()
            match = re.search(r"ORD-\d{6}", row["message"])
            request_ts = (
                simulated_request_datetime(match.group(0)) if match else None
            )
            result = run_agent(
                row["message"], row["scenario_id"],
                request_datetime=request_ts,
            )
            append_run(result)
            results.append(result)
            print(
                f'{result["scenario_id"]}: {result["status"]}; '
                f'calls={result["llm_calls"]}; tokens={result["tokens_used"]}'
            )
    print(f"Saved {len(results)} scenario records to {RUN_LOG}")
    print("The persistent refund ledger was not changed by this demo.")
    return results


def run_limit_demos():
    """Demonstrate token-limit and timeout fallbacks with a temporary ledger."""
    s1 = next(row for row in _scenario_rows() if row["scenario_id"] == "S1")
    order_match = re.search(r"ORD-\d{6}", s1["message"])
    request_ts = simulated_request_datetime(order_match.group(0))

    with _isolated_demo_ledger("limit-demo-ledger"):
        mock_llm.reset()
        token = run_agent(
            s1["message"], "S1_TOKEN_DEMO", token_limit=200,
            run_id="demo-token-limit", request_datetime=request_ts,
        )
        print("Token-limit demo:", token["status"])

        mock_llm.reset()
        timeout = run_agent(
            s1["message"], "S1_TIMEOUT_DEMO", time_limit_seconds=0,
            run_id="demo-timeout", request_datetime=request_ts,
        )
        print("Timeout demo:", timeout["status"])

    print("The persistent refund ledger was not changed by these demos.")
    return token, timeout


def run_resume_demo():
    """Demonstrate crash/resume using a temporary ledger."""
    s1 = next(row for row in _scenario_rows() if row["scenario_id"] == "S1")
    order_match = re.search(r"ORD-\d{6}", s1["message"])
    request_ts = simulated_request_datetime(order_match.group(0))
    run_id = f"demo-resume-s1-{uuid.uuid4().hex[:8]}"
    cp = checkpoint_path(run_id)
    if cp.exists():
        cp.unlink()

    with _isolated_demo_ledger("resume-demo-ledger"):
        mock_llm.reset()
        try:
            run_agent(
                s1["message"], "S1_RESUME_DEMO", run_id=run_id,
                crash_after_tool_executions=2,
                request_datetime=request_ts,
            )
        except SimulatedCrash as error:
            print(error)
        result = resume_agent(run_id)

    print("Tools across crash/resume:", result["tools_executed"])
    print("Final result:", result["status"], result["final_answer"])
    print("The persistent refund ledger was not changed by this demo.")
    return result


def interactive():
    """Run the agent interactively; require a request timestamp for date checks."""
    print("Refund Support Agent - simulation only. Type exit to stop.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        match = re.search(r"ORD-\d{6}", question)
        if not match:
            print("Please include an order ID such as ORD-123456.")
            continue
        request_ts = input(
            "Enter refund request date/time (YYYY-MM-DD HH:MM:SS): "
        ).strip()
        try:
            _parse_local_datetime(request_ts, "request_datetime")
        except ValueError as error:
            print(error)
            continue
        mock_llm.reset()
        result = run_agent(
            question, "interactive", request_datetime=request_ts
        )
        print(result["status"], result["final_answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--demos", action="store_true")
    parser.add_argument("--resume-demo", action="store_true")
    args = parser.parse_args()

    if args.batch:
        run_all_scenarios()
    elif args.demos:
        run_limit_demos()
    elif args.resume_demo:
        run_resume_demo()
    else:
        interactive()
