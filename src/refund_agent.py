
# Instamart Refund Support Agent - offline simulation only

from pathlib import Path
from datetime import datetime, timezone
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


@contextmanager
def _isolated_demo_ledger(label):
    """Use a fresh temporary ledger for repeatable demonstrations.

    The persistent refund ledger is restored even if a demo raises an error.
    This keeps demo/test runs from changing the real audit history.
    """
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
        "description": (
            "Calculate a full refund for delivery later than 45 minutes."
        ),
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
            },
            "required": ["order_id", "refund_inr"],
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
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOLS = [
    {"type": "function", "function": spec}
    for spec in TOOLS_SPEC
]


# ------------------------------------------------------------
# Tool functions
# ------------------------------------------------------------

def get_order(order_id):
    """Retrieve an order from the SQLite database."""
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        key: row[key]
        for key in (
            "order_id",
            "status",
            "order_amount",
            "delivery_minutes",
        )
    }


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


def issue_refund(order_id, refund_inr):
    """Issue an eligible simulated refund of INR 1000 or less."""
    order = get_order(order_id)

    if "error" in order:
        return {"error": "order not found"}

    if _has_previous_refund(order_id):
        return {
            "error": "A refund has already been issued for this order."
        }

    if order["status"] != "completed":
        return {"error": "Only completed orders qualify for refunds."}

    minutes = order["delivery_minutes"]

    if minutes is None or float(minutes) <= 45:
        return {
            "error": "Order does not qualify for a late-delivery refund."
        }

    actual = round(float(order["order_amount"]), 2)

    if actual > 1000:
        return {
            "error": "Refund above INR 1000 requires human approval."
        }

    if abs(float(refund_inr) - actual) > 0.01:
        return {
            "error": "Refund amount does not match the order amount."
        }

    _ledger("issue_refund", order_id, actual)
    return {"refund_inr": actual}


def escalate_to_human(order_id, refund_inr):
    """Record an eligible refund above INR 1000 for human review."""
    order = get_order(order_id)

    if "error" in order:
        return {"error": "order not found"}

    if _has_previous_refund(order_id):
        return {
            "error": "A refund or escalation has already been recorded for this order."
        }

    if order["status"] != "completed":
        return {
            "error": "Only completed orders qualify for refunds."
        }

    minutes = order["delivery_minutes"]

    if minutes is None or float(minutes) <= 45:
        return {
            "error": "Order does not qualify for a late-delivery refund."
        }

    actual = round(float(order["order_amount"]), 2)

    if actual <= 1000:
        return {"error": "Refund is not eligible for escalation."}

    if abs(float(refund_inr) - actual) > 0.01:
        return {
            "error": "Refund amount does not match the order amount."
        }

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


def fallback(status, steps):
    """Create a safe human-handoff message for agent limits."""
    summaries = {
        "FALLBACK_MAX_ITERATIONS": "The agent reached its model-call limit.",
        "FALLBACK_TOKEN_LIMIT": (
            "The token budget was exceeded before the requested tool ran."
        ),
        "FALLBACK_TIMEOUT": "The agent reached its time limit.",
    }

    return (
        f"{status}: {summaries[status]} "
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

    with checkpoint_path(state["run_id"]).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def finish(state, status, answer):
    """Save final state and return the public result."""
    state["status"] = status
    state["final_answer"] = answer
    save_checkpoint(state)

    return {
        key: state[key]
        for key in (
            "run_id",
            "scenario_id",
            "status",
            "final_answer",
            "steps",
            "llm_calls",
            "tokens_used",
            "tools_executed",
        )
    }


def continue_agent(state, crash_after_tool_executions=None):
    """Continue an agent run, saving each tool observation."""
    started = time.monotonic()

    while state["llm_calls"] < state["max_iterations"]:
        if (
            time.monotonic() - started
            >= state["time_limit_seconds"]
        ):
            return finish(
                state,
                "FALLBACK_TIMEOUT",
                fallback("FALLBACK_TIMEOUT", state["steps"]),
            )

        response = mock_llm.chat(state["messages"], tools=TOOLS)
        state["llm_calls"] += 1

        usage = field(response, "usage")
        state["tokens_used"] += int(
            field(usage, "total_tokens", 0) or 0
        )

        msg = response.choices[0].message
        calls = field(msg, "tool_calls", []) or []

        if state["tokens_used"] > state["token_limit"]:
            return finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                fallback("FALLBACK_TOKEN_LIMIT", state["steps"]),
            )

        if not calls:
            answer = field(msg, "content", "") or ""
            return finish(state, status_for(answer), answer)

        calls = [safe_tool_call(call) for call in calls]

        state["messages"].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
            }
        )

        for call in calls:
            try:
                result = run_tool(call["name"], call["arguments"])
            except (
                TypeError,
                ValueError,
                KeyError,
                sqlite3.Error,
                json.JSONDecodeError,
            ) as error:
                result = {"error": str(error)}

            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(
                        result, ensure_ascii=False
                    ),
                }
            )

            state["steps"].append(
                {
                    "step": len(state["steps"]) + 1,
                    "tool_name": call["name"],
                    "arguments": call["arguments"],
                    "observation": result,
                }
            )

            state["tools_executed"].append(call["name"])
            state["step_number"] = len(state["steps"])

            # Persist every tool observation before continuing.
            save_checkpoint(state)

            # Keep the crash demonstration deterministic.
            if (
                crash_after_tool_executions
                == len(state["tools_executed"])
            ):
                raise SimulatedCrash(
                    "Simulated crash; checkpoint saved after tool execution."
                )

            # Critical fix: do not send tool errors to mock_llm as
            # successful results. Its supplied implementation expects
            # refund_inr in successful refund-tool responses.
            if isinstance(result, dict) and "error" in result:
                error_message = str(result["error"])

                status = (
                    "NOT_FOUND"
                    if "not found" in error_message.lower()
                    else "DENIED"
                )

                return finish(
                    state,
                    status,
                    f"{status}: {error_message}",
                )

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
):
    """Start a new agent run."""
    run_id = run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}"

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
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
                    "Never claim real money was transferred."
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
                "run_id",
                "scenario_id",
                "status",
                "final_answer",
                "steps",
                "llm_calls",
                "tokens_used",
                "tools_executed",
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
            "scenario_id",
            "status",
            "final_answer",
            "llm_calls",
            "tokens_used",
            "steps",
        )
    }

    with RUN_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_all_scenarios():
    """Run the seven supplied scenarios against a clean, temporary ledger."""
    OUT.mkdir(parents=True, exist_ok=True)

    path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        scenarios = list(csv.DictReader(file))

    RUN_LOG.write_text("", encoding="utf-8")
    results = []

    with _isolated_demo_ledger("batch-demo-ledger"):
        for row in scenarios:
            mock_llm.reset()
            result = run_agent(row["message"], row["scenario_id"])
            append_run(result)
            results.append(result)
            print(
                f'{result["scenario_id"]}: {result["status"]}; '
                f'calls={result["llm_calls"]}; '
                f'tokens={result["tokens_used"]}'
            )

    print(f"Saved {len(results)} scenario records to {RUN_LOG}")
    print("The persistent refund ledger was not changed by this demo.")
    return results


def run_limit_demos():
    """Demonstrate token-limit and timeout fallbacks."""
    scenario_path = ROOT / "data" / "raw" / "agent_scenarios.csv"

    with scenario_path.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        s1 = next(
            row["message"]
            for row in csv.DictReader(file)
            if row["scenario_id"] == "S1"
        )

    mock_llm.reset()

    token = run_agent(
        s1,
        "S1_TOKEN_DEMO",
        token_limit=200,
        run_id="demo-token-limit",
    )
    print("Token-limit demo:", token["status"])

    mock_llm.reset()

    timeout = run_agent(
        s1,
        "S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
        run_id="demo-timeout",
    )
    print("Timeout demo:", timeout["status"])

    return token, timeout


def run_resume_demo():
    """Demonstrate crash/resume using a temporary ledger, not real history."""
    scenario_path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    with scenario_path.open(newline="", encoding="utf-8-sig") as file:
        s1 = next(
            row["message"]
            for row in csv.DictReader(file)
            if row["scenario_id"] == "S1"
        )

    run_id = f"demo-resume-s1-{uuid.uuid4().hex[:8]}"
    cp = checkpoint_path(run_id)
    if cp.exists():
        cp.unlink()

    with _isolated_demo_ledger("resume-demo-ledger"):
        mock_llm.reset()
        try:
            run_agent(
                s1,
                "S1_RESUME_DEMO",
                run_id=run_id,
                crash_after_tool_executions=2,
            )
        except SimulatedCrash as error:
            print(error)

        result = resume_agent(run_id)

    print("Tools across crash/resume:", result["tools_executed"])
    print("Final result:", result["status"], result["final_answer"])
    print("The persistent refund ledger was not changed by this demo.")
    return result


def interactive():
    """Run the agent interactively."""
    print("Refund Support Agent - simulation only. Type exit to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not re.search(r"ORD-\d{6}", question):
            print("Please include an order ID such as ORD-123456.")
            continue

        mock_llm.reset()

        result = run_agent(question, "interactive")
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
