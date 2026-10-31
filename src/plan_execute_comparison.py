"""Deterministic Plan-and-Execute baseline compared with saved ReAct runs.

This is an offline rule-based planner using the same order, policy and refund
business tools. It does not call an LLM; its zero LLM-token count must not be
compared as if it were an LLM-powered Plan-and-Execute model.
"""

import csv
import json
import re
from pathlib import Path

import refund_agent as agent

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
REACT_LOG = ROOT / "outputs" / "agent_runs.jsonl"
OUTPUT_FILE = ROOT / "outputs" / "react_vs_plan_execute.csv"
SELECTED_SCENARIOS = {"S1", "S5", "S6"}


def read_scenarios():
    with SCENARIOS_FILE.open(newline="", encoding="utf-8-sig") as f:
        return {
            row["scenario_id"]: row
            for row in csv.DictReader(f)
            if row["scenario_id"] in SELECTED_SCENARIOS
        }


def read_react_runs():
    if not REACT_LOG.exists():
        raise FileNotFoundError(
            "Missing outputs/agent_runs.jsonl. Run refund_agent.py --batch first."
        )
    with REACT_LOG.open(encoding="utf-8") as f:
        return {
            row["scenario_id"]: row
            for row in (json.loads(line) for line in f if line.strip())
            if row.get("scenario_id") in SELECTED_SCENARIOS
        }


def execute_plan(row):
    """Create a fixed plan, then execute it step by step with conditional stops."""
    message = row["message"]
    match = re.search(r"ORD-\d{6}", message)
    if not match:
        return {"status": "NEEDS_INFO", "tool_names": [], "plan_steps": 1}

    order_id = match.group(0)
    request_ts = agent.simulated_request_datetime(order_id)
    planned_sequence = [
        "get_order", "search_policy", "calculate_refund",
        "issue_refund_or_escalate",
    ]
    executed = []

    # Step 1: execute the pre-built order lookup plan.
    order = agent.get_order(order_id)
    executed.append("get_order")
    if "error" in order:
        return {
            "status": "NOT_FOUND", "tool_names": executed,
            "plan_steps": len(planned_sequence),
        }

    # Step 2: execute the policy lookup plan.
    agent.search_policy("refund policy late delivery seven days")
    executed.append("search_policy")
    if str(order["status"]).lower() != "completed":
        return {
            "status": "DENIED", "tool_names": executed,
            "plan_steps": len(planned_sequence),
        }

    # Step 3: calculate eligibility and the amount.
    calculation = agent.calculate_refund(
        order["order_amount"], order["delivery_minutes"]
    )
    executed.append("calculate_refund")
    amount = float(calculation["refund_inr"])
    if amount <= 0:
        return {
            "status": "DENIED", "tool_names": executed,
            "plan_steps": len(planned_sequence),
        }

    # Step 4: follow the plan's conditional final action.
    if amount > 1000:
        outcome = agent.escalate_to_human(
            order_id, amount, request_datetime=request_ts
        )
        executed.append("escalate_to_human")
        status = "ESCALATE" if "error" not in outcome else "DENIED"
    else:
        outcome = agent.issue_refund(
            order_id, amount, request_datetime=request_ts
        )
        executed.append("issue_refund")
        status = "REFUND" if "error" not in outcome else "DENIED"

    if isinstance(outcome, dict) and outcome.get("review_required"):
        status = "NEEDS_HUMAN_REVIEW"

    return {
        "status": status,
        "tool_names": executed,
        "plan_steps": len(planned_sequence),
    }


def main():
    scenarios = read_scenarios()
    react_runs = read_react_runs()
    missing = SELECTED_SCENARIOS - set(react_runs)
    if missing:
        raise ValueError(f"Missing ReAct runs for: {sorted(missing)}")

    rows = []
    with agent._isolated_demo_ledger("plan-execute-comparison"):
        for scenario_id in ("S1", "S5", "S6"):
            result = execute_plan(scenarios[scenario_id])
            react = react_runs[scenario_id]
            react_tools = [step.get("tool_name") for step in react.get("steps", [])]
            rows.append({
                "scenario_id": scenario_id,
                "expected_outcome": scenarios[scenario_id].get("expected_outcome", ""),
                "react_result": react.get("status"),
                "react_matches_expected": (
                    react.get("status") == scenarios[scenario_id].get("expected_outcome", "")
                ),
                "react_llm_calls": react.get("llm_calls", 0),
                "react_llm_tokens": react.get("tokens_used", 0),
                "react_tool_executions": len(react_tools),
                "plan_execute_result": result["status"],
                "plan_execute_matches_expected": (
                    result["status"] == scenarios[scenario_id].get("expected_outcome", "")
                ),
                "plan_execute_planning_passes": 1,
                "plan_execute_llm_calls": 0,
                "plan_execute_llm_tokens": 0,
                "plan_execute_tool_executions": len(result["tool_names"]),
                "plan_execute_tools": ";".join(result["tool_names"]),
                "planner_type": "deterministic rule-based baseline",
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("ReAct vs Plan-and-Execute comparison saved:", OUTPUT_FILE)
    print("The Plan-and-Execute baseline is deterministic and uses no LLM tokens.")
    print("Results:")
    for row in rows:
        print(
            row["scenario_id"],
            "ReAct:", row["react_result"],
            f"{row['react_llm_calls']} LLM calls / {row['react_llm_tokens']} tokens;",
            "Plan-and-Execute:", row["plan_execute_result"],
            f"1 planning pass / {row['plan_execute_tool_executions']} tool executions."
        )


if __name__ == "__main__":
    main()
