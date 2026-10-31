"""Non-destructive unit and tool tests for the seven-day refund window.

All successful refund/escalation calls run against temporary ledgers.
The persistent outputs/refund_ledger.jsonl is never modified.
"""

import refund_agent as agent

ORDER = {"estimated_delivery_at": "2025-04-03 00:00:00"}


def test_window_boundaries():
    cases = [
        ("within 2 days", "2025-04-05 00:00:00", None),
        ("exactly 7 days", "2025-04-10 00:00:00", None),
        ("after 8 days", "2025-04-11 00:00:00", "outside the policy window"),
        ("before delivery", "2025-04-02 23:59:59", "earlier than delivery"),
        ("missing request date", None, "request_datetime is required"),
    ]

    for label, request_ts, expected_error in cases:
        result = agent.check_seven_day_window(ORDER, request_ts)
        if expected_error is None:
            assert result is None, f"{label}: expected eligible, got {result}"
        else:
            assert result is not None, f"{label}: expected rejection/review"
            assert expected_error.lower() in result["error"].lower(), (
                f"{label}: unexpected message {result['error']}"
            )
        print(f"PASS: {label}")

    missing_delivery = agent.check_seven_day_window(
        {}, "2025-04-05 00:00:00"
    )
    assert missing_delivery and missing_delivery.get("review_required")
    print("PASS: missing delivery timestamp requires human review")


def test_refund_tools_use_temporary_ledger():
    order = agent.get_order("ORD-100048")
    assert "error" not in order, "Test order ORD-100048 is missing"
    delivery = agent._parse_local_datetime(
        order["estimated_delivery_at"], "estimated_delivery_at"
    )
    valid_request = (delivery + agent.timedelta(days=2)).isoformat(sep=" ")
    late_request = (delivery + agent.timedelta(days=8)).isoformat(sep=" ")

    with agent._isolated_demo_ledger("test-seven-day-issue") as ledger:
        expired = agent.issue_refund(
            "ORD-100048", 128.25, request_datetime=late_request
        )
        assert "error" in expired and "7 days" in expired["error"], expired
        assert not ledger.exists() or not ledger.read_text(encoding="utf-8").strip(), (
            "Expired request must not append a ledger record"
        )
        valid = agent.issue_refund(
            "ORD-100048", 128.25, request_datetime=valid_request
        )
        assert valid.get("refund_inr") == 128.25, valid
        lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1, "Valid request should append exactly one ledger record"
    print("PASS: valid refund is recorded; expired request is not recorded")

    high_order = agent.get_order("ORD-100145")
    assert "error" not in high_order, "Test order ORD-100145 is missing"
    high_delivery = agent._parse_local_datetime(
        high_order["estimated_delivery_at"], "estimated_delivery_at"
    )
    high_request = (high_delivery + agent.timedelta(days=2)).isoformat(sep=" ")
    with agent._isolated_demo_ledger("test-seven-day-escalation") as ledger:
        result = agent.escalate_to_human(
            "ORD-100145", 1999.00, request_datetime=high_request
        )
        assert result.get("refund_inr") == 1999.00, result
        lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1
    print("PASS: eligible high-value request is escalated")


def main():
    test_window_boundaries()
    test_refund_tools_use_temporary_ledger()
    print("All seven-day refund eligibility tests passed.")


if __name__ == "__main__":
    main()
