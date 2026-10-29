
"""
Instamart Capstone - Part C
Support Ticket Triage with Guardrails

Reads:
    data/raw/tickets.csv
    outputs/customer_segments.csv

Creates:
    outputs/triage_budget_demo.csv
    outputs/triage_results.csv
    outputs/triage_metrics.txt

Do not modify mock_llm.py.
"""

import json
import random
import re
import time
import sys
from pathlib import Path

# Add the project root folder to Python's import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm


# ---------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Assignment requires the exact patterns and replacement order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket.
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category":"...","reason":"..."}
The category must be one of the four allowed categories.
Do not include markdown or text outside the JSON.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket.
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category":"...","reason":"..."}
The category must be one of the four allowed categories.
Do not include markdown or text outside the JSON.

EXAMPLES:
Ticket: I was charged twice for my order.
Output: {"category":"billing","reason":"duplicate charge"}
Ticket: My delivery is late.
Output: {"category":"delivery","reason":"late delivery"}
Ticket: Someone made an unauthorized transaction on my card.
Output: {"category":"fraud","reason":"unauthorized transaction"}
Ticket: The app crashes when I try to log in.
Output: {"category":"technical","reason":"app crash"}
""".strip()


# ---------------------------------------------------------
# 2. Redact personal information
# ---------------------------------------------------------

def redact_text(text):
    """Redact email, phone, and order ID in the required order."""
    result = text or ""
    counts = {}

    for name, pattern, replacement in PII_PATTERNS:
        matches = pattern.findall(result)
        counts[name] = len(matches)
        result = pattern.sub(replacement, result)

    return result, counts


def remaining_pii_count(text):
    """Count matches remaining after redaction."""
    return sum(
        len(pattern.findall(text or ""))
        for _, pattern, _ in PII_PATTERNS
    )


# ---------------------------------------------------------
# 3. Guardrails and output parsing
# ---------------------------------------------------------

def parse_response(content):
    """
    Return (category, status).
    Recover chatty keyword responses when possible.
    Invalid categories are sent for human review.
    """
    content = str(content or "").strip()

    try:
        data = json.loads(content)
        category = str(data.get("category", "")).strip().lower()

        if category in CATEGORIES:
            return category, "classified_json"

        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError, AttributeError):
        # Find a category word anywhere in a chatty response.
        # Word boundaries also handle phrases such as billing-related.
        found = re.search(
            r"\b(billing|delivery|technical|fraud)\b",
            content.lower(),
        )

        if found:
            return found.group(1), "classified_keyword"

        return None, "flagged_invalid_output"


def is_empty(text):
    return pd.isna(text) or not str(text).strip()


# ---------------------------------------------------------
# 4. One ticket classification with transient retries
# ---------------------------------------------------------

def classify_one(redacted_text, system_prompt, rng):
    """
    The user message contains exactly the redacted ticket text.
    Retry transient failures up to five attempts total.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

    # Safety check: user message must contain only the ticket text.
    assert messages[-1]["content"] == redacted_text
    assert remaining_pii_count(messages[-1]["content"]) == 0

    retry_logs = []
    tokens_used = 0

    for attempt in range(5):
        try:
            response = mock_llm.chat(messages)

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_used += int(
                    getattr(usage, "total_tokens", 0) or 0
                )

            content = response.choices[0].message.content
            category, status = parse_response(content)

            return {
                "category": category,
                "status": status,
                "tokens_used": tokens_used,
                "retry_logs": retry_logs,
            }

        except Exception as exc:
            # Retry only transient LLM failures. Do not hide other errors.
            transient_type = getattr(mock_llm, "TransientLLMError", None)
            is_transient = (
                transient_type is not None
                and isinstance(exc, transient_type)
            )

            if not is_transient:
                raise

            if attempt == 4:
                retry_logs.append(
                    f"Transient failure: maximum retries reached: {exc}"
                )
                return {
                    "category": None,
                    "status": "flagged_invalid_output",
                    "tokens_used": tokens_used,
                    "retry_logs": retry_logs,
                }

            # Retry number 0 wait is in the required 0.045-0.0575 sec range.
            wait = 0.045 + rng.random() * 0.0125
            retry_logs.append(
                f"Transient failure; retry number {attempt}; "
                f"jitter wait {wait:.4f} seconds"
            )
            print(retry_logs[-1])
            time.sleep(wait)

    raise RuntimeError("Unexpected end of retry loop")


# ---------------------------------------------------------
# 5. Load customer segments and route tickets
# ---------------------------------------------------------

def load_segments():
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SEGMENTS_FILE}. Run Part B first."
        )

    segments = pd.read_csv(SEGMENTS_FILE)
    required = {"customer_id", "segment_name", "is_premium"}

    if not required.issubset(segments.columns):
        raise ValueError(
            f"Customer segments must contain: {sorted(required)}"
        )

    return segments.drop_duplicates("customer_id").set_index("customer_id")


def premium_value(value):
    """Interpret CSV booleans safely."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def route_ticket(status, category, is_premium):
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    if status in human_statuses:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# ---------------------------------------------------------
# 6. Run a complete batch
# ---------------------------------------------------------

def run_batch(tickets, segments, prompt, budget=None, seed=42):
    """
    Run from a fresh mock state.
    A budget is checked before every model call.
    """
    mock_llm.reset()
    rng = random.Random(seed)

    rows = []
    total_tokens = 0
    redaction_totals = {
        "email": 0,
        "phone": 0,
        "order_id": 0,
    }
    pii_remaining = 0
    retry_log_count = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket["text"]

        text = "" if pd.isna(original_text) else str(original_text)
        redacted_text, counts = redact_text(text)

        for key in redaction_totals:
            redaction_totals[key] += counts[key]

        pii_remaining += remaining_pii_count(redacted_text)

        category = None
        reason_status = None
        ticket_tokens = 0

        # Guardrails are checked before any model call.
        if is_empty(original_text):
            reason_status = "skipped_empty"

        elif mock_llm.est_tokens(text) > 400:
            reason_status = "flagged_long"

        elif budget is not None and total_tokens >= budget:
            reason_status = "budget_exhausted"

        else:
            result = classify_one(redacted_text, prompt, rng)
            category = result["category"]
            reason_status = result["status"]
            ticket_tokens = result["tokens_used"]
            total_tokens += ticket_tokens
            retry_log_count += len(result["retry_logs"])

        segment_name = ""
        is_premium = False

        if customer_id in segments.index:
            segment = segments.loc[customer_id]
            segment_name = str(segment["segment_name"])
            is_premium = premium_value(segment["is_premium"])

        route, priority = route_ticket(
            reason_status, category, is_premium
        )

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": reason_status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": ticket_tokens,
            "true_category": ticket.get("true_category"),
        })

    result_df = pd.DataFrame(rows)

    # No personal information should remain in any message sent.
    assert pii_remaining == 0, (
        f"Found {pii_remaining} PII matches after redaction."
    )

    return {
        "rows": result_df,
        "total_tokens": total_tokens,
        "redaction_totals": redaction_totals,
        "pii_remaining": pii_remaining,
        "retry_log_count": retry_log_count,
    }


# ---------------------------------------------------------
# 7. Accuracy and confusion table
# ---------------------------------------------------------

def calculate_metrics(result):
    df = result["rows"]
    classified = df[df["status"].isin(
        ["classified_json", "classified_keyword"]
    )].copy()

    if len(classified):
        correct = (
            classified["category"].astype(str).str.lower()
            == classified["true_category"].astype(str).str.lower()
        ).sum()
        accuracy = correct / len(classified)
    else:
        correct = 0
        accuracy = 0.0

    return len(classified), int(correct), accuracy


def confusion_table(df):
    classified = df[df["status"].isin(
        ["classified_json", "classified_keyword"]
    )]

    return pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )


def save_final_csv(result, path):
    output = result["rows"][
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    ]
    output.to_csv(path, index=False)


# ---------------------------------------------------------
# 8. Main: compare prompts, demonstrate budget, final output
# ---------------------------------------------------------

def main():
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(f"Missing ticket file: {TICKETS_FILE}")

    tickets = pd.read_csv(TICKETS_FILE)
    required = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    if not required.issubset(tickets.columns):
        raise ValueError(f"Ticket file must contain {sorted(required)}")

    segments = load_segments()

    print("=" * 65)
    print("PART C: SUPPORT TICKET TRIAGE")
    print("=" * 65)
    print(f"Tickets loaded: {len(tickets)}")
    print(f"Empty messages: {tickets['text'].apply(is_empty).sum()}")

    # Full zero-shot run, no budget.
    print("\n--- ZERO-SHOT RUN (no token budget) ---")
    zero = run_batch(tickets, segments, ZERO_SHOT_PROMPT, budget=None)
    zero_count, zero_correct, zero_accuracy = calculate_metrics(zero)

    print(f"Classified: {zero_count}/{len(tickets)}")
    print(f"Correct: {zero_correct}")
    print(f"Accuracy: {zero_accuracy:.1%}")
    print(f"Total tokens: {zero['total_tokens']}")
    print(
        "Tokens per classified ticket: "
        f"{zero['total_tokens'] / max(zero_count, 1):.2f}"
    )
    print("Status counts:")
    print(zero["rows"]["status"].value_counts().to_string())
    print("Confusion table:")
    print(confusion_table(zero["rows"]).to_string())

    # Full few-shot run, no budget.
    print("\n--- FEW-SHOT RUN (no token budget) ---")
    few = run_batch(tickets, segments, FEW_SHOT_PROMPT, budget=None)
    few_count, few_correct, few_accuracy = calculate_metrics(few)

    print(f"Classified: {few_count}/{len(tickets)}")
    print(f"Correct: {few_correct}")
    print(f"Accuracy: {few_accuracy:.1%}")
    print(f"Total tokens: {few['total_tokens']}")
    print(
        "Tokens per classified ticket: "
        f"{few['total_tokens'] / max(few_count, 1):.2f}"
    )
    print("Status counts:")
    print(few["rows"]["status"].value_counts().to_string())

    # Budget demonstration. If 6000 exhausts no tickets, try 5000.
    print("\n--- ZERO-SHOT TOKEN BUDGET DEMO ---")
    budget = 6000
    demo = run_batch(
        tickets, segments, ZERO_SHOT_PROMPT, budget=budget, seed=42
    )
    exhausted = int(
        (demo["rows"]["status"] == "budget_exhausted").sum()
    )

    if exhausted == 0:
        budget = 5000
        print("No tickets exhausted at 6000 tokens; retrying with 5000.")
        demo = run_batch(
            tickets, segments, ZERO_SHOT_PROMPT, budget=budget, seed=42
        )
        exhausted = int(
            (demo["rows"]["status"] == "budget_exhausted").sum()
        )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    save_final_csv(demo, budget_path)
    print(f"Budget cap: {budget}")
    print(f"Budget-exhausted tickets: {exhausted}")
    print(f"Saved: {budget_path}")

    # Final required output: few-shot with high token budget.
    print("\n--- FINAL FEW-SHOT OUTPUT ---")
    final = run_batch(
        tickets, segments, FEW_SHOT_PROMPT, budget=100000, seed=42
    )
    final_path = OUTPUT_DIR / "triage_results.csv"
    save_final_csv(final, final_path)

    print(f"Saved: {final_path}")
    print(f"Final rows: {len(final['rows'])}")
    print("Final status counts:")
    print(final["rows"]["status"].value_counts().to_string())
    print("Final route counts:")
    print(final["rows"]["route"].value_counts().to_string())

    # Verify required redaction totals using the raw tickets.
    raw_counts = {"email": 0, "phone": 0, "order_id": 0}
    for value in tickets["text"]:
        _, counts = redact_text("" if pd.isna(value) else str(value))
        for key in raw_counts:
            raw_counts[key] += counts[key]

    print("\n--- PII REDACTION AUDIT ---")
    print(f"Email matches redacted: {raw_counts['email']}")
    print(f"Phone matches redacted: {raw_counts['phone']}")
    print(f"Order IDs redacted: {raw_counts['order_id']}")
    print(f"Total redactions: {sum(raw_counts.values())}")
    print(f"PII matches remaining after redaction: {final['pii_remaining']}")

    # Save a report with the measured results from this run.
    report_path = OUTPUT_DIR / "triage_metrics.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("PART C - SUPPORT TICKET TRIAGE METRICS\n\n")
        report.write(f"Total tickets: {len(tickets)}\n")
        report.write(f"Zero-shot classified: {zero_count}\n")
        report.write(f"Zero-shot correct: {zero_correct}\n")
        report.write(f"Zero-shot accuracy: {zero_accuracy:.4%}\n")
        report.write(f"Zero-shot total tokens: {zero['total_tokens']}\n")
        report.write(
            "Zero-shot tokens/classified: "
            f"{zero['total_tokens'] / max(zero_count, 1):.2f}\n\n"
        )
        report.write(f"Few-shot classified: {few_count}\n")
        report.write(f"Few-shot correct: {few_correct}\n")
        report.write(f"Few-shot accuracy: {few_accuracy:.4%}\n")
        report.write(f"Few-shot total tokens: {few['total_tokens']}\n")
        report.write(
            "Few-shot tokens/classified: "
            f"{few['total_tokens'] / max(few_count, 1):.2f}\n\n"
        )
        report.write(f"Budget cap used: {budget}\n")
        report.write(f"Budget-exhausted tickets: {exhausted}\n\n")
        report.write("Redaction totals:\n")
        for key, value in raw_counts.items():
            report.write(f"{key}: {value}\n")
        report.write(f"Total redactions: {sum(raw_counts.values())}\n")
        report.write(
            f"PII matches remaining: {final['pii_remaining']}\n\n"
        )
        report.write("Zero-shot confusion table:\n")
        report.write(confusion_table(zero["rows"]).to_string())
        report.write("\n\nZero-shot status counts:\n")
        report.write(zero["rows"]["status"].value_counts().to_string())
        report.write("\n\nFew-shot status counts:\n")
        report.write(few["rows"]["status"].value_counts().to_string())
        report.write("\n\nFew-shot route counts:\n")
        report.write(final["rows"]["route"].value_counts().to_string())

    print(f"Metrics report saved: {report_path}")
    print("\nPart C pipeline finished.")


if __name__ == "__main__":
    main()
