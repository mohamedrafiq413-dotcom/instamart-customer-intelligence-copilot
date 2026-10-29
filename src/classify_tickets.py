
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm


INPUT_FILE = PROJECT_ROOT / "data" / "clean" / "processed_tickets.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "clean" / "classified_tickets.csv"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_ATTEMPTS = 3
MAX_TOKENS = 400
BASE_DELAY = 0.05


def safe_text(value):
    """Convert a value to text safely."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def classify_one(ticket_text):
    """Classify a ticket and validate the model response."""

    messages = [
        {
            "role": "system",
            "content": (
                "CLASSIFY: You are a customer-support ticket classifier. "
                "Classify each ticket into exactly one category: billing, "
                "delivery, technical, or fraud. Return a JSON object with "
                'a category field, such as {"category": "billing"}.\n\n'
                "EXAMPLES:\n"
                'Ticket: "I was charged twice for my order."\n'
                'Response: {"category": "billing"}\n'
                'Ticket: "My package has not arrived."\n'
                'Response: {"category": "delivery"}\n'
                'Ticket: "The app crashes when I open it."\n'
                'Response: {"category": "technical"}\n'
                'Ticket: "Someone used my account without permission."\n'
                'Response: {"category": "fraud"}'
            ),
        },
        {
            "role": "user",
            "content": f"Ticket message:\n{ticket_text}",
        },
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = mock_llm.chat(messages)
            content = safe_text(response.choices[0].message.content)

            if not content:
                return {
                    "predicted_category": "needs_review",
                    "classification_status": "flagged_invalid_output",
                    "attempts": attempt,
                }

            # First, attempt to parse JSON.
            try:
                parsed = json.loads(content)

                if isinstance(parsed, dict):
                    category = safe_text(parsed.get("category")).lower()

                    if category in ALLOWED_CATEGORIES:
                        return {
                            "predicted_category": category,
                            "classification_status": "classified_json",
                            "attempts": attempt,
                        }

            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            # Fall back to finding exactly one allowed category word.
            words = re.findall(r"[a-z]+", content.lower())
            found_categories = {
                word for word in words if word in ALLOWED_CATEGORIES
            }

            if len(found_categories) == 1:
                return {
                    "predicted_category": found_categories.pop(),
                    "classification_status": "classified_keyword",
                    "attempts": attempt,
                }

            return {
                "predicted_category": "needs_review",
                "classification_status": "flagged_invalid_output",
                "attempts": attempt,
            }

        except mock_llm.TransientLLMError:
            if attempt < MAX_ATTEMPTS:
                delay = (
                    BASE_DELAY
                    * (2 ** (attempt - 1))
                    * random.uniform(0.9, 1.15)
                )
                time.sleep(delay)
            else:
                return {
                    "predicted_category": "needs_review",
                    "classification_status": "flagged_invalid_output",
                    "attempts": attempt,
                }

    return {
        "predicted_category": "needs_review",
        "classification_status": "flagged_invalid_output",
        "attempts": MAX_ATTEMPTS,
    }


def main():
    """Classify processed tickets and save the results."""

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return 1

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "ticket_id",
        "redacted_text",
        "route",
        "processing_status",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"ERROR: Missing required columns: {sorted(missing_columns)}")
        return 1

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        print(f"ERROR: Input contains duplicate columns: {duplicates}")
        return 1

    if hasattr(mock_llm, "reset"):
        mock_llm.reset()

    results = []

    for _, row in df.iterrows():
        ticket_text = safe_text(row["redacted_text"])
        processing_status = safe_text(row["processing_status"])

        if not ticket_text or processing_status == "manual_review":
            result = {
                "predicted_category": "none",
                "classification_status": "manual_review",
                "attempts": 0,
            }
            final_route = "manual_review"

        elif mock_llm.est_tokens(ticket_text) > MAX_TOKENS:
            result = {
                "predicted_category": "needs_review",
                "classification_status": "flagged_long",
                "attempts": 0,
            }
            final_route = "manual_review"

        else:
            result = classify_one(ticket_text)

            if result["classification_status"] not in {
                "classified_json",
                "classified_keyword",
            }:
                final_route = "manual_review"
            elif result["predicted_category"] == "fraud":
                final_route = "priority_fraud_review"
            else:
                final_route = "classification_queue"

        results.append(
            {
                "predicted_category": result["predicted_category"],
                "classification_status": result["classification_status"],
                "attempts": result["attempts"],
                "route": final_route,
            }
        )

    # Update existing columns rather than concatenating duplicate names.
    output_df = df.copy()

    for column in (
        "predicted_category",
        "classification_status",
        "attempts",
        "route",
    ):
        output_df[column] = [result[column] for result in results]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("Ticket classification completed.")
    print(f"Total tickets: {len(output_df)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nClassification status counts:")
    print(output_df["classification_status"].value_counts().to_string())

    print("\nPredicted category counts:")
    print(output_df["predicted_category"].value_counts().to_string())

    print("\nRouting counts:")
    print(output_df["route"].value_counts().to_string())

    print("\nAttempt counts:")
    print(output_df["attempts"].value_counts().sort_index().to_string())

    long_with_attempts = output_df.loc[
        output_df["classification_status"] == "flagged_long",
        "attempts",
    ].sum()

    print("\nLong tickets with model attempts:")
    print(int(long_with_attempts))

    # Final basic output checks.
    duplicate_columns = output_df.columns[output_df.columns.duplicated()].tolist()

    if duplicate_columns:
        print(f"ERROR: Duplicate output columns: {duplicate_columns}")
        return 1

    if len(output_df) != len(df):
        print("ERROR: The output row count does not match the input.")
        return 1

    print("\nOutput checks: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
