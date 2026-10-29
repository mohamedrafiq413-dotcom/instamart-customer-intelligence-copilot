
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = OUTPUT_DIR / "processed_tickets.csv"


def redact_pii(text):
    """Redact common personal information from ticket messages."""
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text)

    # Redact email addresses first
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        text,
    )

    # Redact long digit sequences before phone numbers
    text = re.sub(
        r"(?<!\d)\d{12,19}(?!\d)",
        "[REDACTED_NUMBER]",
        text,
    )

    # Redact phone numbers, including common international formats
    text = re.sub(
        r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)",
        "[REDACTED_PHONE]",
        text,
    )

    return text


def process_tickets():
    """Prepare tickets for classification and route incomplete messages."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Ticket file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df["ticket_id"].duplicated().any():
        raise ValueError("Duplicate ticket IDs found.")

    # Redact before writing the processed file
    df["redacted_text"] = df["text"].apply(redact_pii)
    missing_text = df["redacted_text"].str.strip().eq("")

    df["predicted_category"] = "needs_classification"
    df["route"] = "classification_queue"
    df["processing_status"] = "ready_for_classification"

    df.loc[missing_text, "predicted_category"] = "none"
    df.loc[missing_text, "route"] = "manual_review"
    df.loc[missing_text, "processing_status"] = "missing_text"

    # Initial fraud routing uses the supplied label for workflow setup.
    # The classifier independently determines the final category and route.
    known_fraud = (
        ~missing_text
        & df["true_category"].astype(str).str.lower().eq("fraud")
    )
    df.loc[known_fraud, "route"] = "priority_fraud_review"

    # Never save the original, unredacted message in this output
    df = df.drop(columns=["text"])
    df.to_csv(OUTPUT_FILE, index=False)

    print("Ticket processing completed.")
    print(f"Total tickets: {len(df)}")
    print(f"Tickets ready for classification: {(~missing_text).sum()}")
    print(f"Tickets sent for manual review: {missing_text.sum()}")
    print(f"Priority fraud-review tickets: {known_fraud.sum()}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nProcessing status counts:")
    print(df["processing_status"].value_counts().to_string())

    print("\nRouting counts:")
    print(df["route"].value_counts().to_string())


if __name__ == "__main__":
    process_tickets()
