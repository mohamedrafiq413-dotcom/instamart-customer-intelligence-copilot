
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = OUTPUT_DIR / "processed_tickets.csv"


# ---------------------------------------------------------
# 2. Redact personal information
# ---------------------------------------------------------
def redact_pii(text):
    """Redact email addresses, phone numbers, and order IDs."""

    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text)

    # Redact email addresses
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text,
    )

    # Redact order IDs such as ORD-105035
    text = re.sub(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text,
        flags=re.IGNORECASE,
    )

    # Redact long digit sequences
    text = re.sub(
        r"(?<!\d)\d{12,19}(?!\d)",
        "[REDACTED_NUMBER]",
        text,
    )

    # Redact Indian 10-digit phone numbers
    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text,
    )

    return text


# ---------------------------------------------------------
# 3. Process support tickets
# ---------------------------------------------------------
def process_tickets():
    """Prepare tickets for classification and route incomplete messages."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Ticket file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve literal text values such as 'NA'
    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    # Required columns
    required_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Check duplicate ticket IDs
    if df["ticket_id"].duplicated().any():
        raise ValueError("Duplicate ticket IDs found.")

    # -----------------------------------------------------
    # 4. Redact before saving
    # -----------------------------------------------------
    df["redacted_text"] = df["text"].apply(redact_pii)

    # Identify empty messages
    missing_text = df["redacted_text"].str.strip().eq("")

    # -----------------------------------------------------
    # 5. Initial ticket classification status and routing
    # -----------------------------------------------------
    df["predicted_category"] = "needs_classification"
    df["route"] = "classification_queue"
    df["processing_status"] = "ready_for_classification"

    # Empty messages require manual review
    df.loc[
        missing_text,
        "predicted_category"
    ] = "none"

    df.loc[
        missing_text,
        "route"
    ] = "manual_review"

    df.loc[
        missing_text,
        "processing_status"
    ] = "missing_text"

    # Initial fraud routing uses the supplied label
    # for workflow setup only.
    # The classifier independently predicts the category.
    known_fraud = (
        ~missing_text
        & df["true_category"]
        .astype(str)
        .str.lower()
        .eq("fraud")
    )

    df.loc[
        known_fraud,
        "route"
    ] = "priority_fraud_review"

    # -----------------------------------------------------
    # 6. Remove original unredacted message
    # -----------------------------------------------------
    df = df.drop(columns=["text"])

    # Save processed tickets
    df.to_csv(OUTPUT_FILE, index=False)

    # -----------------------------------------------------
    # 7. Display processing summary
    # -----------------------------------------------------
    print("Ticket processing completed.")

    print(f"Total tickets: {len(df)}")

    print(
        f"Tickets ready for classification: "
        f"{(~missing_text).sum()}"
    )

    print(
        f"Tickets sent for manual review: "
        f"{missing_text.sum()}"
    )

    print(
        f"Priority fraud-review tickets: "
        f"{known_fraud.sum()}"
    )

    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nProcessing status counts:")
    print(
        df["processing_status"]
        .value_counts()
        .to_string()
    )

    print("\nRouting counts:")
    print(
        df["route"]
        .value_counts()
        .to_string()
    )


# ---------------------------------------------------------
# 8. Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    process_tickets()
