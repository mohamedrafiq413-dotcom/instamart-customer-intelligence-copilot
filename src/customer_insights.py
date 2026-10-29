from pathlib import Path
import pandas as pd

# Step 1: Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "clean" / "customer_segments.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "clean" / "customer_insights.txt"


def main():
    # Step 2: Load customer segmentation results
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Customer segments file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Step 3: Calculate summary metrics
    total_customers = len(df)
    total_spend = df["monetary"].sum()
    total_completed_orders = df["frequency"].sum()

    segment_summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_completed_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
    ).round(2)

    segment_summary["percentage"] = (
        segment_summary["customers"] / total_customers * 100
    ).round(2)

    # Step 4: Create business recommendations
    recommendations = {
        "High Value": (
            "Offer loyalty rewards and personalized product "
            "recommendations to encourage repeat purchases."
        ),
        "At Risk": (
            "Consider a re-engagement campaign and investigate "
            "whether these customers experienced service issues."
        ),
        "Regular": (
            "Encourage repeat purchases with relevant offers "
            "and useful product recommendations."
        ),
        "No Completed Orders": (
            "Review the customer journey and offer helpful "
            "onboarding guidance where appropriate."
        ),
    }

    # Step 5: Build the report
    lines = [
        "=" * 60,
        "INSTAMART CUSTOMER INTELLIGENCE SUMMARY",
        "=" * 60,
        "",
        f"Total customers analyzed: {total_customers}",
        f"Total completed orders: {total_completed_orders}",
        f"Total recorded completed-order spending: ₹{total_spend:,.2f}",
        "",
        "CUSTOMER SEGMENT SUMMARY",
        "-" * 60,
        segment_summary.to_string(),
        "",
        "BUSINESS RECOMMENDATIONS",
        "-" * 60,
    ]

    for segment, row in segment_summary.iterrows():
        recommendation = recommendations.get(
            segment, "Review this segment to determine suitable actions."
        )
        lines.extend([
            "",
            f"{segment}: {row['customers']} customers "
            f"({row['percentage']}%)",
            recommendation,
        ])

    # Step 6: Save the report
    report = "\n".join(lines)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    # Step 7: Display the report
    print(report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
