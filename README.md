# Instamart Customer Intelligence & Support Copilot

## Overview

This project uses synthetic Instamart data to analyse customers and orders, predict customer churn, classify support tickets, retrieve company policies, and support refund decisions.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, TF-IDF and Mock LLM.

## Reproduce

Run these commands from the project root in Windows PowerShell.

1. Create and activate the environment:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies and generate data:

```powershell
python -m pip install -r requirements.txt
python .\generate_data.py
```

3. Prepare and validate the data:

```powershell
python .\src\audit_data.py
python .\src\clean_customers.py
python .\src\clean_products.py
python .\src\clean_orders.py
python .\src\validate_data.py
python .\src\build_database.py
python .\src\sql_analysis.py
```

4. Run customer analysis and prediction:

```powershell
python .\src\customer_intelligence.py
python .\src\customer_insights.py
python .\src\lapse_model.py
```

5. Run ticket classification:

```powershell
python .\src\triage_pipeline.py
```

6. Run policy QA and refund-agent tests:

```powershell
python .\src\policy_qa.py
python .\src\test_refund_rules.py
python .\src\refund_agent.py --batch
python .\src\refund_agent.py --demos
python .\src\refund_agent.py --resume-demo
```

## Part A — Data Preparation and SQL

- Raw orders: 13,084 rows, 119 duplicates and 656 pending rows.
- Other raw order issues: 120 non-numeric amounts, 106 negative delivery times and 2,541 blank delivery times.
- Raw customer issues: 61 blank ages, 6 invalid ages, 24 city spellings, 10 gender spellings and 30 blank genders.
- Clean customers: 1,200; products: 40; orders: 12,309.
- Final orders: 11,110 completed and 1,199 cancelled.
- Completed-order revenue: **₹2,998,668.80**.

Customers were standardised by city and gender; invalid ages were replaced using city medians. Orders were deduplicated by keeping the latest record per order ID, missing amounts were calculated using product prices, and missing delivery times were filled for completed orders only.

**Top-selling product:** Instant Noodles 4 Pack — 867 units.

**Outputs:** `data/clean/`, `data/instamart.db`, `sql/queries.sql`.

## Part B — Customer Intelligence

- Customers analysed: 1,200.
- RFM customers: 1,175.
- Customer segments: 4.
- PCA explained variance: 85.23%.
- Lapse cohort: 1,156; lapsed customers: 443 (38.32%).
- Training rows: 924; test rows: 232.

Random Forest test results: accuracy 74.14%, precision 65.93%, recall 67.42% and F1 66.67%. Logistic Regression F1: 61.62%.

**Outputs:** `customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`.

## Part C — Support Ticket Triage

- Tickets processed: 164; empty tickets: 4; long tickets: 4.
- Zero-shot: 126/147 correct (85.7%).
- Few-shot: 147/147 correct (100%).
- Tokens: 12,364 zero-shot; 21,568 few-shot.
- Transient failures recovered: 19.
- PII redactions: 278; remaining matches: 0.

**Final routing:**

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |

The 6,000-token budget demonstration flagged 80 tickets. Actual usage was 6,038 because the budget is checked before each model call.

**Outputs:** `triage_results.csv`, `triage_budget_demo.csv`, `triage_metrics.txt`.

## Part D — Policy QA and Refund Agent

Five policy documents were split into 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2.

Standard retrieval achieved 12/12 expected documents, 12/12 expected facts and 2/2 correct fallback answers. Zero-overlap retrieval achieved 10/12 expected facts. The missed facts concerned delivery delays over 45 minutes and payment deductions after failed payments.

Refund rules: only completed orders qualify; delivery over 45 minutes qualifies for a full refund; amounts above ₹1,000 require human escalation. Simulated requests are treated as being within the seven-day policy window.

**Refund scenarios:**

| Scenario | Result | Calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND ₹128.25 | 5 | 1,239 |
| S2 | REFUND ₹135.00 | 5 | 1,239 |
| S3 | ESCALATE ₹1,999.00 | 5 | 1,226 |
| S4 | DENIED | 4 | 877 |
| S5 | DENIED | 3 | 535 |
| S6 | NOT_FOUND | 2 | 196 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,372 |

**Safety demonstrations:**
- Token limit: `FALLBACK_TOKEN_LIMIT`.
- Timeout: `FALLBACK_TIMEOUT`.

**Crash and resume:** `get_order` → `search_policy` → `calculate_refund` → `issue_refund`. Final result: REFUND ₹128.25, with exactly four tool executions.

**ReAct versus Plan-and-Execute:** ReAct uses model calls to choose the next step from tool observations. Measured calls were S1: 5, S5: 3 and S6: 2. Plan-and-Execute can reduce calls by generating a complete plan first, but unexpected results such as missing or cancelled orders require validation and replanning. Its token usage was not measured here. ReAct is suitable for this workflow because order status and refund eligibility are checked before each financial action.

**Outputs:** `agent_runs.jsonl`, `refund_ledger.jsonl`, policy evaluation files and `state/`.

## Limitations and Next Steps

- All data is synthetic.
- The LLM is an offline mock model.
- Model results need further validation before production use.
- The retention list is illustrative, not a leakage-free evaluation.
- Future work includes improved model testing, monitoring and stronger refund safeguards.