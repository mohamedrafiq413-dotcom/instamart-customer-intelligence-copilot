# Instamart Customer Intelligence & Support Copilot

## Overview

This project uses synthetic Instamart data to analyse customers and orders, predict customer churn, classify support tickets, retrieve company policies, and support refund decisions.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, TF-IDF and Mock LLM.

## Reproduce

Run these commands from the project root in Windows PowerShell.

### 1. Create and activate the environment

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies and generate data

```powershell
python -m pip install -r requirements.txt
python .\generate_data.py
```

### 3. Prepare and validate the data

```powershell
python .\src\audit_data.py
python .\src\clean_customers.py
python .\src\clean_products.py
python .\src\clean_orders.py
python .\src\validate_data.py
python .\src\build_database.py
python .\src\sql_analysis.py
```

### 4. Run customer analysis and prediction

```powershell
python .\src\customer_intelligence.py
python .\src\customer_insights.py
python .\src\lapse_model.py
```

### 5. Run ticket classification

```powershell
python .\src\triage_pipeline.py
```

### 6. Run policy QA and refund-agent tests

```powershell
python .\src\policy_qa.py
python .\src\test_refund_rules.py
python .\src\refund_agent.py --batch
python .\src\refund_agent.py --demos
python .\src\refund_agent.py --resume-demo
```

Type `exit` when the interactive policy QA prompt appears.

## Part A — Data Preparation and SQL

- Raw orders: 13,084 rows, 119 exact duplicates and 656 pending orders.
- Other raw order issues: 120 invalid or missing amounts, 106 negative delivery times and 2,541 blank delivery times.
- Raw customer issues: 61 blank ages, 6 ages outside the 10–100 range, 24 city spellings, 10 nonblank gender spellings and 30 blank genders.
- Clean customers: 1,200; products: 40; orders: 12,309.
- Final orders: 11,110 completed and 1,199 cancelled.
- Completed-order revenue: **₹2,998,668.80**.

Customers were standardised by city and gender. Invalid or missing ages were handled using city medians. Orders were deduplicated by retaining the latest record per order ID. Missing amounts were calculated using product prices, while missing delivery times were filled only for completed orders.

Data validation completed with **53 checks passed and 0 failures**.

Seven SQL analyses and pandas summaries were performed, including city/category revenue, customer spending, daily running revenue, outlier analysis and numeric correlations.

**Top-selling product:** Instant Noodles 4 Pack — 867 units.

**Outputs:** `data/clean/`, `data/instamart.db`, `sql/queries.sql`.

## Part B — Customer Intelligence

- Customers analysed: 1,200.
- RFM customers: 1,175.
- Customer segments: 4.
- PCA explained variance: 85.23%.
- Lapse cohort: 1,156.
- Lapsed customers: 443 (38.32%).
- Training rows: 924; test rows: 232.

### Lapse prediction results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.40% | 73.71% |
| Precision | 59.38% | 65.22% |
| Recall | 64.04% | 67.42% |
| F1-score | 61.62% | 66.30% |

Random Forest was selected using training cross-validation F1 of 0.6869. The selected model was saved and successfully reloaded with identical test predictions.

A balanced Logistic Regression experiment achieved 85.39% recall and 69.41% F1.

### Retention list

The operational Premium retention list uses RFM segmentation based only on completed orders before September 1, 2025. Customers are ranked using predicted lapse probabilities.

The top 10 customers and their retrospective September outcomes are stored separately. In the tested sample, 2 of the selected 10 were actually lapsed. This is a small-sample evaluation, not a guarantee of campaign performance.

**Outputs:** `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/retention_list_evaluation.csv`.

## Part C — Support Ticket Triage

- Tickets processed: 164; empty tickets: 4; long tickets: 4.
- Zero-shot: 126/147 correct (85.7%).
- Few-shot: 147/147 correct (100%).
- Tokens: 12,364 zero-shot; 21,568 few-shot.
- Transient failures recovered: 19.
- PII redactions: 278; remaining matches: 0.

### Final routing

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| **Total** | **164** |

### Strict token-budget demonstration

The token budget is capped at **6,000 tokens**. A conservative pre-call reservation checks that enough budget remains before every model attempt, including retries.

- Configured budget: 6,000 tokens.
- Actual usage: 5,957 tokens.
- Remaining budget: 43 tokens.
- Budget-exhausted tickets: 81.

Budget-exhausted tickets are routed to human review. The unrestricted zero-shot and few-shot classification results remained unchanged after the budget correction.

**Outputs:** `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_metrics.txt`.

## Part D — Policy QA and Refund Agent

Five policy documents were split into 15 chunks using 40-word chunks with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2.

### Policy retrieval evaluation

| Evaluation | Standard retrieval | Zero-overlap retrieval |
|---|---:|---:|
| Expected documents found | 12/12 | 12/12 |
| Expected facts found | 12/12 | 10/12 |
| Correct fallback answers | 2/2 | 2/2 |

The two zero-overlap fact misses concerned delivery delays over 45 minutes and payment deductions after failed payments.

### Refund rules

Only completed orders qualify for refunds. Delivery times over 45 minutes qualify for a full refund. Refund amounts above ₹1,000 require human escalation.

Seven-day refund eligibility tests cover within-window requests, the exact seven-day boundary, expired requests, missing dates, delivery-time validation, refund recording and high-value escalation.

### Refund scenarios

| Scenario | Result | Calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND ₹128.25 | 5 | 1,239 |
| S2 | REFUND ₹135.00 | 5 | 1,239 |
| S3 | ESCALATE ₹1,999.00 | 5 | 1,226 |
| S4 | DENIED | 4 | 877 |
| S5 | DENIED | 3 | 535 |
| S6 | NOT_FOUND | 2 | 196 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,372 |

### Safety and recovery demonstrations

- Token-limit fallback: `FALLBACK_TOKEN_LIMIT`.
- Timeout fallback: `FALLBACK_TIMEOUT`.
- Crash and resume: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`.
- Final resumed result: REFUND ₹128.25.
- Completed tool executions across crash/resume: four, without repeating completed actions.

### ReAct versus Plan-and-Execute

ReAct uses model calls to select the next step based on tool observations. Measured calls included S1: 5, S5: 3 and S6: 2. Plan-and-Execute can reduce calls by generating a plan first, but unexpected outcomes such as missing or cancelled orders require validation and replanning. Plan-and-Execute token usage was not measured in this project.

ReAct is suitable for this workflow because order status and refund eligibility are checked before each financial action.

**Outputs:** `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/policy_eval_results.csv`, `outputs/policy_eval_zero_overlap.csv` and `state/`.

## Limitations and Next Steps

- All project data is synthetic.
- The LLM is an offline mock model, not a live external AI integration.
- Model results require further validation before production use.
- The retention list is a prototype campaign output evaluated on a small test sample.
- Future improvements include broader model validation, monitoring, real-world data evaluation and stronger refund safeguards.