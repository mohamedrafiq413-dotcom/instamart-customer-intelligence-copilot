# Instamart Customer Intelligence & Support Copilot

## Overview

An end-to-end data analytics and AI support automation prototype using synthetic Instamart data. It covers customer analytics, churn prediction, ticket classification, policy retrieval and refund assistance.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, TF-IDF and Mock LLM.

## Reproduce

Run from the project root in Windows PowerShell.

**1. Create environment and install dependencies**

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**2. Generate and prepare data**

```powershell
python .\generate_data.py
python .\src\audit_data.py
python .\src\clean_customers.py
python .\src\clean_products.py
python .\src\clean_orders.py
python .\src\build_database.py
python .\src\validate_data.py
python .\src\sql_analysis.py
```

**3. Run customer intelligence**

```powershell
python .\src\customer_intelligence.py
python .\src\customer_insights.py
python .\src\lapse_model.py
```

**4. Run ticket triage**

```powershell
python .\src\triage_pipeline.py
```

**5. Run policy QA and refund agent**

```powershell
python .\src\policy_qa.py
python .\src\test_refund_rules.py
python .\src\refund_agent.py --batch
python .\src\refund_agent.py --demos
python .\src\refund_agent.py --resume-demo
```

Type `exit` to leave the interactive policy QA prompt. The supplied `generate_data.py` and `mock_llm.py` are retained unmodified.

## Part A — Data Preparation and SQL

**Raw data audit:**
- Customers: 1,200 rows; 61 missing ages, 6 ages outside 10–100, 24 city spellings, 10 nonblank gender spellings and 30 missing genders.
- Products: 40 rows, with no missing values or exact duplicate rows.
- Orders: 13,084 rows; 119 exact duplicates, 656 pending orders, 120 invalid or missing amounts, 106 negative delivery times and 2,541 blank delivery times.
- Tickets: 164 rows, including 4 missing messages.
- Policy evaluation: 14 questions; 2 missing expected facts.
- Agent scenarios: 7 rows.

**Cleaning results:**
- Customers: 1,200; products: 40; orders: 12,309.
- Orders: 11,110 completed and 1,199 cancelled.
- No pending orders or missing completed-order amounts.
- Completed revenue: **₹2,998,668.80**.
- Data validation: **53 passed, 0 failed**.

Customer cities and genders were standardised, and invalid or missing ages were handled using city medians. Orders were deduplicated by retaining the latest record per order ID. Missing amounts were calculated using product prices; missing delivery times were filled only for completed orders.

**SQL and pandas analysis:**
Seven SQL queries returned 30, 30, 11, 25, 550, 1,175 and 183 rows respectively. The analysis covered customer spending, city/category revenue, products, customer ranking and daily running revenue.

- Top-selling product: Instant Noodles 4 Pack — **867 units**.
- Highest city/category revenue: Bengaluru / Electronics — **₹233,909.45**.
- City revenue: Hyderabad ₹528,589.90; Bengaluru ₹523,643.40; Delhi ₹502,829.80; Pune ₹500,918.05; Mumbai ₹479,058.50; Chennai ₹463,629.15.
- Completed order amount: mean ₹269.91; median ₹135.00; skewness 4.61.
- Interquartile range: ₹64.80–₹297.00; upper outlier fence ₹645.30; 897 upper outliers and 0 lower outliers.
- Largest absolute pairwise correlation: order amount versus discount percentage, **−0.095**. Correlation does not establish causation.
- Q6 produced 8 differences between RANK and ROW_NUMBER. Q7 covered 183 daily rows and finished at cumulative revenue of ₹2,998,668.80.

**Outputs:** `data/clean/`, `data/instamart.db`, `sql/queries.sql`.

## Part B — Customer Intelligence

Customer analysis covered 1,200 customers. Initial customer segments were Regular 633, At Risk 301, High Value 241 and No Completed Orders 25.

**RFM segmentation:**
- RFM customers: 1,175.
- K-means clusters: 4.
- PCA explained variance: PC1 63.43%, PC2 21.80%, cumulative 85.23%.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.7606 | 0.4650 |
| 3 | 1268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4605 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (₹) |
|---|---:|---:|---:|---:|
| Recently Active | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Frequent | 419 | 8.77 | 19.40 | 3,410.02 |
| Developing | 182 | 109.04 | 2.20 | 685.99 |

**Lapse prediction:**
- Snapshot: October 1, 2025; prediction cutoff: September 1, 2025.
- Cohort: 1,156; lapsed: 443 (38.32%).
- Training: 924; test: 232.
- Baseline accuracy: 61.64%; lapsed-class precision, recall and F1: 0%.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.40% | 73.71% |
| Precision | 59.38% | 65.22% |
| Recall | 64.04% | 67.42% |
| F1 | 61.62% | 66.30% |

Confusion matrices, with rows representing actual classes and columns predicted classes (not lapsed, lapsed):

```text
Logistic Regression: [[104, 39],
                      [ 32, 57]]

Random Forest:       [[111, 32],
                      [ 29, 60]]
```

Best parameters: Logistic Regression `C=10`; Random Forest `max_depth=8`, `min_samples_leaf=10`. Cross-validation F1: 0.6565 and 0.6869 respectively. Random Forest was selected using training cross-validation.

The balanced Logistic Regression experiment achieved 85.39% recall and 69.41% F1, compared with 64.04% recall and 61.62% F1 before balancing.

**Retention list:** Premium customers for the operational list are identified using pre-September RFM, avoiding September information in audience selection. The top 10 test customers are ranked by predicted lapse probability. Actual September outcomes are stored separately; 2 of the 10 were retrospectively labelled lapsed. This small sample is not a guarantee of campaign performance.

**Outputs:** `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/retention_list_evaluation.csv`.

## Part C — Support Ticket Triage

- Tickets: 164; empty: 4; long: 4.
- Zero-shot: 147 classified, 126 correct, **85.7% accuracy**, 12,364 tokens.
- Few-shot: 147 classified, 147 correct, **100% accuracy**, 21,568 tokens.
- Tokens per classified ticket: 84.11 zero-shot and 146.72 few-shot.
- Recovered transient failures: 19 per full run.

Zero-shot confusion matrix:

```text
                 Predicted
Actual        billing delivery fraud technical
billing           38       0     0       0
delivery           0      36     0       0
fraud             11       0    23       0
technical         10       0     0      29
```

Classification statuses per unrestricted run: 135 classified JSON, 12 classified keyword, 9 invalid output, 4 skipped empty and 4 flagged long.

**PII redaction:** 55 email, 91 phone and 132 order ID matches; 278 total redactions and 0 remaining matches.

**Final routing:**

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| **Total** | **164** |

**Strict token-budget demonstration:**

The configured budget is 6,000 tokens. Conservative pre-call token reservation includes possible responses and retry attempts.

- Actual usage: **5,957 tokens**.
- Remaining budget: 43 tokens.
- Budget-exhausted tickets: 81.
- Budget-demo statuses: 66 classified JSON, 5 classified keyword, 81 budget exhausted, 4 skipped empty, 4 flagged long and 4 invalid output.

Budget-exhausted tickets are routed to human review. Unrestricted classification results remain unchanged. The strict reservation is designed for the supplied mock LLM and its token estimator.

**Outputs:** `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_metrics.txt`.

## Part D — Policy QA and Refund Agent

Five policy documents were divided into 15 chunks using 40-word chunks with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2.

**Policy retrieval evaluation:**

| Evaluation | Standard retrieval | Zero-overlap |
|---|---:|---:|
| Expected documents found | 12/12 | 12/12 |
| Expected facts found | 12/12 | 10/12 |
| Correct fallback answers | 2/2 | 2/2 |

The zero-overlap misses concerned delivery delays over 45 minutes and payment deductions after failed payments.

**Refund rules:** Only completed orders qualify. Delivery over 45 minutes qualifies for a full refund. Refunds above ₹1,000 require human escalation. Seven-day eligibility tests cover requests within the window, the exact seven-day boundary, expired requests, missing dates, delivery validation, refund recording and high-value escalation. All 7 tests passed.

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

Token-limit and timeout demonstrations produced `FALLBACK_TOKEN_LIMIT` and `FALLBACK_TIMEOUT`.

**Crash and resume:** `get_order` → `search_policy` → `calculate_refund` → `issue_refund`. The resumed result was REFUND ₹128.25, with 4 tool executions and no repeated completed actions.

**ReAct versus Plan-and-Execute:** ReAct selects each next step from tool observations. Measured calls were S1: 5, S5: 3 and S6: 2. Plan-and-Execute can plan ahead, but unexpected outcomes such as missing or cancelled orders require validation and replanning. Its token usage was not measured. ReAct supports this workflow by checking order status and refund eligibility before financial actions.

**Outputs:** `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/policy_eval_results.csv`, `outputs/policy_eval_zero_overlap.csv`, `outputs/react_vs_plan_execute.csv`, `state/`.

## Limitations and Next Steps

- All data is synthetic.
- The LLM is an offline mock, not a live external AI integration.
- Model results require broader validation before production use.
- The retention list is a prototype campaign output evaluated on 10 test customers.
- Future work includes external model integration, monitoring, real-world evaluation and stronger refund safeguards.