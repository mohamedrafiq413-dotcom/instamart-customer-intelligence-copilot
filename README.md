# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

This project builds a data analytics and customer support copilot for a simulated Instamart business. It covers data cleaning, SQL analysis, customer segmentation, lapse prediction, support ticket classification, policy question answering, and a simulated refund agent.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, Matplotlib, and `mock_llm.py`.

## 2. Setup and Reproduction

### Requirements
- Python
- Git
- VS Code

### Install and activate the environment (Windows PowerShell)

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the project

Run the scripts from the project root in the following order:

```powershell
python .\generate_data.py
python .\src\audit_data.py
python .\src\clean_customers.py
python .\src\clean_products.py
python .\src\clean_orders.py
python .\src\validate_data.py
python .\src\build_database.py
python .\src\sql_analysis.py
python .\src\customer_intelligence.py
python .\src\customer_insights.py
python .\src\lapse_model.py
python .\src\process_tickets.py
python .\src\classify_tickets.py
python .\src\triage_pipeline.py
python .\src\policy_qa.py
python .\src\refund_agent.py --batch
python .\src\refund_agent.py --demos
python .\src\refund_agent.py --resume-demo
```

The project validation completed with **53 checks passed and 0 failed**.

## 3. Part A — Data Quality and SQL Analytics

The project generated:
- 1,200 customers
- 40 products
- 13,084 raw orders
- 164 support tickets

After cleaning, the orders dataset contained **12,309 rows**:
- 11,110 completed orders
- 1,199 cancelled orders

The cleaned customer and product datasets had no missing values or duplicate rows. Missing delivery times in the cleaned orders belonged to cancelled orders.

The SQLite database was created at `data/instamart.db`.

**Key findings:**
- Completed-order revenue: **₹2,998,668.80**
- Top-selling product by quantity: Instant Noodles 4 Pack (**867 units**)
- Order amounts were right-skewed, with a small number of high-value outliers.

SQL queries were used to analyse order status, product sales, customer activity, revenue, and order patterns.

## 4. Part B — Customer Intelligence and Lapse Prediction

Customer intelligence grouped customers into Regular, At Risk, High Value, and No Completed Orders segments.

A separate RFM analysis produced four segments:

| Segment | Customers | Recency (days) | Frequency | Monetary (₹) |
|---|---:|---:|---:|---:|
| Premium | 137 | 24.11 | 6.79 | 7,148.22 |
| Recently Active | 461 | 29.27 | 3.57 | 1,000.26 |
| Frequent | 419 | 9.13 | 19.40 | 3,410.02 |
| Developing | 183 | 109.28 | 2.21 | 707.33 |

The lapse prediction task used historical customer activity to identify customers likely to become inactive.

**Model results:**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.40% | 59.38% | 64.04% | 61.62% |
| Random Forest | 74.14% | 65.93% | 67.42% | 66.67% |

Random Forest performed better on the test set and was selected for the retention analysis.

Main outputs:
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## 5. Part C — Support Ticket Processing and Triage

The project processed **164 support tickets**:
- 160 tickets were ready for classification.
- 4 tickets with missing text were sent for manual review.
- 39 fraud-related tickets were routed for priority review.

Ticket processing includes text redaction, classification, and routing.

**Triage evaluation:**

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 12,364 | 21,568 |

The triage pipeline also includes a configured **6,000-token budget**. In the recorded budget demonstration, 80 tickets reached the budget-exhausted status and the aggregate token count was 6,038. This is a demonstration of budget handling, not a claim that aggregate usage stayed below the configured limit.

PII redaction covered email addresses, phone numbers, and order IDs. The evaluation reported zero remaining PII matches.

Main outputs:
- `data/clean/processed_tickets.csv`
- `data/clean/classified_tickets.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_metrics.txt`

## 6. Part D — Policy QA and Simulated Refund Agent

The policy QA system uses five policy documents divided into overlapping text chunks.

Evaluation covered 14 questions:
- 12 answerable questions
- 2 unanswerable questions

With chunk overlap, the system retrieved the expected documents and facts for all 12 answerable questions and correctly handled both unanswerable questions.

The refund agent demonstrates policy-based decisions, including refunds, denials, escalations, and fallback handling.

Seven batch scenarios covered different outcomes:
- Refund approved
- Escalation to a human
- Refund denied
- Order not found
- Maximum-iteration fallback

Additional demonstrations tested token-limit fallback, timeout fallback, and checkpoint-based crash/resume behaviour.

The refund ledger is a simulated audit log. Its existing records include repeated events for some order IDs, so the demonstration does not establish exactly-once refund processing or duplicate-refund prevention. No live payment service is connected.

Main outputs:
- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint files

## 7. Limitations and Future Improvements

- The project uses simulated Instamart data and a mock LLM.
- Model performance may change with real-world data.
- The refund agent is a demonstration and does not issue real payments.
- Production deployment would require stronger transaction controls, duplicate-refund protection, monitoring, and integration with secure payment services.

## Conclusion

This project demonstrates an end-to-end customer intelligence and support workflow, combining data analytics, machine learning, ticket triage, policy QA, and a simulated refund agent.