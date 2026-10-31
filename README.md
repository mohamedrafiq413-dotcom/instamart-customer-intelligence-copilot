# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

This project builds a customer intelligence and support copilot for a simulated Instamart business. It covers data cleaning, SQL analytics, customer segmentation, lapse prediction, ticket triage, policy QA, and a simulated refund agent.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, and Mock LLM.

## 2. Setup and Reproduction

### Requirements
- Python 3.11+
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

Run scripts from the project root in this order:

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

Project validation: **53 checks passed, 0 failed.**

## 3. Part A - Data Quality and SQL Analytics

Generated data:
- 1,200 customers
- 40 products
- 13,084 raw orders
- 164 support tickets

After cleaning, 12,309 orders remained:
- 11,110 completed
- 1,199 cancelled

The SQLite database was created at `data/instamart.db`.

**Key findings:**
- Completed-order revenue: ₹2,998,668.80
- Top-selling product: Instant Noodles 4 Pack (867 units)
- Revenue was analysed by customer, city, product and date using seven SQL queries.

## 4. Part B - Customer Intelligence and Lapse Prediction

Customer intelligence identified Regular, At Risk, High Value and No Completed Orders segments.

RFM analysis used 1,175 customers and four K-Means clusters.

| Segment | Customers | Recency | Frequency | Monetary (₹) |
|---|---:|---:|---:|---:|
| Recently Active | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Frequent | 419 | 8.77 | 19.40 | 3,410.02 |
| Developing | 182 | 109.04 | 2.20 | 685.99 |

The lapse prediction model used historical customer activity to identify inactive customers.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.40% | 59.38% | 64.04% | 61.62% |
| Random Forest | 73.71% | 65.22% | 67.42% | 66.30% |

Random Forest was selected based on its higher cross-validation F1-score.

The balanced Logistic Regression experiment improved recall from 64.04% to 85.39%.

The model was saved and reloaded successfully. The retention list contains the top 10 Premium test customers by lapse probability. One was actually lapsed. This list is illustrative because Premium membership uses the full observation period.

**Main outputs:**
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## 5. Part C - Support Ticket Processing and Triage

The project processed 164 support tickets. 160 were ready for classification, and 4 required manual review.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 12,364 | 21,568 |

The few-shot approach improved accuracy but used more tokens.

**PII redaction results:**
- 55 email addresses
- 91 phone numbers
- 132 order IDs
- 278 total replacements
- Zero remaining PII matches

The pipeline also includes retry handling, long-message protection, token-budget checks and ticket routing.

The 6,000-token budget demonstration produced 80 budget-exhausted tickets and 6,038 aggregate tokens due to pre-call budget checking.

**Main outputs:**
- `data/clean/processed_tickets.csv`
- `data/clean/classified_tickets.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_metrics.txt`

## 6. Part D - Policy QA and Simulated Refund Agent

The Policy QA system uses five policy documents, 40-word chunks and 10-word overlap. It retrieves relevant information using TF-IDF and cosine similarity.

Evaluation covered 14 questions: 12 answerable and 2 unanswerable.

| Evaluation | With overlap | Zero overlap |
|---|---:|---:|
| Expected documents found | 12/12 | 12/12 |
| Expected facts found | 12/12 | 10/12 |
| Correct fallback responses | 2/2 | 2/2 |

The simulated refund agent uses five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`.

Seven scenarios tested refunds, denials, escalation, missing orders and maximum-iteration fallback. Additional tests covered token limits, timeouts and crash-resume functionality.

The agent uses a ReAct approach, selecting tools based on previous observations. Plan-and-Execute was considered as an alternative, but a separate implementation and measured comparison were not completed.

**Limitations:** The seven-day refund window is not enforced because delivery and request timestamps are unavailable. The handoff fallback also needs a more detailed work summary. The refund ledger contains repeated historical events for some orders. No real payments are issued.

**Main outputs:**
- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint files

## 7. Limitations and Future Improvements

- The project uses simulated data and an offline Mock LLM.
- Real-world model performance may differ.
- Refund processing requires stronger duplicate prevention and transaction controls.
- Seven-day refund eligibility and detailed human handoff need further implementation.
- Production deployment would require secure payment integration and monitoring.

## Conclusion

This project demonstrates an end-to-end customer intelligence and support workflow using data analytics, machine learning, ticket triage, policy retrieval and a simulated refund agent.