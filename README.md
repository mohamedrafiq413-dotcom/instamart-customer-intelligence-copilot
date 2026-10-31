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

**Project validation:** 53 checks passed, 0 failed.

The policy QA script opens an interactive session after evaluation. Type `exit` to finish. The project uses simulated data and an offline Mock LLM; no paid API is required.

## 3. Part A - Data Quality and SQL Analytics

Generated data:
- 1,200 customers
- 40 products
- 13,084 raw orders
- 164 support tickets

After cleaning, 12,309 orders remained:
- 11,110 completed orders
- 1,199 cancelled orders

The SQLite database was created at `data/instamart.db`.

**Key findings:**
- Completed-order revenue: ₹2,998,668.80
- Top-selling product: Instant Noodles 4 Pack (867 units)
- The analysis covered customer activity, product sales, city revenue, order trends and cumulative revenue.
- Seven SQL queries were executed successfully.

## 4. Part B - Customer Intelligence and Lapse Prediction

Customer intelligence identified Regular, At Risk, High Value and No Completed Orders segments.

RFM analysis included 1,175 customers and four K-Means clusters.

| Segment | Customers | Recency | Frequency | Monetary (₹) |
|---|---:|---:|---:|---:|
| Recently Active | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Frequent | 419 | 8.77 | 19.40 | 3,410.02 |
| Developing | 182 | 109.04 | 2.20 | 685.99 |

The first two PCA components explained 85.23% of the variance.

**Lapse prediction results:**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.40% | 59.38% | 64.04% | 61.62% |
| Random Forest | 73.71% | 65.22% | 67.42% | 66.30% |

Random Forest was selected based on its higher training cross-validation F1-score (0.6869).

Oversampling the training data improved Logistic Regression recall from 64.04% to 85.39%.

The model was saved and reloaded with identical predictions. The retention list contains the top 10 Premium test customers by lapse probability; one was actually lapsed. This list is illustrative because Premium membership uses the full observation period.

**Main outputs:**
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## 5. Part C - Support Ticket Processing and Triage

The project processed 164 support tickets. 160 were ready for classification, and 4 required manual review.

**Triage evaluation:**

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 12,364 | 21,568 |

The few-shot approach improved accuracy but used more tokens.

**PII redaction:**
- 55 email addresses
- 91 phone numbers
- 132 order IDs
- 278 total replacements
- Zero remaining PII matches

The pipeline includes retry handling, long-message protection, token-budget checks and ticket routing. The 6,000-token budget demonstration produced 80 budget-exhausted tickets and 6,038 aggregate tokens due to pre-call budget checking.

**Main outputs:**
- `data/clean/processed_tickets.csv`
- `data/clean/classified_tickets.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_metrics.txt`

## 6. Part D - Policy QA and Simulated Refund Agent

### Policy QA

Five policy documents were divided into 40-word chunks with 10-word overlap, producing 15 chunks. TF-IDF and cosine similarity retrieved the three most relevant chunks.

Evaluation covered 12 answerable and 2 unanswerable questions.

| Evaluation | With overlap | Zero overlap |
|---|---:|---:|
| Expected documents found | 12/12 | 12/12 |
| Expected facts found | 12/12 | 10/12 |
| Correct fallback responses | 2/2 | 2/2 |

Overlap helped preserve two policy facts that were missed at chunk boundaries without overlap.

### Refund Support Agent

The simulated agent uses five tools:

- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

Seven scenarios tested refunds, denials, escalation, missing orders and maximum-iteration fallback.

| Scenario | Result | LLM calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND - INR 128.25 | 5 | 1,102 |
| S2 | REFUND - INR 135.00 | 5 | 1,102 |
| S3 | ESCALATE | 5 | 1,089 |
| S4 | DENIED | 4 | 769 |
| S5 | DENIED | 3 | 460 |
| S6 | NOT_FOUND | 1 | 71 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,246 |

Additional tests verified token-limit fallback, timeout fallback and checkpoint-based crash/resume.

The agent follows a ReAct approach, selecting tools based on previous observations. The saved runs required 5 calls for S1, 3 for S5 and 1 for S6. ReAct allows the agent to adapt to tool results, while Plan-and-Execute establishes a plan before execution. ReAct was preferred for this workflow because it supports step-by-step decisions. A separate Plan-and-Execute implementation and numerical benchmark were not completed.

The human-handoff fallback now includes completed tool observations and relevant policy sources.

**Limitations:** The seven-day refund rule cannot currently be enforced because separate delivery and refund-request dates are unavailable. The refund ledger contains repeated historical events for some orders. No real payments are issued.

**Main outputs:**
- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint files

## 7. Limitations and Future Improvements

- The project uses simulated data and an offline Mock LLM.
- Real-world model performance may differ.
- Seven-day refund eligibility needs delivery and request-date information.
- Production deployment would require stronger transaction controls, duplicate-refund protection, monitoring and secure payment integration.

## Conclusion

This project demonstrates an end-to-end customer intelligence and support workflow using data analytics, machine learning, ticket triage, policy retrieval and a simulated refund agent.