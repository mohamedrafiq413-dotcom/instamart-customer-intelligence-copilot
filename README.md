# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

This project builds a customer intelligence and support copilot for a simulated Instamart business. It combines data analytics, machine learning, ticket classification, policy retrieval and a simulated refund agent.

**Technologies:** Python, Pandas, NumPy, SQLite, Scikit-learn, TF-IDF and Mock LLM.

## 2. Setup and Reproduction

**Requirements:** Python 3.11+, Git and VS Code.

### Install and activate environment (Windows PowerShell)

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the project

Run the scripts from the project root in this order:

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
python .\src\plan_execute_comparison.py
```

The policy QA script opens an interactive session after evaluation. Type `exit` to finish.

**Validation:** 53 checks passed, 0 failed.

## 3. Part A - Data Quality and SQL Analytics

Generated data included 1,200 customers, 40 products, 13,084 raw orders and 164 support tickets.

After cleaning, 12,309 orders remained:
- 11,110 completed orders
- 1,199 cancelled orders

The SQLite database was created at `data/instamart.db`.

**Key findings:**
- Completed-order revenue: ₹2,998,668.80
- Top-selling product: Instant Noodles 4 Pack (867 units)
- Seven SQL queries analysed revenue, customers, products and order trends.

## 4. Part B - Customer Intelligence and Lapse Prediction

Customer intelligence identified Regular, At Risk, High Value and No Completed Orders segments.

RFM analysis grouped 1,175 customers into four clusters. The first two PCA components explained 85.23% of the variance.

**Lapse prediction results:**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 69.40% | 59.38% | 64.04% | 61.62% |
| Random Forest | 73.71% | 65.22% | 67.42% | 66.30% |

Random Forest was selected based on its higher cross-validation F1-score. Oversampling improved Logistic Regression recall to 85.39%.

The trained model was saved and reloaded successfully. A retention list of the top 10 Premium test customers was also generated.

**Main outputs:**
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## 5. Part C - Support Ticket Processing and Triage

The system processed 164 support tickets, including 160 classification-ready tickets and 4 manual-review tickets.

**Classification results:**

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 12,364 | 21,568 |

The pipeline includes classification, routing, retry handling and token-budget controls.

**PII redaction:**
- 55 email addresses
- 91 phone numbers
- 132 order IDs
- 278 total replacements
- Zero remaining PII matches

The 6,000-token budget demonstration produced 80 budget-exhausted tickets and 6,038 aggregate tokens due to pre-call budget checking.

**Main outputs:**
- `data/clean/processed_tickets.csv`
- `data/clean/classified_tickets.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_metrics.txt`

## 6. Part D - Policy QA and Simulated Refund Agent

### Policy QA

Five policy documents were divided into 40-word chunks with 10-word overlap. TF-IDF and cosine similarity retrieved the top three relevant chunks.

Evaluation covered 12 answerable and 2 unanswerable questions.

| Metric | With overlap | Zero overlap |
|---|---:|---:|
| Expected facts found | 12/12 | 10/12 |
| Correct fallback responses | 2/2 | 2/2 |

### Refund Support Agent

The agent uses five tools for order lookup, policy search, refund calculation, refund issuance and human escalation.

Seven scenarios tested refunds, denials, escalation, missing orders and fallback handling. Additional tests covered:
- Seven-day refund eligibility
- Token-limit and timeout handling
- Crash-resume functionality
- Human-handoff summaries

### ReAct vs Plan-and-Execute

Both approaches were compared using three scenarios: refund, cancelled order and missing order. Both produced the expected outcomes in all three cases.

ReAct used 5, 3 and 2 LLM calls respectively. Plan-and-Execute used one deterministic planning pass followed by 4, 2 and 1 tool executions. The comparison uses a rule-based baseline rather than an LLM-powered Plan-and-Execute model.

**Main outputs:**
- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/react_vs_plan_execute.csv`
- `state/` checkpoint files

## 7. Limitations

- The project uses simulated business data and an offline Mock LLM.
- Delivery timestamps are estimated from order timestamps and delivery duration; batch refund-request dates are synthetic.
- The refund ledger contains historical repeated events for some orders.
- No real payments are issued.
- Production deployment would require stronger transaction controls, monitoring and secure payment integration.

## Conclusion

This project demonstrates an end-to-end customer intelligence and support workflow using data analytics, machine learning, NLP, policy retrieval and simulated AI agent tools.