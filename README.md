# Instamart Customer Intelligence & Support Copilot

## 1. Overview

The Instamart Customer Intelligence & Support Copilot is a data analytics and AI-assisted customer support project. It combines data cleaning, SQL analysis, customer segmentation, lapse prediction, support ticket classification, policy question answering, and a simulated refund-agent workflow.

The project demonstrates how data analytics and language-model-assisted workflows can help an online grocery business understand customers, identify retention opportunities, classify support requests, retrieve relevant policies, and handle refund scenarios safely.

### Project objectives

- Audit and clean raw customer, product, order, and support-ticket data.
- Store cleaned data in a SQLite database.
- Analyse sales, order behaviour, products, and cities using SQL.
- Segment customers and identify customers who may be at risk of lapsing.
- Classify and route customer support tickets.
- Evaluate policy question answering with and without overlapping text chunks.
- Demonstrate a simulated refund agent with policy checks, fallback handling, and crash/resume behaviour.

### Technology stack

- **Language:** Python
- **Data analysis:** pandas, NumPy
- **Database:** SQLite
- **Machine learning:** scikit-learn
- **Visualisation:** Matplotlib and other project dependencies, where used
- **Language-model simulation:** `mock_llm.py`
- **Development environment:** Visual Studio Code
- **Version control:** Git and GitHub

Install project dependencies using `requirements.txt`.

### Project structure

The main project files are organised as follows. Some files under `data/` and `outputs/` are generated when the scripts run.

```text
Instamart-Capstone/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── processed_tickets.csv
│   │   ├── classified_tickets.csv
│   │   └── customer_segments.csv
│   ├── policies/
│   └── instamart.db
├── outputs/
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── triage_metrics.txt
│   ├── policy_eval_results.csv
│   ├── policy_eval_zero_overlap.csv
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
├── state/
│   └── Agent state and checkpoint JSON files
├── sql/
│   └── queries.sql
├── src/
│   ├── audit_data.py
│   ├── validate_data.py
│   ├── clean_customers.py
│   ├── clean_products.py
│   ├── clean_orders.py
│   ├── build_database.py
│   ├── sql_analysis.py
│   ├── customer_intelligence.py
│   ├── customer_insights.py
│   ├── lapse_model.py
│   ├── process_tickets.py
│   ├── classify_tickets.py
│   ├── triage_pipeline.py
│   ├── policy_qa.py
│   └── refund_agent.py
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md
```

This tree illustrates the main files and expected generated outputs. The precise files present can vary depending on which scripts have been run.

## 2. Reproduce

### 2.1 Prerequisites

Install Python and Git, then open the project folder in Visual Studio Code.

Open the VS Code terminal and navigate to the project directory.

### 2.2 Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
```

Activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install dependencies and check the environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The final dependency check completed successfully in the tested environment, reporting:

```text
No broken requirements found.
```

### 2.3 Generate and audit the data

Run these commands in order when reproducing the project from generated raw data:

```powershell
python .\generate_data.py
python .\src\audit_data.py
```

### 2.4 Clean and validate the data

```powershell
python .\src\clean_customers.py
python .\src\clean_products.py
python .\src\clean_orders.py
python .\src\validate_data.py
```

The validation script completed with **53 checks passed and 0 failed** in the tested project copy.

### 2.5 Build the database and run SQL analysis

```powershell
python .\src\build_database.py
python .\src\sql_analysis.py
```

### 2.6 Run customer analytics and lapse prediction

```powershell
python .\src\customer_intelligence.py
python .\src\customer_insights.py
python .\src\lapse_model.py
```

### 2.7 Run support-ticket processing and triage

```powershell
python .\src\process_tickets.py
python .\src\classify_tickets.py
python .\src\triage_pipeline.py
```

### 2.8 Run policy question answering

```powershell
python .\src\policy_qa.py
```

### 2.9 Run refund-agent demonstrations

Run the batch scenarios:

```powershell
python .\src\refund_agent.py --batch
```

Run the fallback demonstrations:

```powershell
python .\src\refund_agent.py --demos
```

Run the crash/resume demonstration:

```powershell
python .\src\refund_agent.py --resume-demo
```

These commands demonstrate the tested workflows. Follow the actual script outputs if generated filenames or behaviour differ in another environment.

## 3. Part A — Data Quality and SQL Analytics

### 3.1 Dataset summary

The generated dataset contains the following records:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The order-cleaning process reduced the raw dataset from 13,084 rows to 12,309 cleaned orders.

### 3.2 Data quality findings

The raw data audit identified several data quality issues:

- Multiple spellings of city names.
- Inconsistent gender values and missing gender values.
- Missing ages and ages outside the audit's 10–100 range.
- Duplicate order rows and repeated order IDs.
- Missing, invalid, or non-positive order amounts requiring investigation.
- Missing and negative delivery-time values.
- Support tickets with missing text or missing order references.

Specific raw-data findings included:

| Issue | Observed count or value |
|---|---:|
| Distinct raw city spellings | 24 |
| Nonblank raw gender spellings | 10 |
| Blank raw gender values | 30 |
| Blank ages | 61 |
| Ages outside the 10–100 audit range | 6 |
| Exact duplicate order rows | 119 |
| Raw orders with invalid or missing amounts | 120 |
| Blank raw delivery-time values | 2,541 |
| Negative raw delivery-time values | 106 |
| Tickets with missing text | 4 |

These counts describe the raw-data audit. They are not necessarily counts of unresolved issues in the cleaned data.

### 3.3 Cleaned data validation

The validation script reported:

- **Customers:** 1,200 rows, 5 columns, no missing values, and no duplicate rows.
- **Products:** 40 rows, 4 columns, no missing values, and no duplicate rows.
- **Orders:** 12,309 rows and 9 columns.
- **Completed orders:** 11,110.
- **Cancelled orders:** 1,199.
- **Missing delivery times in cleaned orders:** 1,199, corresponding to cancelled orders.
- **Validation result:** 53 passed, 0 failed.

### 3.4 SQLite database

The project creates `data/instamart.db` with three main tables:

| Table | Rows |
|---|---:|
| `customers` | 1,200 |
| `products` | 40 |
| `orders` | 12,309 |

The database allows customer, product, and order data to be analysed together using SQL.

### 3.5 Sales and order findings

Recorded spending across completed orders was **₹2,998,668.80**.

| Order status | Count |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Total cleaned orders | 12,309 |

#### City-level completed-order results

| City | Revenue (₹) | Mean order value (₹) | Completed orders |
|---|---:|---:|---:|
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai | 463,629.15 | 272.72 | 1,700 |

Hyderabad recorded the highest revenue in this result set, while Mumbai recorded the highest mean order value.

#### Top products by quantity

| Rank | Product | Quantity |
|---:|---|---:|
| 1 | Instant Noodles 4 Pack | 867 |
| 2 | Peanut Butter 340g | 845 |
| 3 | Cold Coffee Can 180ml | 836 |
| 4 | Sunflower Oil 1L | 820 |
| 5 | Toned Milk 500ml | 811 |

#### Order amount distribution

For completed orders:

| Statistic | Value |
|---|---:|
| Count | 11,110 |
| Mean | ₹269.91 |
| Median | ₹135.00 |
| First quartile | ₹64.80 |
| Third quartile | ₹297.00 |
| Skewness | 4.61 |
| Upper IQR fence | ₹645.30 |
| Observations above upper fence | 897 |
| Observations below zero | 0 |

The positive skewness indicates that the order amount distribution has a long upper tail. The 897 observations above the IQR upper fence are statistical outliers, not necessarily data errors.

The correlation between order amount and discount percentage was approximately −0.095, indicating a weak negative linear relationship in this dataset. Correlation alone does not establish that discounts caused lower order amounts.

### 3.6 SQL query analysis

The SQL analysis script ran seven query groups successfully:

| Query group | Rows returned |
|---|---:|
| Q1 | 30 |
| Q2 | 30 |
| Q3 | 11 |
| Q4 | 25 |
| Q5 | 550 |
| Q6 | 1,175 |
| Q7 | 183 |

The analysis covers business questions such as sales summaries, product performance, customer/order relationships, ranking, and running revenue. The exact question definitions and SQL are maintained in the project SQL files and analysis script.

The running-revenue analysis ended at ₹2,998,668.80, matching the recorded completed-order spending.

## 4. Part B — Customer Intelligence and Lapse Prediction

### 4.1 Customer segmentation

The customer intelligence script generated four customer segments.

| Segment | Customers | Percentage |
|---|---:|---:|
| Regular | 633 | 52.75% |
| At Risk | 301 | 25.08% |
| High Value | 241 | 20.08% |
| No Completed Orders | 25 | 2.08% |
| **Total** | **1,200** | **100%** |

Percentages are rounded, so their displayed sum may differ slightly from 100%.

#### Segment behaviour

| Segment | Average completed orders | Average spend (₹) | Average recency (days) |
|---|---:|---:|---:|
| Regular | 10.21 | 1,971.38 | 15.87 |
| At Risk | 2.87 | 1,285.30 | 88.40 |
| High Value | 15.71 | 5,659.37 | 11.46 |
| No Completed Orders | 0.00 | 0.00 | Not available |

The results suggest several possible business actions:

- **At Risk:** Consider re-engagement campaigns and investigate potential service issues.
- **High Value:** Consider loyalty benefits and relevant personalised recommendations.
- **No Completed Orders:** Investigate onboarding and the customer journey.
- **Regular:** Encourage repeat purchases with relevant offers.

These are analytical recommendations, not evidence that a particular campaign will necessarily improve retention.

### 4.2 RFM segmentation and lapse prediction

The lapse modelling workflow uses recency, frequency, and monetary behaviour, together with additional customer/order features, to explore customer lapse risk.

The tested snapshot date was 1 October 2025, with a lapse cutoff of 1 September 2025.

The analysis used:

- 1,200 cleaned customers.
- 10,301 orders before the cutoff.
- 9,281 completed orders before the cutoff.
- 1,829 completed orders in September.
- A lapse cohort of 1,156 customers.
- 443 lapsed customers, for a cohort lapse rate of 38.32%.
- 924 training records and 232 test records.

#### RFM-style segments

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary value (₹) |
|---|---:|---:|---:|---:|
| Premium | 137 | 24.11 | 6.79 | 7,148.22 |
| Recently Active | 461 | 29.27 | 3.57 | 1,000.26 |
| Frequent | 419 | 9.13 | 19.40 | 3,410.02 |
| Developing | 183 | 109.28 | 2.21 | 707.33 |

The clustering comparison evaluated values of *k* from 2 through 8. The best silhouette score in that comparison was 0.5288 at *k* = 4. The four-segment solution was used for the reported segmentation.

### 4.3 Model comparison

The tested models included a baseline, logistic regression, and random forest.

| Model or experiment | Result |
|---|---|
| Always predict not lapsed baseline | Accuracy 61.64%; lapsed-class F1 0 |
| Logistic regression test accuracy | 69.40% |
| Logistic regression test precision | 59.38% |
| Logistic regression test recall | 64.04% |
| Logistic regression test F1 | 61.62% |
| Random forest test accuracy | 74.14% |
| Random forest test precision | 65.93% |
| Random forest test recall | 67.42% |
| Random forest test F1 | 66.67% |

The random forest was selected based on the training cross-validation comparison. Its reported test F1 was 0.6667.

The random forest's leading feature importances included frequency (0.2875), monetary value (0.2751), and recency (0.1186).

The results are specific to this generated dataset and evaluation split. They should not be interpreted as guaranteed performance on real customers.

### 4.4 Retention output

The script saved the model to:

`outputs/lapse_model.joblib`

It also saved the retention list to:

`outputs/retention_list.csv`

The model reload check reported identical predictions after saving and reloading.

**Important limitation:** The reported Premium segment labels use RFM information from the full observation window, including September, whereas the lapse model features use information before the cutoff and exclude the segment label. The retention list is a demonstration output, not a validated production targeting list.

## 5. Part C — Support Ticket Processing and Triage

### 5.1 Ticket preprocessing

The raw support-ticket dataset contains 164 tickets.

The preprocessing script reported:

| Processing outcome | Tickets |
|---|---:|
| Ready for classification | 160 |
| Manual review due to missing text | 4 |
| Priority fraud-review tickets | 39 |
| Total | 164 |

The script saved the processed data to:

`data/clean/processed_tickets.csv`

Preprocessing creates a redacted text field for downstream processing and routes tickets according to the implemented rules. The original raw dataset should be preserved for traceability.

### 5.2 Ticket classification

The classification script saved:

`data/clean/classified_tickets.csv`

Reported classification statuses:

| Status | Count |
|---|---:|
| Classified JSON | 135 |
| Classified by keyword | 12 |
| Flagged invalid output | 9 |
| Manual review | 4 |
| Flagged long | 4 |
| **Total** | **164** |

Predicted categories:

| Category | Count |
|---|---:|
| Technical | 39 |
| Billing | 38 |
| Delivery | 36 |
| Fraud | 34 |
| Needs review | 13 |
| None | 4 |

The output checks passed. Tickets flagged for invalid output, excessive length, or missing text require appropriate review rather than being treated as reliably classified.

### 5.3 Zero-shot and few-shot triage comparison

The triage pipeline compared zero-shot and few-shot classification.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy among classified tickets | 85.7% | 100% |
| Tokens used | 12,364 | 21,568 |
| Tokens per classified ticket | 84.11 | 146.72 |

In this experiment, few-shot classification produced higher accuracy at a greater token cost. It used approximately 74.4% more tokens than the zero-shot run.

These results describe the recorded evaluation and should not be assumed to generalise to other ticket distributions or language models.

### 5.4 Token budget and final routing

The demonstration used a configured token budget of 6,000 and reported 80 budget-exhausted tickets. The saved `outputs/triage_budget_demo.csv` contains 164 rows and records a total of 6,038 tokens. The recorded total is slightly above the configured cap, so the budget should be interpreted as the pipeline's configured control rather than assuming the final recorded total must be at or below 6,000.

Generated files:

- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`
- `outputs/triage_metrics.txt`

The final few-shot output contained 164 rows, with these reported routes:

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| **Total** | **164** |

The PII audit detected 278 matches before redaction across email, phone, and order-ID patterns. It reported zero remaining matches after redaction in the audited output.

Pattern-based redaction is not a guarantee that all personally identifiable information will be detected. Production systems would need stronger privacy testing and access controls.

## 6. Part D — Policy QA and Simulated Refund Agent

### 6.1 Policy question answering approach

The policy QA workflow loads policy documents, splits them into text chunks, retrieves relevant chunks for a question, and evaluates the answer against the expected document and expected fact.

The tested configuration used:

- 5 policy documents.
- 15 total chunks.
- 40-word chunks.
- 10-word overlap between adjacent chunks.

The chunk distribution was:

| Policy document | Chunks |
|---|---:|
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |
| Refund | 4 |
| **Total** | **15** |

### 6.2 Policy QA main evaluation results

The evaluation contained 14 questions: 12 answerable questions and 2 unanswerable questions.

| Metric | Result |
|---|---:|
| Answerable questions | 12 |
| Expected document found in top 3 | 12/12 |
| Expected fact found in answer | 12/12 |
| Unanswerable questions handled with the correct fallback | 2/2 |

The main evaluation achieved 100% on these recorded checks for this test set.

### 6.3 Comparison with zero-overlap chunks

A second evaluation tested chunking without overlap.

| Metric | With 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document found in top 3 | 12/12 | 12/12 |
| Expected fact found in answer | 12/12 | 10/12 |
| Correct fallback for unanswerable questions | 2/2 | 2/2 |

The zero-overlap experiment retrieved the expected document in all 12 answerable cases, but the answer contained the expected fact in only 10 of those 12 cases (83.33%).

The two missed facts were:

- The delivery full-refund threshold omitted the condition “more than 45 minutes.”
- The failed-payment answer omitted the condition “within 48 hours.”

The expected document was still found in the top three results for both cases. This suggests that retrieval alone is not sufficient: the answer must also preserve important conditions in the source text.

The comparison supports using overlap for this particular dataset and implementation. It is not proof that overlapping chunks will always outperform non-overlapping chunks for every retrieval system.

Generated evaluation files:

- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`

### 6.4 Policy QA limitations

- Evaluation uses a small set of 14 questions.
- Retrieval and answer checks are based on the project's evaluation logic.
- Performance may change with different policies, questions, chunk sizes, or retrieval settings.
- Policy answers should retain relevant thresholds, deadlines, exceptions, and eligibility conditions.
- Uncertain or unsupported answers should be escalated rather than invented.

### 6.5 Simulated refund agent purpose

The refund agent demonstrates a multi-step support workflow using order lookup, policy search, refund calculation, and a simulated refund action.

It includes scenarios for refund approval, escalation, denial, missing orders, iteration limits, token limits, timeouts, and crash/resume behaviour.

The workflow is a demonstration of control flow and tool orchestration, not a live payment-processing integration.

### 6.6 Batch scenario results

The batch run recorded seven scenarios:

| Scenario | Outcome | Tool calls | Tokens |
|---|---|---:|---:|
| S1 | Refund | 5 | 1,102 |
| S2 | Refund | 5 | 1,102 |
| S3 | Escalate | 5 | 1,089 |
| S4 | Denied | 4 | 769 |
| S5 | Denied | 3 | 460 |
| S6 | Not found | 1 | 71 |
| S7 | Fallback: maximum iterations | 6 | 3,246 |

The run saved scenario records to:

`outputs/agent_runs.jsonl`

The persistent refund ledger is stored at:

`outputs/refund_ledger.jsonl`

Agent state and checkpoint files are stored in:

`state/`

### 6.7 Fallback and crash/resume demonstrations

The demo command reported these outcomes:

| Demonstration | Reported outcome |
|---|---|
| Token-limit demonstration | `FALLBACK_TOKEN_LIMIT` |
| Timeout demonstration | `FALLBACK_TIMEOUT` |

The crash/resume demonstration reported that a simulated crash occurred after tool execution and a checkpoint was saved. The workflow then resumed with the tool sequence:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The final demonstration result was:

`REFUND: INR 128.25 issued.`

This is a simulated result, not evidence that a real customer payment was issued. The script reported that the persistent refund ledger was not changed by the demonstration.

### 6.8 Refund-agent safety considerations

A production refund agent would need, at minimum:

- Verified order and customer identity.
- Policy validation before refund decisions.
- Strict monetary limits and permission checks.
- Idempotency controls to prevent duplicate refunds.
- Durable audit logs and recovery checkpoints.
- Human approval for uncertain or high-risk cases.
- Secure handling of customer and payment information.
- Integration tests against a sandbox payment service before any live deployment.

The demo does not establish that the workflow is ready for real financial transactions.

## 7. Limitations and Next Steps

### 7.1 Overall findings

The project produced several useful findings from the generated data and tested workflows:

- **Data quality matters.** Normalising customer fields, removing exact duplicate rows, and validating order records establishes a more consistent foundation for analysis.
- **Completed-order revenue can be reconciled.** The SQL analysis and customer insights reported ₹2,998,668.80 in completed-order spending.
- **Customer segments support different retention strategies.** The At Risk and High Value groups have distinct order frequency, spend, and recency patterns.
- **Lapse prediction outperformed the baseline in the recorded test.** The selected random forest achieved a test F1 of 0.6667, although real-world performance remains unverified.
- **Ticket routing needs fallback paths.** Invalid outputs, missing messages, and long tickets were handled through explicit statuses and review routes.
- **Policy answers must preserve exact conditions.** The zero-overlap comparison missed two expected facts even though the expected documents were retrieved.
- **Refund workflows need safeguards.** Token limits, timeouts, maximum iterations, and crash/resume behaviour demonstrate why fallback handling and auditability are important.

### 7.2 Limitations

- The datasets are generated for the project and do not establish performance on real Instamart customer data.
- The lapse model's results depend on the snapshot, features, split, and generated data distribution.
- Ticket classification and policy QA evaluation sets are limited in size.
- Text redaction and policy retrieval can miss information or conditions.
- The refund agent uses a simulated workflow and does not prove live payment execution.
- Successful syntax, dependency, and validation checks do not guarantee that every production scenario has been tested.

### 7.3 Future improvements

- Add automated unit and integration tests for edge cases.
- Evaluate lapse prediction on a separate, representative dataset with time-based validation.
- Track precision and recall at the retention-list threshold and estimate campaign costs.
- Expand policy QA tests with more questions, paraphrases, exceptions, and unsupported queries.
- Add retrieval-quality and answer-faithfulness measurements.
- Improve multilingual ticket handling and PII redaction testing.
- Add structured logs, reproducible configuration, and model/version tracking.
- Use a sandbox environment and explicit human approval for any future payment integration.
- Document expected outputs for each script and add a single reproducible test command.

### 7.4 Conclusion

The Instamart Customer Intelligence & Support Copilot combines data engineering, SQL analytics, customer segmentation, machine learning, ticket triage, policy retrieval, and simulated agent orchestration in one project.

The tested scripts completed their reported checks, including 53 data validation checks, successful Python compilation, and a clean dependency check. The experiments also identify areas requiring further evaluation, particularly generalisation of the lapse model, reliability of ticket classification, completeness of policy answers, and safeguards around refund actions.

The project is a learning and demonstration system. Additional testing, representative data, privacy review, operational monitoring, and appropriate approvals would be required before using similar workflows in production.
