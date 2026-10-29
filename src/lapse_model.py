
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Paths and fixed dates
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")


# ---------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------
customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

completed = orders[
    orders["status"].astype(str).str.lower().eq("completed")
    & orders["order_ts"].notna()
].copy()

# Features for lapse prediction must use orders strictly before
# September 1, 2025. September is reserved for defining the label.
pre_cutoff = orders[
    orders["order_ts"] < LAPSE_CUTOFF
].copy()

completed_pre = completed[
    completed["order_ts"] < LAPSE_CUTOFF
].copy()

september_completed = completed[
    (completed["order_ts"] >= LAPSE_CUTOFF)
    & (completed["order_ts"] < SNAPSHOT_DATE)
].copy()

active_in_september = set(
    september_completed["customer_id"]
)

print("=" * 70)
print("INSTAMART CUSTOMER SEGMENTATION AND LAPSE PREDICTION")
print("=" * 70)
print("Snapshot date:", SNAPSHOT_DATE)
print("Lapse cutoff:", LAPSE_CUTOFF)
print("Clean customers:", len(customers))
print("Orders before cutoff:", len(pre_cutoff))
print("Completed orders before cutoff:", len(completed_pre))
print("Completed September orders:", len(september_completed))


# ---------------------------------------------------------
# 3. Whole-window RFM segmentation
# ---------------------------------------------------------
# RFM is calculated as of the fixed snapshot date.
# This whole-window segment output is for reporting and routing.
# Segment labels are not used as predictive model features.
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order"]
).dt.total_seconds() / 86400

rfm = rfm[[
    "customer_id", "recency", "frequency", "monetary"
]].copy()

rfm = customers[["customer_id"]].merge(
    rfm, on="customer_id", how="left"
)

# Customers without completed orders receive zero frequency/spend.
# Recency is set to days between snapshot and their last possible
# observation boundary, so it is never missing.
rfm["frequency"] = rfm["frequency"].fillna(0)
rfm["monetary"] = rfm["monetary"].fillna(0)
rfm["recency"] = rfm["recency"].fillna(
    (SNAPSHOT_DATE - LAPSE_CUTOFF).total_seconds() / 86400
)

rfm_numeric = rfm[["recency", "frequency", "monetary"]]

rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(rfm_numeric)

# Required K-means model comparison: K = 2 through 8.
print("\nK-MEANS COMPARISON")
kmeans_results = []

for k in range(2, 9):
    candidate = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = candidate.fit_predict(rfm_scaled)
    silhouette = silhouette_score(rfm_scaled, labels)

    kmeans_results.append({
        "k": k,
        "inertia": candidate.inertia_,
        "silhouette": silhouette,
    })

kmeans_results_df = pd.DataFrame(kmeans_results)
print(kmeans_results_df.to_string(index=False, float_format="%.4f"))

# K=4 is retained for the final segmentation so the segment report
# remains comparable with the existing assignment output.
FINAL_K = 4
kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10,
)
rfm["segment_id"] = kmeans.fit_predict(rfm_scaled)

segment_summary = (
    rfm.groupby("segment_id")
    .agg(
        customers=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .sort_index()
)

# Premium is explicitly the cluster with the highest mean spend.
premium_segment_id = int(
    segment_summary["mean_monetary"].idxmax()
)

# Give remaining clusters descriptive names using their observed
# frequency and recency profiles. Cluster IDs themselves are arbitrary.
other_ids = [
    int(sid) for sid in segment_summary.index
    if int(sid) != premium_segment_id
]

frequency_ranked = sorted(
    other_ids,
    key=lambda sid: segment_summary.loc[sid, "mean_frequency"],
    reverse=True,
)

recency_ranked = sorted(
    other_ids,
    key=lambda sid: segment_summary.loc[sid, "mean_recency"],
)

segment_names = {premium_segment_id: "Premium"}

if frequency_ranked:
    segment_names[frequency_ranked[0]] = "Frequent"

remaining = [
    sid for sid in other_ids if sid not in segment_names
]

if remaining:
    most_recent_id = min(
        remaining,
        key=lambda sid: segment_summary.loc[sid, "mean_recency"],
    )
    segment_names[most_recent_id] = "Recently Active"

remaining = [
    sid for sid in other_ids if sid not in segment_names
]

for sid in remaining:
    segment_names[sid] = "Developing"

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_segment_id)

segment_summary["segment_name"] = segment_summary.index.map(
    segment_names
)

print("\nFINAL SEGMENT PROFILES")
print(segment_summary.to_string(float_format="%.2f"))
print("\nPremium segment ID:", premium_segment_id)
print("Premium mean monetary value:",
      round(segment_summary.loc[premium_segment_id, "mean_monetary"], 2))

print("\nSuggested segment actions:")
for sid, row in segment_summary.iterrows():
    name = row["segment_name"]
    if name == "Premium":
        action = "Protect loyalty with personalized service and relevant rewards."
    elif name == "Frequent":
        action = "Encourage repeat purchases with relevant cross-sell offers."
    elif name == "Recently Active":
        action = "Maintain engagement with timely, relevant recommendations."
    else:
        action = "Test onboarding, value messaging, and targeted re-engagement."
    print(f"{name} (cluster {sid}): {action}")

rfm.to_csv(OUTPUTS / "customer_segments.csv", index=False)

# PCA on scaled RFM features.
pca = PCA(n_components=2)
pca.fit(rfm_scaled)
variance = pca.explained_variance_ratio_

print("\nPCA explained variance:")
print("PC1:", round(variance[0], 4))
print("PC2:", round(variance[1], 4))
print("Cumulative:", round(variance.sum(), 4))
print("Saved:", OUTPUTS / "customer_segments.csv")


# ---------------------------------------------------------
# 4. Create lapse labels and pre-cutoff features
# ---------------------------------------------------------
# Cohort: customers with at least one completed pre-cutoff order.
cohort_ids = set(completed_pre["customer_id"])
cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

last_completed = completed_pre.groupby(
    "customer_id"
)["order_ts"].max()

frequency = completed_pre.groupby(
    "customer_id"
)["order_id"].nunique()

monetary = completed_pre.groupby(
    "customer_id"
)["order_amount"].sum()

avg_discount = completed_pre.groupby(
    "customer_id"
)["discount_pct"].mean()

avg_delivery = completed_pre.groupby(
    "customer_id"
)["delivery_minutes"].mean()

favourite_category = (
    completed_pre.dropna(subset=["category"])
    .groupby("customer_id")["category"]
    .agg(lambda s: s.value_counts().index[0])
)

# Cancellation rate uses only pre-cutoff orders with final statuses.
final_statuses = {"completed", "cancelled"}
final_orders = pre_cutoff[
    pre_cutoff["status"].astype(str).str.lower().isin(final_statuses)
].copy()

cancelled_rate = (
    final_orders.assign(
        is_cancelled=final_orders["status"]
        .astype(str).str.lower().eq("cancelled")
    )
    .groupby("customer_id")["is_cancelled"]
    .mean()
)

features = cohort.set_index("customer_id").copy()

features["recency"] = (
    LAPSE_CUTOFF - last_completed.reindex(features.index)
).dt.total_seconds() / 86400

features["frequency"] = frequency.reindex(features.index)
features["monetary"] = monetary.reindex(features.index)
features["avg_discount_pct"] = avg_discount.reindex(features.index)
features["cancellation_rate"] = cancelled_rate.reindex(features.index)
features["avg_delivery_minutes"] = avg_delivery.reindex(features.index)
features["favourite_category"] = favourite_category.reindex(
    features.index
)
features["tenure_days"] = (
    LAPSE_CUTOFF - features["signup_date"]
).dt.days

# Lapsed means no completed order in September 2025.
features["lapsed"] = [
    0 if customer_id in active_in_september else 1
    for customer_id in features.index
]

# IDs and dates are not model inputs.
features = features.drop(columns=["signup_date"])

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age",
]
categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

X = features[numeric_features + categorical_features].copy()
y = features["lapsed"].astype(int)

assert len(X) == len(y)
assert X.columns.is_unique
assert not X.isna().any().any(), "Model features contain missing values before preprocessing."
assert "segment_name" not in X.columns
assert "segment_id" not in X.columns
assert "is_premium" not in X.columns

print("\nLAPSE COHORT")
print("Cohort rows:", len(features))
print("Lapsed customers:", int(y.sum()))
print("Lapse rate (%):", round(100 * y.mean(), 2))

X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        y,
        features.index,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# Show the training-fitted frequency scaler mean against the full cohort mean.
# The model pipeline is fitted only on training folds; the full-cohort mean is
# reported for context and is never used to fit the predictive scaler.
full_cohort_frequency_mean = float(X["frequency"].mean())
print("Full-cohort frequency mean (descriptive only):", round(full_cohort_frequency_mean, 4))

baseline_pred = np.zeros(len(y_test), dtype=int)
print("\nBASELINE: always predict not lapsed")
print("Accuracy:", round(accuracy_score(y_test, baseline_pred), 4))
print(
    "Lapsed-class precision:",
    round(precision_score(y_test, baseline_pred, zero_division=0), 4),
)
print(
    "Lapsed-class recall:",
    round(recall_score(y_test, baseline_pred, zero_division=0), 4),
)
print(
    "Lapsed-class F1:",
    round(f1_score(y_test, baseline_pred, zero_division=0), 4),
)


# ---------------------------------------------------------
# 5. Preprocessing pipelines
# ---------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
], remainder="drop")

# The preprocessor is inside each model pipeline. GridSearchCV clones
# and fits it separately within every training fold.
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])


# ---------------------------------------------------------
# 6. Grid search using training data only
# ---------------------------------------------------------
print("\nTUNING LOGISTIC REGRESSION")
log_grid = GridSearchCV(
    logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)
log_grid.fit(X_train, y_train)

print("Best parameters:", log_grid.best_params_)
print("Best CV F1:", round(log_grid.best_score_, 4))

print("\nTUNING RANDOM FOREST")
forest_grid = GridSearchCV(
    forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)
forest_grid.fit(X_train, y_train)

print("Best parameters:", forest_grid.best_params_)
print("Best CV F1:", round(forest_grid.best_score_, 4))

forest_numeric_pipeline = (
    forest_grid.best_estimator_
    .named_steps["preprocessor"]
    .named_transformers_["numeric"]
)
learned_frequency_mean = float(
    forest_numeric_pipeline.named_steps["scaler"].mean_[numeric_features.index("frequency")]
)
print("Training-fitted frequency scaler mean:", round(learned_frequency_mean, 4))

# Choose by training-only cross-validation F1, not test-set metrics.
if forest_grid.best_score_ >= log_grid.best_score_:
    selected_name = "random_forest"
    selected_model = forest_grid.best_estimator_
else:
    selected_name = "logistic_regression"
    selected_model = log_grid.best_estimator_

print("\nSelected model based on training CV F1:", selected_name)


# ---------------------------------------------------------
# 7. Evaluate both models on the untouched test set
# ---------------------------------------------------------
def report_model(name, model, X_eval, y_eval):
    pred = model.predict(X_eval)
    print(f"\n{name}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_eval, pred, labels=[0, 1]))
    print("Accuracy:", round(accuracy_score(y_eval, pred), 4))
    print(
        "Lapsed precision:",
        round(precision_score(y_eval, pred, zero_division=0), 4),
    )
    print(
        "Lapsed recall:",
        round(recall_score(y_eval, pred, zero_division=0), 4),
    )
    print(
        "Lapsed F1:",
        round(f1_score(y_eval, pred, zero_division=0), 4),
    )
    return pred


log_pred = report_model(
    "LOGISTIC REGRESSION", log_grid.best_estimator_, X_test, y_test
)
forest_pred = report_model(
    "RANDOM FOREST", forest_grid.best_estimator_, X_test, y_test
)

selected_pred = selected_model.predict(X_test)
print("\nFINAL SELECTED MODEL:", selected_name)
print("Selection used training CV only.")
print(
    "Selected model test F1:",
    round(f1_score(y_test, selected_pred, zero_division=0), 4),
)


# ---------------------------------------------------------
# 8. Feature importance and logistic coefficient signs
# ---------------------------------------------------------
forest = forest_grid.best_estimator_.named_steps["model"]
forest_pre = forest_grid.best_estimator_.named_steps["preprocessor"]
forest_names = forest_pre.get_feature_names_out()

importances = pd.Series(
    forest.feature_importances_,
    index=forest_names,
).sort_values(ascending=False)

print("\nTOP 5 RANDOM-FOREST FEATURE IMPORTANCES")
print(importances.head(5).to_string())

logistic = log_grid.best_estimator_.named_steps["model"]
log_pre = log_grid.best_estimator_.named_steps["preprocessor"]
log_names = log_pre.get_feature_names_out()

coefficients = pd.Series(
    logistic.coef_[0],
    index=log_names,
).sort_values()

print("\nTHREE MOST NEGATIVE LOGISTIC COEFFICIENTS")
print(coefficients.head(3).to_string())
print("\nTHREE MOST POSITIVE LOGISTIC COEFFICIENTS")
print(coefficients.tail(3).sort_values(ascending=False).to_string())


# ---------------------------------------------------------
# 9. Oversample the training minority only
# ---------------------------------------------------------
# Use the logistic pipeline's training-fitted preprocessing.
# Test rows are transformed but never used to fit the preprocessor
# or the oversampled classifier.
log_pre = log_grid.best_estimator_.named_steps["preprocessor"]

X_train_transformed = log_pre.transform(X_train)
X_test_transformed = log_pre.transform(X_test)

train_array = (
    X_train_transformed.toarray()
    if hasattr(X_train_transformed, "toarray")
    else np.asarray(X_train_transformed)
)
test_array = (
    X_test_transformed.toarray()
    if hasattr(X_test_transformed, "toarray")
    else np.asarray(X_test_transformed)
)

y_train_array = y_train.to_numpy()
rng = np.random.RandomState(42)

negative_idx = np.flatnonzero(y_train_array == 0)
positive_idx = np.flatnonzero(y_train_array == 1)

if len(positive_idx) < len(negative_idx):
    extra = rng.choice(
        positive_idx,
        size=len(negative_idx) - len(positive_idx),
        replace=True,
    )
    balanced_idx = np.concatenate([
        np.arange(len(y_train_array)),
        extra,
    ])
elif len(negative_idx) < len(positive_idx):
    extra = rng.choice(
        negative_idx,
        size=len(positive_idx) - len(negative_idx),
        replace=True,
    )
    balanced_idx = np.concatenate([
        np.arange(len(y_train_array)),
        extra,
    ])
else:
    balanced_idx = np.arange(len(y_train_array))

balanced_X = train_array[balanced_idx]
balanced_y = y_train_array[balanced_idx]

balanced_logistic = LogisticRegression(
    max_iter=1000,
    random_state=42,
)
balanced_logistic.fit(balanced_X, balanced_y)
balanced_pred = balanced_logistic.predict(test_array)

print("\nCLASS-BALANCE EXPERIMENT")
print("Original training class counts:")
print(pd.Series(y_train_array).value_counts().sort_index().to_dict())
print("Balanced training class counts:")
print(pd.Series(balanced_y).value_counts().sort_index().to_dict())

print("Original logistic precision / recall / F1:",
      round(precision_score(y_test, log_pred, zero_division=0), 4),
      round(recall_score(y_test, log_pred, zero_division=0), 4),
      round(f1_score(y_test, log_pred, zero_division=0), 4))

print("Balanced logistic precision / recall / F1:",
      round(precision_score(y_test, balanced_pred, zero_division=0), 4),
      round(recall_score(y_test, balanced_pred, zero_division=0), 4),
      round(f1_score(y_test, balanced_pred, zero_division=0), 4))


# ---------------------------------------------------------
# 10. Save and reload the selected model
# ---------------------------------------------------------
model_path = OUTPUTS / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)
original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

predictions_identical = np.array_equal(
    original_predictions, reloaded_predictions
)

print("\nSaved model:", model_path)
print("Reloaded model predictions identical:", predictions_identical)
assert predictions_identical, "Reloaded model predictions differ!"


# ---------------------------------------------------------
# 11. Build the Premium retention list
# ---------------------------------------------------------
# Premium is defined using whole-window RFM. It includes September
# information and is therefore a post-hoc reporting label, not a
# leakage-free feature or independent evaluation cohort.
premium_ids = set(
    rfm.loc[rfm["is_premium"], "customer_id"].astype(str)
)

test_ids = pd.Series(id_test.astype(str), index=X_test.index)
test_probabilities = selected_model.predict_proba(X_test)[:, 1]

retention = pd.DataFrame({
    "customer_id": test_ids.to_numpy(),
    "lapse_probability": test_probabilities,
    "actually_lapsed": y_test.to_numpy(),
})

retention = (
    retention[retention["customer_id"].isin(premium_ids)]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
)

retention.to_csv(OUTPUTS / "retention_list.csv", index=False)

print("\nRetention list saved:", OUTPUTS / "retention_list.csv")
print("Premium test customers listed:", len(retention))
print(
    "Truly lapsed customers in retention list:",
    int(retention["actually_lapsed"].sum()),
)
print("\nPart B model script finished.")
