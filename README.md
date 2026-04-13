# DLP Detection Model Banking Cloud App Events


DLP Detection Model using Banking Cloud App Events
A machine learning pipeline to detect potential data exfiltration events in a banking environment using cloud application upload logs. The model identifies suspicious file upload behaviour by combining rule based risk signals with user behavioural patterns, going beyond a simple domain blacklist to learn complex feature interactions.

## Problem Statement
Data Loss Prevention (DLP) in banking is a critical compliance challenge. Employees handling sensitive client data like account balances, PII, loan applications, have legitimate access to files that, if uploaded to the wrong destination, represent a serious security and regulatory risk.
Traditional DLP systems rely on static blacklists (block these domains, flag these file types). These generate high false positive rates and miss sophisticated exfiltration attempts where an employee uses a safe-looking destination or splits data across multiple small uploads.
This project takes a machine learning approach by training a model on a combination of file-level, domain-level, time-based and user behavioural features to assign a risk probability to each upload event, enabling analysts to prioritise investigations and triage more effectively.

## Dataset

Mock dataset based on realistic banking DLP patterns, combining two versions:

| Column | Type | Description |
|---|---|---|
| Action_ID | bigint | Unique event identifier |
| Timestamp | timestamp | Time of the upload event |
| ActionType | string | Type of action (e.g. FileUploaded) |
| ObjectName | string | Name of the uploaded file |
| Target_Domain | string | Destination domain |
| AccountDisplayName | string | User who performed the upload |
| Position | string | User's job role |
| Risk_Label | bigint | 0 = safe, 1 = risky |

- **Total events:** 6,000 (1,000 original + 5,000 more realistic mock)
- **Class distribution:** ~73% safe (0), ~27% risky (1)
- **Noise built in:** some risky domain uploads are labelled safe (legitimate use) and some safe domain uploads are labelled risky (sensitive file + suspicious behaviour), reflecting real-world situations.

## Project Structure

```
├── Feature_Engineering_v2.ipynb      # Feature engineering pipeline
├── Model_Training_GBT.ipynb          # GBT model training, evaluation and explainability
├── Model_Training_CatBoost.ipynb     # CatBoost model with early stopping
├── Model_Comparison.ipynb            # GBT vs CatBoost side by side comparison
├── risky_domains.csv                 # Blacklist of non-business sanctioned domains
├── sensitive_keywords.csv            # Sensitive keywords matched against file names
└── data/
    ├── CloudAppEvents_BankingDLP.csv     # Original dataset (1000 events)
    └── CloudAppEvents_BankingDLP_v2.csv  # Realistic mock dataset (5000 events)
```


## Features Engineered

### File features
| Feature | Description |
|---|---|
| `file_extension` | Extracted file extension (categorical) |
| `is_high_risk_extension` | 1 if extension is in high-risk list (pdf, csv, sql, xlsx etc.) |
| `file_name_has_sensitive_keyword` | 1 if file name contains a keyword from `sensitive_keywords.csv` |
| `double_extension_check` | 1 if file appears masked (e.g. `data.csv.zip`) |

### Domain features
| Feature | Description |
|---|---|
| `is_risky_target_domain` | 1 if destination domain is in `risky_domains.csv` blacklist |
| `is_first_time_domain` | 1 if this is the first time this user has uploaded to this domain |

### Time features
| Feature | Description |
|---|---|
| `is_after_hours` | 1 if upload occurred between 7 PM and 5 AM |
| `day_of_week` | Day of week |
| `is_monday_or_friday` | 1 if Monday or Friday signals higher risk days for departing employees |

### User behaviour features (rolling window)
| Feature | Description |
|---|---|
| `user_upload_count_24h` | Number of uploads by this user in the last 24 hours |
| `user_unique_domains_7d` | Number of distinct target domains this user uploaded to in the last 7 days |

### Position features
| Feature | Description |
|---|---|
| `Position` | Job role (categorical) |
| `is_high_risk_position` | 1 if role is associated with higher data access risk |

---

## Model

Two models trained and compared side by side in `Model_Comparison.ipynb`.

### GBT (Gradient Boosting Trees)
Baseline model using sklearn's `GradientBoostingClassifier` with `OrdinalEncoder` for categorical features.

```python
GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
```

### CatBoost
CatBoost with native categorical handling and early stopping to prevent overfitting.

```python
CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, eval_metric="AUC", early_stopping_rounds=50)
```

---

## Results

Evaluated on a held-out 20% test set (1,200 events, stratified split):

| Metric | GBT | CatBoost | Winner |
|---|---|---|---|
| AUC-ROC | 0.9326 | 0.9340 | CatBoost |
| AUC-PR | 0.8624 | 0.8643 | CatBoost |
| Precision (class 1) |  0.7966 | 0.7826  | GBT |
| Recall (class 1) | 0.7264 | 0.7358 | CatBoost |
| False Positives | 59 | 65 | GBT |
| False Negatives | 87 | 84 | CatBoost |

Takeaway: Both models perform similarly. CatBoost has a marginal edge on AUC-ROC and AUC-PR and catches 3 more real incidents. GBT generates fewer false alarms. For DLP where missing an incident is more costly than a false alarm, CatBoost is the preferred model.

---

## Explainability

Individual predictions explained using SHAP, showing exactly which features pushed the risk probability up or down for each event.

Example SHAP output (probability: 0.46):

| Feature | Value | SHAP Value |
|---|---|---|
| file_extension | csv | +1.25 |
| file_name_has_sensitive_keyword | 1 | +0.98 |
| is_monday_or_friday | 1 | +0.16 |
| Position | System Architect | +0.14 |
| is_after_hours | 0 | -0.22 |
| is_risky_target_domain | 0 | -0.77 |

---

## Tech Stack

- **Platform:** Databricks Serverless
- **Data processing:** PySpark, Delta tables, Unity Catalog
- **Modelling:** scikit-learn, CatBoost
- **Explainability:** SHAP
