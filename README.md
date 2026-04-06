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
├── Feature_Engineering_v2.ipynb   # Feature engineering pipeline
├── Model_Training_v1.ipynb        # Model training, evaluation and explainability
├── risky_domains.csv              # Blacklist of non-business sanctioned domains
├── sensitive_keywords.csv         # Sensitive keywords matched against file names
└── data/
    └── CloudAppEvents_BankingDLP.csv   # Original dataset
    └── CloudAppEvents_BankingDLP_v2.csv # Mock dataset
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
WIP...
