# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import Window
import os
import csv

# COMMAND ----------

spark.sql("SHOW TABLES IN workspace.default").show(truncate=False)

# COMMAND ----------

#Created 2nd mock data table with more realistic data, since accuracy for model was too perfect. In future will come up with more elegent solution.
df_v1 = spark.table("workspace.default.cloud_app_events_banking_dlp")
df_v2 = spark.table("workspace.default.cloud_app_events_banking_dlp_v_2")

df = df_v1.union(df_v2)

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = os.path.dirname(notebook_path)

print(f"v1 rows: {df_v1.count()}")
print(f"v2 rows: {df_v2.count()}")
print(f"Combined rows: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Schema
# MAGIC | Column             | Type      |
# MAGIC |--------------------|-----------|
# MAGIC | Action_ID          | bigint    |
# MAGIC | Timestamp          | timestamp |
# MAGIC | ActionType         | string    |
# MAGIC | ObjectName         | string    |
# MAGIC | Target_Domain      | string    |
# MAGIC | AccountDisplayName | string    |
# MAGIC | Position           | string    |
# MAGIC | Risk_Label         | bigint    |

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Reference Data
# MAGIC Load risky domains and sensitive keywords from CSVs.

# COMMAND ----------

# Risky domains 
with open(f"/Workspace{repo_root}/risky_domains.csv") as f:
    risky_domains_list = [row["Domain"].strip() for row in csv.DictReader(f)]
risky_domains_pattern = "|".join([d.replace(".", r"\.") for d in risky_domains_list])

#Sensitive keywords
with open(f"/Workspace{repo_root}/sensitive_keywords.csv") as f:
    keyword_list = [row["keyword"].strip().lower() for row in csv.DictReader(f)]
keyword_pattern = "|".join(keyword_list)

# COMMAND ----------

# MAGIC %md
# MAGIC # File Name & Extension
# MAGIC Extracts the file extension and base name from `ObjectName`.
# MAGIC PDFs, CSVs, and text files are higher risk than system files
# MAGIC because they are the primary containers for sensitive client PII.

# COMMAND ----------

df_features = (
    df
    # Extracts everything after the last dot
    .withColumn("file_extension", F.lower(F.substring_index(F.col("ObjectName"), ".", -1)))
    # Extracts everything before the last dot
    .withColumn("file_name",      F.regexp_replace(F.col("ObjectName"), r"\.[^.]+$", ""))
)

display(df_features.select("ObjectName", "file_name", "file_extension").limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC # Double Extension Detection
# MAGIC Identifies "masked" files (e.g. `data.csv.zip`). Flags attempts
# MAGIC to bypass file-type filters by hiding sensitive extensions
# MAGIC inside another filename.

# COMMAND ----------

extension_pattern = r"\.(pdf|docx|xlsx|doc|xls|ppt|pptx|csv|zip|7z|rar|py|sh|bat|ps1|sql|txt)"

df_features = df_features.withColumn(
    "double_extension_check",
    F.when(F.col("file_name").rlike(extension_pattern), 1).otherwise(0)
)
display(df_features.filter(F.col("double_extension_check") == 1))


# COMMAND ----------

# MAGIC %md
# MAGIC # File Risk Scoring
# MAGIC Flags files based on extension risk level,
# MAGIC and presence of sensitive keywords in the file name.

# COMMAND ----------

high_risk_extensions = ["pdf", "csv", "xlsx", "xls", "docx", "doc", "sql", "txt", "json"]
high_risk_pattern = "|".join([f"^{ext}$" for ext in high_risk_extensions])

df_features = (
    df_features
    # Is the extension in the high-risk list?
    .withColumn(
        "is_high_risk_extension",
        F.when(F.col("file_extension").rlike(high_risk_pattern), 1).otherwise(0)
    )
    # File name contains a sensitive keyword loaded from CSV
    .withColumn(
        "file_name_has_sensitive_keyword",
        F.when(F.lower(F.col("file_name")).rlike(keyword_pattern), 1).otherwise(0)
    )
)

display(df_features.select(
    "ObjectName", "file_extension", "is_high_risk_extension",
     "file_name_has_sensitive_keyword"
).limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC # Risky Destination Domain
# MAGIC Checks `Target_Domain` against the risky domains blacklist.
# MAGIC Domains not used in standard business processes carry a higher
# MAGIC risk of data leakage.

# COMMAND ----------

df_features = df_features.withColumn(
    "is_risky_target_domain",
    F.when(F.lower(F.col("Target_Domain")).rlike(risky_domains_pattern), 1).otherwise(0)
)

display(df_features.filter(F.col("is_risky_target_domain") == 1))

# COMMAND ----------

# MAGIC %md
# MAGIC # Time Features
# MAGIC After-hours uploads are higher risk and have lower probability of being work-related.
# MAGIC Monday and Friday are higher-risk days, becasue departing employees often
# MAGIC exfiltrate on their first day of notice or just before leaving.

# COMMAND ----------

df_features = (
    df_features
    .withColumn("hour", F.hour(F.col("Timestamp")))
    .withColumn(
        "is_after_hours",
        F.when((F.col("hour") >= 19) | (F.col("hour") <= 5), 1).otherwise(0)
    )
    .drop("hour")
    .withColumn("day_of_week", F.dayofweek(F.col("Timestamp")))  # 1=Sun, 2=Mon, ..., 7=Sat
    .withColumn(
        "is_monday_or_friday",
        F.when(F.col("day_of_week").isin([2, 6]), 1).otherwise(0)
    )
)

display(df_features.select(
    "Timestamp", "is_after_hours", "day_of_week", "is_monday_or_friday"
).limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC # Position Risk Flag
# MAGIC Certain roles have higher access to sensitive data and are more commonly
# MAGIC associated with insider threats — finance, HR, IT admin, and contractors
# MAGIC in particular. Flag these as high-risk positions.
# MAGIC
# MAGIC Update `high_risk_positions` to reflect role naming conventions in your org.

# COMMAND ----------

high_risk_positions = [
    "Relationship Banker","System Architect","Loan Officer","Wealth Manager"
]
position_pattern = "|".join([p.lower() for p in high_risk_positions])

df_features = df_features.withColumn(
    "is_high_risk_position",
    F.when(F.lower(F.col("Position")).rlike(position_pattern), 1).otherwise(0)
)

display(df_features.select("Position", "is_high_risk_position").distinct().limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC # User Behaviour Features (Window Functions)
# MAGIC Captures whether a user's behaviour is unusual *relative to their own history*.
# MAGIC
# MAGIC - `user_upload_count_24h` — rolling upload count per user in the last 24 hours
# MAGIC - `user_unique_domains_7d` — distinct target domains per user in the last 7 days
# MAGIC - `is_first_time_domain` — first time this user has uploaded to this domain

# COMMAND ----------

# Convert Timestamp to unix seconds for range-based windows
df_features = df_features.withColumn("_ts_unix", F.unix_timestamp(F.col("Timestamp")))

seconds_24h = 86400
seconds_7d  = 7 * 86400

# Upload count per user in last 24 hours 
window_24h = (
    Window
    .partitionBy("AccountDisplayName")
    .orderBy("_ts_unix")
    .rangeBetween(-seconds_24h, 0)
)
df_features = df_features.withColumn(
    "user_upload_count_24h",
    F.count("*").over(window_24h)
)

# Distinct target domains per user in last 7 days 
window_7d = (
    Window
    .partitionBy("AccountDisplayName")
    .orderBy("_ts_unix")
    .rangeBetween(-seconds_7d, 0)
)
df_features = df_features.withColumn(
    "user_unique_domains_7d",
    F.approx_count_distinct("Target_Domain").over(window_7d)
)

# First time this user uploads to this domain
window_first_domain = (
    Window
    .partitionBy("AccountDisplayName", "Target_Domain")
    .orderBy("_ts_unix")
)
df_features = (
    df_features
    .withColumn("_domain_rank", F.rank().over(window_first_domain))
    .withColumn(
        "is_first_time_domain",
        F.when(F.col("_domain_rank") == 1, 1).otherwise(0)
    )
    .drop("_domain_rank", "_ts_unix")   # drop intermediary columns
)

display(df_features.select(
    "AccountDisplayName", "Timestamp", "Target_Domain",
    "user_upload_count_24h", "user_unique_domains_7d", "is_first_time_domain"
).limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Clean Feature Set
# MAGIC All features combined, nulls dropped, ready for model training.

# COMMAND ----------

df_features_clean = (
    df_features
    .select(
        # Label
        "Risk_Label",

        # Identifiers, not used as model features
        "Action_ID",
        "AccountDisplayName",
        "ObjectName",
        "Target_Domain",
        "Timestamp",

        # Categorical features
        "ActionType",
        "file_extension",
        "Position",

        # Numerical / binary features
        "is_high_risk_extension",
        "file_name_has_sensitive_keyword",
        "double_extension_check",
        "is_risky_target_domain",
        "is_first_time_domain",
        "is_after_hours",
        "day_of_week",
        "is_monday_or_friday",
        "is_high_risk_position",
        "user_upload_count_24h",
        "user_unique_domains_7d",
    )
    .dropna()
)

print(f"Final feature rows: {df_features_clean.count()}")
df_features_clean.groupBy("Risk_Label").count().show()
display(df_features_clean.limit(100))

# COMMAND ----------

df_features_clean.write.mode("overwrite").saveAsTable("workspace.default.dlp_features_clean")

# COMMAND ----------

# MAGIC %md
# MAGIC # Tests

# COMMAND ----------

display(df_features.filter(F.col("double_extension_check") == 1))

# COMMAND ----------

display(df_features.filter(df.ObjectName.contains(".pdf.")))

# COMMAND ----------

df_features.select("Target_Domain").distinct().show()

# COMMAND ----------

display(df_features.filter(F.col("file_name_has_sensitive_keyword") == 1))

# COMMAND ----------

display(df_features.filter(F.col("user_upload_count_24h") > 10))

# COMMAND ----------

display(df_features.filter(F.col("is_high_risk_position") == 1).select("Position").distinct())

# COMMAND ----------

display(df_features.filter(F.col("Risk_Label") == 0).count())

# COMMAND ----------

display(df_features.select("ActionType"))
