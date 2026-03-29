# Databricks notebook source
#import pyspark.sql.functions as F
from pyspark.sql.functions import col, lower, when, substring_index, regexp_replace, explode, split, hour
import os

# COMMAND ----------

table_path = "workspace.default.cloud_app_events_banking_dlp"
current_directory = os.getcwd() 
df = spark.table(table_path)

# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = os.path.dirname(notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #Get File Name and Extension 
# MAGIC - This Feature will be extracting file extension from file name. For DLP PDFs, CSVs, and Text files are higher risk than system files (like .exe) 
# MAGIC because they are the primary containers for sensitive client PII.

# COMMAND ----------

df_features = df.withColumn("file_name", regexp_replace(col("ObjectName"),r"\.[^.]+$", "")
                            ).withColumn("file_extension", substring_index(col("ObjectName"), ".", -1))
display(df_features.limit(100))


# COMMAND ----------

# MAGIC %md
# MAGIC #Double Extension Detection:
# MAGIC - Identifies "masked" files ('data.csv.zip'). This flags attempts 
# MAGIC to bypass file-type filters by hiding sensitive extensions inside 
# MAGIC another filename.
# MAGIC

# COMMAND ----------

extension_pattern = r"\.(pdf|docx|xlsx|doc|xls|ppt|pptx|csv|zip|7z|rar|py|sh|bat|ps1|sql|txt)"

df_features = df_features.withColumn(
    "double_extension_check", 
    when(col("file_name").rlike(extension_pattern), 1).otherwise(0)
)

  
                                  
display(df_features.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Risky Domains Flag:
# MAGIC
# MAGIC * Checks the **Target Domain** and flags it if the domain exists in `risky_domains.csv`. Since these domains are not used in a standard business process, there is a higher risk of a data leak associated with them.

# COMMAND ----------


csv_path = os.path.join(current_directory, "risky_domains.csv")
risky_domains_df = spark.read.csv(csv_path, header=True, inferSchema=True)


# COMMAND ----------

risky_domains_df = risky_domains_df.withColumn(
    "Domains", 
    explode(split(col("Domain"), r",\s*"))
)

risky_domains_list = [row[0] for row in risky_domains_df.select("Domains").collect()]

regex_pattern = "|".join([d.replace(".", r"\.") for d in risky_domains_list])

df_features = df_features.withColumn(
    "is_risky_destination",
    when(lower(col("Target_Domain")).rlike(regex_pattern), 1).otherwise(0)
)

display(df_features)

# COMMAND ----------

# MAGIC %md
# MAGIC # Flagging data exfiltration after work hours
# MAGIC After-work uploads could be considered more risky since there is a lower probability that they are work-related.
# MAGIC

# COMMAND ----------

df_features = df_features.withColumn(
    "after_work_hour", hour(col("Timestamp"))
).withColumn(
    "is_after_hours", 
    when((col("after_work_hour") >= 19) | (col("after_work_hour") <= 5), 1).otherwise(0)
)

display(df_features)

# COMMAND ----------

# MAGIC %md
# MAGIC #Clean Code Cell
# MAGIC with comments
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, lower, when, substring_index, regexp_replace, explode, split, hour

# Define extension pattern to match common file extensions
extension_pattern = r"\.(pdf|docx|xlsx|doc|xls|ppt|pptx|csv|zip|7z|rar|py|sh|bat|ps1|sql|txt)"

# Flagging data exfiltration risks by matching target domains against a dynamic blacklist of non-business sanctioned apps.

risky_domains_df = risky_domains_df.withColumn(
    "Domains", 
    explode(split(col("Domain"), r",\s*"))
)

risky_domains_list = [row[0] for row in risky_domains_df.select("Domains").collect()]
regex_pattern = "|".join([d.replace(".", r"\.") for d in risky_domains_list])




df_features_clean = df_features.withColumn(
    # Extracts everything after the last dot
    "file_extension", lower(substring_index(col("ObjectName"), ".", -1))
).withColumn(
    # Extracts everything BEFORE the last dot by searching for the last dot and replacing whats after with an empty string
    "file_name", regexp_replace(col("ObjectName"), r"\.[^.]+$", "")
).withColumn(
    # Check 1: Is there an extension left after removing one?
    "double_extension_check", 
    when(col("file_name").rlike(extension_pattern), 1).otherwise(0)
).withColumn(
    # Check 2: Is the risky target domain?
    "is_risky_target_domain",
    when(lower(col("Target_Domain")).rlike(regex_pattern), 1).otherwise(0)
).withColumn(
    # Check 3: if between 7 PM and 5 AM 
    "is_after_hours", 
    when((col("after_work_hour") >= 19) | (col("after_work_hour") <= 5), 1).otherwise(0)
)
display(df_features_clean.limit(100))
   

# COMMAND ----------

# MAGIC %md
# MAGIC #Tests Below

# COMMAND ----------

display(df_features.filter(col("double_extension_check") == 1))

# COMMAND ----------

display(df_features.filter(df.ObjectName.contains(".pdf.")))

# COMMAND ----------

df_features.select("Target_Domain").distinct().show()
