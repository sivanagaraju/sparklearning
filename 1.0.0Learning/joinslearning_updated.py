from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    rand,
    md5,
    substring
)

# 1. Initialize Spark session
spark = (
    SparkSession.builder
        .appName("LargeDataFrameJoinExample")
        .getOrCreate()
)

# Disable auto-broadcast so Spark won’t try to do a small broadcast join
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# 2. Configuration for row counts and distinct key range
num_rows = 10_000_000       # 10 million rows per DataFrame
distinct_key_count = 100_000

# 3. Create FIRST DataFrame with multiple columns to increase data size
df1 = (
    spark.range(num_rows)
         .withColumnRenamed("id", "seq_id")
         # A join key so we can merge with df2
         .withColumn("join_key", col("seq_id") % distinct_key_count)

         # 5 random string columns (32 characters each)
         .withColumn("str_col1", substring(md5(rand(seed=0)), 1, 32))
         .withColumn("str_col2", substring(md5(rand(seed=1)), 1, 32))
         .withColumn("str_col3", substring(md5(rand(seed=2)), 1, 32))
         .withColumn("str_col4", substring(md5(rand(seed=3)), 1, 32))
         .withColumn("str_col5", substring(md5(rand(seed=4)), 1, 32))

         # 5 numeric columns
         .withColumn("num_col1", rand(seed=5) * 100)
         .withColumn("num_col2", rand(seed=6) * 100)
         .withColumn("num_col3", rand(seed=7) * 100)
         .withColumn("num_col4", rand(seed=8) * 100)
         .withColumn("num_col5", rand(seed=9) * 100)
)

# 4. Create SECOND DataFrame (same row count / columns, or adapt as needed)
df2 = (
    spark.range(num_rows)
         .withColumnRenamed("id", "seq_id")
         # Same join key distribution
         .withColumn("join_key", col("seq_id") % distinct_key_count)

         # 5 random string columns
         .withColumn("str_col1", substring(md5(rand(seed=10)), 1, 32))
         .withColumn("str_col2", substring(md5(rand(seed=11)), 1, 32))
         .withColumn("str_col3", substring(md5(rand(seed=12)), 1, 32))
         .withColumn("str_col4", substring(md5(rand(seed=13)), 1, 32))
         .withColumn("str_col5", substring(md5(rand(seed=14)), 1, 32))

         # 5 numeric columns
         .withColumn("num_col1", rand(seed=15) * 100)
         .withColumn("num_col2", rand(seed=16) * 100)
         .withColumn("num_col3", rand(seed=17) * 100)
         .withColumn("num_col4", rand(seed=18) * 100)
         .withColumn("num_col5", rand(seed=19) * 100)
)

# 5. Join WITHOUT sorting first (to see typical shuffle behavior)
print("=== JOIN WITHOUT SORTING ===")
df_join_no_sort = df1.join(df2, on="join_key", how="inner")

# Look at the physical plan
df_join_no_sort.explain(True)

# Trigger an action to measure performance / cause a shuffle
count_no_sort = df_join_no_sort.count()
print(f"Join count (no sort): {count_no_sort}")

# 6. Now sort both DataFrames on join_key, then join
print("=== JOIN WITH SORTING ===")
df1_sorted = df1.sort("join_key")
df2_sorted = df2.sort("join_key")

df_join_sorted = df1_sorted.join(df2_sorted, on="join_key", how="inner")
df_join_sorted.explain(True)

count_sorted = df_join_sorted.count()
print(f"Join count (with sort): {count_sorted}")

# 7. Stop Spark session (optional)
spark.stop()
