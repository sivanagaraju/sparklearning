from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, md5, substring, rand, when, expr, floor, date_add, to_date, year, month
)
import datetime

# --------------------------------------------------------------------------------
# 1. Initialize Spark Session
# --------------------------------------------------------------------------------
spark = (
    SparkSession.builder
        .appName("SCD2_with_MultiLevelPartitioning_Example")
        # .config("spark.sql.shuffle.partitions", "200")  # Adjust as needed
        .getOrCreate()
)

# Disable auto-broadcast so we see full shuffle or sort-merge joins for large data
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# --------------------------------------------------------------------------------
# 2. Configuration
# --------------------------------------------------------------------------------
num_rows_large = 10_000_000   # for large dimension/fact
distinct_product_ids = 100_000
distinct_store_ids = 10_000
distinct_customer_ids = 1_000_000  # Large number of customers

# SMALL dimension sizes
num_rows_stores = 10_000  # smaller table
num_rows_dates  = 1_500   # e.g., about 4+ years of daily dates

# --------------------------------------------------------------------------------
# 3. Create a Date Dimension
#    We'll produce ~1,500 days from 2019-01-01 onward
# --------------------------------------------------------------------------------
start_date = datetime.date(2019, 1, 1)
date_list = [(start_date + datetime.timedelta(days=x)) for x in range(num_rows_dates)]

# Create a small DF with one row per date
df_dates = spark.createDataFrame([
    (d, d.year, d.month, d.day, d.isoweekday()) 
    for d in date_list
], ["full_date", "year", "month", "day_of_month", "day_of_week"])

df_dates = df_dates.withColumn("date_id", expr("year * 10000 + month * 100 + day_of_month"))
# Alternatively, you can keep date_id as a simple increment if you prefer.

# Write out partitioned by year, month if desired
(
    df_dates
    .write
    .mode("overwrite")
    .partitionBy("year", "month")
    .parquet("/path/to/output/dim_dates")
)

# --------------------------------------------------------------------------------
# 4. Create STORES Dimension (smaller dimension)
# --------------------------------------------------------------------------------
df_stores = (
    spark.range(num_rows_stores)
         .withColumnRenamed("id", "store_id")
         .withColumn("store_region", (col("store_id") % 5).cast("string"))
         .withColumn("store_type", (col("store_id") % 2).cast("string"))

         # Some random string columns for extra size
         .withColumn("store_str1", substring(md5(rand(seed=50)), 1, 32))
         .withColumn("store_str2", substring(md5(rand(seed=51)), 1, 32))

         # Some numeric columns
         .withColumn("store_size", rand(seed=52) * 10000)  # e.g., square footage
)

# Partition by region or type if you like:
(
    df_stores
    .write
    .mode("overwrite")
    .partitionBy("store_region", "store_type")
    .parquet("/path/to/output/dim_stores")
)

# --------------------------------------------------------------------------------
# 5. Create CUSTOMERS Dimension (large, but no SCD2 here for simplicity)
# --------------------------------------------------------------------------------
df_customers = (
    spark.range(distinct_customer_ids)  # 1 million unique customers
         .withColumnRenamed("id", "customer_id")

         # Random string columns
         .withColumn("cust_name", substring(md5(rand(seed=100)), 1, 16))
         .withColumn("cust_email", substring(md5(rand(seed=101)), 1, 24))
         .withColumn("cust_region", (col("customer_id") % 5).cast("string"))

         # Random numeric columns
         .withColumn("credit_score", floor(rand(seed=102) * 800))
)

# Suppose we partition by region to help certain queries
(
    df_customers
    .write
    .mode("overwrite")
    .partitionBy("cust_region")
    .parquet("/path/to/output/dim_customers")
)

# --------------------------------------------------------------------------------
# 6. Create PRODUCTS Dimension with SCD Type 2
#    We'll generate 10 million "rows", but effectively each product_id
#    can appear up to 3 times with different version periods.
# --------------------------------------------------------------------------------

df_products_base = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "seq_id")
         .withColumn("product_id", (col("seq_id") % distinct_product_ids))

         # brand/category with small cardinalities for partitioning
         .withColumn("brand", (col("seq_id") % 50).cast("string"))
         .withColumn("category", (col("seq_id") % 10).cast("string"))

         # random columns
         .withColumn("prod_str1", substring(md5(rand(seed=1)), 1, 32))
         .withColumn("prod_str2", substring(md5(rand(seed=2)), 1, 32))
         .withColumn("prod_price", rand(seed=4) * 100)

         # We'll define a "version_number" to simulate multiple SCD2 versions
         .withColumn("version_number", (col("seq_id") % 3) + 1)
)

# For each version_number, define start_date, end_date, is_current
df_products_scd2 = (
    df_products_base
    .withColumn("start_date", expr("date_add(to_date('2020-01-01'), version_number * 10)"))
    .withColumn(
        "end_date",
        when(col("version_number") < 3,
             expr("date_add(start_date, 9)"))  # end_date = start_date + 9 days
        .otherwise(to_date(lit("9999-12-31")))
    )
    .withColumn("is_current", when(col("version_number") < 3, lit(False)).otherwise(lit(True)))
    .drop("seq_id")  # not needed
)

# Multi-level partition by brand, category, is_current
(
    df_products_scd2
    .write
    .mode("overwrite")
    .partitionBy("brand", "category", "is_current")
    .parquet("/path/to/output/dim_products_scd2")
)

# --------------------------------------------------------------------------------
# 7. Create FACT SALES with multi-level partitioning
#    We'll reference: product_id, store_id, date_id, customer_id
#    And we can have some measures: quantity, sales_amount, discount_amount, etc.
# --------------------------------------------------------------------------------

df_sales = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "transaction_id")

         # foreign keys
         .withColumn("product_id", col("transaction_id") % distinct_product_ids)
         .withColumn("store_id", col("transaction_id") % distinct_store_ids)
         .withColumn("customer_id", col("transaction_id") % distinct_customer_ids)

         # random region assignment (for partitioning)
         .withColumn("region", (col("transaction_id") % 5).cast("string"))

         # link to date dimension - pick a random offset into the date list
         # so we ensure date_id lines up with the date dimension
         .withColumn("date_offset", floor(rand(seed=500) * num_rows_dates))
         .withColumn("date_id", expr("20190000 + date_offset"))  # rough logic
         # (Or you can do more precise logic to ensure date_id is valid.)

         # measures
         .withColumn("quantity", floor(rand(seed=501) * 10))   # e.g. 0..9
         .withColumn("sales_amount", rand(seed=502) * 500)
         .withColumn("discount_amount", rand(seed=503) * 50)
)

# Derive year/month from date_id if you want real partition columns. 
# Because we used "20190000 + offset", let's approximate year/month:
# Real approach: join date_id to df_dates to get the actual year/month,
# but for demonstration, let's do a quick hack:
df_sales_part = (
    df_sales
    .withColumn("sale_year", expr("floor(date_id / 10000)"))
    .withColumn("sale_month", expr("floor((date_id % 10000) / 100))"))
)

# Now, multi-level partition: region, sale_year, sale_month
(
    df_sales_part
    .write
    .mode("overwrite")
    .partitionBy("region", "sale_year", "sale_month")
    .parquet("/path/to/output/fact_sales")
)

# --------------------------------------------------------------------------------
# 8. Optional: Demonstrate a join (to show the plan)
#    For instance, let's do a Sales -> SCD2 Products join on product_id
#    with date-range matching for SCD2, using start_date/end_date vs. an approximate date.
#    But here we have date_id, so let's do a simplified demonstration:
# --------------------------------------------------------------------------------

# We'll read back some data to show how you'd join. 
# (In real use, you'd read from the parquet outputs with spark.read.parquet(...))

df_products_scd2_loaded = spark.read.parquet("/path/to/output/dim_products_scd2")
df_sales_loaded = spark.read.parquet("/path/to/output/fact_sales")

# A typical SCD2 join requires date-range matching:
# Sales.date_id (converted to an actual date) between product.start_date and product.end_date.
# But we stored date_id as an integer, so let's do a rough approach:
# We'll convert 'start_date' to an int "YYYYMMDD" or something similar. 
# That requires some function calls or UDF, but let's do a conceptual example:

df_products_scd2_loaded = df_products_scd2_loaded \
    .withColumn("start_int", expr("year(start_date) * 10000 + month(start_date) * 100 + day(start_date)")) \
    .withColumn("end_int", expr("year(end_date) * 10000 + month(end_date) * 100 + day(end_date)"))

# Now we can join with condition:
# sales.product_id = products.product_id
# AND sales.date_id >= product.start_int
# AND sales.date_id < product.end_int

df_scd_join = df_sales_loaded.alias("s") \
    .join(
        df_products_scd2_loaded.alias("p"),
        on=(
            (col("s.product_id") == col("p.product_id")) &
            (col("s.date_id") >= col("p.start_int")) &
            (col("s.date_id") < col("p.end_int"))
        ),
        how="inner"
    )

df_scd_join.explain(True)
print("Joined row count:", df_scd_join.count())

# --------------------------------------------------------------------------------
# 9. Stop Spark
# --------------------------------------------------------------------------------
spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, md5, substring, expr, when, to_date
import datetime

# --------------------------------------------------------------------------------
# 1. Initialize Spark session
# --------------------------------------------------------------------------------
spark = (
    SparkSession.builder
        .appName("DifferentJoinStrategiesExample")
        # .config("spark.sql.shuffle.partitions", "200")  # Adjust if needed
        .getOrCreate()
)

# --------------------------------------------------------------------------------
# 2. Generate or load data
#
#    In this snippet, we generate simpler DataFrames with enough rows
#    to see different join strategies. If you have already created the
#    large SCD2 Products, Sales, and Stores, simply read them from
#    wherever they’re stored (e.g., Parquet on disk).
# --------------------------------------------------------------------------------

# For demonstration, let's do ~2 million rows so it doesn't blow up a local machine.
# Increase to 10 million or more if you have a real cluster.
num_rows_large = 2_000_000
distinct_product_ids = 100_000
distinct_store_ids = 10_000

# --- PRODUCTS (large dimension)
df_products = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "seq_id")
         .withColumn("product_id", (col("seq_id") % distinct_product_ids))

         # A few random columns
         .withColumn("prod_str", substring(md5(rand(seed=1)), 1, 32))
         .withColumn("prod_price", rand(seed=2) * 100)
)

# --- STORES (small dimension)
df_stores = (
    spark.range(10_000)  # only 10k
         .withColumnRenamed("id", "store_id")
         .withColumn("store_str", substring(md5(rand(seed=3)), 1, 32))
)

# --- SALES (fact, ~2 million)
df_sales = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "transaction_id")
         .withColumn("product_id", (col("transaction_id") % distinct_product_ids))
         .withColumn("store_id", (col("transaction_id") % distinct_store_ids))
         .withColumn("quantity", (rand(seed=4) * 10).cast("int"))
         .withColumn("sales_amount", rand(seed=5) * 500)
)

# Cache them in memory if you want repeated tests (optional)
df_products.cache()
df_stores.cache()
df_sales.cache()

# --------------------------------------------------------------------------------
# 3. BROADCAST JOIN EXAMPLE
#    We expect Spark to broadcast the smaller df_stores (~10k rows).
#    We'll set a large autoBroadcastJoinThreshold so Spark will do it.
# --------------------------------------------------------------------------------

print("=== BROADCAST JOIN EXAMPLE ===")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024 * 1024)  
# set to 10 GB (or similarly large) so the 10k-row table is broadcast

df_broadcast_join = df_sales.join(df_stores, on="store_id", how="inner")

df_broadcast_join.explain(True)
count_broadcast = df_broadcast_join.count()
print(f"Broadcast Join row count = {count_broadcast}")

# Look in Spark UI -> SQL tab or explain plan:
# You should see "BroadcastHashJoin" or "BroadcastNestedLoopJoin" (for a CROSS join),
# but typically BroadcastHashJoin for this scenario.

# --------------------------------------------------------------------------------
# 4. SHUFFLE/HASH JOIN EXAMPLE
#    We disable broadcast by setting threshold to -1, forcing Spark
#    to shuffle both sides. For a small dimension, Spark might do a
#    Shuffle Hash Join. For big ones, it often does Sort Merge Join.
# --------------------------------------------------------------------------------

print("=== SHUFFLE/HASH JOIN EXAMPLE (Stores & Sales, no broadcast) ===")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)  # disable broadcast

df_shuffle_join = df_sales.join(df_stores, on="store_id", how="inner")
df_shuffle_join.explain(True)
count_shuffle = df_shuffle_join.count()
print(f"Shuffle/Hash Join row count = {count_shuffle}")

# In the plan, you may see "ShuffledHashJoin" or "SortMergeJoin", depending on Spark version
# and internal heuristics. If the tables are small enough, Spark might pick ShuffledHashJoin.

# --------------------------------------------------------------------------------
# 5. SORT-MERGE JOIN EXAMPLE (Sales & Products, both large)
#    We sort both DataFrames on product_id to encourage a SortMergeJoin.
#    We'll keep broadcast disabled so it can't short-circuit that way.
# --------------------------------------------------------------------------------

print("=== SORT-MERGE JOIN EXAMPLE (Sales & Products) ===")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)  # ensure no broadcast

# Sort the large tables by product_id
df_sales_sorted = df_sales.sort("product_id")
df_products_sorted = df_products.sort("product_id")

df_sort_merge = df_sales_sorted.join(df_products_sorted, on="product_id", how="inner")
df_sort_merge.explain(True)
count_sort_merge = df_sort_merge.count()
print(f"SortMerge Join row count = {count_sort_merge}")

# The plan typically shows "SortMergeJoin". 
# Even if you don't pre-sort, Spark often uses SortMergeJoin for large data
# when broadcast is disabled. But explicitly sorting can reduce some overhead
# if the data remains partitioned consistently.

# --------------------------------------------------------------------------------
# 6. OPTIONAL: Compare performance in Spark UI
#    - Check each stage's "Tasks", "Duration", "Shuffle Read/Write Size".
#    - The .count() triggers full execution, so you can measure how long
#      each join takes.
# --------------------------------------------------------------------------------

print("\nAll joins complete. Check Spark UI for performance metrics.\n")

# (Optional) Stop Spark if desired
# spark.stop()
