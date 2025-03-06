from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    rand,
    floor
)

# --------------------------------------------------------------------------------
# 1. Initialize Spark session
# --------------------------------------------------------------------------------
spark = (
    SparkSession.builder
        .appName("LargeSortMergeJoinWithSmallDimension")
        # You can adjust shuffle partitions to tune performance
        # .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
)

# Disable auto-broadcast so we force shuffle or sort-merge joins
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# --------------------------------------------------------------------------------
# 2. Configuration for row counts
# --------------------------------------------------------------------------------
num_rows_large = 10_000_000  # For Products and Sales
num_rows_small = 10_000      # For Stores (small dimension)

# For controlling the distribution of product_id and store_id
distinct_product_ids = 100_000
distinct_store_ids = 10_000  # Matches the # of rows in df_stores to ensure 1:1 store_id

# --------------------------------------------------------------------------------
# 3. Create the PRODUCTS DataFrame (~10M rows)
# --------------------------------------------------------------------------------
df_products = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "seq_id")
         .withColumn("product_id", (col("seq_id") % distinct_product_ids))
         .withColumn("brand", (col("seq_id") % 3).cast("string"))    # small set of brands
         .withColumn("category", (col("seq_id") % 5).cast("string")) # small set of categories
         .withColumn("price", rand(seed=42) * 100)                   # random float in [0..100)
)

# --------------------------------------------------------------------------------
# 4. Create the SALES DataFrame (~10M rows)
# --------------------------------------------------------------------------------
df_sales = (
    spark.range(num_rows_large)
         .withColumnRenamed("id", "transaction_id")
         .withColumn("product_id", (col("transaction_id") % distinct_product_ids))
         .withColumn("store_id", (col("transaction_id") % distinct_store_ids))
         .withColumn("quantity", floor(rand(seed=999) * 10))         # random int in [0..9]
         .withColumn("total_amount", rand(seed=999) * 500)           # random float in [0..500)
)

# --------------------------------------------------------------------------------
# 5. Create the STORES DataFrame (~10k rows) - smaller dimension
# --------------------------------------------------------------------------------
df_stores = (
    spark.range(num_rows_small)
         .withColumnRenamed("id", "store_id")
         .withColumn("store_name", (col("store_id") + 1000).cast("string"))
         .withColumn("region", (col("store_id") % 5).cast("string")) # e.g., 5 regions
         .withColumn("manager_id", (col("store_id") + 500).cast("string"))
)

# --------------------------------------------------------------------------------
# 6. Example: Join Sales -> Products (large vs. large)
# --------------------------------------------------------------------------------
print("=== JOIN SALES and PRODUCTS (No Sorting) ===")
df_join_no_sort = df_sales.join(df_products, on="product_id", how="inner")
df_join_no_sort.explain(True)

count_no_sort = df_join_no_sort.count()
print(f"[Sales-Products Join No Sort] Row count: {count_no_sort}")

# --- Now with sorting to encourage sort-merge join
df_products_sorted = df_products.sort("product_id")
df_sales_sorted = df_sales.sort("product_id")

print("=== JOIN SALES and PRODUCTS (With Sorting) ===")
df_join_sorted = df_sales_sorted.join(df_products_sorted, on="product_id", how="inner")
df_join_sorted.explain(True)

count_sorted = df_join_sorted.count()
print(f"[Sales-Products Join With Sort] Row count: {count_sorted}")

# --------------------------------------------------------------------------------
# 7. Example: Join Sales -> Stores (large vs. small)
# --------------------------------------------------------------------------------
# This join might attempt broadcast if we didn't disable it,
# because df_stores is small. But we've set autoBroadcastJoinThreshold = -1,
# so it should do a shuffle or sort-merge join.
print("=== JOIN SALES and STORES (No Sorting) ===")
df_join_sales_stores = df_sales.join(df_stores, on="store_id", how="inner")
df_join_sales_stores.explain(True)

count_sales_stores = df_join_sales_stores.count()
print(f"[Sales-Stores Join No Sort] Row count: {count_sales_stores}")

# --------------------------------------------------------------------------------
# 8. Example: 3-way join among Sales, Products, and Stores
# --------------------------------------------------------------------------------
# We'll do a single SELECT that joins all three together.
# Typically you'd do something like:
#    SELECT product_id, store_id, brand, category, region, total_amount
#    FROM Sales
#    JOIN Products on Sales.product_id = Products.product_id
#    JOIN Stores on Sales.store_id = Stores.store_id
print("=== 3-WAY JOIN (Sales -> Products -> Stores) ===")
df_3_way = (df_sales
            .join(df_products, "product_id", "inner")
            .join(df_stores, "store_id", "inner"))

df_3_way.explain(True)
count_3_way = df_3_way.count()
print(f"[3-WAY JOIN] Row count: {count_3_way}")

# --------------------------------------------------------------------------------
# 9. Stop Spark session (optional)
# --------------------------------------------------------------------------------
spark.stop()
