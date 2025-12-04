# nearest_neighbor_database_creator.py
# Hybrid pipeline: PySpark (ALS) -> Annoy nearest neighbors -> SQLite
# 
# In order to properly utilize PySpark, and to optimize performance 
# with Annoy, the assistance of large language models were used to 
# improve some code in this file. When I wrote the code completely 
# by myself, my implementation using PySpark was improperly configured 
# which caused my computers disk to reach 100% utilization. My computer 
# had an SSD, so it would have been damaged to it if it was sustained 
# for a longer period of time. The implementation has been changed so 
# now it runs without causing strain on computers hardware. 
#
# Requirements:
#   pip install pyspark annoy psutil numpy
#
# Produces:
#   movies_nearest_neighbors.db
#   users_nearest_neighbors.db

import os
import sqlite3
import math
from collections import defaultdict

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StructType, StructField
from pyspark.ml.recommendation import ALS

# Annoy
from annoy import AnnoyIndex
import numpy as np

# ----------------- Configuration -----------------
RATINGS_FILE = "csv-files/ratings.csv"          # same working directory
MOVIE_NEIGHBORS_DB = "db-files/movies_nearest_neighbors.db"
USER_NEIGHBORS_DB = "db-files/users_nearest_neighbors.db"

# ALS hyperparameters
ALS_RANK = 40         # embedding dimensionality (lower => less memory, faster)
ALS_MAX_ITER = 10
ALS_REG = 0.1

# ANN hyperparameters
NN_K = 50             # top-k neighbors to compute
ANNOY_TREES = 50      # more trees => higher accuracy, slower build

# Spark (protect SSD) -- we use single worker and small shuffle partitions to avoid thrash
SPARK_MASTER = "local[1]"   # SINGLE worker: minimal parallelism -> minimal temp files
SPARK_DRIVER_MEMORY = "8g"  # set safely depending on your machine
SPARK_EXECUTOR_MEMORY = "8g"
SPARK_SHUFFLE_PARTITIONS = "4"
SPARK_LOCAL_DIR = None      # set to None to use default TEMP; you may set e.g. "C:/tmp/spark" if desired
# -------------------------------------------------

def create_spark():
    builder = SparkSession.builder.master(SPARK_MASTER).appName("ALS-then-Annoy")
    builder = builder.config("spark.driver.memory", SPARK_DRIVER_MEMORY)
    builder = builder.config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    builder = builder.config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
    builder = builder.config("spark.default.parallelism", "1")
    # memory tuning to favor caching, reduce spilling
    builder = builder.config("spark.memory.fraction", "0.8")
    builder = builder.config("spark.memory.storageFraction", "0.6")
    if SPARK_LOCAL_DIR:
        builder = builder.config("spark.local.dir", SPARK_LOCAL_DIR)
    spark = builder.getOrCreate()
    # set warn to reduce noisy logs
    spark.sparkContext.setLogLevel("WARN")
    return spark

def read_ratings_df(spark):
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True)
    ])
    # read CSV; tolerate missing header / extra whitespace
    df = spark.read.csv(RATINGS_FILE, schema=schema, header=False)
    # drop rows with null user or movie or rating
    df = df.na.drop(subset=["userId", "movieId", "rating"])
    return df

def train_als(df):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=ALS_RANK,
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG,
        coldStartStrategy="drop",   # drop NaN preds
        nonnegative=False
    )
    model = als.fit(df)
    return model

def collect_factors_to_driver(df_factors):
    """
    df_factors: DataFrame with schema [id (int), features (array<float>)]
    Returns: dict id -> np.array(features, dtype=float32)
    """
    rows = df_factors.collect()   # collecting item/user factor table to driver
    mapping = {}
    for r in rows:
        idx = int(r["id"])
        feats = r["features"]
        # convert to numpy array
        vec = np.array(feats, dtype=np.float32)
        mapping[idx] = vec
    return mapping

def build_annoy_index(vecs_dict, dim, n_trees=ANNOY_TREES):
    """
    vecs_dict: mapping id -> vector (numpy array)
    returns: annoy index object and id_to_idx map and idx_to_id list
    """
    # Annoy requires contiguous integer indices [0..n-1]
    ids = list(vecs_dict.keys())
    idx_to_id = ids[:]                 # idx -> movieId
    id_to_idx = {mid: i for i, mid in enumerate(ids)}  # movieId -> idx

    index = AnnoyIndex(dim, metric="angular")
    for i, mid in enumerate(ids):
        index.add_item(i, vecs_dict[mid].tolist())
    index.build(n_trees)
    return index, id_to_idx, idx_to_id

def compute_and_store_neighbors(index, id_to_idx, idx_to_id, vec_dict, db_path, topk=NN_K):
    """
    Compute top-k neighbors for each id and store into sqlite db_path.
    Stores (id, neighbor_id, similarity, dummy_co_count)
    similarity computed as cosine similarity from vectors for higher accuracy.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS neighbors (
        id INTEGER,
        neighbor_id INTEGER,
        similarity REAL,
        PRIMARY KEY (id, neighbor_id)
    );
    """)
    c.execute("DELETE FROM neighbors;")
    conn.commit()

    dim = next(iter(vec_dict.values())).shape[0]

    # pre-normalize vectors for cosine
    norms = {}
    for mid, v in vec_dict.items():
        n = np.linalg.norm(v)
        norms[mid] = n if n > 0 else 1.0

    insert = "INSERT OR REPLACE INTO neighbors (id, neighbor_id, similarity) VALUES (?, ?, ?)"
    total = len(idx_to_id)
    for i, mid in enumerate(idx_to_id):
        # ask for topk+1 because first will be itself
        idx = id_to_idx[mid]
        neighbors = index.get_nns_by_item(idx, topk + 1, include_distances=False)
        stored = 0
        for neigh_idx in neighbors:
            neigh_id = idx_to_id[neigh_idx]
            if neigh_id == mid:
                continue
            # cosine similarity
            sim = float(np.dot(vec_dict[mid], vec_dict[neigh_id]) / (norms[mid] * norms[neigh_id] + 1e-12))
            c.execute(insert, (int(mid), int(neigh_id), float(sim)))
            stored += 1
            if stored >= topk:
                break
        # commit periodically
        if (i % 500) == 0:
            conn.commit()
    conn.commit()
    conn.close()

def save_small_debug_info(mapping, fname):
    # optional helper: save id list and dims to disk
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"count={len(mapping)}\n")
        next_id = next(iter(mapping.keys()))
        f.write(f"dim={mapping[next_id].shape[0]}\n")

def main():
    print("Starting Spark (Hadoop framework) in safe configuration...")
    spark = create_spark()
    print("Reading ratings file...")
    df = read_ratings_df(spark)
    print("Number of ratings rows:", df.count())
    print("Training ALS to produce embeddings (this is the Hadoop step)...")
    model = train_als(df)
    print("Collecting item factors to driver...")
    item_factors_df = model.itemFactors  # columns: id, features
    item_vecs = collect_factors_to_driver(item_factors_df)
    print("Items collected:", len(item_vecs))
    print("Collecting user factors to driver...")
    user_factors_df = model.userFactors
    user_vecs = collect_factors_to_driver(user_factors_df)
    print("Users collected:", len(user_vecs))

    # build Annoy index for items
    if len(item_vecs) == 0:
        print("No item vectors found -- aborting.")
        spark.stop()
        return

    dim = next(iter(item_vecs.values())).shape[0]
    print(f"Building Annoy index for items (dim={dim}) ...")
    item_index, item_id_to_idx, item_idx_to_id = build_annoy_index(item_vecs, dim, n_trees=ANNOY_TREES)
    print("Computing top-k neighbors for items and saving to", MOVIE_NEIGHBORS_DB)
    compute_and_store_neighbors(item_index, item_id_to_idx, item_idx_to_id, item_vecs, MOVIE_NEIGHBORS_DB, topk=NN_K)

    # build Annoy index for users (optional)
    if len(user_vecs) > 0:
        dim_u = next(iter(user_vecs.values())).shape[0]
        print(f"Building Annoy index for users (dim={dim_u}) ...")
        user_index, user_id_to_idx, user_idx_to_id = build_annoy_index(user_vecs, dim_u, n_trees=ANNOY_TREES)
        print("Computing top-k neighbors for users and saving to", USER_NEIGHBORS_DB)
        compute_and_store_neighbors(user_index, user_id_to_idx, user_idx_to_id, user_idx_to_id and user_vecs, USER_NEIGHBORS_DB, topk=NN_K)
    else:
        print("No user vectors found; skipping user neighbor creation.")

    print("Done. Closing Spark.")
    spark.stop()

if __name__ == "__main__":
    main()
