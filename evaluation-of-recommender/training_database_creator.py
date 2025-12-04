# training_database_creator.py
#
# Creates nearest neighbor databases with 80% of the data used for training.
# This is used so the remaining 20% can be left for evaluation with 
# evaluate_with_app_recommender.py. The code in this file is based off of the
# PySpark and Annoy implementation written in nearest_neighbor_database_creator.py.
# All the information related to nearest_neighbor_database_creator.py, such as
# notes written in the comments of the file, also applies here as well.

# Outputs:
#   db-files/movies_nearest_neighbors_train.db
#   db-files/users_nearest_neighbors_train.db

# Requirements:
#   pip install pyspark annoy numpy tqdm
#   (Also needs csv-files/ratings.csv and db-files/movies.db present.)


import os
import sqlite3
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm

# Spark imports (must be installed)
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StructType, StructField
from pyspark.ml.recommendation import ALS

# ----------------- CONFIG -----------------
RATINGS_CSV = "csv-files/ratings.csv"       # original full ratings CSV (userId,movieId,rating,timestamp)
MOVIES_DB = "db-files/movies.db"            # existing movies DB (for title mapping if needed)
OUT_MOVIE_NEIGH_DB = "db-files/movies_nearest_neighbors_train.db"
OUT_USER_NEIGH_DB = "db-files/users_nearest_neighbors_train.db"
TRAIN_RATINGS_CSV = "csv-files/ratings_train.csv"
TEST_RATINGS_CSV = "csv-files/ratings_test.csv"

TRAIN_FRAC = 0.8           # per-user fraction to keep in training
SEED = 42
NN_K = 50                  # number of neighbors to compute/store
ANNOY_TREES = 50          # trees for Annoy
ALS_RANK = 40
ALS_MAX_ITER = 10
ALS_REG = 0.1

# Spark tuning (safe defaults)
SPARK_MASTER = "local[1]"
SPARK_DRIVER_MEMORY = "6g"
SPARK_EXECUTOR_MEMORY = "6g"
SPARK_SHUFFLE_PARTITIONS = "4"

random.seed(SEED)
np.random.seed(SEED)

# ----------------- Helpers -----------------
def split_train_test_per_user(ratings_csv: str, train_csv: str, test_csv: str, train_frac=0.8, seed=42):
    """
    Per-user holdout: for each user, randomly keep ~train_frac of their ratings in train and the rest in test.
    If a user has only 1 rating, keep it in train (no test).
    Writes two CSV files (same format as input).
    """
    import csv
    by_user = defaultdict(list)
    with open(ratings_csv, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)  # assumes header exists; if not, remove this line
        for row in r:
            if not row:
                continue
            uid = int(row[0]); mid = int(row[1]); rating = float(row[2]); ts = int(row[3]) if len(row) > 3 and row[3] != "" else 0
            by_user[uid].append((uid, mid, rating, ts))
    # shuffle and split
    train_rows = []
    test_rows = []
    rnd = random.Random(seed)
    for uid, rows in by_user.items():
        n = len(rows)
        if n == 1:
            train_rows.extend(rows)
            continue
        idxs = list(range(n))
        rnd.shuffle(idxs)
        cut = max(1, int(math.floor(train_frac * n)))
        train_idx = set(idxs[:cut])
        for i, row in enumerate(rows):
            if i in train_idx:
                train_rows.append(row)
            else:
                test_rows.append(row)
    # write CSVs (include header)
    with open(train_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["userId","movieId","rating","timestamp"])
        w.writerows(train_rows)
    with open(test_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["userId","movieId","rating","timestamp"])
        w.writerows(test_rows)
    return len(train_rows), len(test_rows)

# ----------------- Spark + ALS + Annoy pipeline (adapted) -----------------
def create_spark_session():
    builder = SparkSession.builder.master(SPARK_MASTER).appName("ALS-Annoy-Eval")
    builder = builder.config("spark.driver.memory", SPARK_DRIVER_MEMORY)
    builder = builder.config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    builder = builder.config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
    builder = builder.config("spark.default.parallelism", "1")
    builder = builder.config("spark.memory.fraction", "0.8")
    builder = builder.config("spark.memory.storageFraction", "0.6")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def read_ratings_df(spark, ratings_file):
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True)
    ])
    df = spark.read.csv(ratings_file, schema=schema, header=True)
    df = df.na.drop(subset=["userId","movieId","rating"])
    return df

def train_als_model(spark_df):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=ALS_RANK,
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG,
        coldStartStrategy="drop",
        nonnegative=False
    )
    model = als.fit(spark_df)
    return model

def collect_factors(df_factors):
    """
    df_factors has columns id and features (array<float>)
    return dict id -> numpy array
    """
    mapping = {}
    rows = df_factors.collect()
    for r in rows:
        idx = int(r["id"])
        feats = r["features"]
        mapping[idx] = np.array(feats, dtype=np.float32)
    return mapping

def build_annoy_index(vecs_dict, dim, n_trees=ANNOY_TREES):
    ids = list(vecs_dict.keys())
    idx_to_id = ids[:]  # internal idx -> original id
    id_to_idx = {mid: i for i, mid in enumerate(ids)}
    index = AnnoyIndex(dim, metric="angular")
    for i, mid in enumerate(ids):
        index.add_item(i, vecs_dict[mid].tolist())
    index.build(n_trees)
    return index, id_to_idx, idx_to_id

def compute_and_store_neighbors(index, id_to_idx, idx_to_id, vec_dict, db_path, topk=NN_K):
    # compute cosine similarities and store (id, neighbor_id, similarity)
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

    norms = {}
    for mid, v in vec_dict.items():
        n = np.linalg.norm(v)
        norms[mid] = n if n > 0 else 1.0

    insert = "INSERT OR REPLACE INTO neighbors (id, neighbor_id, similarity) VALUES (?, ?, ?)"
    total = len(idx_to_id)
    for i, mid in enumerate(idx_to_id):
        idx = id_to_idx[mid]
        # request topk+1 since first will be itself
        neighbors = index.get_nns_by_item(idx, topk+1, include_distances=False)
        stored = 0
        for neigh_idx in neighbors:
            neigh_id = idx_to_id[neigh_idx]
            if neigh_id == mid:
                continue
            sim = float(np.dot(vec_dict[mid], vec_dict[neigh_id]) / (norms[mid] * norms[neigh_id] + 1e-12))
            c.execute(insert, (int(mid), int(neigh_id), float(sim)))
            stored += 1
            if stored >= topk:
                break
        if (i % 500) == 0:
            conn.commit()
    conn.commit()
    conn.close()

# ----------------- Load neighbor DB into memory (same helper as in app.py) -----------------
def load_neighbors_db(neighbors_db_path: str) -> Dict[int, List[Tuple[int, float]]]:
    mapping: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    if not os.path.exists(neighbors_db_path):
        return {}
    try:
        conn = sqlite3.connect(neighbors_db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, neighbor_id, similarity FROM neighbors")
        for idcol, nid, sim in cur.fetchall():
            try:
                iid = int(idcol)
                nn = int(nid)
                s = float(sim)
            except Exception:
                continue
            mapping[iid].append((nn, s))
        conn.close()
    except Exception:
        return {}
    for i in mapping:
        mapping[i].sort(key=lambda t: -abs(t[1]))
    return dict(mapping)

# ----------------- Prediction functions (copied/adapted from app.py) -----------------
def compute_user_means(user_ratings: Dict[int, Dict[int, float]]) -> Tuple[Dict[int, float], float]:
    means: Dict[int, float] = {}
    total = 0.0
    count = 0
    for u, rs in user_ratings.items():
        if rs:
            s = sum(rs.values()); c = len(rs)
            means[u] = s / c
            total += s; count += c
        else:
            means[u] = 0.0
    global_mean = (total / count) if count > 0 else 3.0
    return means, global_mean

def predict_item_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       movie_neighbors: Dict[int, List[Tuple[int, float]]],
                       k: int = NN_K) -> float:
    neighs = movie_neighbors.get(movie_id, [])[:k]
    num = 0.0
    denom = 0.0
    ur = user_ratings.get(user_id, {})
    for neigh_id, sim in neighs:
        r = ur.get(neigh_id)
        if r is None:
            continue
        num += sim * r
        denom += abs(sim)
    if denom == 0:
        if ur:
            return max(0.5, min(5.0, sum(ur.values()) / len(ur)))
        return 3.0
    pred = num / denom
    return max(0.5, min(5.0, pred))

def predict_user_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       user_neighbors: Dict[int, List[Tuple[int, float]]],
                       user_means: Dict[int, float],
                       global_mean: float,
                       k: int = NN_K) -> float:
    neighs = user_neighbors.get(user_id, [])[:k]
    num = 0.0
    denom = 0.0
    for neigh_id, sim in neighs:
        neigh_ratings = user_ratings.get(neigh_id, {})
        r = neigh_ratings.get(movie_id)
        if r is None:
            continue
        neigh_mean = user_means.get(neigh_id, 0.0)
        num += sim * (r - neigh_mean)
        denom += abs(sim)
    if denom == 0:
        um = user_means.get(user_id)
        if um is not None and um != 0:
            return max(0.5, min(5.0, um))
        return max(0.5, min(5.0, global_mean))
    base = user_means.get(user_id, 0.0)
    pred = base + (num / denom)
    return max(0.5, min(5.0, pred))

# ----------------- Evaluation metrics -----------------
def evaluate(preds: List[float], truths: List[float]):
    import math
    if len(preds) == 0:
        return {"rmse": None, "mae": None}
    preds = np.array(preds)
    truths = np.array(truths)
    mse = np.mean((preds-truths)**2)
    mae = np.mean(np.abs(preds-truths))
    return {"rmse": math.sqrt(mse), "mae": float(mae)}

# ----------------- Main flow -----------------
def main():
    # 1) Split ratings into train/test
    print("Splitting ratings into train/test (per-user holdout)...")
    train_count, test_count = split_train_test_per_user(RATINGS_CSV, TRAIN_RATINGS_CSV, TEST_RATINGS_CSV, train_frac=TRAIN_FRAC, seed=SEED)
    print(f"Train rows: {train_count}, Test rows: {test_count}")

    # 2) Start Spark and train ALS on TRAIN_RATINGS_CSV
    print("Starting Spark session...")
    spark = create_spark_session()
    print("Reading train ratings CSV into Spark")
    df_train = read_ratings_df(spark, TRAIN_RATINGS_CSV)
    print("Training ALS on training data (this may take some time)...")
    model = train_als_model(df_train)
    print("Collect item and user factors from ALS model...")
    item_factors_df = model.itemFactors  # id, features (id is movieId)
    user_factors_df = model.userFactors  # id, features (id is userId)
    item_vecs = collect_factors(item_factors_df)
    user_vecs = collect_factors(user_factors_df)
    print(f"Collected item vectors: {len(item_vecs)}, user vectors: {len(user_vecs)}")

    if len(item_vecs) == 0:
        print("No item vectors returned by ALS. Exiting.")
        spark.stop()
        return

    # 3) Build Annoy index for items and compute neighbors, store to OUT_MOVIE_NEIGH_DB
    dim = next(iter(item_vecs.values())).shape[0]
    print(f"Building Annoy index for items (dim={dim}) ...")
    item_index, item_id_to_idx, item_idx_to_id = build_annoy_index(item_vecs, dim, n_trees=ANNOY_TREES)
    print(f"Computing top-{NN_K} item neighbors and storing to {OUT_MOVIE_NEIGH_DB} ...")
    compute_and_store_neighbors(item_index, item_id_to_idx, item_idx_to_id, item_vecs, OUT_MOVIE_NEIGH_DB, topk=NN_K)

    # 4) Build Annoy + neighbors for users if present
    if len(user_vecs) > 0:
        dim_u = next(iter(user_vecs.values())).shape[0]
        print(f"Building Annoy index for users (dim={dim_u}) ...")
        user_index, user_id_to_idx, user_idx_to_id = build_annoy_index(user_vecs, dim_u, n_trees=ANNOY_TREES)
        print(f"Computing top-{NN_K} user neighbors and storing to {OUT_USER_NEIGH_DB} ...")
        compute_and_store_neighbors(user_index, user_id_to_idx, user_idx_to_id, user_vecs, OUT_USER_NEIGH_DB, topk=NN_K)
    else:
        print("No user vectors returned by ALS; skipping user neighbor DB.")

    # stop spark
    spark.stop()

    # 5) Load train ratings into memory (user_ratings dict) and compute means
    train_user_ratings = defaultdict(dict)
    import csv
    with open(TRAIN_RATINGS_CSV, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            u = int(row[0]); m = int(row[1]); rating = float(row[2])
            train_user_ratings[u][m] = rating
    user_means_map, global_mean = compute_user_means(train_user_ratings)
    print(f"Computed user means (train). Global mean = {global_mean:.4f}")

    # 6) Load neighbor DBs into memory
    print("Loading item neighbors DB into memory...")
    movie_neighbors = load_neighbors_db(OUT_MOVIE_NEIGH_DB)  # mapping movieId -> [(neighbor_movieId, sim), ...]
    print(f"Loaded neighbors for {len(movie_neighbors)} items.")
    print("Loading user neighbors DB into memory (if present)...")
    user_neighbors = load_neighbors_db(OUT_USER_NEIGH_DB)
    print(f"Loaded neighbors for {len(user_neighbors)} users.")

    # NOTE: The neighbor DB stores neighbor ids as original user/movie ids (not internal indices).
    # Our predict functions assume movie_neighbors keyed by movieId and user_neighbors keyed by userId.
    # The load_neighbors_db returns exactly that mapping so we can use directly.

    # 7) Evaluate on TEST set
    print("Reading test set and evaluating predictions...")
    test_rows = []
    with open(TEST_RATINGS_CSV, newline='', encoding='utf-8') as f:
        r = __import__('csv').reader(f)
        next(r)
        for row in r:
            test_rows.append((int(row[0]), int(row[1]), float(row[2])))

    item_preds = []
    user_preds = []
    baseline_preds = []
    truths = []
    fallback_item_count = 0
    fallback_user_count = 0

    for (u,m,true_r) in tqdm(test_rows, desc="Evaluate", unit="row"):
        truths.append(true_r)
        baseline_preds.append(global_mean)

        # item-based: predict using movie_neighbors where neighbors are movieIds
        ib_pred = predict_item_based(u, m, train_user_ratings, movie_neighbors, k=NN_K)
        # the predict_item_based function falls back to user's mean or 3.0 if denom==0; log those fallbacks:
        if m not in movie_neighbors or all(train_user_ratings.get(u, {}).get(nid) is None for nid, _ in movie_neighbors.get(m, [])):
            # denom likely zero -> fallback used
            fallback_item_count += 1
        item_preds.append(ib_pred)

        # user-based: predict using user_neighbors (neighbors keyed by neighbor userId)
        ub_pred = predict_user_based(u, m, train_user_ratings, user_neighbors, user_means_map, global_mean, k=NN_K)
        if u not in user_neighbors or all(train_user_ratings.get(nid, {}).get(m) is None for nid, _ in user_neighbors.get(u, [])):
            fallback_user_count += 1
        user_preds.append(ub_pred)

    item_metrics = evaluate(item_preds, truths)
    user_metrics = evaluate(user_preds, truths)
    baseline_metrics = evaluate(baseline_preds, truths)

    def fmt_metrics(metrics):
        if metrics["rmse"] is None:
            return "n/a"
        return f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"

    print("\n=== EVALUATION RESULTS ===")
    print(f"Test rows: {len(truths)}")
    print("Baseline (global mean):", fmt_metrics(baseline_metrics))
    print("Item-based CF (neighbors from ALS+Annoy trained on 80%):", fmt_metrics(item_metrics))
    print("User-based CF (neighbors from ALS+Annoy trained on 80%):", fmt_metrics(user_metrics))
    print()
    print("Fallback (denom==0) counts where predictor used simple fallback:")
    print(f" - item-based fallback count: {fallback_item_count} / {len(truths)}")
    print(f" - user-based fallback count: {fallback_user_count} / {len(truths)}")
    print("\nNeighbor DB files written to:")
    print(" -", OUT_MOVIE_NEIGH_DB)
    print(" -", OUT_USER_NEIGH_DB)
    print("\nDone.")

if __name__ == "__main__":
    main()
