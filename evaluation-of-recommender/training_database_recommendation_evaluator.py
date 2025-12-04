# training_database_recommendation_evaluator.py
#
# Evaluates the prediction accuracy of movies_nearest_neighbors_train.db and
# users_nearest_neighbors_train.db. The recommendation code in this file is 
# based off of the recommendation code in app.py. The code here is modified so
# it runs as a standalone recommender with the additional functionality of evaluating
# the accuracy of the recommendations of the test set. Any notes written in the comments 
# of app.py about the implementation, such as how the code was written, applies here as
# well.

import csv
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ---------- USER CONFIG ----------
TRAIN_RATINGS_CSV = "csv-files/ratings_train.csv"
TEST_RATINGS_CSV = "csv-files/ratings_test.csv"
MOVIE_NEIGHBORS_DB = "db-files/movies_nearest_neighbors_train.db"
USER_NEIGHBORS_DB = "db-files/users_nearest_neighbors_train.db"

K_USED = 50  # neighbors to use in prediction (same as in app.py)
# ---------------------------------

def load_ratings_train(path: str) -> Tuple[Dict[int, Dict[int, float]], float, Dict[int, float]]:
    user_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
    total = 0.0
    count = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            raise RuntimeError("Train CSV is empty")
        try:
            int(first[0])
            rows_iter = [first]
        except Exception:
            rows_iter = []
        if rows_iter:
            u = int(rows_iter[0][0]); m = int(rows_iter[0][1]); rt = float(rows_iter[0][2])
            user_ratings[u][m] = rt
            total += rt; count += 1
        for line in reader:
            if not line:
                continue
            try:
                u = int(line[0]); m = int(line[1]); rt = float(line[2])
            except Exception:
                continue
            user_ratings[u][m] = rt
            total += rt; count += 1

    user_means: Dict[int, float] = {}
    for u, rs in user_ratings.items():
        if rs:
            s = sum(rs.values()); c = len(rs)
            user_means[u] = s / c
        else:
            user_means[u] = 0.0
    global_mean = (total / count) if count > 0 else 3.5
    return dict(user_ratings), global_mean, user_means

def load_neighbors_db(path: str) -> Dict[int, List[Tuple[int, float]]]:
    mapping: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT id, neighbor_id, similarity FROM neighbors")
        for idcol, nid, sim in cur.fetchall():
            try:
                iid = int(idcol); nn = int(nid); s = float(sim)
            except Exception:
                continue
            mapping[iid].append((nn, s))
        conn.close()
    except Exception as e:
        print(f"Failed to read neighbors DB {path}: {e}", file=sys.stderr)
        return {}
    for i in mapping:
        mapping[i].sort(key=lambda t: -abs(t[1]))
    return dict(mapping)

def predict_item_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       movie_neighbors: Dict[int, List[Tuple[int, float]]],
                       k: int = K_USED) -> Optional[float]:
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
            return sum(ur.values()) / len(ur)
        return None
    return num / denom

def predict_user_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       user_neighbors: Dict[int, List[Tuple[int, float]]],
                       user_means: Dict[int, float],
                       global_mean: float,
                       k: int = K_USED) -> Optional[float]:
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
        if um is not None:
            return um
        return None
    base = user_means.get(user_id, 0.0)
    return base + (num / denom)

def clamp_and_round_half(x: Optional[float]) -> float:
    if x is None:
        return None
    r = round(x * 2.0) / 2.0
    if r < 0.5: r = 0.5
    if r > 5.0: r = 5.0
    return r

def main():
    print("Evaluator (MAE only, rounded-to-0.5) starting.")
    print(f"Loading training ratings from {TRAIN_RATINGS_CSV} ...")
    user_ratings, global_mean, user_means = load_ratings_train(TRAIN_RATINGS_CSV)
    print(f"Users in train: {len(user_ratings)}, global mean={global_mean:.4f}")

    print("Loading movie neighbor DB:", MOVIE_NEIGHBORS_DB)
    movie_neighbors = load_neighbors_db(MOVIE_NEIGHBORS_DB)
    print(f"Loaded neighbors for {len(movie_neighbors)} movies")
    print("Loading user neighbor DB:", USER_NEIGHBORS_DB)
    user_neighbors = load_neighbors_db(USER_NEIGHBORS_DB)
    print(f"Loaded neighbors for {len(user_neighbors)} users")

    # MAE accumulators
    item_sae = 0.0
    user_sae = 0.0
    total = 0

    print(f"Streaming test rows from {TEST_RATINGS_CSV} ...")
    with open(TEST_RATINGS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            peek = next(reader)
        except StopIteration:
            print("Test CSV empty; exiting.")
            return
        is_header = False
        try:
            int(peek[0])
        except Exception:
            is_header = True

        if not is_header:
            try:
                u = int(peek[0]); m = int(peek[1]); true_rt = float(peek[2])
            except Exception:
                pass
            else:
                p_item = predict_item_based(u, m, user_ratings, movie_neighbors, k=K_USED)
                if p_item is None:
                    p_item = user_means.get(u, None)
                    if p_item is None:
                        p_item = global_mean
                p_item_r = clamp_and_round_half(p_item)

                p_user = predict_user_based(u, m, user_ratings, user_neighbors, user_means, global_mean, k=K_USED)
                if p_user is None:
                    p_user = user_means.get(u, None)
                    if p_user is None:
                        p_user = global_mean
                p_user_r = clamp_and_round_half(p_user)

                item_sae += abs(p_item_r - true_rt)
                user_sae += abs(p_user_r - true_rt)
                total += 1

        for row in reader:
            if not row:
                continue
            try:
                u = int(row[0]); m = int(row[1]); true_rt = float(row[2])
            except Exception:
                continue

            p_item = predict_item_based(u, m, user_ratings, movie_neighbors, k=K_USED)
            if p_item is None:
                p_item = user_means.get(u, None)
                if p_item is None:
                    p_item = global_mean
            p_item_r = clamp_and_round_half(p_item)

            p_user = predict_user_based(u, m, user_ratings, user_neighbors, user_means, global_mean, k=K_USED)
            if p_user is None:
                p_user = user_means.get(u, None)
                if p_user is None:
                    p_user = global_mean
            p_user_r = clamp_and_round_half(p_user)

            item_sae += abs(p_item_r - true_rt)
            user_sae += abs(p_user_r - true_rt)
            total += 1

            if (total % 500000) == 0:
                print(f"Processed {total:,} test rows...")

    if total == 0:
        print("No valid test rows processed. Exiting.")
        return

    item_mae = item_sae / total
    user_mae = user_sae / total

    print("\n=== RESULTS (rounded to 0.5 only) ===")
    print(f"Processed test rows: {total:,}")
    print(f"Item-based CF (rounded to 0.5): MAE={item_mae:.4f}")
    print(f"User-based CF (rounded to 0.5): MAE={user_mae:.4f}")

if __name__ == "__main__":
    main()
