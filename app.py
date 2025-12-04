# app.py

from flask import Flask, jsonify, request, render_template_string
from threading import Lock
import threading
import os
import sys
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import html

app = Flask(__name__)
_state_lock = Lock()

# ================================================================================================================
# Flask Related Global Variables - BEGIN

# ----- Application state (global) -----
# app_state: one of 'initial_loading', 'logged_out', 'loading_user', 'logged_in'
app_state = "initial_loading"
# status values: WAITING / IN PROGRESS / DONE
status = {
    "movies": "WAITING",
    "movie_neighbors": "WAITING",
    "user_neighbors": "WAITING",
    "user_means": "WAITING",
}
current_user = None  # integer user id when loading or logged in
error_message = ""   # shown in logged_out if backend returns an error
# Panels shown when logged in (will contain HTML strings)
panels = {
    "items_rated": "",
    "reco_user": "",
    "reco_item": "",
    "reco_similar_users": "",
}

# Flask Related Global Variables - END
# ================================================================================================================
#
#
# ================================================================================================================
# Recommender Related Global Variables - BEGIN

# ---------------- Configuration ----------------
MOVIES_DB = "db-files/movies.db"
MOVIE_NEIGHBORS_DB = "db-files/movies_nearest_neighbors.db"
USER_NEIGHBORS_DB = "db-files/users_nearest_neighbors.db"

K_USED = 50        # neighbors to use for predictions
TOP_N = 10         # recommendations to return
TARGET_USER = 1    # demo user id
SIMILAR_TOP_K = 10 # how many similar users to show in diagnostics
SHOW_TOP_SHARED = 5
MIN_OVERLAP_FOR_PEARSON = 2
# -----------------------------------------------

# Global variables that will be populated by the loader thread
user_ratings: Dict[int, Dict[int, float]] = {}
movies_map: Dict[int, str] = {}
movie_neighbors: Dict[int, List[Tuple[int, float]]] = {}
user_neighbors: Dict[int, List[Tuple[int, float]]] = {}
user_means: Dict[int, float] = {}
global_mean: float = 3.0

# Recommender Related Global Variables - END
# ================================================================================================================
#
#
# ================================================================================================================
# Flask Related Functions - BEGIN


# ----- Helper functions the recommendation code can call directly -----
# Implementation note: Due to my own unfamiliarity with flask, and the 
# fact that I worked on the project by myself, some Flask related code 
# was generated with the assistance of large language models to ensure 
# the web interface looks and works as intended. This includes the 
# Flask related functions within this section, and Flask related code
# later in the implementation of this file.
def set_status(key: str, value: str):
    """Set one of the 4 loading statuses.
    key: "movies" | "movie_neighbors" | "user_neighbors" | "user_means"
    value: "WAITING" | "IN PROGRESS" | "DONE" (or any string)
    """
    with _state_lock:
        if key not in status:
            raise KeyError(f"Unknown status key: {key}")
        status[key] = value

def set_status_movies(val: str): set_status("movies", val)
def set_status_movie_neighbors(val: str): set_status("movie_neighbors", val)
def set_status_user_neighbors(val: str): set_status("user_neighbors", val)
def set_status_user_means(val: str): set_status("user_means", val)

def goto_initial_loading():
    global app_state, current_user, error_message
    with _state_lock:
        app_state = "initial_loading"
        current_user = None
        error_message = ""

def goto_logged_out():
    global app_state, current_user, error_message, panels
    with _state_lock:
        app_state = "logged_out"
        current_user = None
        error_message = ""
        # Keep statuses as-is (they reflect dataset loading); panels cleared
        for k in panels:
            panels[k] = ""

def start_loading_user(user_id: int):
    global app_state, current_user, error_message
    with _state_lock:
        app_state = "loading_user"
        current_user = int(user_id)
        error_message = ""

def set_logged_in_user(user_id: int):
    """Call this when backend finished loading user data and user is fully logged-in."""
    global app_state, current_user, error_message
    with _state_lock:
        app_state = "logged_in"
        current_user = int(user_id)
        error_message = ""

def logout_user():
    goto_logged_out()

def login_failed(msg: str):
    """Call this to indicate login failed and show message on logged_out screen."""
    global app_state, error_message, current_user
    with _state_lock:
        app_state = "logged_out"
        error_message = str(msg)
        current_user = None

def update_panel(panel_key: str, text: str):
    """Update one of the 4 panels shown in the logged-in state.
    panel_key: items_rated | reco_user | reco_item | reco_similar_users
    text: HTML string (or plain text) to display in the panel
    """
    if panel_key not in panels:
        raise KeyError("Invalid panel key")
    with _state_lock:
        panels[panel_key] = str(text)

# ----- Placeholder backend API you can replace / override by pasting your code below -----
# Replace the existing backend_login(...) with this version:

def backend_login(user_id: int):
    """
    Called when the user presses Log In.
    - Validates user_id
    - Ensures the main datasets have been loaded
    - Moves UI to loading_user
    - Spawns a background thread to compute recommendations and populate the UI panels
    - When done, sets the UI to logged_in via set_logged_in_user(...)
    """
    try:
        uid = int(user_id)
    except Exception:
        return {"success": False, "error": "User ID must be an integer."}
    if uid < 0:
        return {"success": False, "error": "User ID must be >= 0."}

    # Ensure the global data loaded by the startup loader is present
    with _state_lock:
        data_loaded = bool(user_ratings) and bool(movies_map)

    if not data_loaded:
        return {"success": False, "error": "Dataset still loading. Please wait for initial loading to finish."}

    # Put UI into loading_user state
    start_loading_user(uid)

    # Background worker that computes recommendations and populates the panels
    def _compute_and_populate(u):
        try:
            # Compute recommendations & similar-user diagnostics for this user
            item_recs = recommend_top_n_item_based(u, user_ratings, movie_neighbors, movies_map, top_n=TOP_N, k=K_USED)
            user_recs = recommend_top_n_user_based(u, user_ratings, user_neighbors, user_means, global_mean, movies_map, top_n=TOP_N, k=K_USED)
            sims = recommend_similar_users(u, user_ratings, user_neighbors, movies_map, top_k=SIMILAR_TOP_K)

            # ===== Items rated table =====
            items_rows = []
            ur = user_ratings.get(u, {})
            if ur:
                # sort rated items by rating desc then movieId asc
                for mid, r in sorted(ur.items(), key=lambda x: (-x[1], x[0])):
                    title = movies_map.get(mid, "")
                    items_rows.append((mid, title, r))
            # create HTML table
            if items_rows:
                th = "<tr><th>Movie ID</th><th>Movie Title</th><th>Rating</th></tr>"
                tb = "".join(f"<tr><td>{mid}</td><td>{html.escape(title)}</td><td>{r:.1f}</td></tr>"
                             for (mid, title, r) in items_rows)
                items_html = f"<table><thead>{th}</thead><tbody>{tb}</tbody></table>"
            else:
                items_html = "<div>(no ratings)</div>"

            # ===== User-based recommendations table (will be displayed in right column top) =====
            if user_recs:
                th = "<tr><th>Movie ID</th><th>Movie Title</th><th>Predicted Score</th></tr>"
                # Round predicted score to nearest 0.5 and display with one decimal
                tb = "".join(
                    f"<tr><td>{mid}</td><td>{html.escape(title) if title else ''}</td><td>{round(score*2)/2:.1f}</td></tr>"
                    for (mid, title, score) in user_recs)
                reco_user_html = f"<table><thead>{th}</thead><tbody>{tb}</tbody></table>"
            else:
                reco_user_html = "<div>(no recommendations)</div>"

            # ===== Item-based recommendations table (right column middle) =====
            if item_recs:
                th = "<tr><th>Movie ID</th><th>Movie Title</th><th>Predicted Score</th></tr>"
                # Round predicted score to nearest 0.5 and display with one decimal
                tb = "".join(
                    f"<tr><td>{mid}</td><td>{html.escape(title) if title else ''}</td><td>{round(score*2)/2:.1f}</td></tr>"
                    for (mid, title, score) in item_recs)
                reco_item_html = f"<table><thead>{th}</thead><tbody>{tb}</tbody></table>"
            else:
                reco_item_html = "<div>(no recommendations)</div>"

            # ===== Similar users diagnostics table (right column bottom) =====
            if sims:
                # Removed '#' column and removed 'Similarity' column as requested
                th = "<tr><th>User ID</th><th>Overlap</th><th>Shared movies</th></tr>"

                def format_shared(shared_list, uid_val, nid_val):
                    """
                    shared_list: list of (mid, title, ra, rb)
                    uid_val: current logged-in user id (A)
                    nid_val: neighbor/recommended user id (B)
                    returns HTML: a scrollable container with a nested table of all shared movies
                    """
                    if not shared_list:
                        return "<div class='shared-scroll' data-nid='{}' style='min-height:20px; max-height:80px; overflow:auto;'></div>".format(html.escape(str(nid_val)))
                    rows = "".join(
                        f"<tr><td>{mid}</td><td>{html.escape(title)}</td><td>{ra:.1f}</td><td>{rb:.1f}</td></tr>"
                        for (mid, title, ra, rb) in shared_list
                    )
                    # build the header labels using the actual user ids (escaped)
                    rating_a_label = f"Rating from User {html.escape(str(uid_val))}"
                    rating_b_label = f"Rating from User {html.escape(str(nid_val))}"
                    nested = (
                        "<div class='shared-scroll' data-nid='{nid}' style='max-height:200px; overflow:auto;'>"
                        "<table class='nested-table'><thead><tr>"
                        f"<th>Movie ID</th><th>Title</th><th>{rating_a_label}</th><th>{rating_b_label}</th></tr></thead>"
                        f"<tbody>{rows}</tbody></table>"
                        "</div>"
                    ).format(nid=html.escape(str(nid_val)))
                    return nested

                tb_rows = []
                for (nid, sim_val, overlap, pearson, shared) in sims:
                    # NOTE: `shared` now contains ALL overlapping movies (see recommend_similar_users)
                    shared_html = format_shared(shared, u, nid)
                    tb_rows.append(f"<tr>"
                                   f"<td>{nid}</td>"
                                   f"<td>{overlap}</td>"
                                   f"<td>{shared_html}</td>"
                                   f"</tr>")
                tb = "".join(tb_rows)
                sims_html = f"<table><thead>{th}</thead><tbody>{tb}</tbody></table>"
            else:
                sims_html = "<div>(no similar users)</div>"

            # Update the UI panels (thread-safe; functions use _state_lock internally)
            update_panel("items_rated", items_html)
            update_panel("reco_user", reco_user_html)
            update_panel("reco_item", reco_item_html)
            update_panel("reco_similar_users", sims_html)

            # Finally move UI to logged_in
            set_logged_in_user(u)

        except Exception as e:
            # On error, present readable message in the logged_out state
            # (login_failed will set app_state back to logged_out and show the message)
            login_failed(f"Failed to prepare user data: {e}")

    # Start the background thread and return immediate success to the front-end
    thr = threading.Thread(target=_compute_and_populate, args=(uid,), daemon=True)
    thr.start()
    return {"success": True}


# Flask Related Functions - END
# ================================================================================================================
#
#
# ================================================================================================================
# Recommender Related Functions - BEGIN

# -----------------------
# Shared Data / Loaders
# -----------------------
# These load everything once into memory so all recommendation functions reuse the same data.
# Implementation note: Due to the massive size of the data set, and because this
# code is used in the Flask app web interface, the assistance of large language 
# models was used. This includes the loading and recommendation related code, 
# multithreading, and more. This can be observed throughout the Python file.

def load_ratings_and_titles(movies_db_path: str) -> Tuple[Dict[int, Dict[int, float]], Dict[int, str]]:
    """
    Load ratings and movie titles from movies.db into memory.
    Returns:
      user_ratings: dict[userId] -> { movieId: rating }
      movies_map: dict[movieId] -> title
    """
    conn = sqlite3.connect(movies_db_path)
    cur = conn.cursor()

    movies_map: Dict[int, str] = {}
    try:
        cur.execute("SELECT movieId, title FROM movies")
        for mid, title in cur.fetchall():
            try:
                movies_map[int(mid)] = str(title)
            except Exception:
                pass
    except sqlite3.OperationalError:
        # movies table missing -> empty map
        pass

    user_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
    try:
        cur.execute("SELECT userId, movieId, rating FROM ratings")
        for u, m, r in cur.fetchall():
            try:
                user_ratings[int(u)][int(m)] = float(r)
            except Exception:
                pass
    except sqlite3.OperationalError:
        # ratings table missing -> empty ratings
        pass

    conn.close()
    return dict(user_ratings), movies_map

def load_neighbors_db(neighbors_db_path: str) -> Dict[int, List[Tuple[int, float]]]:
    """
    Load neighbors table (rigid schema) into memory.
    Assumes table: neighbors(id, neighbor_id, similarity)
    Returns: dict id -> list of (neighbor_id, similarity), sorted by |similarity| desc.
    """
    mapping: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
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
        # if DB missing or corrupt, return empty dict
        return {}

    # sort each list by absolute similarity descending
    for i in mapping:
        mapping[i].sort(key=lambda t: -abs(t[1]))
    return dict(mapping)

# -----------------------
# Utilities
# -----------------------

def compute_user_means(user_ratings: Dict[int, Dict[int, float]]) -> Tuple[Dict[int, float], float]:
    """
    Compute per-user mean rating and global mean.
    """
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

# -----------------------
# ITEM-BASED RECOMMENDER
# -----------------------
# Section: Item-based CF (preload movie neighbors -> fast prediction)
# Functions: predict_item_based(), recommend_top_n_item_based()

def predict_item_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       movie_neighbors: Dict[int, List[Tuple[int, float]]],
                       k: int = K_USED) -> float:
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

def recommend_top_n_item_based(user_id: int,
                               user_ratings: Dict[int, Dict[int, float]],
                               movie_neighbors: Dict[int, List[Tuple[int, float]]],
                               movies_map: Dict[int, str],
                               top_n: int = TOP_N,
                               k: int = K_USED) -> List[Tuple[int, Optional[str], float]]:
    all_movies = set(movies_map.keys())
    for _, rs in user_ratings.items():
        all_movies.update(rs.keys())
    rated = set(user_ratings.get(user_id, {}).keys())
    candidates = sorted(all_movies - rated)
    scored = []
    for m in candidates:
        p = predict_item_based(user_id, m, user_ratings, movie_neighbors, k=k)
        scored.append((m, p))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    return [(mid, movies_map.get(mid), score) for mid, score in top]

# -----------------------
# USER-BASED RECOMMENDER
# -----------------------
def predict_user_based(user_id: int,
                       movie_id: int,
                       user_ratings: Dict[int, Dict[int, float]],
                       user_neighbors: Dict[int, List[Tuple[int, float]]],
                       user_means: Dict[int, float],
                       global_mean: float,
                       k: int = K_USED) -> float:
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

def recommend_top_n_user_based(user_id: int,
                               user_ratings: Dict[int, Dict[int, float]],
                               user_neighbors: Dict[int, List[Tuple[int, float]]],
                               user_means: Dict[int, float],
                               global_mean: float,
                               movies_map: Dict[int, str],
                               top_n: int = TOP_N,
                               k: int = K_USED) -> List[Tuple[int, Optional[str], float]]:
    all_movies = set(movies_map.keys())
    for _, rs in user_ratings.items():
        all_movies.update(rs.keys())
    rated = set(user_ratings.get(user_id, {}).keys())
    candidates = sorted(all_movies - rated)
    scored = []
    for m in candidates:
        p = predict_user_based(user_id, m, user_ratings, user_neighbors, user_means, global_mean, k=k)
        scored.append((m, p))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    return [(mid, movies_map.get(mid), score) for mid, score in top]

# -----------------------
# SIMILAR USER DIAGNOSTICS
# -----------------------
def pearson_on_overlap(ratings_a: Dict[int, float], ratings_b: Dict[int, float], min_overlap: int = MIN_OVERLAP_FOR_PEARSON) -> Optional[float]:
    common = set(ratings_a.keys()).intersection(ratings_b.keys())
    n = len(common)
    if n < min_overlap:
        return None
    xa = [ratings_a[mid] for mid in common]
    xb = [ratings_b[mid] for mid in common]
    mean_a = sum(xa) / n
    mean_b = sum(xb) / n
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for a, b in zip(xa, xb):
        da = a - mean_a
        db = b - mean_b
        num += da * db
        den_a += da * da
        den_b += db * db
    if den_a <= 0 or den_b <= 0:
        return None
    return num / ((den_a ** 0.5) * (den_b ** 0.5))

def get_top_shared_movies(user_ratings: Dict[int, Dict[int, float]],
                          uid_a: int, uid_b: int, movies_map: Dict[int, str],
                          top_n: Optional[int] = None) -> List[Tuple[int, str, float, float]]:
    """
    Returns shared movies between uid_a and uid_b.

    If top_n is None => return ALL shared movies (unsliced).
    Otherwise return up to top_n items (sorted by average rating desc, tie-break by smaller diff).
    """
    ra = user_ratings.get(uid_a, {})
    rb = user_ratings.get(uid_b, {})
    common = set(ra.keys()).intersection(rb.keys())
    rows = []
    for mid in common:
        a = ra[mid]; b = rb[mid]
        avg = (a + b) / 2.0
        diff = abs(a - b)
        rows.append((mid, movies_map.get(mid, ""), a, b, avg, diff))
    rows.sort(key=lambda t: (-t[4], t[5]))
    if top_n is None:
        sliced = rows
    else:
        sliced = rows[:top_n]
    return [(mid, title, ra[mid], rb[mid]) for (mid, title, _, _, _, _) in sliced]

def recommend_similar_users(target_user: int,
                            user_ratings: Dict[int, Dict[int, float]],
                            user_neighbors: Dict[int, List[Tuple[int, float]]],
                            movies_map: Dict[int, str],
                            top_k: int = SIMILAR_TOP_K) -> List[Tuple[int, float, int, Optional[float], List[Tuple[int, str, float, float]]]]:
    neighs = user_neighbors.get(target_user, [])[:top_k]
    results = []
    for nid, sim in neighs:
        ra = user_ratings.get(target_user, {})
        rb = user_ratings.get(nid, {})
        overlap = len(set(ra.keys()).intersection(rb.keys()))
        pearson = pearson_on_overlap(ra, rb)
        # REQUEST ALL shared movies here (top_n=None) so the UI can show every overlapping movie
        shared = get_top_shared_movies(user_ratings, target_user, nid, movies_map, top_n=None) if overlap > 0 else []
        results.append((nid, float(sim), overlap, pearson, shared))
    return results

# Recommender Related Functions - END
# ================================================================================================================
#
#
# ----- Flask HTTP API used by the front-end (AJAX) -----
@app.route("/api/status", methods=["GET"])
def api_status():
    with _state_lock:
        return jsonify({
            "app_state": app_state,
            "status": status,
            "current_user": current_user,
            "error_message": error_message,
            "panels": panels,
        })

@app.route("/api/login", methods=["POST"])
def api_login():
    body = request.get_json(force=True)
    user_id = body.get("user_id", None)
    if user_id is None:
        return jsonify({"success": False, "error": "Missing user_id"}), 400
    # attempt to call backend_login (user is expected to paste/replace that function)
    try:
        result = backend_login(user_id)
    except Exception as e:
        # If backend_login raises, return error
        return jsonify({"success": False, "error": f"backend_login error: {e}"}), 500
    if not isinstance(result, dict):
        return jsonify({"success": False, "error": "backend_login must return a dict."}), 500
    if result.get("success"):
        # UI moves to loading_user. The backend_login may already have called start_loading_user.
        return jsonify({"success": True})
    else:
        msg = result.get("error", "Login failed")
        login_failed(msg)
        return jsonify({"success": False, "error": msg})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    logout_user()
    return jsonify({"success": True})

@app.route("/api/set_status", methods=["POST"])
def api_set_status():
    body = request.get_json(force=True)
    key = body.get("key")
    value = body.get("value")
    if key not in status:
        return jsonify({"success": False, "error": "invalid key"}), 400
    set_status(key, value)
    return jsonify({"success": True, "status": status})

@app.route("/api/update_panel", methods=["POST"])
def api_update_panel():
    body = request.get_json(force=True)
    panel = body.get("panel")
    text = body.get("text", "")
    if panel not in panels:
        return jsonify({"success": False, "error": "invalid panel"}), 400
    update_panel(panel, text)
    return jsonify({"success": True, "panels": panels})

@app.route("/api/goto_logged_out", methods=["POST"])
def api_goto_logged_out():
    goto_logged_out()
    return jsonify({"success": True})

@app.route("/")
def index():
    # Inline HTML/CSS/JS template - note: removed duplicate logout in top of card; kept single topbar logout
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Movie Recommender</title>
  <style>
    /* Barebones styling as requested */
    body { font-family: Arial, Helvetica, sans-serif; margin: 18px; background: #f6f7fb; color: #111; }
    .card { background: white; border-radius: 8px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 12px; }
    .muted { color: #666; font-size: 0.95rem; }
    input[type=number] { padding: 6px 8px; font-size: 1rem; width: 160px; }
    button { padding: 8px 12px; font-size: 1rem; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
    button.primary { background: #1677ff; color: white; border-color: #0f62d0; }
    .label { font-weight: 600; margin-bottom: 6px; display:block; }
    .panel { background:#f4f6fb; padding:10px; border-radius:6px; min-height:60px; white-space: normal; overflow:auto; }
    .status-line { margin:6px 0; }
    .small { font-size:0.9rem; color:#444; }
    .error { color: #b00020; margin-top:8px; }
    .topbar { display:flex; gap:12px; align-items:center; margin-bottom:12px; }

    /* table styling inside panels */
    .panel table { width:100%; border-collapse: collapse; }
    .panel th, .panel td { padding: 6px 8px; border-bottom: 1px solid #e6e9ef; text-align: left; vertical-align: top; }
    .panel thead th { background: #eef2fb; font-weight: 700; }
    .nested-table { margin-top:6px; border: 1px solid #e6e9ef; border-radius:6px; overflow:auto; width:100%; }
    .nested-table th, .nested-table td { padding:4px 6px; font-size:0.9rem; border-bottom:1px solid #eee; }

    /* scrollable shared movies container */
    .shared-scroll { border: 1px solid #e6e9ef; border-radius:6px; padding:6px; background: #fff; }

    /* layout for logged-in panels */
    .logged-in-grid { display:flex; gap:12px; align-items:flex-start; }
    .left-col { flex-basis:50%; flex-grow:0; }
    .right-col { flex-basis:50%; display:flex; flex-direction:column; gap:12px; flex-grow:0; }
    .right-col .panel-box { background: transparent; } /* container for each right panel */
    .full-width { width:100%; margin-top:12px; }

    /* top logout button styling (aligned to the right) */
    #btn-logout-top { margin-left: auto; }
  </style>
</head>
<body>
  <div class="topbar">
    <h2>Movie Recommender Web Interface</h2>
    <div class="muted"></div>
    <!-- Single top logout (hidden unless logged in) -->
    <button id="btn-logout-top" style="display:none;" onclick="logout()">Log Out</button>
  </div>

  <!-- Initial loading state -->
  <div id="view-initial-loading" class="card" style="display:none;">
    <div class="label">Loading dataset. Please wait. This process may take a multiple minutes.</div>
    <div class="status-line">Loading movies & ratings into memory (this step takes the longest): <span id="st-movies">WAITING</span></div>
    <div class="status-line">Loading movie neighbors into memory: <span id="st-movie-neighbors">WAITING</span></div>
    <div class="status-line">Loading user neighbors into memory: <span id="st-user-neighbors">WAITING</span></div>
    <div class="status-line">Computing user means: <span id="st-user-means">WAITING</span></div>
    <div class="small muted">(Status will update as loading is completed)</div>
  </div>

  <!-- Logged out state -->
  <div id="view-logged-out" class="card" style="display:none;">
    <div class="label">Log In</div>
    <div>
      <label class="small label">User ID</label>
      <input id="input-user-id" type="number" step="1" min="0" placeholder="Enter user id (integer)">
      <button id="btn-login" class="primary" onclick="attemptLogin()">Log In</button>
    </div>
    <div id="login-error" class="error" style="display:none;"></div>
  </div>

  <!-- Loading user state -->
  <div id="view-loading-user" class="card" style="display:none;">
    <div class="label" id="loading-user-label">Loading data for User...</div>
    <div class="muted">Please wait while recommendation data is loaded for the user....</div>
  </div>

  <!-- Logged in state -->
  <div id="view-logged-in" style="display:none;">
    <div class="card">
      <div style="display:flex; align-items:center; gap:12px;">
        <div class="label" id="logged-in-as">Logged in as X</div>
        <div style="flex:1"></div>
        <!-- removed duplicate logout here; topbar logout remains -->
      </div>

      <div class="logged-in-grid" style="margin-top:12px;">
        <!-- LEFT: Items rated (50%) -->
        <div class="left-col">
          <div class="small label">Items rated by User <span id="lbl-user-id-in-panel">X</span>...</div>
          <div class="panel" id="panel-items-rated"></div>
        </div>

        <!-- RIGHT: three stacked panels (50%) -->
        <div class="right-col">
          <div class="panel-box">
            <div class="small label">Recommendations using similar user nearest neighbor...</div>
            <div class="panel" id="panel-reco-user"></div>
          </div>

          <div class="panel-box">
            <div class="small label">Recommendations using similar item nearest neighbor...</div>
            <div class="panel" id="panel-reco-item"></div>
          </div>

          <div class="panel-box">
            <div class="small label">Recommendation of other users who share similar preference...</div>
            <div class="panel" id="panel-reco-similar-users"></div>
          </div>
        </div>
      </div>

      <div style="margin-top:10px;">
        <!-- bottom logout (user can also log out from here) -->
        <button onclick="logout()">Log Out</button>
      </div>
    </div>
  </div>

<script>
  // Front-end state polling and UI logic
  let pollInterval = 1000; // ms
  let pollHandle = null;

  // keep a snapshot to avoid unnecessary DOM updates while in logged_in
  window.__lastSnapshot = {
    app_state: null,
    panels_json: null,
    status_json: null,
    current_user: null,
    error_message: null
  };

  function updateUIFromStatus(data) {
    // data: {app_state, status, current_user, error_message, panels}
    const s = data.status || {};
    document.getElementById('st-movies').innerText = s.movies || 'WAITING';
    document.getElementById('st-movie-neighbors').innerText = s.movie_neighbors || 'WAITING';
    document.getElementById('st-user-neighbors').innerText = s.user_neighbors || 'WAITING';
    document.getElementById('st-user-means').innerText = s.user_means || 'WAITING';

    const state = data.app_state || 'logged_out';

    // Show/hide views
    document.getElementById('view-initial-loading').style.display = (state === 'initial_loading') ? 'block' : 'none';
    document.getElementById('view-logged-out').style.display = (state === 'logged_out') ? 'block' : 'none';
    document.getElementById('view-loading-user').style.display = (state === 'loading_user') ? 'block' : 'none';
    document.getElementById('view-logged-in').style.display = (state === 'logged_in') ? 'block' : 'none';

    // Show/hide topbar logout button when logged in
    document.getElementById('btn-logout-top').style.display = (state === 'logged_in') ? 'inline-block' : 'none';

    // Loading user label and logged-in label
    if (state === 'loading_user' || state === 'logged_in') {
      const uid = data.current_user === null ? '' : data.current_user;
      document.getElementById('loading-user-label').innerText = `Loading data for User ${uid}...`;
      document.getElementById('logged-in-as').innerText = `Logged in as ${uid}`;
      document.getElementById('lbl-user-id-in-panel').innerText = `${uid}`;
    }

    // Panels for logged in (render HTML tables)
    const panels = data.panels || {};
    // items, user/item recs update normally
    document.getElementById('panel-items-rated').innerHTML = panels.items_rated || '';
    document.getElementById('panel-reco-user').innerHTML = panels.reco_user || '';
    document.getElementById('panel-reco-item').innerHTML = panels.reco_item || '';

    // For similar-users panel we preserve per-neighbor scroll positions.
    const simEl = document.getElementById('panel-reco-similar-users');
    // Save scrollTop for each existing shared-scroll box keyed by data-nid
    const scrollMap = {};
    simEl.querySelectorAll('.shared-scroll[data-nid]').forEach(el => {
      const nid = el.getAttribute('data-nid');
      if (nid) scrollMap[nid] = el.scrollTop;
    });

    // Replace content for similar-users
    simEl.innerHTML = panels.reco_similar_users || '';

    // After replacement, restore scrollTop for matching data-nid elements
    simEl.querySelectorAll('.shared-scroll[data-nid]').forEach(el => {
      const nid = el.getAttribute('data-nid');
      if (nid && (nid in scrollMap)) {
        try { el.scrollTop = scrollMap[nid]; } catch (e) { /* ignore */ }
      }
    });

    // Error display for logged out
    if (state === 'logged_out' && data.error_message) {
      const el = document.getElementById('login-error');
      el.style.display = 'block';
      el.innerText = data.error_message;
    } else {
      const el = document.getElementById('login-error');
      el.style.display = 'none';
      el.innerText = '';
    }
  }

  async function pollStatusOnce() {
    try {
      const r = await fetch('/api/status');
      const data = await r.json();

      // Decide whether to update the UI.
      // Strategy:
      //  - Always update on first load or if app_state changed.
      //  - When app_state === 'logged_in', only update if current_user, panels, status, or error_message changed.
      //  - For other states (loading, logged_out), update normally.
      const last = window.__lastSnapshot || { app_state: null, panels_json: null, status_json: null, current_user: null, error_message: null };

      let shouldUpdate = false;

      // first-time or app_state change -> update
      if (last.app_state === null || data.app_state !== last.app_state) {
        shouldUpdate = true;
      } else if (data.app_state === 'logged_in') {
        // compare current_user and panels/status content
        const panels_json = JSON.stringify(data.panels || {});
        const status_json = JSON.stringify(data.status || {});
        if (String(data.current_user) !== String(last.current_user)
            || panels_json !== last.panels_json
            || status_json !== last.status_json
            || data.error_message !== last.error_message) {
          shouldUpdate = true;
        } else {
          // no change -> skip DOM updates to avoid resets
          shouldUpdate = false;
        }
      } else {
        // other states (e.g., loading, logged_out) - update as normal even if the state string identical,
        // because status labels or errors could have changed; compare status/error to be safe
        const status_json = JSON.stringify(data.status || {});
        if (status_json !== last.status_json || data.error_message !== last.error_message || JSON.stringify(data.panels || {}) !== last.panels_json) {
          shouldUpdate = true;
        } else {
          // if nothing changed, skip
          shouldUpdate = false;
        }
      }

      if (shouldUpdate) {
        updateUIFromStatus(data);
        // update snapshot
        window.__lastSnapshot = {
          app_state: data.app_state,
          panels_json: JSON.stringify(data.panels || {}),
          status_json: JSON.stringify(data.status || {}),
          current_user: data.current_user,
          error_message: data.error_message
        };
      }
    } catch (e) {
      console.error("Failed to fetch status:", e);
    }
  }

  function startPolling() {
    if (pollHandle) clearInterval(pollHandle);
    pollHandle = setInterval(pollStatusOnce, pollInterval);
    pollStatusOnce();
  }

  async function attemptLogin() {
    const input = document.getElementById('input-user-id');
    let uid = input.value;
    if (uid === '') {
      document.getElementById('login-error').style.display = 'block';
      document.getElementById('login-error').innerText = "Please enter a numeric User ID.";
      return;
    }
    // POST to /api/login
    try {
      const resp = await fetch('/api/login', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({user_id: uid})
      });
      const data = await resp.json();
      if (data.success) {
        // UI will move to loading_user; poll will pick that up.
        document.getElementById('login-error').style.display = 'none';
      } else {
        document.getElementById('login-error').style.display = 'block';
        document.getElementById('login-error').innerText = data.error || "Login failed";
      }
    } catch (e) {
      document.getElementById('login-error').style.display = 'block';
      document.getElementById('login-error').innerText = "Network error when attempting login.";
      console.error(e);
    }
  }

  async function logout() {
    await fetch('/api/logout', {method:'POST'});
    // Poll will show the logged_out state
    // clear lastSnapshot so UI will fully re-render on next poll
    window.__lastSnapshot = { app_state: null, panels_json: null, status_json: null, current_user: null, error_message: null };
  }

  // start polling when page loads
  startPolling();
</script>
</body>
</html>
    """)

# Background loader thread that runs the snippet and updates labels as requested
def background_loader():
    global user_ratings, movies_map, movie_neighbors, user_neighbors, user_means, global_mean
    try:
        # UPDATE "Loading movies & ratings into memory: Y" LABEL SO Y = "IN PROGRESS"
        print("Updating UI: Loading movies & ratings -> IN PROGRESS")
        set_status_movies("IN PROGRESS")

        print("Loading movies & ratings into memory from", MOVIES_DB)
        ur, mm = load_ratings_and_titles(MOVIES_DB)

        if not ur:
            print("No ratings found in movies.db — aborting demo.")
            # ensure UI reflects failure
            set_status_movies("DONE")
            # exit entire process since dataset needed
            print("Exiting because no ratings found.")
            os._exit(1)  # use os._exit to reliably stop the server process
            return

        # assign into globals under lock
        with _state_lock:
            user_ratings = ur
            movies_map = mm

        # UPDATE "Loading movies & ratings into memory: Y" LABEL SO Y = "DONE"
        print("Updating UI: Loading movies & ratings -> DONE")
        set_status_movies("DONE")

        # UPDATE "Loading movie neighbors into memory: Y" LABEL SO Y = "IN PROGRESS"
        print("Updating UI: Loading movie neighbors -> IN PROGRESS")
        set_status_movie_neighbors("IN PROGRESS")

        print("Loading movie neighbors into memory from", MOVIE_NEIGHBORS_DB)
        mn = load_neighbors_db(MOVIE_NEIGHBORS_DB)
        with _state_lock:
            movie_neighbors = mn

        # UPDATE "Loading movie neighbors into memory: Y" LABEL SO Y = "DONE"
        print("Updating UI: Loading movie neighbors -> DONE")
        set_status_movie_neighbors("DONE")

        # UPDATE "Loading user neighbors into memory: Y" LABEL SO Y = "IN PROGRESS"
        print("Updating UI: Loading user neighbors -> IN PROGRESS")
        set_status_user_neighbors("IN PROGRESS")

        print("Loading user neighbors into memory from", USER_NEIGHBORS_DB)
        un = load_neighbors_db(USER_NEIGHBORS_DB)
        with _state_lock:
            user_neighbors = un

        # UPDATE "Loading user neighbors into memory: Y" LABEL SO Y = "DONE"
        print("Updating UI: Loading user neighbors -> DONE")
        set_status_user_neighbors("DONE")

        # UPDATE "Computing user means: Y" LABEL SO Y = "IN PROGRESS"
        print("Updating UI: Computing user means -> IN PROGRESS")
        set_status_user_means("IN PROGRESS")

        print("Computing user means ...")
        um, gm = compute_user_means(user_ratings)
        with _state_lock:
            user_means = um
            global_mean = gm

        # UPDATE "Computing user means: Y" LABEL SO Y = "DONE"
        print("Updating UI: Computing user means -> DONE")
        set_status_user_means("DONE")

        # TAKE USER TO THE Logged out state
        print("Loader finished — switching UI to logged_out")
        goto_logged_out()

    except Exception as e:
        print("Exception in background loader:", e, file=sys.stderr)
        # reflect error state in UI; put labels to DONE/ERROR
        try:
            set_status_movies("DONE")
            set_status_movie_neighbors("DONE")
            set_status_user_neighbors("DONE")
            set_status_user_means("DONE")
        except Exception:
            pass
        # exit to be safe
        os._exit(1)

# ----- Run server (development) -----
if __name__ == "__main__":
    # By default start in initial_loading state
    goto_initial_loading()

    # Start background loader but avoid double-start when Flask's reloader runs.
    # Werkzeug sets WERKZEUG_RUN_MAIN in the reloaded child process; only start the loader there.
    should_start_loader = True
    # If the reloader is active, only start once in the reloader child process:
    if "WERKZEUG_RUN_MAIN" in os.environ:
        if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            should_start_loader = False

    if should_start_loader:
        t = threading.Thread(target=background_loader, daemon=True)
        t.start()

    # If you want the app to start at logged_out for testing, uncomment:
    # goto_logged_out()
    app.run(debug=True)
# end of app.py
