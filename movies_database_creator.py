import csv
import sqlite3


# DATABASE TABLE STRUCTURE

# TABLE: movies
#   - Movie ID (Primary Key)
#   - Movie Title
# Each line contains one Movie ID, with its corresponding title.

# TABLE: genres
#   - Movie ID (Foreign Key: references Movie ID from the movies table)
#   - Genre
# Each line contains a Movie ID and one of its genres.
# There can be multiple entries with the same Movie ID to contain multiple genres.

# TABLE: ratings
#   - User ID
#   - Movie ID (Foreign Key: references Movie ID from the movies table)
#   - Rating
#   - Timestamp
# Each line shows a User ID and one Movie ID that the user rated. 
# Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
# There can be multiple entries by the same User ID to rate multiple different movies.



# VALID GENRES:
#    "Action",
#    "Adventure",
#    "Animation",
#    "Children's",
#    "Comedy",
#    "Crime",
#    "Documentary",
#    "Drama",
#    "Fantasy",
#    "Film-Noir",
#    "Horror",
#    "Musical",
#    "Mystery",
#    "Romance",
#    "Sci-Fi",
#    "Thriller",
#    "War",
#    "Western",
#    "(no genres listed)"


db_file = "db-files/movies.db"       # SQLite database file to create


# --- CONNECT TO SQLITE DATABASE ---
conn = sqlite3.connect(db_file)
cursor = conn.cursor()


# --- CREATE TABLE IN SQLITE DATABASE ---
cursor.execute("""DROP TABLE IF EXISTS movies;""")
cursor.execute("""DROP TABLE IF EXISTS genres;""")
cursor.execute("""DROP TABLE IF EXISTS ratings;""")


cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        movieId INTEGER PRIMARY KEY,
        title TEXT
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS genres (
        movieId INTEGER REFERENCES movies(movieId),
        genre TEXT
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        userId INTEGER,
        movieId INTEGER REFERENCES movies(movieId),
        rating REAL,
        timestamp INTEGER
    );
""")

# ============================================================
#   MOVIES (and genres)
# ============================================================

# --- OPEN CSV AND INSERT ROWS  ---
movies_csv_file = "csv-files/movies.csv"
movies_rows = []
genres_rows = []
with open(movies_csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    # If CSV has a header row, skip it:
    header = next(reader)
    # If no header, remove the above line.

    for line in reader:
        movieId = int(line[0])
        title = str(line[1])
        movies_rows.append((movieId, title))

        genres = str(line[2]).split("|")
        for genre in genres:
            genres_rows.append((movieId, genre))

# Bulk insert
cursor.executemany("""
    INSERT INTO movies (movieId, title)
    VALUES (?, ?);
""", movies_rows)

cursor.executemany("""
    INSERT INTO genres (movieId, genre)
    VALUES (?, ?);
""", genres_rows)

# Commit changes
conn.commit()
movies_rows.clear()
genres_rows.clear()

print("Finished parsing movies and genres")


# ============================================================
#   RATINGS
# ============================================================

# --- OPEN CSV AND INSERT ROWS  ---
ratings_csv_file = "csv-files/ratings.csv"
ratings_rows = []
with open(ratings_csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    # If CSV has a header row, skip it:
    header = next(reader)
    # If no header, remove the above line.

    for line in reader:
        userId = int(line[0])
        movieId = int(line[1])
        rating = float(line[2])
        timestamp = int(line[3])

        ratings_rows.append((userId, movieId, rating, timestamp))

# Bulk insert
cursor.executemany("""
    INSERT INTO ratings (userId, movieId, rating, timestamp)
    VALUES (?, ?, ?, ?);
""", ratings_rows)

# Commit changes
conn.commit()
ratings_rows.clear()

print("Finished parsing ratings")



# ============================================================
#   DONE
# ============================================================

conn.close()
print("Success. All done.")