import sqlite3
from pathlib import Path

from utils.parsing_utils import process_movie

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "movies.db"
MOVIE_LIST_PATH = DATA_DIR / "movie_list.txt"

def _connect():
    """Return a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    """Create tables for movies, directors, genres, and their relationships."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS movies (
                page_title TEXT PRIMARY KEY,
                movie_title TEXT ,
                classification TEXT,
                budget INTEGER,
                box_office INTEGER,
                release_date DATE,
                running_time TEXT,
                rotten_tomatoes INTEGER,
                metacritic INTEGER
            );

            CREATE TABLE IF NOT EXISTS directors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS genres (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS movie_directors (
                page_title TEXT NOT NULL,
                director_id INTEGER NOT NULL,
                PRIMARY KEY (page_title, director_id),
                FOREIGN KEY (page_title) REFERENCES movies(page_title) ON DELETE CASCADE,
                FOREIGN KEY (director_id) REFERENCES directors(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS movie_genres (
                page_title TEXT NOT NULL,
                genre_id INTEGER NOT NULL,
                PRIMARY KEY (page_title, genre_id),
                FOREIGN KEY (page_title) REFERENCES movies(page_title) ON DELETE CASCADE,
                FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def add_movie_to_database(movie_info):
    """Insert or update a single movie row."""
    if not movie_info or not movie_info.get("title"):
        return

    # Normalize list fields from the parser.
    directors = movie_info.get("director") or []
    genres = movie_info.get("genre") or []
    page_title = movie_info.get("page_title", "")

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO movies (
                page_title,
                movie_title,
                classification,
                budget,
                box_office,
                release_date,
                running_time,
                rotten_tomatoes,
                metacritic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page_title,
                movie_info.get("title", ""),
                movie_info.get("classification", ""),
                movie_info.get("budget"),
                movie_info.get("box_office"),
                movie_info.get("release_date"),
                movie_info.get("running_time", ""),
                movie_info.get("rotten_tomatoes"),
                movie_info.get("metacritic"),
            ),
        )

        # Ensure director and genre lookup values exist.
        director_ids = []
        for name in directors:
            cleaned = (name or "").strip()
            if not cleaned:
                continue
            cur.execute(
                "INSERT OR IGNORE INTO directors (name) VALUES (?)",
                (cleaned,),
            )
            cur.execute("SELECT id FROM directors WHERE name = ?", (cleaned,))
            row = cur.fetchone()
            if row:
                director_ids.append(row[0])

        genre_ids = []
        for name in genres:
            cleaned = (name or "").strip()
            if not cleaned:
                continue
            cur.execute(
                "INSERT OR IGNORE INTO genres (name) VALUES (?)",
                (cleaned,),
            )
            cur.execute("SELECT id FROM genres WHERE name = ?", (cleaned,))
            row = cur.fetchone()
            if row:
                genre_ids.append(row[0])

        # Re-link relationships (INSERT OR REPLACE on movies triggers a delete,
        # so associations are cleared via ON DELETE CASCADE).
        for director_id in director_ids:
            cur.execute(
                "INSERT OR IGNORE INTO movie_directors (page_title, director_id) VALUES (?, ?)",
                (page_title, director_id),
            )
        for genre_id in genre_ids:
            cur.execute(
                "INSERT OR IGNORE INTO movie_genres (page_title, genre_id) VALUES (?, ?)",
                (page_title, genre_id),
            )

        conn.commit()
    finally:
        conn.close()


def create_movie_database():

    # Initialize database
    initialize_database()

    with MOVIE_LIST_PATH.open("r") as f:
        movie_list = f.readlines()

    for movie in movie_list:
        movie_info = process_movie(movie.strip())
        add_movie_to_database(movie_info)
        print(f"Added {movie.strip()} to database")


if __name__ == "__main__":
    create_movie_database()