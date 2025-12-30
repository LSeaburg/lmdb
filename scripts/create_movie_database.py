import sqlite3
from pathlib import Path

from utils.parsing_utils import process_movie

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "movies.db"
MOVIE_LIST_PATH = DATA_DIR / "movie_list.txt"

def initialize_database():
    """Create the movies table if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                title TEXT PRIMARY KEY,
                classification TEXT,
                director TEXT,
                budget INTEGER,
                box_office INTEGER,
                release_date DATE,
                running_time TEXT,
                genre TEXT,
                rotten_tomatoes INTEGER,
                metacritic INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def add_movie_to_database(movie_info):
    """Insert or update a single movie row."""
    if not movie_info or not movie_info.get("title"):
        return

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO movies (
                title,
                classification,
                director,
                budget,
                box_office,
                release_date,
                running_time,
                genre,
                rotten_tomatoes,
                metacritic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                movie_info.get("title", ""),
                movie_info.get("classification", ""),
                movie_info.get("director", ""),
                movie_info.get("budget"),
                movie_info.get("box_office"),
                movie_info.get("release_date"),
                movie_info.get("running_time", ""),
                movie_info.get("genre", ""),
                movie_info.get("rotten_tomatoes"),
                movie_info.get("metacritic"),
            ),
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


if __name__ == "__main__":
    create_movie_database()