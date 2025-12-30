"""CLI helper to build the SQLite index database from a multistream index dump."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.wikipedia_utils import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_PATH,
    build_index_database,
)


def create_wiki_database() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SQLite title + FTS5 index for the Wikipedia dump."
    )
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help="Path to enwiki multistream index .bz2 file (default: %(default)s)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Where to create the SQLite database (default: %(default)s)",
    )
    args = parser.parse_args()

    build_index_database(args.index_path, args.db_path)
    print(f"Database ready at {Path(args.db_path).resolve()}")


if __name__ == "__main__":
    create_wiki_database()


