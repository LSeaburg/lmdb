import re
from pathlib import Path
from typing import List

from utils.wikipedia_utils import get_wiki_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MOVIE_LIST_PATH = DATA_DIR / "movie_list.txt"

# Instead of taking the Lists of American films page, we'll just hardcode the years
def process_year_list():
    year_list = ["List of American films of the 1890s"]
    for year in range(1900, 2025):
        year_list.append(f"List of American films of {year}")

    return year_list

# Takes in a page title and returns a list of movies in that year
def process_year(page_title: str) -> List[str]:
    """
    Parse a yearly "List of American films of XXXX" wikitext page and
    return the linked movie titles from the tables.
    """
    wikitext, _ = get_wiki_text(page_title)
    if not wikitext:
        return []

    # Movie titles appear as the first cell in each table row and are
    # italicized links like: | ''[[Title (year film)|Display]]'' || ...
    pattern = re.compile(
        r"^\|\s*''+\[\[([^|\]]+)(?:\|[^\]]*)?\]\]''+",
        flags=re.MULTILINE,
    )

    movies: List[str] = []
    seen = set()

    for match in pattern.finditer(wikitext):
        title = match.group(1).strip()
        if title and title not in seen:
            seen.add(title)
            movies.append(title)

    return movies


def create_list_of_movies():
    list_of_years = process_year_list()

    movie_list = []
    for year in list_of_years:
        movie_list.extend(process_year(year))

    # Write movie_list to a file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MOVIE_LIST_PATH.open("w") as f:
        for movie in movie_list:
            f.write(movie + "\n")


if __name__ == "__main__":
    create_list_of_movies()