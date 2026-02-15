# Logan's Movie Database

Uses local Wikipedia and local LLM to create a database that includes originality of film (original, adapted, sequel, reboot, franchise).

## Requirements

This project requires Python 3.9+.

This project also requires a local Ollama install with `gemma3:12b` pulled and `ollama` running.

This project also requires Wikipedia download files in `./data`:
- `enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2`
- `enwiki-YYYYMMDD-pages-articles-multistream-index.txt.bz2`

This project uses `enwiki-20251101-*`, but you can point to other dumps via `DEFAULT_DUMP_PATH` in `utils/wikipedia_utils.py` or CLI args to `create_wiki_database.py`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# ensure the model is pulled locally
ollama pull gemma3:12b
```

## Running

Run from repo root with the data files in `./data`:

```bash
# 1) Index the Wikipedia dump
python scripts/create_wiki_database.py \
  --dump data/enwiki-20251101-pages-articles-multistream.xml.bz2 \
  --index data/enwiki-20251101-pages-articles-multistream-index.txt.bz2

# 2) Build the list of American films
python scripts/create_list_of_movies.py

# 3) Scrape movie articles and build the movie DB (takes ~6 hours on an M4 Mac mini)
python scripts/create_movie_database.py
```

Outputs:
- `data/wiki.db`: index of the Wikipedia multistream dump
- `data/movie_list.txt`: list of American films (pre-2025)
- `data/movies.db`: parsed movie metadata + classifications

Notes:
- Step 3 makes network calls to Wikidata for some Rotten Tomatoes scores (needs internet).
- Most work is local (LLM + local dump); ensure `ollama` is running.

Scope: The movie list targets American films before 2025; exact inclusion follows the category-based collection in `create_list_of_movies.py`.