"""Utility helpers for working with Wikipedia multistream dumps and wikitext."""

from __future__ import annotations

import bz2
import io
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_DB_PATH = str(DATA_DIR / "wiki_articles.db")
DEFAULT_DUMP_PATH = str(DATA_DIR / "enwiki-20251101-pages-articles-multistream.xml.bz2")
DEFAULT_INDEX_PATH = str(
    DATA_DIR / "enwiki-20251101-pages-articles-multistream-index.txt.bz2"
)


def build_index_database(index_path: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Build (or reuse) a SQLite database containing title offsets and an FTS5 index.
    """
    if os.path.exists(db_path):
        print(f"Found existing database: {db_path}")
        return

    print(f"Building SQLite index at {db_path}...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Schema
    c.execute(
        "CREATE TABLE wiki_articles (title TEXT PRIMARY KEY, offset INTEGER, page_id INTEGER)"
    )
    c.execute(
        "CREATE VIRTUAL TABLE wiki_articles_fts USING fts5(title, content='wiki_articles', content_rowid='rowid')"
    )

    batch = []
    total = 0

    with bz2.open(index_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                offset, page_id, title = line.strip().split(":", 2)
            except ValueError:
                continue
            batch.append((title, int(offset), int(page_id)))
            total += 1
            if len(batch) >= 10000:
                c.executemany(
                    "INSERT OR IGNORE INTO wiki_articles(title, offset, page_id) VALUES (?, ?, ?)",
                    batch,
                )
                batch.clear()
                print(f"Inserted {total:,} wiki article titles...")

    if batch:
        c.executemany(
            "INSERT OR IGNORE INTO wiki_articles(title, offset, page_id) VALUES (?, ?, ?)",
            batch,
        )

    print("Populating FTS5 search index...")
    c.execute(
        "INSERT INTO wiki_articles_fts(rowid, title) SELECT rowid, title FROM wiki_articles"
    )

    conn.commit()
    conn.close()
    print(f"Done. Indexed {total:,} wiki article titles.")


def escape_fts_query(text: str) -> str:
    """Escape special FTS5 characters."""
    safe = text.replace('"', '""')
    return f'"{safe}"'


def search_titles(query: str, db_path: str, limit: int = 5) -> List[str]:
    """Search for article titles using FTS5."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    escaped = escape_fts_query(query)
    query_str = f"{escaped} OR {escaped}*"
    results = [
        row[0]
        for row in c.execute(
            "SELECT title FROM wiki_articles_fts WHERE title MATCH ? LIMIT ?",
            (query_str, limit),
        )
    ]
    conn.close()
    return results


def get_offset_for_title(title: str, db_path: str) -> Optional[int]:
    """Return the byte offset for a given title."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT offset FROM wiki_articles WHERE title = ?", (title,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def extract_article_xml(title: str, dump_path: str, offset: Optional[int]) -> Optional[str]:
    """Seek to offset and decompress relevant XML block."""
    if offset is None:
        print(f"No offset found for: {title}")
        return None

    with open(dump_path, "rb") as dump_file:
        dump_file.seek(offset)
        decompressor = bz2.BZ2Decompressor()
        xml_bytes = b""

        for _ in range(40):
            chunk = dump_file.read(1024 * 1024)
            if not chunk:
                break
            xml_bytes += decompressor.decompress(chunk)
            if b"</page>" in xml_bytes:
                break

    return xml_bytes.decode("utf-8", errors="ignore")


def extract_wikitext_from_xml(xml_text: str, title: str) -> Optional[str]:
    """Parse one decompressed XML block to get article text."""
    wrapped = "<root>" + xml_text + "</root>"
    try:
        for _, elem in ET.iterparse(io.StringIO(wrapped), events=("end",)):
            if elem.tag == "page":
                t = elem.find("title")
                if t is not None and t.text == title:
                    text_tag = elem.find("./revision/text")
                    if text_tag is not None and text_tag.text:
                        return text_tag.text
                elem.clear()
    except ET.ParseError:
        return None
    return None


def is_redirect(wikitext: str) -> bool:
    return bool(wikitext and wikitext.strip().lower().startswith("#redirect"))


def resolve_redirect_target(wikitext: str) -> Optional[str]:
    m = re.search(r"#redirect\s*\[\[(.*?)\]\]", wikitext, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_article_follow_redirects(
    title: str, dump_path: str, db_path: str, max_depth: int = 3
) -> Tuple[Optional[str], str]:
    """Extract wikitext, following redirects up to max_depth."""
    for _ in range(max_depth):
        offset = get_offset_for_title(title, db_path)
        xml_block = extract_article_xml(title, dump_path, offset)
        if not xml_block:
            return None, title
        wikitext = extract_wikitext_from_xml(xml_block, title)
        if not wikitext:
            return None, title
        if is_redirect(wikitext):
            target = resolve_redirect_target(wikitext)
            if target and get_offset_for_title(target, db_path) is not None:
                title = target
                continue
            return None, title
        return wikitext, title
    print("Max redirect depth reached.")
    return None, title


def get_wiki_text(
    page_title: str,
    dump_path: str = DEFAULT_DUMP_PATH,
    db_path: str = DEFAULT_DB_PATH,
    max_redirect_depth: int = 3,
) -> Tuple[Optional[str], str]:
    """Convenience helper to resolve a title and fetch its wikitext."""
    return extract_article_follow_redirects(
        page_title,
        dump_path=dump_path,
        db_path=db_path,
        max_depth=max_redirect_depth,
    )
