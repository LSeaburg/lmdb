import re
import requests
import ollama
from datetime import datetime
from typing import Dict, Optional, List

from utils.wikipedia_utils import get_wiki_text

MODEL = "gemma3:12b"

SEQUEL_PROMPT = """
You are classifying a film description into one category (0-5).
Use ONLY the provided text. Do not use outside knowledge.
Identical input always → identical number.
Output only the digit (0-5) on its own line. No extra text.

Decision order (apply the first that matches):

NOTE: Do NOT classify as 1 or 3 because of the phrase “first installment”, 
“launch of a new series”, or “beginning of a trilogy”.  These films should still 
be classifed as 4 (adapted) or 5 (original).
For example, a film described as “based on [a novel] and the first installment 
of a trilogy” is still classified under Rule 4 (adapted screenplay).

1 = Sequel / Prequel within the SAME film continuity.
   Apply only when the text explicitly says:
   “sequel to [film]”, “prequel to [film]”, “follow-up to [film]”,
   “direct continuation”, “picks up after [film]”, or
   “set before/after the events of [film]”.
   These must describe continuation of an *earlier* or *later* story.

2 = Remake / Reboot / Reimagining.
   Must contain the words “remake”, “reboot”, or “reimagining”.

3 = Existing IP / Franchise / Universe entry (not 1 or 2).
   Mentions a known universe, franchise, or character property
   (e.g. “in the Marvel universe”, “based on the DC character”)
   but not a direct sequel/prequel.
   Franchise must have existed at the time of the film's release.

4 = Adapted screenplay (not 1-3).
   Says “based on” or “adapted from” a novel, book,
   story, play, comic, article, memoir, biography, prior screenplay,
   show, or other written/produced work.
   DO NOT classify as adapted if the text only mentions “personal experiences”,
   “semi-autobiographical elements”, “true events”, or the lives of real people — those go to (5).

5 = Original screenplay (none of 1-4).
   Original story NOT derived from any prior work - if a prior novel exists classify as 4, adapted.
   Includes films described as “based on true events”, “based on real people”,
   or “inspired by the director's life or history”.
   Also includes parodies, homages, and film à clef works.

0 = Insufficient information or not about a film.

If multiple rules apply, stop at the first valid match (1>2>3>4>5).
Print only the digit.
"""

category = {
    0: "undetermined",
    1: "sequel",
    2: "remake",
    3: "franchise",
    4: "adapted",
    5: "original",
}

# Send a query to the model and return the response
def _query_model(model, query):
    result = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": query}],
        stream=False,  # disable streaming
        # keep_alive=False,  # starts a new context each time
        options={"temperature": 0},
    )
    return result["message"]["content"].strip()

# Only return the first token of the response
def _query_model_first_token(model, query):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": query}],
        stream=True,  # enable streaming
        options={"temperature": 0},
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            return token.strip()  # return only the first token
    return ""


# Classify a movie into a category based on the wikitext
def classify_movie(wikitext, debug=False):
    para = lead_paragraph(wikitext)
    
    if debug:
        prompt = SEQUEL_PROMPT + "\n\nThis is debug mode, please print the number followed by a newline and provide your reasoning."
        model_response = _query_model(MODEL, prompt + para)
        return model_response

    prompt = SEQUEL_PROMPT + "\n\nThis is debug mode, please print the number followed by a newline and provide your reasoning."
    model_response = _query_model_first_token(MODEL, prompt + para)
    return category[int(model_response)]


def _extract_infobox_fields(wikitext: str) -> Dict[str, str]:
    """
    Grab the top-level Infobox film template and return a dict of
    lowercased field -> raw value.
    """
    start = wikitext.lower().find("{{infobox film")
    if start == -1:
        return {}

    # Walk to the matching closing braces to capture the template safely.
    depth = 0
    i = start
    end: Optional[int] = None
    length = len(wikitext)
    while i < length - 1:
        if wikitext.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if wikitext.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                end = i
                break
            continue
        i += 1

    if end is None:
        return {}

    infobox = wikitext[start:end]
    fields: Dict[str, str] = {}
    current_key: Optional[str] = None
    for line in infobox.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("|"):
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            current_key = key.lstrip("|").strip().lower()
            fields[current_key] = value.strip()
            continue
        # Handle multi-line values (e.g., plainlist bullet items) as continuations.
        if current_key and stripped:
            fields[current_key] += "\n" + stripped
    return fields


def _clean_value(value: str) -> str:
    """Simplify wikitext values into plain text."""
    if not value:
        return ""
    cleaned = value
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*/>", "", cleaned)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    # Drop common footnote templates that can trail names (e.g., {{efn|...}}).
    cleaned = re.sub(r"\{\{\s*efn[^}]*\}\}", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Unwrap simple templates like {{nowrap|...}} that often appear in titles.
    cleaned = re.sub(r"\{\{\s*nowrap\|([^{}]+?)\}\}", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = cleaned.replace("'''", "").replace("''", "")
    cleaned = re.sub(r"<[^>]+>", "", cleaned)  # strip remaining HTML tags
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_money_value(text: str) -> Optional[int]:
    """
    Convert a money-like string to an integer number of dollars.
    Handles values such as "$10 million", "$123,456,789", and ranges like "6–7 million"
    by averaging the endpoints.
    """
    if not text:
        return None

    # Prefer explicit USD-denominated values ({{US$|...}}) over other currencies.
    usd_matches = re.findall(r"\{\{\s*US\$\s*\|([^|}]+)", text, flags=re.IGNORECASE)
    if usd_matches:
        numbers = [_parse_money_value(v.strip()) for v in usd_matches]
        numbers = [n for n in numbers if n is not None]
        if numbers:
            return int(round(sum(numbers) / len(numbers)))

    # Also prefer bullet-list items that carry an explicit USD wiki-link marker
    # (e.g. [[United States dollar|$]]35 million) over non-USD line items.
    usd_item_re = re.compile(
        r"^\s*\*[^\n]*(?:\[\[(?:United States dollar|US dollar)[^\]]*\]\]|US\$|USD)[^\n]*",
        re.IGNORECASE | re.MULTILINE,
    )
    usd_items = usd_item_re.findall(text)
    if usd_items:
        # Strip the leading "* " so the recursive call doesn't re-match this pattern.
        numbers = [_parse_money_value(item.lstrip().lstrip("*").strip()) for item in usd_items]
        numbers = [n for n in numbers if n is not None]
        if numbers:
            return int(round(sum(numbers) / len(numbers)))

    # If a non-USD currency is present, return the $ equivalent or None.
    # Currency conversion is not supported — non-USD values without a $ equivalent return None.
    _non_usd_re = re.compile(
        r"[¥£€₹₩₽]"
        r"|\[\[(?:Japanese yen|Chinese yuan|British pound|Pound sterling|"
        r"Italian lira|French franc|German mark|Soviet ruble|Spanish peseta|Euro)[^\]]*\]\]"
        r"|\{\{\s*(?:JPY|CNY|EUR|GBP|INR|KRW|RUB|ITL|FRF|DEM|¥|£|€)[^}]*\}\}",
        re.IGNORECASE,
    )
    if _non_usd_re.search(text):
        m_usd = re.search(r"\$\s*[0-9][\d,.]*(?:\s*(?:million|billion|m|bn))?", text, re.IGNORECASE)
        if m_usd:
            val = _parse_money_value(m_usd.group(0))
            if val is not None:
                return val
        return None

    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*/>", "", cleaned)
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("\u00a0", " ").replace("&nbsp;", " ")
    cleaned = re.sub(r"\{\{\s*nbsp\s*\}\}", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\{\{\s*nbsp\|[^{}]*\}\}", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\{\{\s*efn[^{}]*\}\}", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    def _to_number(int_part: str, dec_part: str, unit: Optional[str], fallback_unit: Optional[str]) -> Optional[float]:
        integer_part = int_part.replace(",", "")
        decimal_part = dec_part or ""
        try:
            base = float(f"{integer_part}.{decimal_part}") if decimal_part else float(integer_part)
        except ValueError:
            return None

        unit_lower = (unit or fallback_unit or "").lower()
        multiplier = 1
        if unit_lower in ("million", "m"):
            multiplier = 1_000_000
        elif unit_lower in ("billion", "bn"):
            multiplier = 1_000_000_000
        return base * multiplier

    # Range like "$6–7 million" -> average endpoints.
    m_range = re.search(
        r"([$\£\€]?)\s*([0-9][\d,]*)(?:\.(\d+))?\s*(million|billion|m|bn)?\s*[-–—]\s*([0-9][\d,]*)(?:\.(\d+))?\s*(million|billion|m|bn)?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_range:
        currency = m_range.group(1)  # reserved; kept for future but unused
        int1, dec1, unit1 = m_range.group(2), m_range.group(3), m_range.group(4)
        int2, dec2, unit2 = m_range.group(5), m_range.group(6), m_range.group(7)
        common_unit = unit1 or unit2
        # Don't apply a fallback unit to a number that already contains commas
        # (e.g. "985,000" in "985,000–1 million" is already a fully-expressed value).
        v1 = _to_number(int1, dec1, unit1, common_unit if "," not in int1 else None)
        v2 = _to_number(int2, dec2, unit2, common_unit if "," not in int2 else None)
        if v1 is not None and v2 is not None:
            return int(round((v1 + v2) / 2))

    # Single value
    m_single = re.search(
        r"([$\£\€]?)\s*([0-9][\d,]*)(?:\.(\d+))?\s*(million|billion|m|bn)?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not m_single:
        return None

    int_part = m_single.group(2)
    dec_part = m_single.group(3)
    unit = m_single.group(4)
    value = _to_number(int_part, dec_part, unit, unit)
    return int(round(value)) if value is not None else None


def _parse_title(fields: Dict[str, str], wikitext: str) -> str:
    """
    Read the movie title from the infobox name/title fields, falling back to a
    bold/italic lead title when no infobox title is present.
    """
    for key in ("name", "title"):
        if key in fields and fields[key]:
            cleaned = _clean_value(fields[key])
            if cleaned:
                return cleaned

    # Fallback: grab the bold/italic first term from the lead paragraph wikitext.
    lead = _lead_paragraph_wikitext(wikitext)
    m_lead = re.search(r"^\s*'{2,5}\s*([^']+?)\s*'{2,5}", lead)
    if m_lead:
        return _clean_value(m_lead.group(1))

    return ""


def _parse_budget(fields: Dict[str, str]) -> Optional[int]:
    for key in ["budget"]:
        if key in fields:
            amount = _parse_money_value(fields[key])
            if amount is not None:
                return amount
    return None


def _parse_box_office(fields: Dict[str, str]) -> Optional[int]:
    for key in ["box office", "box_office", "gross"]:
        if key in fields:
            amount = _parse_money_value(fields[key])
            if amount is not None:
                return amount
    return None


def _parse_director(fields: Dict[str, str]) -> List[str]:
    raw = fields.get("director", "") or fields.get("directed by", "")

    # Normalize the raw value into a string representation we can parse.
    if isinstance(raw, (list, tuple)):
        raw_text = "\n".join(str(item) for item in raw if item is not None)
    else:
        raw_text = "" if raw is None else str(raw)

    if not raw_text:
        return []

    # Detect plainlist / bullet-formatted entries and pull out individual lines.
    items: List[str] = []
    raw_lower = raw_text.lower()
    if "{{plainlist" in raw_lower or "\n" in raw_text or raw_text.strip().startswith("*"):
        normalized = re.sub(r"\{\{\s*plainlist\s*\|?", "", raw_text, flags=re.IGNORECASE)
        normalized = normalized.replace("}}", "")
        for line in normalized.splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("*").strip()
            if line:
                items.append(line)
    else:
        items.append(raw_text)

    # Clean and split names on common separators.
    cleaned_names: List[str] = []
    for item in items:
        cleaned = _clean_value(item)
        if not cleaned:
            continue
        parts = re.split(r"[\/,;]| and ", cleaned)
        for part in parts:
            name = part.strip()
            if name:
                cleaned_names.append(name)

    # Dedupe while preserving order.
    return list(dict.fromkeys(cleaned_names))


MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def _normalize_date_from_template(text: str) -> str:
    """Handle common date templates like {{Film date|YYYY|MM|DD}} -> ISO date."""
    if not text:
        return ""
    # Pull the first template that looks date-like.
    m = re.search(
        r"\{\{(?:film date|film release|release date and age|release date|start date and age|start date)[^}]*\}\}",
        text,
        flags=re.IGNORECASE,
    )
    candidate = m.group(0) if m else text
    parts = [p for p in candidate.strip("{}").split("|") if p]
    # Extract numeric year, month, day.
    nums: List[str] = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            nums.append(p)
            continue
        lower = p.lower()
        if lower in MONTHS:
            nums.append(MONTHS[lower])
    if len(nums) >= 2:
        year = nums[0]
        month = nums[1].zfill(2)
        day = nums[2].zfill(2) if len(nums) >= 3 else "01"
        return f"{year}-{month}-{day}"
    return _clean_value(candidate)


def _parse_release_date(fields: Dict[str, str], wikitext: str) -> Optional[str]:
    # Try infobox raw
    for key in ["release date", "released", "release_date"]:
        if key in fields and fields[key]:
            parsed = _normalize_date_from_template(fields[key])
            if parsed:
                return parsed if re.match(r"\d{4}-\d{2}-\d{2}$", parsed) else None
    # Look for Month DD, YYYY
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        wikitext,
    )
    if m:
        try:
            return datetime.strptime(m.group(0), "%B %d, %Y").date().isoformat()
        except Exception:
            return None
    # ISO-like fallback
    m = re.search(r"\d{4}-\d{2}-\d{2}", wikitext)
    if m:
        return m.group(0)
    # Year-only fallback
    m = re.search(r"\b(\d{4})\b", wikitext)
    if m:
        return f"{m.group(1)}-01-01"
    return None


def _parse_running_time(fields: Dict[str, str], wikitext: str) -> Optional[int]:
    """
    Parse runtime to an integer minute count.
    """
    raw = ""
    for key in ["running time", "running_time", "runtime"]:
        if key in fields:
            raw = fields[key]
            break
    for source in (raw, wikitext):
        if not source:
            continue
        m = re.search(r"(\d{2,3})\s*(?:minutes|min)", source, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _lead_paragraph_wikitext(wikitext: str) -> str:
    """
    Return the raw wikitext for the lead paragraph (before the first blank line),
    skipping initial templates/files but preserving link markup.
    """
    if not wikitext:
        return ""

    i = 0
    length = len(wikitext)
    while i < length:
        if wikitext.startswith("<!--", i):
            end = wikitext.find("-->", i)
            if end == -1:
                break
            i = end + 3
            continue

        if wikitext.startswith("{|", i):
            end = wikitext.find("|}", i)
            if end == -1:
                break
            i = end + 2
            continue

        if wikitext.startswith("{{", i):
            balance = 1
            i += 2
            while i < length and balance > 0:
                if wikitext.startswith("{{", i):
                    balance += 1
                    i += 2
                elif wikitext.startswith("}}", i):
                    balance -= 1
                    i += 2
                else:
                    i += 1
            continue

        if wikitext.startswith("[[", i):
            snippet = wikitext[i + 2 : i + 12]
            if ":" in snippet:
                balance = 1
                i += 2
                while i < length and balance > 0:
                    if wikitext.startswith("[[", i):
                        balance += 1
                        i += 2
                    elif wikitext.startswith("]]", i):
                        balance -= 1
                        i += 2
                    else:
                        i += 1
                continue

        if wikitext[i] == "\n":
            i += 1
            continue

        break

    start_text = i
    end_text = wikitext.find("\n\n", start_text)
    paragraph = wikitext[start_text:] if end_text == -1 else wikitext[start_text:end_text]
    return paragraph.strip()


def _first_sentence(text: str) -> str:
    """Return the first sentence-ish chunk, skipping common abbreviations like Dr."""
    if not text:
        return ""

    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*/>", "", cleaned)
    # Drop HTML comments so punctuation inside them does not prematurely end the sentence.
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

    abbreviations = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "messrs",
        "u.s",
        "u.k",
        "u.s.a",
        "u.k.",
    }

    length = len(cleaned)
    i = 0
    while i < length:
        ch = cleaned[i]
        if ch in ".!?":
            # Capture the word/token immediately before the punctuation.
            prefix = cleaned[:i]
            m_word = re.search(r"([A-Za-z\.]+)$", prefix)
            prev_token = m_word.group(1).lower() if m_word else ""
            is_abbrev = prev_token in abbreviations or (len(prev_token) == 1 and prev_token.isalpha())
            if not is_abbrev:
                # Include trailing quotes/brackets right after the punctuation.
                end = i + 1
                while end < length and cleaned[end] in "\"'’”]) ) ":
                    end += 1
                return cleaned[:end].strip()
        i += 1

    return cleaned.strip()


def _extract_genres_from_first_sentence(wikitext: str) -> List[str]:
    """
    Pull genre-like links from the first sentence of the article lead.
    Genre candidates are links with at least one lowercase-starting word
    or whose targets look like genre/film pages.
    """
    paragraph = _lead_paragraph_wikitext(wikitext)
    sentence = _first_sentence(paragraph)
    if not sentence:
        return []

    def _first_unlinked_film(text: str) -> int:
        """Return index of first 'film' token not inside [[...]], else -1."""
        lower = text.lower()
        in_link = 0
        n = len(lower)
        i = 0
        while i < n:
            if lower.startswith("[[", i):
                in_link += 1
                i += 2
                continue
            if lower.startswith("]]", i) and in_link:
                in_link -= 1
                i += 2
                continue
            if not in_link and lower.startswith("film", i):
                prev = lower[i - 1] if i > 0 else " "
                nxt = lower[i + 4] if i + 4 < n else " "
                if not prev.isalpha() and not nxt.isalpha():
                    return i
            i += 1
        return -1

    genres: List[str] = []
    seen = set()
    film_pos = _first_unlinked_film(sentence)
    if film_pos == -1:
        # Fallback to legacy behavior if no standalone 'film' word was found.
        film_pos = sentence.lower().find(" film")

    for m in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", sentence):
        target = m.group(1).strip()
        label = m.group(2).strip() if m.group(2) else target
        label = label.replace("'''", "").replace("''", "")
        label = re.sub(r"\s+", " ", label).strip()
        label = re.sub(r"^[\"“”'’]+|[\"“”'’]+$", "", label)
        if not label:
            continue

        label_lower = label.lower()
        words = label.split()
        has_lower_word = any(w and w[0].islower() for w in words)
        target_lower = target.lower()
        looks_like_genre = (
            has_lower_word
            or "(genre" in target_lower
            or " film" in target_lower
            or label_lower.endswith(" film")
            or label_lower.endswith("film")
        )
        if not looks_like_genre:
            continue

        # If the sentence contains "film", only collect links that appear
        # before that first occurrence OR themselves contain a film/genre marker.
        if film_pos != -1 and m.start() > film_pos:
            if not ("film" in label_lower or " film" in target_lower or "(genre" in target_lower):
                continue

        normalized = label_lower
        if normalized.endswith(" film"):
            normalized = normalized[: -len(" film")].strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        genres.append(normalized)

    return genres


def _parse_genre(wikitext: str) -> List[str]:
    """Return a list of genre labels derived from the lead sentence."""
    from_links = _extract_genres_from_first_sentence(wikitext)
    return from_links if from_links else []


def lead_paragraph(wikitext: str) -> str:
    """
    Return a cleaned lead paragraph (prose) from the article wikitext.
    Uses the raw lead extraction but removes bold/italic/wiki/link markup and refs.
    """
    raw = _lead_paragraph_wikitext(wikitext)
    if not raw:
        return ""
    cleaned = raw.replace("'''", "").replace("''", "")
    cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"\[http\S+ ([^\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\{\{\s*efn[^}]*\}\}", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*/>", "", cleaned)
    return cleaned.strip()


USER_AGENT = "wikipedia-parser/0.1 (+https://example.com)"


def _fetch_rt_score_from_wikidata(article_title: str) -> Optional[int]:
    """
    Resolve Rotten Tomatoes score from Wikidata when the article uses the
    {{RT data}} / {{Rotten Tomatoes data}} template (no score present in raw text).
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": article_title,
                "prop": "pageprops",
                "format": "json",
                # Follow redirects so loose titles like "Iron Man 2 (film)" still
                # resolve to the canonical page and expose the wikibase_item id.
                "redirects": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        qid = page.get("pageprops", {}).get("wikibase_item")
        if not qid:
            return None

        wd_resp = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        wd_resp.raise_for_status()
        entity = wd_resp.json().get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})

        # P444 (review score) with qualifier P447 (reviewed by) == Rotten Tomatoes (Q105584)
        for claim in claims.get("P444", []):
            snak = claim.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            qualifiers = claim.get("qualifiers", {})
            providers = [
                qual.get("datavalue", {}).get("value", {}).get("id")
                for qual in qualifiers.get("P447", [])
                if qual.get("snaktype") == "value"
            ]
            if "Q105584" not in providers:
                continue
            val = snak.get("datavalue", {}).get("value")
            try:
                # Values may be stored as "84%" or numeric.
                numeric = float(str(val).rstrip("%"))
                return min(100, max(0, int(round(numeric))))
            except (TypeError, ValueError):
                continue
    except Exception:
        # Network errors, API errors, or unexpected shapes should not break parsing.
        return None
    return None


def _extract_rt_score(wikitext: str, article_title: Optional[str] = None) -> Optional[int]:
    # Highest priority: explicit RT data templates with a provided score.
    if article_title and re.search(r"\{\{(?:RT data|Rotten Tomatoes data)\|score", wikitext, flags=re.IGNORECASE):
        val = _fetch_rt_score_from_wikidata(article_title)
        if val is not None:
            return val
    m_rt_data = re.search(r"\{\{\s*(?:RT data|Rotten Tomatoes data)\s*\|\s*score\s*=\s*([0-9]{1,3})", wikitext, flags=re.IGNORECASE)
    if m_rt_data:
        return min(100, int(m_rt_data.group(1)))
    # Template form like {{Rotten Tomatoes prose|74|7.1/10|378|consensus=...}}
    m_tpl = re.search(r"\{\{\s*Rotten Tomatoes prose\|\s*([0-9]{1,3})", wikitext, flags=re.IGNORECASE)
    if m_tpl:
        return min(100, int(m_tpl.group(1)))
    # Look near an RT mention in the raw wikitext (handles % and percent templates).
    raw_lower = wikitext.lower()
    idx_raw = raw_lower.find("rotten tomatoes")
    if idx_raw != -1:
        window_raw = wikitext[max(0, idx_raw - 400) : idx_raw + 800]
        m_pct = re.search(r"(\d{1,3})\s*%", window_raw)
        if m_pct:
            return min(100, int(m_pct.group(1)))
        m_word = re.search(r"(\d{1,3})\s*(?:percent|per\s*cent)", window_raw, flags=re.IGNORECASE)
        if m_word:
            return min(100, int(m_word.group(1)))
        m_tpl_pct = re.search(r"\{\{\s*(?:percent|percentage)\s*\|\s*([0-9]{1,3})", window_raw, flags=re.IGNORECASE)
        if m_tpl_pct:
            return min(100, int(m_tpl_pct.group(1)))
    # Simple approach: strip refs/links, find the Rotten Tomatoes mention, and grab the nearest percent.
    prose = re.sub(r"<ref[^>]*>.*?</ref>", "", wikitext, flags=re.DOTALL)
    prose = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", prose)
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.DOTALL)
    prose_lower = prose.lower()
    idx = prose_lower.find("rotten tomatoes")
    if idx != -1:
        window = prose[max(0, idx - 300) : idx + 800]
        m_pct = re.search(r"(\d{1,3})\s*%", window)
        if m_pct:
            return min(100, int(m_pct.group(1)))
    # If no explicit phrase match, try a loose search anywhere mentioning Rotten Tomatoes and a percent nearby.
    m = re.search(r"Rotten Tomatoes.{0,800}?(\d{1,3})\s*%", prose, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return min(100, int(m.group(1)))
    m = re.search(r"(\d{1,3})\s*%.{0,800}?Rotten Tomatoes", prose, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return min(100, int(m.group(1)))
    # Handle pages using {{RT data|score}} / {{Rotten Tomatoes data|score}} as a simple fallback.
    if article_title and re.search(r"\{\{(?:RT data|Rotten Tomatoes data)\|score", wikitext, flags=re.IGNORECASE):
        return _fetch_rt_score_from_wikidata(article_title)
    return None


def _extract_metacritic_score(wikitext: str) -> Optional[int]:
    m = re.search(r"\{\{Metacritic[^}]*?(\d{1,3})[^}]*\}\}", wikitext, flags=re.IGNORECASE)
    if m:
        return min(100, int(m.group(1)))
    # Common prose form: "... Metacritic ... score of 76 out of 100 ..."
    m = re.search(
        r"Metacritic[^0-9]{0,200}?(\d{1,3})\s*out of\s*100",
        wikitext,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return min(100, int(m.group(1)))
    m = re.search(
        r"Metacritic[^.\d]{0,40}?(\d{1,3})\s*/\s*100", wikitext, flags=re.IGNORECASE
    )
    if m:
        return min(100, int(m.group(1)))
    m = re.search(r"Metacritic[^.\d]{0,40}?score[^.\d]{0,10}?(\d{1,3})", wikitext, flags=re.IGNORECASE)
    if m:
        return min(100, int(m.group(1)))
    m = re.search(
        r"Metacritic[^.\n]{0,120}?score of (\d{1,3}) out of 100",
        wikitext,
        flags=re.IGNORECASE,
    )
    if m:
        return min(100, int(m.group(1)))
    return None


# Process a movie and return a column with the movie title, 
# classification, budget, box office, release date, running time, genre,
# Rotten tomatoes score, and Metacritic score
def process_movie(article_title: str, use_llm: bool = True):
    wt, resolved_title = get_wiki_text(article_title)
    if not wt:
        return {
            "page_title": article_title,
            "title": "",
            "classification": "",
            "director": [],
            "budget": "",
            "box_office": "",
            "release_date": "",
            "running_time": "",
            "genre": [],
            "rotten_tomatoes": "",
            "metacritic": "",
        }

    classification = classify_movie(wt) if use_llm else None

    fields = _extract_infobox_fields(wt)
    parsed_title = _parse_title(fields, wt) or article_title

    director = _parse_director(fields)
    budget = _parse_budget(fields)
    box_office = _parse_box_office(fields)
    release_date = _parse_release_date(fields, wt)
    running_time = _parse_running_time(fields, wt) or ""
    genre = _parse_genre(wt)
    # Use the resolved (redirect-followed) title for Wikidata lookups like RT score.
    lookup_title = resolved_title or article_title
    rotten_tomatoes = _extract_rt_score(wt, lookup_title)
    metacritic = _extract_metacritic_score(wt)

    return {
        "page_title": lookup_title,
        "title": parsed_title,
        "classification": classification,
        "director": director,
        "budget": budget,
        "box_office": box_office,
        "release_date": release_date,
        "running_time": running_time,
        "genre": genre,
        "rotten_tomatoes": rotten_tomatoes,
        "metacritic": metacritic,
    }