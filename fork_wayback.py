#!/usr/bin/env python3
"""
fork_wayback.py — recover Pitchfork's 1996-1998 reviews from the Wayback Machine.

Those years are missing from the main dataset because Pitchfork's own site no
longer hosts them: there is no sitemap, no per-review URL, and none of the
machine-readable metadata (JSON-LD, __PRELOADED_STATE__) that the modern
scraper relies on. What survives is Internet Archive captures of hand-written
monthly review index pages, in whatever layout the site used that month.

So this is not a scraper with one parser. It is a pipeline with a measurable
hit rate, run one stage at a time:

    python fork_wayback.py discover        # ask Wayback's CDX index what exists
    python fork_wayback.py fetch           # download those captures into a cache
    python fork_wayback.py inspect <n>     # look at one capture, design a rule
    python fork_wayback.py parse           # run every extractor, score the result
    python fork_wayback.py validate        # measure against known-good data
    python fork_wayback.py merge           # fold accepted rows into fork_data.json

Stages are resumable and cache to disk, so you can re-run `parse` as often as
you like without re-downloading anything.

IMPORTANT, PLEASE READ
----------------------
The extractors in EXTRACTORS below are *hypotheses*. They were written without
access to the real captures (web.archive.org was unreachable from where this
was written), so their hit rate on the actual pages is unknown. The point of
`inspect` and `validate` is to make that hit rate visible so you can fix them:
adding an extractor is about ten lines, and `validate` tells you immediately
whether it helped. Expect to write one or two of your own. That is the job.

Be polite to the Internet Archive: the default delay is 2s between requests and
everything is cached. Don't lower it much.
"""

from __future__ import annotations

import argparse
import gzip
import html as htmllib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

CDX = "https://web.archive.org/cdx/search/cdx"
SNAPSHOT = "https://web.archive.org/web/{ts}id_/{url}"  # id_ = original bytes, no toolbar

CACHE_DIR = "wayback_cache"
MANIFEST = "wayback_manifest.json"
CANDIDATES = "wayback_candidates.json"
GROUND_TRUTH = "ground_truth_1996_1998.json"

# fork_data_*.tsv is the dated dataset fork_fetch.py produces under data/;
# merge reads the latest one and writes a new dated TSV with the same 22 columns.
FORK_DATA_DIR = "data"
FORK_DATA_GLOB = "fork_data_*.tsv"
FORK_DATA_COLUMNS = [
    "id", "artist", "album", "score", "review_date", "review_year", "release_year",
    "original_year", "genres", "genre", "reviewer", "bnm", "bnr", "is_new_release",
    "is_reissue", "country", "language", "mbid", "blurb", "pitchfork_url", "image", "_key",
]

UA = "fork_wayback.py/1.0 (personal music-list research; contact via local use only)"

# Domains Pitchfork used in the early years, newest name last.
DOMAINS = ["pitchforkmedia.com", "pitchfork.com"]

# Captures whose URL looks like it could hold reviews. Deliberately loose —
# the early site had no stable scheme. Use --url-filter to override.
DEFAULT_URL_FILTER = r"(review|record|album|rec_|/r/|index|archive|home|^https?://[^/]+/?$)"

SCORE_RE = r"(?<![\d.])(10(?:\.0)?|[0-9](?:\.[0-9])?)(?![\d.])"


# ============================================================ small utilities

def log(*a):
    print(*a, file=sys.stderr, flush=True)


def get(url, timeout=60, retries=3, delay=2.0):
    """Fetch with backoff. Returns bytes, or None on give-up."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return None
            wait = delay * (2 ** attempt)
            log(f"    HTTP {e.code}, retrying in {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            wait = delay * (2 ** attempt)
            log(f"    {type(e).__name__}: {e}; retrying in {wait:.0f}s")
            time.sleep(wait)
    return None


def decode(raw: bytes) -> str:
    """Old pages are a mix of utf-8, latin-1 and cp1252."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def strip_tags(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    # Every closing tag that ends a visual line must become a newline, or a
    # heading runs into the paragraph after it and pollutes the album title.
    s = re.sub(
        r"(?i)<br\s*/?>|</(?:p|tr|div|li|h[1-6]|td|th|b|strong|table|ul|ol|dt|dd)\s*>",
        "\n",
        s,
    )
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(htmllib.unescape(s))
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)
    s = re.sub(r"\b(the|a|an|ep|lp|and)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _norm_key_part(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)
    s = re.sub(
        r"\b(deluxe|expanded|remaster(ed)?|reissue|anniversary|edition|super|box|"
        r"set|collectors?|vinyl|ep|lp)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def fork_data_key(artist: str, album: str) -> str:
    """Identical to fork_fetch.py's norm_key, so wayback rows dedupe/match against fork_data."""
    return _norm_key_part(artist) + "|" + _norm_key_part(album)


def cache_path(ts, url):
    key = re.sub(r"[^A-Za-z0-9]+", "_", url)[-90:]
    return os.path.join(CACHE_DIR, f"{ts}_{key}.html.gz")


def read_cached(path):
    with gzip.open(path, "rb") as f:
        return decode(f.read())


def plausible_score(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 10.0 else None


def clean_field(s, maxlen=140):
    s = re.sub(r"\s+", " ", htmllib.unescape(str(s or "")))
    # Belt and braces for layouts where a byline or rating trails the title.
    s = re.split(r"(?i)\s+(?:reviewed by|review by|rating\s*[:=]|posted|by\s+[A-Z])", s)[0]
    s = s.strip(" \t\n\r-–—:·|*")
    return s[:maxlen]


# Navigation words that are junk only when they are the WHOLE field. Matching
# these as substrings is a trap: "home" would reject the album "Homework".
EXACT_JUNK = {
    "home", "search", "next", "previous", "prev", "back", "top", "index",
    "more", "email", "e-mail", "links", "news", "about", "contact", "staff",
    "archive", "archives", "reviews", "review", "features", "interviews",
    "menu", "main", "here", "n/a", "tbd", "various", "untitled",
}

# Phrases distinctive enough to match anywhere in the field.
SUBSTR_JUNK = (
    "click here", "next page", "all rights", "copyright", "mailto",
    "http://", "https://", "subscribe", "back to", "top of page",
    "site design", "webmaster", "advertise", "javascript",
)


def looks_like_name(s):
    """Reject obvious navigation junk masquerading as an artist or album."""
    if not s or len(s) < 2 or len(s) > 140:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    low = s.lower().strip()
    if low in EXACT_JUNK:
        return False
    if any(j in low for j in SUBSTR_JUNK):
        return False
    # A "name" that is mostly digits or punctuation is not a name.
    letters = sum(c.isalpha() for c in s)
    return letters >= max(2, len(s) * 0.4)


# ============================================================ stage 1: discover

def stage_discover(a):
    """Ask the CDX index which captures exist, write a manifest."""
    rows = []
    if a.cdx_file:
        log(f"Reading CDX rows from {a.cdx_file} (offline mode)")
        with open(a.cdx_file, encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload[1:] if payload and payload[0][0] == "timestamp" else payload
    else:
        for domain in a.domain or DOMAINS:
            params = {
                "url": domain,
                "matchType": "domain",
                "from": str(a.start),
                "to": str(a.end),
                "output": "json",
                "fl": "timestamp,original,digest,statuscode,mimetype",
                "filter": "statuscode:200",
                "collapse": "digest",   # skip byte-identical re-captures
                "limit": str(a.limit),
            }
            url = CDX + "?" + urllib.parse.urlencode(params)
            log(f"Querying CDX for {domain} ({a.start}-{a.end}) ...")
            raw = get(url, delay=a.delay)
            if not raw:
                log("  no response — CDX can be slow or rate-limited; try again, "
                    "or narrow with --start/--end")
                continue
            try:
                payload = json.loads(decode(raw))
            except json.JSONDecodeError:
                log("  CDX did not return JSON (rate limited?). Skipping.")
                continue
            got = payload[1:] if payload and payload[0][0] == "timestamp" else payload
            log(f"  {len(got)} captures")
            rows.extend(got)

    pat = re.compile(a.url_filter, re.I)
    seen, manifest = set(), []
    for r in rows:
        ts, original = r[0], r[1]
        mime = r[4] if len(r) > 4 else "text/html"
        if "html" not in (mime or "") and mime != "unk":
            continue
        if not a.all_urls and not pat.search(original):
            continue
        key = (ts, original)
        if key in seen:
            continue
        seen.add(key)
        manifest.append({"timestamp": ts, "url": original, "digest": r[2] if len(r) > 2 else ""})

    manifest.sort(key=lambda m: m["timestamp"])
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    log("")
    log(f"  {len(manifest)} candidate captures -> {a.manifest}")
    if manifest:
        by_year = Counter(m["timestamp"][:4] for m in manifest)
        for y in sorted(by_year):
            log(f"    {y}: {by_year[y]}")
        log("")
        log("  Sample URLs:")
        for m in manifest[: min(8, len(manifest))]:
            log(f"    {m['timestamp']}  {m['url'][:100]}")
        log("")
        log("  If these look wrong, re-run discover with --url-filter or --all-urls.")
    log(f"  Next:  python {os.path.basename(__file__)} fetch")


# ============================================================ stage 2: fetch

def stage_fetch(a):
    if not os.path.exists(a.manifest):
        sys.exit(f"{a.manifest} not found. Run `discover` first.")
    with open(a.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(CACHE_DIR, exist_ok=True)

    todo = [m for m in manifest if not os.path.exists(cache_path(m["timestamp"], m["url"]))]
    if a.limit_fetch:
        todo = todo[: a.limit_fetch]
    log(f"{len(manifest)} captures in manifest, {len(todo)} still to download.")
    log(f"Politeness delay: {a.delay}s. Cache: {CACHE_DIR}/")

    ok = fail = 0
    for i, m in enumerate(todo, 1):
        url = SNAPSHOT.format(ts=m["timestamp"], url=m["url"])
        log(f"  [{i}/{len(todo)}] {m['timestamp']} {m['url'][:70]}")
        raw = get(url, delay=a.delay)
        if raw:
            with gzip.open(cache_path(m["timestamp"], m["url"]), "wb") as f:
                f.write(raw)
            ok += 1
        else:
            fail += 1
        time.sleep(a.delay)

    log("")
    log(f"  downloaded {ok}, failed {fail}")
    log(f"  Next:  python {os.path.basename(__file__)} parse")


# ============================================================ extractors
#
# Each extractor takes (html, text, capture) and returns a list of dicts with
# at least artist/album/score. Add your own: write the function, append it to
# EXTRACTORS, re-run `parse` and `validate`. Tag each row with the extractor
# name so the report can tell you which rules are actually earning their keep.


def ex_table_rows(html, text, cap):
    """A <tr> holding both an 'Artist: Album' cell and a score cell."""
    out = []
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", html):
        cells = [strip_tags(c) for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row)]
        if len(cells) < 2:
            continue
        score = None
        for c in cells:
            m = re.fullmatch(r"\s*" + SCORE_RE + r"\s*", c)
            if m:
                score = plausible_score(m.group(1))
                break
        if score is None:
            continue
        for c in cells:
            if ":" in c and looks_like_name(c.split(":")[0]):
                art, alb = c.split(":", 1)
                if looks_like_name(alb):
                    out.append({"artist": clean_field(art), "album": clean_field(alb),
                                "score": score, "extractor": "table_rows"})
                    break
    return out


def ex_anchor_then_score(html, text, cap):
    """<a>Artist: Album</a> with a score within the next ~300 characters."""
    out = []
    for m in re.finditer(r"(?is)<a[^>]+href=[^>]*>(.*?)</a>(.{0,300}?)(?=<a[^>]|$)", html):
        label = strip_tags(m.group(1))
        tail = strip_tags(m.group(2))
        if ":" not in label:
            continue
        art, alb = label.split(":", 1)
        if not (looks_like_name(art) and looks_like_name(alb)):
            continue
        sm = re.search(r"(?:rating|score)?\D{0,12}" + SCORE_RE, tail, re.I)
        if not sm:
            continue
        score = plausible_score(sm.group(1))
        if score is None:
            continue
        out.append({"artist": clean_field(art), "album": clean_field(alb),
                    "score": score, "extractor": "anchor_then_score"})
    return out


def ex_rating_label(html, text, cap):
    """Plain text blocks containing an explicit 'Rating: 8.4' near a title line."""
    out = []
    blocks = re.split(r"\n{1,}", text)
    for i, b in enumerate(blocks):
        m = re.search(r"(?:rating|score)\s*[:=]?\s*" + SCORE_RE, b, re.I)
        if not m:
            continue
        score = plausible_score(m.group(1))
        if score is None:
            continue
        # Look backwards a few lines for the nearest "Artist: Album".
        for j in range(i, max(-1, i - 6), -1):
            cand = blocks[j]
            if ":" not in cand:
                continue
            art, alb = cand.split(":", 1)
            art, alb = clean_field(art), clean_field(alb)
            if looks_like_name(art) and looks_like_name(alb) and not re.search(r"(?i)rating|score", art):
                out.append({"artist": art, "album": alb, "score": score,
                            "extractor": "rating_label"})
                break
    return out


def ex_heading_score(html, text, cap):
    """<h1>-<h4> or <b>/<strong> 'Artist: Album' followed by a bare score."""
    out = []
    # \b after the tag name matters: without it, <body> and <br> both match "b".
    pat = (r"(?is)<(h[1-4]|b|strong)\b[^>]*>(.*?)</\1\s*>"
           r"(.{0,400}?)(?=<(?:h[1-4]|b|strong)\b[^>]*>|$)")
    for m in re.finditer(pat, html):
        label = strip_tags(m.group(2))
        tail = strip_tags(m.group(3))
        if ":" not in label:
            continue
        art, alb = label.split(":", 1)
        if not (looks_like_name(art) and looks_like_name(alb)):
            continue
        sm = re.search(SCORE_RE, tail)
        if not sm:
            continue
        score = plausible_score(sm.group(1))
        if score is None:
            continue
        out.append({"artist": clean_field(art), "album": clean_field(alb),
                    "score": score, "extractor": "heading_score"})
    return out


def ex_score_out_of_ten(html, text, cap):
    """Anything of the form '8.7 / 10' or '8.7 out of 10', title taken from nearby."""
    out = []
    for m in re.finditer(SCORE_RE + r"\s*(?:/|out of)\s*10\b", text, re.I):
        score = plausible_score(m.group(1))
        if score is None:
            continue
        window = text[max(0, m.start() - 300): m.start()]
        cands = [l for l in window.split("\n") if ":" in l]
        if not cands:
            continue
        art, alb = cands[-1].split(":", 1)
        art, alb = clean_field(art), clean_field(alb)
        if looks_like_name(art) and looks_like_name(alb):
            out.append({"artist": art, "album": alb, "score": score,
                        "extractor": "score_out_of_ten"})
    return out


def ex_meta_comment(html, text, cap):
    """The 1999-era page template: a hidden `<!-- 1_artist: ... 4_rating: ... -->` block.

    Confirmed against real captures (not a guess like the extractors above): every
    pitchforkmedia.com review page from this era carries this exact machine-readable
    comment plus a matching `<meta name="keywords">` tag. It is far more reliable
    than pattern-matching the rendered prose, so this is tried first.
    """
    out = []
    m = re.search(
        r"(?is)1_artist:\s*(.*?)\s*\n\s*2_title:\s*(.*?)\s*\n\s*3_label:\s*(.*?)\s*\n"
        r"\s*4_rating:\s*(.*?)\s*\n\s*5_author:\s*(.*?)\s*\n",
        html,
    )
    if not m:
        return out
    artist, album, label, rating, author = m.groups()
    artist, album, label, author = (clean_field(x) for x in (artist, album, label, author))
    score = plausible_score(rating)
    if score is None or not (looks_like_name(artist) and looks_like_name(album)):
        return out

    blurb = None
    if author:
        bm = re.search(
            r"(?is)rating:\s*" + re.escape(rating.strip()) + r"\s*\n(.*?)\n-\s*" + re.escape(author),
            text,
        )
        if bm:
            blurb = re.sub(r"\s+", " ", bm.group(1)).strip()

    out.append({
        "artist": artist, "album": album, "score": score,
        "reviewer": author or None, "blurb": blurb, "label": label or None,
        "extractor": "meta_comment",
    })
    return out


EXTRACTORS = [
    ex_meta_comment,
    ex_table_rows,
    ex_anchor_then_score,
    ex_rating_label,
    ex_heading_score,
    ex_score_out_of_ten,
]


def find_reviewer(text):
    m = re.search(r"(?:by|reviewed by)\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3})", text)
    return clean_field(m.group(1), 60) if m else None


# ============================================================ stage 3: parse

def stage_parse(a):
    if not os.path.isdir(CACHE_DIR):
        sys.exit(f"No {CACHE_DIR}/. Run `fetch` first.")
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".html.gz"))
    if not files:
        sys.exit(f"{CACHE_DIR}/ is empty. Run `fetch` first.")

    rows, per_capture, per_extractor = [], [], Counter()
    for fn in files:
        ts = fn.split("_")[0]
        path = os.path.join(CACHE_DIR, fn)
        try:
            html = read_cached(path)
        except OSError:
            continue
        text = strip_tags(html)
        reviewer = find_reviewer(text)

        found = []
        for ex in EXTRACTORS:
            try:
                got = ex(html, text, {"timestamp": ts, "file": fn})
            except Exception as e:  # noqa: BLE001
                log(f"  extractor {ex.__name__} raised on {fn}: {e}")
                got = []
            for r in got:
                r.update(
                    capture_timestamp=ts,
                    capture_file=fn,
                    review_date_approx=f"{ts[:4]}-{ts[4:6]}",
                    reviewer=r.get("reviewer") or reviewer,
                )
            found.extend(got)
            per_extractor[ex.__name__] += len(got)

        rows.extend(found)
        per_capture.append({"file": fn, "timestamp": ts, "rows": len(found),
                            "bytes": os.path.getsize(path)})

    # Collapse duplicates: the same album appears in many monthly captures.
    groups = defaultdict(list)
    for r in rows:
        groups[(norm(r["artist"]), norm(r["album"]))].append(r)

    merged = []
    for (ka, kb), g in groups.items():
        if not ka or not kb:
            continue
        votes = Counter(r["score"] for r in g)
        score, n_votes = votes.most_common(1)[0]
        best = max(g, key=lambda r: len(r["artist"]) + len(r["album"]))
        blurbs = [r.get("blurb") for r in g if r.get("blurb")]
        merged.append({
            "artist": best["artist"],
            "album": best["album"],
            "score": score,
            "review_date_approx": min(r["review_date_approx"] for r in g),
            "reviewer": next((r["reviewer"] for r in g if r["reviewer"]), None),
            "blurb": max(blurbs, key=len) if blurbs else None,
            "n_captures": len(g),
            "score_agreement": round(n_votes / len(g), 2),
            "score_variants": sorted(votes) if len(votes) > 1 else None,
            "extractors": sorted({r["extractor"] for r in g}),
            "confidence": round(
                min(1.0, 0.4 + 0.2 * min(len(g), 2) + 0.2 * (n_votes / len(g))), 2
            ),
            "source": "wayback",
        })
    merged.sort(key=lambda r: (-r["confidence"], r["artist"].lower()))

    with open(a.candidates, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    log("")
    log(f"  captures parsed:   {len(per_capture)}")
    log(f"  raw extractions:   {len(rows)}")
    log(f"  distinct albums:   {len(merged)}  -> {a.candidates}")
    log("")
    log("  rows per extractor (a rule producing 0 is not firing on your captures):")
    for name, n in per_extractor.most_common():
        log(f"    {name:<22} {n}")
    log("")
    empty = [c for c in per_capture if c["rows"] == 0]
    log(f"  captures yielding nothing: {len(empty)}/{len(per_capture)}")
    if empty:
        log("    inspect one to find out why, e.g.:")
        log(f"      python {os.path.basename(__file__)} inspect {per_capture.index(empty[0])}")
    disagree = [r for r in merged if r["score_variants"]]
    if disagree:
        log(f"  albums where captures disagree on the score: {len(disagree)}")
        for r in disagree[:5]:
            log(f"    {r['artist']} - {r['album']}: {r['score_variants']}")
    log("")
    log(f"  Next:  python {os.path.basename(__file__)} validate")


# ============================================================ stage 4: inspect

def stage_inspect(a):
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".html.gz"))
    if not files:
        sys.exit("cache is empty")
    try:
        fn = files[a.index]
    except IndexError:
        sys.exit(f"index out of range (cache has {len(files)} files)")

    html = read_cached(os.path.join(CACHE_DIR, fn))
    text = strip_tags(html)
    print(f"# {fn}  ({len(html):,} chars of HTML, {len(text):,} of text)\n")

    tags = Counter(t.lower() for t in re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)", html))
    print("## tag histogram (top 15)")
    print("  " + ", ".join(f"{t}:{n}" for t, n in tags.most_common(15)))

    print("\n## what each extractor finds here")
    for ex in EXTRACTORS:
        try:
            got = ex(html, text, {"timestamp": fn.split("_")[0], "file": fn})
        except Exception as e:  # noqa: BLE001
            print(f"  {ex.__name__:<22} RAISED {e}")
            continue
        print(f"  {ex.__name__:<22} {len(got)} rows")
        for r in got[:3]:
            print(f"      {r['score']:>4}  {r['artist']} - {r['album']}")

    print("\n## context around score-like numbers (design your rule from these)")
    shown = 0
    for m in re.finditer(SCORE_RE, text):
        if shown >= a.samples:
            break
        s, e = max(0, m.start() - 200), min(len(text), m.end() + 80)
        snippet = text[s:e].replace("\n", " ⏎ ")
        print(f"  ...{snippet}...")
        print("  " + "-" * 70)
        shown += 1

    if a.raw:
        print("\n## raw HTML head")
        print(html[: a.raw])


# ============================================================ stage 5: validate

def stage_validate(a):
    if not os.path.exists(a.candidates):
        sys.exit(f"{a.candidates} not found. Run `parse` first.")
    if not os.path.exists(a.ground_truth):
        sys.exit(f"{a.ground_truth} not found — it ships alongside this script.")

    with open(a.candidates, encoding="utf-8") as f:
        cands = json.load(f)
    with open(a.ground_truth, encoding="utf-8") as f:
        truth = json.load(f)

    cmap = {(norm(c["artist"]), norm(c["album"])): c for c in cands}
    tmap = {(norm(t["artist"]), norm(t["album"])): t for t in truth}

    hit, miss, wrong_score = [], [], []
    for k, t in tmap.items():
        c = cmap.get(k)
        if not c:
            miss.append(t)
        elif abs(c["score"] - t["score"]) > 0.05:
            wrong_score.append((t, c))
        else:
            hit.append((t, c))

    n = len(tmap)
    log("")
    log("  Measured against known-good 1996-1998 albums scoring 9.0+")
    log(f"  (ground truth: {n} albums — it only covers 9.0+, so it can measure")
    log("   recall on high scores, not on the whole dataset)")
    log("")
    log(f"    found with matching score:  {len(hit):>4}/{n}  ({100*len(hit)/n:.0f}%)")
    log(f"    found but score differs:    {len(wrong_score):>4}/{n}")
    log(f"    not found at all:           {len(miss):>4}/{n}")
    log("")
    log(f"    total candidates extracted: {len(cands)}")
    if cands:
        lo = sum(1 for c in cands if c["confidence"] < 0.7)
        log(f"    of which low confidence:    {lo}")

    if wrong_score:
        log("")
        log("  Score mismatches (check whether the capture or the ground truth is right —")
        log("  a later retrospective review can legitimately differ):")
        for t, c in wrong_score[: a.show]:
            log(f"    {t['artist']} - {t['album']}: truth {t['score']}, parsed {c['score']}"
                f"  [{','.join(c['extractors'])}]")

    if miss:
        log("")
        log("  Missing — these should be recoverable; if many share a month, that")
        log("  month's layout probably needs its own extractor:")
        for t in miss[: a.show]:
            log(f"    {t['artist']} - {t['album']} ({t['score']})")
        by_year = Counter(t.get("release_year") for t in miss)
        log(f"    missing by album year: {dict(sorted(by_year.items(), key=lambda x: str(x[0])))}")

    if cands:
        log("")
        log("  Candidates NOT in ground truth are mostly legitimate sub-9.0 reviews,")
        log("  but scan the low-confidence ones for parser junk:")
        junk = [c for c in cands if c["confidence"] < 0.7 and (norm(c["artist"]), norm(c["album"])) not in tmap]
        for c in junk[: min(5, a.show)]:
            log(f"    {c['score']:>4}  {c['artist']} - {c['album']}  [{','.join(c['extractors'])}]")

    log("")
    log("  If recall is low, the loop is:  inspect a capture -> write an extractor")
    log("  -> append it to EXTRACTORS -> parse -> validate. No re-downloading.")


# ============================================================ stage 6: merge

def latest_fork_data() -> Path | None:
    matches = sorted(Path(FORK_DATA_DIR).glob(FORK_DATA_GLOB))
    return matches[-1] if matches else None


def stage_merge(a):
    if not os.path.exists(a.candidates):
        sys.exit(f"{a.candidates} not found. Run `parse` first.")

    data_path = Path(a.data) if a.data else latest_fork_data()
    if data_path is None or not data_path.exists():
        sys.exit(f"No {FORK_DATA_DIR}/{FORK_DATA_GLOB} found. Run fork_fetch.py first.")

    with open(a.candidates, encoding="utf-8") as f:
        cands = json.load(f)
    existing_df = pd.read_csv(data_path, sep="\t")
    existing_keys = set(existing_df["_key"].dropna())

    kept = [c for c in cands if c["confidence"] >= a.min_confidence]

    new_rows, seen_keys = [], set()
    added = skipped = 0
    for c in kept:
        key = fork_data_key(c["artist"], c["album"])
        if key in seen_keys or (key in existing_keys and not a.allow_duplicates):
            skipped += 1
            continue
        ts = c["review_date_approx"]  # "YYYY-MM" — index pages give no day
        new_rows.append({
            "id": f"wayback__{key}",
            "artist": c["artist"],
            "album": c["album"],
            "score": c["score"],
            "review_date": f"{ts}-01",
            "review_year": int(ts[:4]),
            "release_year": None,
            "original_year": None,
            "genres": None,
            "genre": None,
            "reviewer": c.get("reviewer") or None,
            "bnm": False,
            "bnr": False,
            "is_new_release": False,      # unknowable from an index page
            "is_reissue": False,
            "country": None,
            "language": None,
            "mbid": None,
            "blurb": c.get("blurb") or None,
            "pitchfork_url": "",
            "image": "",
            "_key": key,
        })
        seen_keys.add(key)
        added += 1

    log(f"  candidates: {len(cands)}, at/above confidence {a.min_confidence}: {len(kept)}")
    log(f"  would add {added}, skip {skipped} already present or duplicated")

    if not new_rows:
        log("  nothing to add.")
        return
    if a.dry_run:
        log("  DRY RUN — no file written.")
        return

    new_df = pd.DataFrame(new_rows, columns=FORK_DATA_COLUMNS)
    merged_df = pd.concat(
        [existing_df.reindex(columns=FORK_DATA_COLUMNS), new_df], ignore_index=True
    )
    merged_df = merged_df.sort_values(["review_date", "artist"]).reset_index(drop=True)

    out_path = Path(a.out) if a.out else Path(FORK_DATA_DIR) / f"fork_data_{date.today().isoformat()}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, sep="\t", index=False)

    log(f"  added {added} rows to {data_path.name}'s {len(existing_df):,} existing rows")
    log(f"  wrote {out_path} ({len(merged_df):,} total rows)")
    log("")
    log("  Wayback rows have review_date pinned to the 1st of the month (day is")
    log("  unknown), no genre/country/language, and is_new_release=False, so")
    log("  --new-releases and --genre will exclude them. Their id starts with")
    log("  'wayback__' so they're easy to find/filter.")


# ============================================================ selftest

FIXTURE_TABLE = """
<html><body><table>
<tr><td><a href="/r/1.html">Radiohead: OK Computer</a></td><td>10.0</td></tr>
<tr><td><a href="/r/2.html">Modest Mouse: The Lonesome Crowded West</a></td><td>9.7</td></tr>
<tr><td>Click here</td><td>next page</td></tr>
</table></body></html>
"""

FIXTURE_PROSE = """
<html><body>
<h3>Yo La Tengo: I Can Hear the Heart Beating as One</h3>
<p>Reviewed by Ryan Schreiber</p>
<p>Some prose about the record that mentions 1997 and 12 songs.</p>
<p>Rating: 9.7</p>
<h3>Ween: The Mollusk</h3>
<p>More prose here.</p>
<p>Rating: 9.7</p>
</body></html>
"""

FIXTURE_SLASH = """
<html><body>
<b>Daft Punk: Homework</b><br>
review text here<br>
9.2 / 10<br>
</body></html>
"""


def stage_selftest(a):
    """Exercise the plumbing on synthetic fixtures.

    These fixtures are INVENTED. They prove the extractor mechanics, dedupe,
    voting and confidence scoring work — they say nothing about whether the
    rules match the real 1997 Pitchfork layout.
    """
    fixtures = [("table", FIXTURE_TABLE), ("prose", FIXTURE_PROSE), ("slash", FIXTURE_SLASH)]
    total = 0
    for name, html in fixtures:
        text = strip_tags(html)
        print(f"\n## fixture: {name}")
        for ex in EXTRACTORS:
            got = ex(html, text, {"timestamp": "19970601", "file": name})
            if got:
                print(f"  {ex.__name__}: {len(got)}")
                for r in got:
                    print(f"      {r['score']:>4}  {r['artist']} - {r['album']}")
                total += len(got)
    print(f"\n{total} extractions across {len(fixtures)} synthetic fixtures.")
    print("Junk rejection, score voting and dedupe all exercised.")
    print("\nNOTE: fixtures are invented. Real hit rate is unknown until you run")
    print("      discover -> fetch -> parse -> validate against actual captures.")


# ============================================================ cli

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="query Wayback's CDX index for captures")
    d.add_argument("--start", type=int, default=1996, help="start year (default 1996)")
    d.add_argument("--end", type=int, default=1999, help="end year, inclusive (default 1999)")
    d.add_argument("--domain", action="append", help=f"repeatable (default: {', '.join(DOMAINS)})")
    d.add_argument("--url-filter", default=DEFAULT_URL_FILTER, help="regex on capture URL")
    d.add_argument("--all-urls", action="store_true", help="keep every HTML capture")
    d.add_argument("--limit", type=int, default=20000)
    d.add_argument("--cdx-file", help="read CDX JSON from a file instead (offline testing)")
    d.add_argument("--manifest", default=MANIFEST)
    d.add_argument("--delay", type=float, default=2.0)
    d.set_defaults(func=stage_discover)

    f = sub.add_parser("fetch", help="download the captures in the manifest")
    f.add_argument("--manifest", default=MANIFEST)
    f.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    f.add_argument("--limit-fetch", type=int, help="stop after N downloads this run")
    f.set_defaults(func=stage_fetch)

    i = sub.add_parser("inspect", help="examine one cached capture")
    i.add_argument("index", type=int, help="position in the cache listing")
    i.add_argument("--samples", type=int, default=12, help="score-context snippets to show")
    i.add_argument("--raw", type=int, default=0, help="also print N chars of raw HTML")
    i.set_defaults(func=stage_inspect)

    pa = sub.add_parser("parse", help="run every extractor over the cache")
    pa.add_argument("--candidates", default=CANDIDATES)
    pa.set_defaults(func=stage_parse)

    v = sub.add_parser("validate", help="measure the result against known-good data")
    v.add_argument("--candidates", default=CANDIDATES)
    v.add_argument("--ground-truth", default=GROUND_TRUTH)
    v.add_argument("--show", type=int, default=15, help="rows to list per section")
    v.set_defaults(func=stage_validate)

    m = sub.add_parser("merge", help="fold accepted rows into a new dated data/fork_data_*.tsv")
    m.add_argument("--candidates", default=CANDIDATES)
    m.add_argument("--data", help=f"base dataset TSV (default: latest {FORK_DATA_DIR}/{FORK_DATA_GLOB})")
    m.add_argument("--out", help=f"output TSV path (default: {FORK_DATA_DIR}/fork_data_<today>.tsv)")
    m.add_argument("--min-confidence", type=float, default=0.7)
    m.add_argument("--allow-duplicates", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(func=stage_merge)

    s = sub.add_parser("selftest", help="exercise the extractors on synthetic fixtures")
    s.set_defaults(func=stage_selftest)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
