#!/usr/bin/env python3
"""
fork_fetch.py — download and clean the "The Fork" Pitchfork review dataset.

Source: https://github.com/olievans123/The-Fork  (MIT licence)
Underlying reviews are Pitchfork's; this is for personal/offline analysis.

Builds the cleaned dataset as a pandas DataFrame (HTML entities decoded,
enrichment (country/language) merged in, derived fields added) and writes
it to data/fork_data_YYYY-MM-DD.tsv.

Usage:
    python fork_fetch.py                  # download + clean -> data/fork_data_YYYY-MM-DD.tsv
    python fork_fetch.py --keep-raw       # keep the downloaded raw files
    python fork_fetch.py --offline        # re-clean from already-downloaded raw
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

RAW = "https://raw.githubusercontent.com/olievans123/The-Fork/main/"
ALBUMS_FILE = "albums.full.json"
ENRICH_FILE = "enrichment.json"
DATA_DIR = "data"
# Dated so each run's output is a distinct, versioned snapshot rather than
# silently overwriting the previous one as the upstream data changes.
OUT_TSV = f"{DATA_DIR}/fork_data_{datetime.date.today().isoformat()}.tsv"

UA = "fork_fetch.py (personal dataset analysis)"

# Pitchfork review-date year at/after which a release-year match means "new release".
NEW_RELEASE_TOLERANCE = 1  # years between release year and review year


def log(*a) -> None:
    print(*a, file=sys.stderr)


def download(name: str, dest: Path) -> None:
    url = RAW + name
    log(f"  downloading {name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    log(f"    {total/1e6:.1f} MB -> {dest}")


def clean_text(s) -> str:
    """Decode HTML entities (the raw data has &amp;, &#39;, &#8211; etc.) and tidy whitespace."""
    if s is None:
        return ""
    s = str(s)
    # Some fields are double-escaped, so unescape twice.
    s = html.unescape(html.unescape(s))
    s = s.replace("​", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s: str) -> str:
    """Loose key for de-duplicating the same album across multiple reviews."""
    s = unicodedata.normalize("NFKD", clean_text(s).lower())
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)  # drop [Deluxe Edition], (Remastered) ...
    s = re.sub(
        r"\b(deluxe|expanded|remaster(ed)?|reissue|anniversary|edition|super|box|"
        r"set|collectors?|vinyl|ep|lp)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def build(albums: list[dict], enrichment: dict) -> pd.DataFrame:
    """Clean and merge the raw records into a single tidy DataFrame."""
    rows = []
    for a in albums:
        url = a.get("url") or ""
        enr = enrichment.get(url) or {}
        date = a.get("date") or ""
        review_year = int(date[:4]) if date[:4].isdigit() else None
        release_year = a.get("releaseYear")
        original_year = a.get("originalYear")

        # Reissue / retrospective detection, most reliable signal first.
        gap = (
            review_year - release_year
            if (review_year and isinstance(release_year, int))
            else None
        )
        is_reissue = bool(
            a.get("bnr")
            or original_year
            or (gap is not None and gap >= 2)
        )
        is_new_release = bool(
            not is_reissue
            and gap is not None
            and abs(gap) <= NEW_RELEASE_TOLERANCE
        )

        artist = clean_text(a.get("artist"))
        title = clean_text(a.get("title"))
        # The dataset uses "Unknown" as the artist for compilations.
        if artist.lower() in ("unknown", ""):
            artist = "Various Artists"

        genres = a.get("genres") or []

        rows.append(
            {
                "id": a.get("id"),
                "artist": artist,
                "album": title,
                "score": a.get("score"),
                "review_date": date[:10],
                "review_year": review_year,
                "release_year": release_year,
                "original_year": original_year,
                "genres": "; ".join(genres),
                "genre": genres[0] if genres else None,
                "reviewer": clean_text(a.get("reviewer")),
                "bnm": bool(a.get("bnm")),
                "bnr": bool(a.get("bnr")),
                "is_new_release": is_new_release,
                "is_reissue": is_reissue,
                "country": enr.get("country"),
                "language": enr.get("language"),
                "mbid": enr.get("mbid"),
                "blurb": clean_text(a.get("description")),
                "pitchfork_url": ("https://pitchfork.com" + url) if url else "",
                "image": a.get("image") or "",
                "_key": norm_key(artist) + "|" + norm_key(title),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["review_date", "artist"]).reset_index(drop=True)
    return df


def summarise(df: pd.DataFrame) -> None:
    years = df["review_year"].dropna()
    log("")
    log(f"  records:        {len(df):,}")
    log(f"  review years:   {int(years.min())}–{int(years.max())}")
    log(f"  new releases:   {df['is_new_release'].sum():,}")
    log(f"  reissues:       {df['is_reissue'].sum():,}")
    log(f"  Best New Music: {df['bnm'].sum():,}")
    log(f"  10.0 reviews:   {(df['score'] == 10).sum():,}")
    log(f"  9.0+ reviews:   {(df['score'].fillna(0) >= 9).sum():,}")
    genre_counts = (
        df["genre"].fillna("(untagged)").value_counts().to_dict()
    )
    log("  genres:         " + ", ".join(f"{k} {v:,}" for k, v in genre_counts.items()))
    log(f"  countries:      {df['country'].nunique()}")
    log(f"  reviewers:      {df['reviewer'].nunique()}")
    log("")
    log("  NOTE: coverage starts in 1999. Pitchfork's 1996-1998 reviews are not")
    log("        in this dataset — that is where many of the early 9.x scores live.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=OUT_TSV, help=f"output TSV (default {OUT_TSV})")
    p.add_argument("--keep-raw", action="store_true", help="keep downloaded raw JSON files")
    p.add_argument("--offline", action="store_true", help="skip download, reuse raw files on disk")
    args = p.parse_args()

    albums_path = Path(ALBUMS_FILE)
    enrich_path = Path(ENRICH_FILE)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.offline:
        log("Fetching The Fork dataset from GitHub:")
        for name, dest in ((ALBUMS_FILE, albums_path), (ENRICH_FILE, enrich_path)):
            download(name, dest)
    else:
        for dest in (albums_path, enrich_path):
            if not dest.exists():
                sys.exit(f"--offline given but {dest} is not here. Run without --offline first.")

    log("Cleaning ...")
    albums = json.loads(albums_path.read_text(encoding="utf-8"))
    enrichment = json.loads(enrich_path.read_text(encoding="utf-8"))

    df = build(albums, enrichment)

    df.to_csv(out_path, sep="\t", index=False)
    log(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    if not args.keep_raw and not args.offline:
        for dest in (albums_path, enrich_path):
            try:
                dest.unlink()
            except OSError:
                pass
        log("  removed raw files (use --keep-raw to keep them)")

    summarise(df)
    log("Done. Now run:  python fork.py --help")


if __name__ == "__main__":
    main()
