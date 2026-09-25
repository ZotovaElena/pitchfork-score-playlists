#!/usr/bin/env python3
"""
fork_prepare.py — dedupe, genre-annotate, and format the raw Pitchfork dataset.

Run fork_fetch.py first. Reads the latest raw fork_data_*.tsv (never modifies
it) and writes a processed fork_data_prepared_<date>.tsv, ready for filtering
(fork.py) and visualization (notebooks/01_eda_pitchfork.ipynb):

  1. dedupe   - one row per album (highest score, then earliest review)
  2. genre    - fill missing genre/genres via Last.fm (genre_annotator.py),
                falling back to artist-level tags
  3. format   - consistent column order/dtypes, sorted by review_date/artist;
                any row still without a genre is tagged "Untagged"

Genre lookups are cached to disk (lastfm_genre_cache.json) so re-runs don't
re-hit the Last.fm API for rows already resolved (or already known to have
no tags).

Usage:
    python fork_prepare.py                    # dedupe + genre-fill + save
    python fork_prepare.py --no-genre         # dedupe + format only, skip Last.fm
    python fork_prepare.py --limit-genre 50   # only annotate the first N missing rows
    python fork_prepare.py --dry-run          # report counts, no Last.fm calls, no file written
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import genre_annotator

DATA_DIR = "data"
DATA_GLOB = "fork_data_*.tsv"
GENRE_CACHE_FILE = "lastfm_genre_cache.json"
COLUMNS = [
    "id", "artist", "album", "score", "review_date", "review_year", "release_year",
    "original_year", "genres", "genre", "reviewer", "bnm", "bnr", "is_new_release",
    "is_reissue", "country", "language", "mbid", "blurb", "pitchfork_url", "image", "_key",
]


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def latest_raw_file() -> Path | None:
    """Most recent raw data/fork_data_*.tsv, skipping any previously prepared output."""
    matches = sorted(p for p in Path(DATA_DIR).glob(DATA_GLOB) if "_prepared" not in p.name)
    return matches[-1] if matches else None


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per album: highest score, then earliest review."""
    df = df.sort_values(["score", "review_date"], ascending=[False, True])
    return df.drop_duplicates(subset="_key", keep="first").reset_index(drop=True)


def _missing_genre(df: pd.DataFrame) -> pd.Series:
    return df["genres"].isna() | (df["genres"].astype(str).str.strip() == "")


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def annotate_genres(
    df: pd.DataFrame, cache_path: Path, limit: int | None, delay: float
) -> pd.DataFrame:
    """Fill missing genres/genre via Last.fm, checkpointing a disk cache as it goes.

    The cache stores fetch_raw_tags()'s *raw* Last.fm tags, not the final genre
    string — canonicalize_tags() is re-run over them on every load, so changes to
    the bucketing/filtering rules in genre_annotator.py apply to cached rows too,
    without any new API calls. A cache entry from before this (a plain string) is
    treated as a miss and re-fetched once.
    """
    df = df.copy()
    missing_idx = df.index[_missing_genre(df)].tolist()
    todo = missing_idx if limit is None else missing_idx[:limit]

    cache = load_cache(cache_path)
    log(
        f"  {len(missing_idx)} rows missing genre, annotating {len(todo)} via Last.fm "
        f"({len(cache)} cached lookups already on disk)"
    )

    filled = unresolved = 0
    bar = tqdm(todo, desc="Last.fm genres", unit="album", file=sys.stderr)
    for i, idx in enumerate(bar, 1):
        row = df.loc[idx]
        key = f"{row['artist']}::{row['album']}"
        raw = cache.get(key)
        if not isinstance(raw, dict):
            raw = genre_annotator.fetch_raw_tags(row["artist"], row["album"])
            cache[key] = raw
            time.sleep(delay)
            if i % 50 == 0:
                save_cache(cache, cache_path)
                bar.write(f"    [{i}/{len(todo)}] checkpointed cache")

        genres = genre_annotator.canonicalize_tags(raw, row["artist"], row["album"])
        if genres:
            df.at[idx, "genres"] = genres
            df.at[idx, "genre"] = genres.split("; ")[0]
            filled += 1
        else:
            unresolved += 1
        bar.set_postfix(filled=filled, unresolved=unresolved)

    save_cache(cache, cache_path)
    log(f"  filled {filled}, still unresolved {unresolved}")
    return df


UNTAGGED = "Untagged"


def format_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Consistent column order and row order for the prepared output.

    Any row still without a genre at this point — Last.fm never resolved it,
    --no-genre skipped annotation, or --limit-genre left it untried — gets an
    explicit "Untagged" instead of a blank/NaN, so nothing is silently missing.
    """
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    still_missing = _missing_genre(df)
    df.loc[still_missing, "genres"] = UNTAGGED
    df.loc[still_missing, "genre"] = UNTAGGED

    df = df[COLUMNS]
    return df.sort_values(["review_date", "artist"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", help=f"raw dataset TSV (default: latest {DATA_DIR}/{DATA_GLOB})")
    p.add_argument("--out", help=f"output TSV path (default: {DATA_DIR}/fork_data_prepared_<today>.tsv)")
    p.add_argument("--no-genre", action="store_true", help="skip Last.fm genre annotation")
    p.add_argument("--limit-genre", type=int, help="only annotate the first N missing-genre rows")
    p.add_argument("--delay", type=float, default=0.25, help="seconds between Last.fm calls (default 0.25)")
    p.add_argument("--cache", default=GENRE_CACHE_FILE, help=f"Last.fm lookup cache file (default {GENRE_CACHE_FILE})")
    p.add_argument("--dry-run", action="store_true", help="report counts, no Last.fm calls, no file written")
    a = p.parse_args()

    data_path = Path(a.data) if a.data else latest_raw_file()
    if data_path is None or not data_path.exists():
        sys.exit(f"No {DATA_DIR}/{DATA_GLOB} found. Run fork_fetch.py first.")

    log(f"loading {data_path.name} ...")
    df = pd.read_csv(data_path, sep="\t")
    log(f"  {len(df):,} raw rows")

    df = dedupe(df)
    log(f"  {len(df):,} rows after dedupe")

    if a.dry_run:
        log(f"  DRY RUN — would annotate {_missing_genre(df).sum():,} missing-genre rows; no file written")
        return

    if not a.no_genre:
        df = annotate_genres(df, Path(a.cache), a.limit_genre, a.delay)

    df = format_dataset(df)

    out_path = Path(a.out) if a.out else Path(DATA_DIR) / f"fork_data_prepared_{date.today().isoformat()}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    log(f"wrote {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
