#!/usr/bin/env python3
"""
fork.py — filter the cleaned Pitchfork review dataset and save/export selections.

Run fork_fetch.py first to create a fork_data_YYYY-MM-DD.tsv file. Every run of
this script also saves the full filtered selection to
filtered/fork_selection_<filters>.tsv for downstream use (playlist building,
further analysis, etc).

For charts and distributions over the dataset, see notebooks/01_eda_pitchfork.ipynb
instead — this script is for filtering and exporting concrete selections, not
visualization.

EXAMPLES

  # The 9.0+ club, new releases only, as a playlist-import list
  python fork.py --min-score 9 --new-releases --format playlist -o 9plus.txt

  # Every 10.0, including reissue re-reviews
  python fork.py --min-score 10

  # Best New Music in the 2020s, electronic only
  python fork.py --bnm --genre Electronic --review-year 2020-2029

  # Top 3 highest-scoring new albums of every year (evenly spread list)
  python fork.py --new-releases --top-per-year 3 --format playlist -o top3.txt

  # High scores from Japan
  python fork.py --min-score 8.5 --country JP --sort score

  # One critic's taste
  python fork.py --reviewer "Jenn Pelly" --min-score 8

  # Obscure gems: high score, untagged genre, no Best New Music badge
  python fork.py --min-score 8.5 --no-genre --no-bnm

  # What can I filter on?
  python fork.py --list genres
  python fork.py --list countries
  python fork.py --list reviewers
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = "data"
DATA_GLOB = "fork_data_*.tsv"
FILTERED_DIR = "filtered"


# ----------------------------------------------------------------------------- loading

def latest_data_file() -> Path | None:
    """Most recent dated dataset (data/fork_data_YYYY-MM-DD.tsv)."""
    matches = sorted(Path(DATA_DIR).glob(DATA_GLOB))
    return matches[-1] if matches else None


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def selection_filename(a) -> Path:
    """filtered/fork_selection_<filters>.tsv — encodes whatever was put in the filter flags."""
    parts = []
    if a.min_score is not None:
        parts.append(f"min{a.min_score:g}")
    if a.max_score is not None:
        parts.append(f"max{a.max_score:g}")
    if a.review_year:
        parts.append(f"revyear-{a.review_year}")
    if a.release_year:
        parts.append(f"relyear-{a.release_year}")
    if a.new_releases:
        parts.append("new")
    if a.reissues:
        parts.append("reissues")
    if a.bnm:
        parts.append("bnm")
    if a.no_bnm:
        parts.append("nobnm")
    if a.bnr:
        parts.append("bnr")
    if a.genre:
        parts.append("genre-" + "+".join(a.genre))
    if a.exclude_genre:
        parts.append("xgenre-" + "+".join(a.exclude_genre))
    if a.no_genre:
        parts.append("nogenre")
    if a.country:
        parts.append("country-" + "+".join(a.country))
    if a.language:
        parts.append("lang-" + "+".join(a.language))
    if a.artist:
        parts.append(f"artist-{a.artist}")
    if a.reviewer:
        parts.append(f"reviewer-{a.reviewer}")
    if a.search:
        parts.append(f"search-{a.search}")
    if a.dedupe:
        parts.append("dedupe")
    if a.top_per_year:
        parts.append(f"top{a.top_per_year}peryear")
    if a.top_per_release_year:
        parts.append(f"top{a.top_per_release_year}perrelyear")

    slug = "_".join(parts) if parts else "all"
    slug = re.sub(r"[^A-Za-z0-9+.\-_]+", "-", slug)
    if len(slug) > 150:
        slug = slug[:140] + "-" + hashlib.sha1(slug.encode()).hexdigest()[:8]
    return Path(FILTERED_DIR) / f"fork_selection_{slug}.tsv"


def save_selection(df: pd.DataFrame, path: Path, quiet: bool = False) -> None:
    """Persist the current filtered selection (all columns) to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    if not quiet:
        print(f"selection saved -> {path} ({len(df)} rows)", file=sys.stderr)


def genre_lists(df: pd.DataFrame, lower: bool = False) -> pd.Series:
    """The ';'-joined genres column, as a list per row (empty list if untagged)."""
    s = df["genres"].fillna("")
    if lower:
        s = s.str.lower()
    return s.apply(lambda s: [g for g in s.split("; ") if g])


def parse_year_spec(spec):
    """'2015' | '2010-2019' | '1999,2003,2011' -> set of ints, or None."""
    if not spec:
        return None
    years = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", part)
        if m:
            years.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            years.add(int(part))
        else:
            sys.exit(f"Bad year spec: {part!r}. Use 2015, 2010-2019 or 1999,2003.")
    return years


# ----------------------------------------------------------------------------- filtering

def apply_filters(df: pd.DataFrame, a) -> pd.DataFrame:
    mask = df["score"].notna()

    if a.min_score is not None:
        mask &= df["score"] >= a.min_score
    if a.max_score is not None:
        mask &= df["score"] <= a.max_score

    ry = parse_year_spec(a.review_year)
    if ry:
        mask &= df["review_year"].isin(ry)
    rel = parse_year_spec(a.release_year)
    if rel:
        mask &= df["release_year"].isin(rel)

    if a.no_genre or a.genre or a.exclude_genre:
        gl = genre_lists(df, lower=True)
        if a.no_genre:
            mask &= gl.apply(len) == 0
        if a.genre:
            wanted = {g.lower() for g in a.genre}
            mask &= gl.apply(lambda gs: bool(wanted & set(gs)))
        if a.exclude_genre:
            excluded = {g.lower() for g in a.exclude_genre}
            mask &= gl.apply(lambda gs: not (excluded & set(gs)))

    if a.country:
        countries = {c.upper() for c in a.country}
        mask &= df["country"].fillna("").str.upper().isin(countries)
    if a.language:
        langs = {l.lower() for l in a.language}
        mask &= df["language"].fillna("").str.lower().isin(langs)

    if a.bnm:
        mask &= df["bnm"]
    if a.no_bnm:
        mask &= ~df["bnm"]
    if a.bnr:
        mask &= df["bnr"]
    if a.new_releases:
        mask &= df["is_new_release"]
    if a.reissues:
        mask &= df["is_reissue"]

    if a.artist:
        mask &= df["artist"].str.lower().str.contains(a.artist.lower(), regex=False, na=False)
    if a.reviewer:
        mask &= df["reviewer"].fillna("").str.lower().str.contains(
            a.reviewer.lower(), regex=False, na=False
        )
    if a.search:
        hay = (
            df["artist"].fillna("") + " " + df["album"].fillna("") + " "
            + df["blurb"].fillna("") + " " + df["reviewer"].fillna("")
        ).str.lower()
        mask &= hay.str.contains(a.search.lower(), regex=False, na=False)

    return df[mask]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per album: highest score, then earliest review."""
    df = df.sort_values(["score", "review_date"], ascending=[False, True])
    return df.drop_duplicates(subset="_key", keep="first")


def top_per_year(df: pd.DataFrame, n: int, field: str) -> pd.DataFrame:
    d = df.dropna(subset=[field]).sort_values(["score", "artist"], ascending=[False, True])
    return d.groupby(field, group_keys=False).head(n).sort_values(field)


SORTS = {
    "score": (["score", "artist"], [False, True]),
    "score-asc": (["score", "artist"], [True, True]),
    "date": (["review_date", "artist"], [True, True]),
    "date-desc": (["review_date", "artist"], [False, True]),
    "release": (["release_year", "artist"], [True, True]),
}
ALL_SORTS = sorted(list(SORTS) + ["artist", "random"])


def sort_recs(df: pd.DataFrame, how: str, seed: int | None = None) -> pd.DataFrame:
    if how == "random":
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if how == "artist":
        return df.sort_values(["artist", "album"], key=lambda s: s.str.lower())
    cols, ascending = SORTS[how]
    na_position = "first" if how == "release" else "last"
    return df.sort_values(cols, ascending=ascending, na_position=na_position)


# ----------------------------------------------------------------------------- output

COLS = [
    "score", "artist", "album", "release_year", "review_date",
    "genre", "reviewer", "country", "bnm", "bnr", "pitchfork_url",
]


TABLE_PREVIEW_ROWS = 30


def render(df: pd.DataFrame, fmt: str, out) -> None:
    if fmt == "playlist":
        for _, r in df.iterrows():
            out.write(f"{r['artist']} - {r['album']}\n")
    elif fmt in ("csv", "tsv"):
        df.to_csv(out, sep="," if fmt == "csv" else "\t", columns=COLS, index=False, lineterminator="\n")
    elif fmt == "markdown":
        out.write("| Score | Artist | Album | Year | Genre |\n|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            year = "" if pd.isna(r["release_year"]) else int(r["release_year"])
            out.write(
                f"| {r['score']} | {r['artist']} | {r['album']} "
                f"| {year} | {r['genre'] if pd.notna(r['genre']) else ''} |\n"
            )
    else:  # table -> a capped on-screen preview; the full selection is always saved to TSV
        if df.empty:
            out.write("(no matches)\n")
            return
        preview = df.head(TABLE_PREVIEW_ROWS)
        aw = min(34, int(preview["artist"].str.len().max()))
        tw = min(48, int(preview["album"].str.len().max()))
        for _, r in preview.iterrows():
            flag = "BNM" if r["bnm"] else ("BNR" if r["bnr"] else "   ")
            year = str(int(r["release_year"])) if pd.notna(r["release_year"]) else "    "
            genre = r["genre"] if pd.notna(r["genre"]) else ""
            out.write(
                f"{r['score']:>4}  {flag}  {str(r['artist'])[:aw]:<{aw}}  "
                f"{str(r['album'])[:tw]:<{tw}}  {year:<4}  {genre}\n"
            )
        if len(df) > TABLE_PREVIEW_ROWS:
            out.write(f"... and {len(df) - TABLE_PREVIEW_ROWS} more (see the saved selection TSV)\n")


def show_list(df: pd.DataFrame, what: str) -> None:
    if what == "genres":
        gl = genre_lists(df).apply(lambda gs: gs or ["(untagged)"])
        counts = gl.explode().value_counts()
    elif what == "countries":
        counts = df["country"].fillna("(unknown)").value_counts()
    elif what == "languages":
        counts = df["language"].fillna("(unknown)").value_counts()
    else:
        counts = df["reviewer"].fillna("(unknown)").value_counts()
    for k, v in counts.items():
        print(f"  {k:<32} {v:>6}")
    print(f"  ({len(counts)} distinct)")


# ----------------------------------------------------------------------------- cli

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", help=f"dataset TSV file (default: latest {DATA_DIR}/{DATA_GLOB})")

    s = p.add_argument_group("score")
    s.add_argument("--min-score", type=float, help="e.g. 9 or 8.5")
    s.add_argument("--max-score", type=float, help="e.g. 7 or 6.5")

    y = p.add_argument_group("dates")
    y.add_argument("--review-year", help="when Pitchfork reviewed it: 2015, 2010-2019, 1999,2003")
    y.add_argument("--release-year", help="when the album came out, same formats")

    t = p.add_argument_group("type of review")
    t.add_argument("--new-releases", action="store_true", help="only albums reviewed on release")
    t.add_argument("--reissues", action="store_true", help="only reissue / retrospective reviews")
    t.add_argument("--bnm", action="store_true", help="only Best New Music")
    t.add_argument("--no-bnm", action="store_true", help="exclude Best New Music")
    t.add_argument("--bnr", action="store_true", help="only Best New Reissue")

    m = p.add_argument_group("metadata")
    m.add_argument("--genre", action="append", help="repeatable: --genre Rap --genre Jazz")
    m.add_argument("--exclude-genre", action="append", help="repeatable: exclude reviews tagged with any of these genres")
    m.add_argument("--no-genre", action="store_true", help="only reviews with no genre tag")
    m.add_argument("--country", action="append", help="ISO code, e.g. JP, BR, NG — repeatable")
    m.add_argument("--language", action="append", help="ISO code, e.g. spa, rus, jpn — repeatable")
    m.add_argument("--artist", help="substring match on artist")
    m.add_argument("--reviewer", help="substring match on reviewer")
    m.add_argument("--search", help="substring match on artist, album, blurb, reviewer")

    o = p.add_argument_group("shaping and output")
    o.add_argument("--dedupe", action="store_true", help="one row per album (keeps highest score)")
    o.add_argument("--top-per-year", type=int, metavar="N", help="best N per review year")
    o.add_argument("--top-per-release-year", type=int, metavar="N", help="best N per release year")
    o.add_argument("--limit", type=int, help="keep only the first N rows after sorting")
    o.add_argument("--sort", default="score", choices=ALL_SORTS, help="sort order (default: score)")
    o.add_argument("--seed", type=int, help="seed for --sort random")
    o.add_argument(
        "--format", default="table",
        choices=["table", "playlist", "csv", "tsv", "markdown"],
        help="output format (default: table, a capped on-screen preview)",
    )
    o.add_argument("-o", "--output", help="write to a file instead of the screen")
    o.add_argument("--list", dest="list_what",
                   choices=["genres", "countries", "languages", "reviewers"],
                   help="list available filter values (respects other filters)")
    o.add_argument("-q", "--quiet", action="store_true", help="no summary line on stderr")

    a = p.parse_args()

    data_path = Path(a.data) if a.data else latest_data_file()
    if data_path is None or not data_path.exists():
        sys.exit(f"No {DATA_DIR}/{DATA_GLOB} found. Run:  python fork_fetch.py")

    sel_file = selection_filename(a)

    df = apply_filters(load(data_path), a)
    if a.dedupe:
        df = dedupe(df)

    if a.list_what:
        save_selection(df, sel_file, a.quiet)
        show_list(df, a.list_what)
        return

    if a.top_per_year:
        df = top_per_year(df, a.top_per_year, "review_year")
    if a.top_per_release_year:
        df = top_per_year(df, a.top_per_release_year, "release_year")

    df = sort_recs(df, a.sort, a.seed)
    if a.limit:
        df = df.head(a.limit)

    save_selection(df, sel_file, a.quiet)

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            render(df, a.format, f)
        if not a.quiet:
            print(f"{len(df)} rows -> {a.output}", file=sys.stderr)
    else:
        render(df, a.format, sys.stdout)
        if not a.quiet:
            print(f"\n{len(df)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
