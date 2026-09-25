#!/usr/bin/env python3
"""
fork_playlist.py — build a Spotify playlist from a fork.py selection, one
track per album: the album's single most-popular track.

Two caveats before you rely on this:

  - Spotify's API never exposes real play counts. The closest thing available
    is each track's "popularity" (0-100, Spotify's own relative ranking) —
    that's what "most reproduced" actually means here, not literal streams.
  - Matching your Pitchfork album titles to Spotify's catalog is fuzzy
    (reissue suffixes, messy titles) and won't be perfect. Unmatched or
    low-confidence albums are reported, never silently dropped.

Usage:
    # match + preview only — prints/saves the track list, touches nothing on
    # your Spotify account (catalog search only needs app credentials, no login)
    python fork_playlist.py filtered/fork_selection_min9_max9.9_new.tsv

    # same matching (cached — no repeat API calls), then actually creates and
    # fills the playlist (needs a one-time browser login, playlist-modify scope)
    python fork_playlist.py filtered/fork_selection_....tsv --push --playlist-name "Pitchfork 9+"

    # add to an existing playlist instead of creating a new one
    python fork_playlist.py filtered/fork_selection_....tsv --push --playlist-id 37i9dQZF1...

    # only push albums whose title matched exactly (skip the "fuzzy" tier)
    python fork_playlist.py filtered/fork_selection_....tsv --push --playlist-name "..." --exact-only

--push login on a headless server: it prints a Spotify URL. Open it in a browser
on your own machine, approve, and copy the URL you're redirected to (the page
itself will fail to load — that's expected) back into the terminal prompt. The
redirect URI must match one registered in your Spotify app's dashboard exactly.

Requires SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env for catalog search
(preview mode), plus SPOTIFY_REDIRECT_URI for --push (needs a user login).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from tqdm import tqdm

load_dotenv()

CACHE_FILE = "spotify_match_cache.json"
TOKEN_CACHE_FILE = ".spotify_token_cache"
PLAYLIST_SCOPE = "playlist-modify-private playlist-modify-public"
PLAYLIST_DIR = "playlists"


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- auth

def spotify_client(need_user_auth: bool) -> spotipy.Spotify:
    """Read-only catalog search needs only app credentials; creating/filling a
    playlist needs a real user login (opens a browser once, then caches the token).

    Reads SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET/SPOTIFY_REDIRECT_URI explicitly
    rather than relying on spotipy's own SPOTIPY_*-prefixed default names, since
    that's what's actually set in .env.
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env")

    if need_user_auth:
        redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
        if not redirect_uri:
            sys.exit("SPOTIFY_REDIRECT_URI not set in .env — needed for --push (user login)")
        auth = SpotifyOAuth(
            client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri,
            scope=PLAYLIST_SCOPE, cache_path=TOKEN_CACHE_FILE,
            open_browser=False,     # this runs on a headless server: print the login URL instead
        )
    else:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth)


# ----------------------------------------------------------------------------- caching

def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- matching

# Letters NFKD can't split into base + accent, so they need an explicit ASCII form.
_LIGATURES = str.maketrans({"æ": "ae", "œ": "oe", "ø": "o", "ð": "d", "þ": "th", "ß": "ss", "ł": "l", "đ": "d"})


def _clean(s: str) -> str:
    """Lowercase, accent-free, apostrophe-free text with "&"/"+" read as "and",
    "St." as "Street", and any [bracketed] / (parenthesized) part dropped.
    """
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c)).translate(_LIGATURES)
    s = re.sub(r"['\u2019`\u00b4]", "", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)
    return re.sub(r"\bst\b\.?", "street", s)


def _norm(s: str) -> str:
    """Whitespace/punctuation-free form of a name, for whole-name equality."""
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"^the ", "", _clean(s).strip()))


# A title ending in one of these is the same record as without it: "Rival Dealer EP"
# = "Rival Dealer", "Superfuzz Bigmuff Deluxe Edition" = "Superfuzz Bigmuff".
_EDITION_END = {"ep", "edition", "deluxe", "remastered", "remaster", "version"}
_EDITION_WORDS = _EDITION_END | {"expanded", "anniversary", "collectors", "collector", "special",
                                 "bonus", "track", "tracks", "limited", "super"}


def _tokens(s: str) -> list[str]:
    """A title as a list of words (accents, "and", "Street", leading "The" and a
    trailing EP / edition tag all normalized away)."""
    # "the" is dropped everywhere, not just at the front: once word order is ignored,
    # a leading "The" in one title and a mid-title "the" in the other must still match
    toks = [t for t in re.findall(r"[a-z0-9]+", _clean(s)) if t != "the"]
    if toks and toks[-1] in _EDITION_END:
        stripped = list(toks)
        while stripped and (stripped[-1] in _EDITION_WORDS or re.fullmatch(r"\d+(st|nd|rd|th)", stripped[-1])):
            stripped.pop()
        toks = stripped or toks
    return toks


def _titles_equal(a: str, b: str) -> bool:
    """Same words, in any order ("Human Amusement at Hourly Rates: The Best of
    Guided by Voices" = "The Best of Guided by Voices: Human Amusement ...")."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and sorted(ta) == sorted(tb)


_CONNECTORS = re.compile(r"\s*(?:,|/|;|\s&\s|\s\+\s|\sand\s|\swith\s|\sfeat\.?\s|\sfeaturing\s|\svs\.?\s)\s*", re.I)


def _artist_ok(want: str, got_names: list[str]) -> bool:
    """The Spotify album's artist must be the Pitchfork artist — one of its credited
    artists, all of them together ("Johnny Cash" + "Willie Nelson" for "Johnny Cash
    and Willie Nelson"), or — when Pitchfork adds collaborators — its *main* artist:
    the first name before a separator ("D'Angelo, The Vanguard" -> D'Angelo) or the
    front of a possessive band name ("Ariel Pink's Haunted Graffiti" -> Ariel Pink).
    Whole-name equality only: substring matching would accept "Low" for "Slowdive".
    """
    if not got_names:
        return False
    credited = {_norm(n) for n in got_names} | {_norm(" and ".join(got_names))}
    if _norm(want) in credited - {""}:
        return True
    main = _CONNECTORS.split(want or "", maxsplit=1)[0]
    mains = {_norm(main), _norm(re.split(r"[\u2019']s\b", main)[0])} - {""}
    return bool(mains & credited)


# Numbers that mark a *different* release ("Part Four", "Vol. 10"). "1"/"one" is
# left out on purpose: an unnumbered title is usually the first volume.
_NUMBER_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "ii", "iii", "iv", "vi", "vii", "viii", "ix"}


def _title_related(want: str, got: str) -> bool:
    """One title's words are all inside the other's — edition/subtitle variants like
    "Legends of Country Music" vs "Legends of Country Music: Bob Wills ...".
    Guarded against the two ways that goes wrong: a short generic title inside a
    long one ("Rockabilly" in "Rockin' Bones: 1950s Punk and Rockabilly", so the
    shorter must have 3+ words), and the extra words being a different volume
    number ("Part Four" or "Vol. 10" is not the release we want).
    """
    tw, tg = _tokens(want), _tokens(got)
    short, long_ = (tw, tg) if len(tw) <= len(tg) else (tg, tw)
    if len(short) < 3 or Counter(short) - Counter(long_):
        return False
    extras = Counter(long_) - Counter(short)
    return not {t for t in extras if t.isdigit() or t in _NUMBER_WORDS} - {"1"}


_VARIOUS = "variousartists"


def _is_various(artist: str) -> bool:
    return _norm(artist) == _VARIOUS


def _is_compilation(it: dict) -> bool:
    return it.get("album_type") == "compilation" or _artist_ok(
        "Various Artists", [a["name"] for a in it.get("artists", [])]
    )


def _classify(artist: str, album: str, match: dict) -> str | None:
    """Confidence today's rules give a (possibly cached) match — "exact", "fuzzy",
    or None if it no longer passes. Derived from the match itself, never from the
    confidence label stored with it: that label came from whichever rules were
    current when the match was cached.
    """
    name = match.get("name", "")
    if _is_various(artist):
        credited = [{"name": n} for n in (match.get("artists") or [match.get("artist", "")])]
        if not _is_compilation({"album_type": match.get("album_type"), "artists": credited}):
            return None
    elif not _artist_ok(artist, match.get("artists") or [match.get("artist", "")]):
        return None
    if _titles_equal(name, album):
        return "exact"
    return "fuzzy" if _title_related(album, name) else None


def _match_is_valid(artist: str, album: str, match: dict) -> bool:
    return _classify(artist, album, match) is not None


def _pick_candidate(items: list[dict], artist: str, album: str) -> dict | None:
    """Best verified candidate, or None. A right-artist-wrong-album hit is
    rejected too: a track from the wrong album breaks the "most popular track *on
    the album*" rule, so unmatched (reported) beats a confident-looking mistake.
    """
    if _is_various(artist):
        return _pick_compilation(items, album)
    verified = [
        it for it in items if _artist_ok(artist, [a["name"] for a in it.get("artists", [])])
    ]
    for it in verified:
        if _titles_equal(it["name"], album):
            return _as_match(it, "exact")
    for it in verified:
        if _title_related(album, it["name"]):
            return _as_match(it, "fuzzy")
    return None


def _pick_compilation(items: list[dict], album: str) -> dict | None:
    """"Various Artists" is a placeholder, so the artist can't verify anything —
    Spotify credits these to "Various Artists" or to a label ("Soul Jazz Records
    Presents ..."). Match on the title alone, but only among compilation-style
    albums: an ordinary artist's album that happens to share the title (a band's
    "Goodbye, Babylon" vs Dust-to-Digital's box set) is a different record.
    """
    cands = [it for it in items if _is_compilation(it)]
    for it in cands:
        if _titles_equal(it["name"], album):
            return _as_match(it, "exact")
    for it in cands:
        if _title_related(album, it["name"]):
            return _as_match(it, "fuzzy")
    return None


def _as_match(it: dict, confidence: str) -> dict:
    names = [a["name"] for a in it.get("artists", [])]
    return {
        "id": it["id"], "name": it["name"], "artist": ", ".join(names),
        "artists": names, "album_type": it.get("album_type"), "confidence": confidence,
    }


def find_album(sp: spotipy.Spotify, artist: str, album: str) -> dict | None:
    """Spotify album match verified against the Pitchfork artist, or None.
    "exact" = artist and title both match after normalization; "fuzzy" = right
    artist, related title (edition/subtitle variant). Never falls back to
    Spotify's top hit unverified — that used to put e.g. Lady Gaga tracks on
    Joanna Newsom albums. "Various Artists" rows have no artist to verify, so
    they're matched on title alone (see _pick_compilation).
    """
    various = _is_various(artist)
    if various:
        # artist:"Various Artists" is what surfaces some compilations that a bare
        # title search buries, so all three run and their results are pooled
        queries = (f'artist:"Various Artists" album:"{album}"', f'album:"{album}"', album)
    else:
        queries = (f'artist:"{artist}" album:"{album}"', f"{artist} {album}")

    pool: dict[str, dict] = {}
    for query in queries:
        try:
            found = sp.search(q=query, type="album", limit=10)
        except Exception:
            continue
        items = (found or {}).get("albums", {}).get("items", [])
        if not various:
            match = _pick_candidate(items, artist, album)
            if match:
                return match
        for it in items:
            pool.setdefault(it["id"], it)
    return _pick_candidate(list(pool.values()), artist, album) if various else None


def fetch_tracks(sp: spotipy.Spotify, album_id: str) -> list[dict]:
    """All of an album's tracks with popularity. album_tracks() doesn't include
    popularity, so track ids are fetched first, then re-fetched in batches via
    tracks() (up to 50 at a time) which does return it.
    """
    track_ids = []
    page = sp.album_tracks(album_id, limit=50)
    while page:
        track_ids.extend(t["id"] for t in page["items"] if t.get("id"))
        page = sp.next(page) if page.get("next") else None

    tracks = []
    for i in range(0, len(track_ids), 50):
        full = sp.tracks(track_ids[i : i + 50])["tracks"]
        tracks.extend(
            {"id": t["id"], "name": t["name"], "popularity": t["popularity"]}
            for t in full if t
        )
    return tracks


def build_matches(df: pd.DataFrame, cache_path: Path, delay: float, limit: int | None) -> list[dict]:
    """Match + fetch every (artist, album) pair once. The cache holds the chosen
    album match and its full tracklist with popularity, so re-picking the best
    track needs no new API calls. A cached match that fails today's matching
    rules (_match_is_valid) is treated as a miss and re-searched, so tightening
    the rules only re-queries the affected albums.
    """
    pairs = df[["artist", "album"]].drop_duplicates()
    todo = pairs if limit is None else pairs.head(limit)

    cache = load_cache(cache_path)
    sp = spotify_client(need_user_auth=False)

    results = []
    bar = tqdm(todo.itertuples(index=False), total=len(todo), desc="Spotify match", unit="album")
    for i, (artist, album) in enumerate(bar, 1):
        key = f"{artist}::{album}"
        entry = cache.get(key)
        stale = False
        if isinstance(entry, dict) and entry.get("match"):
            confidence = _classify(artist, album, entry["match"])
            stale = confidence is None
            if not stale:
                entry["match"]["confidence"] = confidence      # refresh a label from older rules
        if not isinstance(entry, dict) or stale:
            match = find_album(sp, artist, album)
            tracks = fetch_tracks(sp, match["id"]) if match else []
            entry = {"match": match, "tracks": tracks}
            cache[key] = entry
            time.sleep(delay)
            if i % 25 == 0:
                save_cache(cache, cache_path)

        results.append({"artist": artist, "album": album, **entry})

    save_cache(cache, cache_path)
    return results


def pick_best_track(entry: dict) -> dict | None:
    tracks = entry.get("tracks") or []
    return max(tracks, key=lambda t: t["popularity"]) if tracks else None


def build_report(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        match = r.get("match")
        best = pick_best_track(r)
        rows.append({
            "artist": r["artist"],
            "album": r["album"],
            "matched": match is not None,
            "confidence": (match or {}).get("confidence"),
            "spotify_artist": (match or {}).get("artist"),
            "spotify_album": (match or {}).get("name"),
            "track_name": best["name"] if best else None,
            "track_id": best["id"] if best else None,
            "popularity": best["popularity"] if best else None,
        })
    return pd.DataFrame(rows)


def select_track_ids(report: pd.DataFrame, exact_only: bool) -> tuple[list[str], int]:
    """Track ids to push, in report order, without repeats. Returns (ids, number
    of duplicates dropped) — one song can be matched for several albums (e.g. a
    compilation that overlaps another), and adding it twice would repeat it.
    """
    rows = report[report["track_id"].notna()]
    if exact_only:
        rows = rows[rows["confidence"] == "exact"]
    ids = rows["track_id"].tolist()
    unique = list(dict.fromkeys(ids))
    return unique, len(ids) - len(unique)


# ----------------------------------------------------------------------------- cli

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("selection", help="a fork.py selection TSV, e.g. filtered/fork_selection_....tsv")
    p.add_argument("--cache", default=CACHE_FILE, help=f"Spotify match cache file (default {CACHE_FILE})")
    p.add_argument("--delay", type=float, default=0.2, help="seconds between Spotify calls (default 0.2)")
    p.add_argument("--limit", type=int, help="only match the first N albums (testing)")
    p.add_argument("--out", help=f"report TSV path (default: {PLAYLIST_DIR}/<selection>_spotify.tsv)")
    p.add_argument("--push", action="store_true", help="create/fill the real playlist (needs browser login)")
    p.add_argument("--playlist-name", help="creates a new playlist (with --push)")
    p.add_argument("--playlist-id", help="add to an existing playlist instead (with --push)")
    p.add_argument("--public", action="store_true", help="new playlist is public (default: private)")
    p.add_argument("--exact-only", action="store_true",
                   help="with --push: only add albums whose title matched exactly, skip 'fuzzy' ones")
    a = p.parse_args()

    selection_path = Path(a.selection)
    if not selection_path.exists():
        sys.exit(f"{selection_path} not found.")
    df = pd.read_csv(selection_path, sep="\t")

    results = build_matches(df, Path(a.cache), a.delay, a.limit)
    report = build_report(results)

    out_path = Path(a.out) if a.out else Path(PLAYLIST_DIR) / f"{selection_path.stem}_spotify.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, sep="\t", index=False)

    matched = int(report["matched"].sum())
    fuzzy = int((report["confidence"] == "fuzzy").sum())
    log(f"{matched}/{len(report)} albums matched on Spotify ({fuzzy} of those only fuzzily — review before trusting)")
    log(f"report -> {out_path}")

    if not a.push:
        log("Preview only — nothing was changed on your Spotify account.")
        log("Re-run with --push --playlist-name '...' (or --playlist-id ...) to actually create/fill the playlist.")
        return

    if not a.playlist_name and not a.playlist_id:
        sys.exit("--push requires --playlist-name (new playlist) or --playlist-id (existing playlist)")

    track_ids, dupes = select_track_ids(report, a.exact_only)
    if dupes:
        log(f"dropped {dupes} duplicate track(s) (same song matched for more than one album)")
    if not track_ids:
        sys.exit("No matched tracks to add — nothing to push.")
    log(f"pushing {len(track_ids)} tracks" + (" (exact matches only)" if a.exact_only else ""))

    sp_user = spotify_client(need_user_auth=True)
    if a.playlist_id:
        playlist_id = a.playlist_id
    else:
        me = sp_user.current_user()
        playlist = sp_user.user_playlist_create(me["id"], a.playlist_name, public=a.public)
        playlist_id = playlist["id"]
        log(f"created playlist '{a.playlist_name}' -> {playlist['external_urls']['spotify']}")

    for i in range(0, len(track_ids), 100):
        sp_user.playlist_add_items(playlist_id, track_ids[i : i + 100])
    log(f"added {len(track_ids)} tracks -> playlist {playlist_id}")


if __name__ == "__main__":
    main()
