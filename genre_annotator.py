import os
import re
from functools import lru_cache

import pylast
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('LASTFM_API_KEY')
api_secret = os.getenv('LASTFM_API_SECRET')
username = os.getenv('LASTFM_USERNAME')
password = os.getenv('LASTFM_PASSWORD')

network = pylast.LastFMNetwork(
    api_key=api_key,
    api_secret=api_secret,
    username=username,
    password_hash=pylast.md5(password) if password else None
)

MAX_GENRES = 6

# Pitchfork's own genre column only ever holds these 9 values (checked against
# the real dataset). Last.fm's tags are much finer-grained free text ("indie
# rock", "post-punk", "trip-hop", ...), so fold any tag containing one of a
# bucket's keywords onto that bucket instead of keeping hundreds of distinct
# subgenre strings that don't match Pitchfork's own vocabulary.
#
# Order matters: earlier buckets win when a tag's keywords span more than one,
# e.g. "pop rap" hits Rap's "rap" and Pop/R&B's "pop" — Rap is checked first,
# so it wins. Broad, single-word buckets (Rock, Pop/R&B) are deliberately last
# so a genuinely more specific genre it also matches ("pop rap", "folk rock")
# doesn't get swallowed by them. This is a heuristic, not a lookup table — it
# won't be right for every edge case (e.g. "electropop" lands in Electronic,
# not Pop/R&B), and untagged tags just fall through Title Cased as-is.
GENRE_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Metal", ("metal", "grindcore", "thrash", "doom", "sludge",  "industrial", "black metal", 
               "death metal", "heavy metal", "metalcore", "post-metal", "post metal", 
               "nu metal", "slowcore", "mathcore", "power metal", "progressive metal", "prog metal", 
               "gammarec", "neocrust")),
    ("Rap", ("rap", "hip hop", "hip-hop", "hiphop", "grime", "totec radio")),
    ("Jazz", ("jazz", "bebop", "swing", "jazzfanatics", "jazz fusion", "free jazz", "smooth jazz", "vocal jazz", "latin jazz",
              "jazz funk", "jazz rock", "jazz rap", "jazzcore", "jazz blues", "jazz pop", "jazz soul")),
    ("Experimental", ("experimental", "noise", "avant-garde", "avant garde", "drone", 
                      "psychedelic", "krautrock", "neo-psychedelia", "avantgarde", 
                      "avantgarde rock", "avantgarde jazz", "contemporary classical", 
                      "modern classical", "electroacoustic", "musique concrète", "minimalistic")),
    ("Electronic", (
        "electronic", "electronica", "techno", "house", "idm", "edm",
        "drum and bass", "dnb", "trance",
        "downtempo", "trip hop", "trip-hop", "synth", "electro", "lo-fi", "lo fi", "chillwave", "chill wave", 
        "synthwave", "synth wave", "minimal", "glitch", "future bass", "vaporwave", "vapor wave", 
        "chillout", "chill out", "breakbeat", "hardstyle", "hard trance", "hard trance", 
        "acid house", "acid", "electroclash", "electro clash", "jungle", "gabber", "gabba", "big beat",
        "dubstep", "drumstep", "trapstep", "future garage", "ghetto house",
    )),
    ("Folk/Country", (
        "folk", "country", "americana", "bluegrass",
        "singer-songwriter", "singer songwriter", "americana folk", 
        "folk rock", "alt-country", "alt country"
    )),
    ("Global", (
        "world music", "afrobeat", "afrobeats", "reggae", "latin",
        "cumbia", "bossa nova", "reggaeton", "highlife", "dub", 
        "african", "africa", "afro", "roots reggae",
    )),
    ("Rock", ("rock", "punk", "grunge", "shoegaze", "emo",  "emocore",
              "indie rock", "post-punk", "post punk", "garage rock", 
              "hard rock", "new wave", "psychedelic rock", "prog rock", 
              "progressive rock", "post-rock", "post rock", "indie", 
              "alt rock", "alternative rock", "classic rock", "blues rock", 
              "folk rock", "surf rock", "alternative", "pop punk", 
              "glam rock", "goth rock", "garage punk", "math rock", "noise rock", "post hardcore",
              "post-hardcore", "stoner rock", "southern rock", "garage", "blues", "progressive",
              "pop rock", "hardcore", "hardcore punk", "emo rock"
              )),
    ("Pop/R&B", ("pop", "r&b", "rnb", "r n b", "r and b", "soul", 
                 "funk", "disco", "indie pop", "synthpop", "synth pop", "dance", 
                 "dance pop", "electropop", "electro pop", "new wave pop", 
                 "bubblegum pop", "k-pop", "dream pop", "pop soul", "pop rap", "pop r&b", "trap")),
]

# Last.fm tags are crowd-sourced and full of non-genre noise: years/decades,
# "seen live", "favourite albums", subjective praise, etc. Reject those rather
# than trying to enumerate every real genre.
JUNK_TAG_SUBSTRINGS = (
    "seen live", "favourite", "favorite", "best of", "of all time",
    "check out", "listen to", "spotify", "amazing", "awesome", "masterpiece",
    "perfect album", "under rated", "under-rated", "underrated", "overrated",
    "beautiful", "classic album", "album of the year", "flawless", "10/10", 
    "american", "canadian", "australian", "all", "female vocalist", 
    "male vocalist", "female vocals", "male vocals", "female singer", "male singer", 
    "japanese", "french", "german", "italian", "spanish", "swedish", "norwegian", "dutch", "chicago", "new york", 
    "london", "paris", "berlin", "tokyo", "seoul", "moscow", "brazilian", "soundtrack", "sound track", 
    "score", "instrumental", "instrumentals", "cover", "covers", "remix", "remixes", 
    "canada", "usa", "uk", "us", "british", "american", "canadian", "australian", "united states", 
    "united kingdom", "european", "europe", "bass", "guitar", "drums", "piano", 
    "violin", "saxophone", "trumpet", "flute", "free improvisation", "compilation", 
    "california", "texas", "florida", "new jersey", "new mexico", "new england", "japan", "improvisation", 
    "my albums", "my music", "my collection", "my library", "my playlist", "my favorite albums",
    "atmospheric", "relaxing", "relaxation", "mellow", "ambient music", 
    "new zealand", "austria", "switzerland", "belgium", "finland", "denmark", "norway", "sweden", "classical", 
    "lounge", "beats"
)

def _is_genre_like(tag: str, artist: str, album: str = "") -> bool:
    t = tag.strip().lower()
    if not t:
        return False
    if t == artist.strip().lower() or (album and t == album.strip().lower()):
        return False
    if re.fullmatch(r"\d{1,4}s?", t):               # "1997", "97", "90s", "2000s"
        return False
    if any(j in t for j in JUNK_TAG_SUBSTRINGS):
        return False
    return True


def _canonicalize(tag: str) -> str | None:
    """Map a tag onto one of Pitchfork's genre buckets, or None if it's not a recognized genre.

    GENRE_BUCKETS' keyword lists double as a whitelist: Last.fm's tag vocabulary
    is too noisy to map manually, so instead of keeping unmatched tags as loose
    text (which just reproduces that noise), anything that doesn't hit a known
    genre keyword is dropped rather than passed through.
    """
    t = tag.strip().lower()
    for bucket, keywords in GENRE_BUCKETS:
        if any(k in t for k in keywords):
            return bucket
    return None


def _top_tags(get_taggable):
    """Call a pylast get_album/get_artist thunk and return its tag names, or [] on any failure."""
    try:
        return [t.item.get_name() for t in get_taggable().get_top_tags(limit=15)]
    except pylast.WSError:
        return []
    except Exception:
        return []


def fetch_raw_tags(artist: str, album: str) -> dict:
    """Unfiltered Last.fm tag names for an album — the expensive, network-bound half.

    Kept separate from canonicalize_tags() so the raw tags can be cached and
    replayed through whatever the *current* bucketing/filtering rules are,
    instead of freezing in whatever a final genre string looked like the day
    it was first fetched. Tries the album's own tags first; Pitchfork album
    titles often don't match Last.fm exactly (reissues, "Deluxe Edition",
    compilations), so falls back to the artist's tags when the album's tags
    don't yield anything genre-like.
    """
    album_tags = _top_tags(lambda: network.get_album(artist, album))
    artist_tags = None
    resolves = any(
        _canonicalize(g) for g in album_tags if _is_genre_like(g, artist, album)
    )
    if not resolves:
        artist_tags = _top_tags(lambda: network.get_artist(artist))
    return {"album_tags": album_tags, "artist_tags": artist_tags}


def canonicalize_tags(raw: dict, artist: str, album: str) -> str:
    """Turn fetch_raw_tags()'s output into a final "; "-joined genre string.

    Pure/offline — no network calls — so it can be re-run over cached raw tags
    any time the bucketing rules (GENRE_BUCKETS, junk filters, ...) change.
    """
    genres = [g for g in raw.get("album_tags") or [] if _is_genre_like(g, artist, album)]
    if raw.get("artist_tags") and not any(_canonicalize(g) for g in genres):
        genres = [g for g in raw["artist_tags"] if _is_genre_like(g, artist)]

    # Whitelist onto Pitchfork's own genre buckets, dropping anything that isn't
    # a recognized genre keyword, then dedupe (several tags can map to one bucket).
    seen, deduped = set(), []
    for g in (_canonicalize(g) for g in genres):
        if g and g not in seen:
            seen.add(g)
            deduped.append(g)

    return "; ".join(deduped[:MAX_GENRES])


@lru_cache(maxsize=4096)
def get_genres(artist: str, album: str) -> str:
    """Genre tags for an album, from Last.fm's crowd tags. See fetch_raw_tags/canonicalize_tags."""
    return canonicalize_tags(fetch_raw_tags(artist, album), artist, album)

def get_genres_from_filename(filename):
    if not filename:
        return ""
    filename = filename.strip().split('/')[-1]  
    filename = filename.split('.')[0]  
    if filename.startswith('aquarium'):
        return "russian rock"
    if filename.startswith('songs_of_protest'):
        return "russian rock"

    genre = filename.replace('__', ',')
    genre = genre.replace('_', ' ')
    genre = genre.lower() 

    return genre 

# print(get_genres_from_filename("lists/Pop__russian_rock__Indie.csv"))  # Example usage