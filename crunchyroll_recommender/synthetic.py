"""Generate a synthetic user-ratings dataset for the catalog.

Real Crunchyroll viewing data is not public, so this simulates it. Each fake
user belongs to one or two hidden "taste groups" (battle shonen fans, cozy
watchers, romcom fans, ...). Groups are deliberately defined by hand-picked
titles rather than by genre, so some of them cut across genres - for example
the cozy group mixes Laid-Back Camp, Frieren, Haikyu!! and The Apothecary
Diaries. Collaborative filtering can learn those links from co-watching;
content-based filtering cannot.

Users mostly watch titles from their groups (more often popular ones), rate
those highly, and rate the occasional out-of-group title lower.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RATINGS = Path(__file__).resolve().parent.parent / "data" / "synthetic_ratings.csv"

TASTE_GROUPS: dict[str, list[str]] = {
    "battle_shonen": [
        "Jujutsu Kaisen", "Demon Slayer: Kimetsu no Yaiba", "Chainsaw Man", "My Hero Academia",
        "Black Clover", "Naruto", "Naruto Shippuden", "Bleach", "One Piece", "Solo Leveling",
        "Kaiju No. 8", "Dandadan", "Hell's Paradise", "Fire Force", "Wind Breaker",
        "Dragon Ball Super", "Jojo's Bizarre Adventure",
    ],
    "isekai": [
        "That Time I Got Reincarnated as a Slime", "Re:ZERO -Starting Life in Another World-",
        "KONOSUBA -God's blessing on this wonderful world!", "Mushoku Tensei: Jobless Reincarnation",
        "The Rising of the Shield Hero", "Overlord", "Sword Art Online", "Log Horizon",
        "The Eminence in Shadow", "Solo Leveling", "Mashle: Magic and Muscles",
    ],
    "cozy": [
        "Laid-Back Camp", "Frieren: Beyond Journey's End", "Bocchi the Rock!", "Skip and Loafer",
        "Mushishi", "Nichijou - My Ordinary Life", "Spy x Family", "The Apothecary Diaries",
        "Ranking of Kings", "Dr. Stone", "Haikyu!!",
    ],
    "romance": [
        "Kaguya-sama: Love is War", "Horimiya", "Toradora!", "My Dress-Up Darling", "Fruits Basket",
        "Rascal Does Not Dream of Bunny Girl Senpai", "Your Lie in April", "Clannad After Story",
        "Spy x Family", "Dandadan",
    ],
    "mind_games": [
        "Death Note", "Code Geass", "Steins;Gate", "Psycho-Pass", "The Promised Neverland", "Erased",
        "Odd Taxi", "The Apothecary Diaries", "Attack on Titan", "Banana Fish",
    ],
    "sports": [
        "Haikyu!!", "Kuroko's Basketball", "Blue Lock", "Yuri!!! on Ice", "Free! - Iwatobi Swim Club",
        "Wind Breaker", "Assassination Classroom",
    ],
    "prestige": [
        "Vinland Saga", "Fullmetal Alchemist: Brotherhood", "Attack on Titan", "Cowboy Bebop",
        "Neon Genesis Evangelion", "Made in Abyss", "Steins;Gate", "Frieren: Beyond Journey's End",
        "Mushishi", "Banana Fish", "Hunter x Hunter", "Gintama",
    ],
    "comedy": [
        "Gintama", "One-Punch Man", "Mob Psycho 100", "KONOSUBA -God's blessing on this wonderful world!",
        "Nichijou - My Ordinary Life", "Mashle: Magic and Muscles", "Assassination Classroom",
        "Kaguya-sama: Love is War", "Spy x Family", "Bocchi the Rock!",
    ],
}


def generate_ratings(
    catalog: pd.DataFrame,
    n_users: int = 2000,
    min_watched: int = 6,
    max_watched: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of (user_id, title, rating) with ratings from 1 to 5."""
    rng = np.random.default_rng(seed)
    titles = catalog["title"].tolist()
    position = {t: i for i, t in enumerate(titles)}
    unknown = {t for members in TASTE_GROUPS.values() for t in members} - set(position)
    if unknown:
        raise ValueError(f"taste groups reference titles missing from the catalog: {sorted(unknown)}")

    membership = np.zeros((len(TASTE_GROUPS), len(titles)))
    for g, members in enumerate(TASTE_GROUPS.values()):
        membership[g, [position[t] for t in members]] = 1.0

    # Popularity nudges which titles get watched and how well they are received.
    quality = pd.to_numeric(catalog["rating"], errors="coerce").fillna(4.5).to_numpy()
    popularity = np.exp(3 * (quality - quality.mean()))
    quality_bias = (quality - quality.mean()) * 2

    rows = []
    for user in range(1, n_users + 1):
        n_groups = rng.choice([1, 2], p=[0.6, 0.4])
        groups = rng.choice(len(TASTE_GROUPS), size=n_groups, replace=False)
        fit = membership[groups].max(axis=0)  # 1 if the title is in one of the user's groups

        appeal = (fit + 0.05) * popularity
        n_watched = min(int(rng.integers(min_watched, max_watched + 1)), len(titles))
        watched = rng.choice(len(titles), size=n_watched, replace=False, p=appeal / appeal.sum())

        generosity = rng.normal(0, 0.4)  # some users rate everything higher
        for i in watched:
            score = 2.3 + 1.9 * fit[i] + quality_bias[i] + generosity + rng.normal(0, 0.6)
            rows.append((user, titles[i], int(np.clip(round(score), 1, 5))))

    return pd.DataFrame(rows, columns=["user_id", "title", "rating"])
