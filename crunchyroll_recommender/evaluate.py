"""Offline evaluation: does collaborative filtering actually help?

For a sample of users, one title they liked is hidden. The model is trained
on the remaining ratings and asked for top-N recommendations from the rest of
the user's history. A "hit" is when the hidden title shows up in the top N.
Hit rate is reported for content-only, CF-only and hybrid blends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .collaborative import LIKE_THRESHOLD, HybridRecommender


def holdout_split(ratings: pd.DataFrame, test_users: int = 300, seed: int = 0):
    """Hide one liked title for each sampled user. Returns (train, held_out)."""
    rng = np.random.default_rng(seed)
    liked = ratings[ratings["rating"] >= LIKE_THRESHOLD]
    # Need at least two likes: one to hide, one to build the profile from.
    eligible = liked.groupby("user_id").size().loc[lambda s: s >= 2].index.to_numpy()
    users = rng.choice(eligible, size=min(test_users, len(eligible)), replace=False)

    candidates = liked[liked["user_id"].isin(users)]
    # Shuffle, then keep the first liked row per user: one random hidden title each.
    held = candidates.sample(frac=1, random_state=int(rng.integers(1 << 31))).drop_duplicates("user_id")
    train = ratings.drop(index=held.index)
    return train, held


def hit_rate(catalog: pd.DataFrame, ratings: pd.DataFrame, cf_weight: float, k: int = 10, **split_kw) -> float:
    train, held = holdout_split(ratings, **split_kw)
    model = HybridRecommender(catalog, train, cf_weight=cf_weight)
    hits = [
        target in model.for_user_id(user, n=k)["title"].tolist()
        for user, target in zip(held["user_id"], held["title"])
    ]
    return float(np.mean(hits))


def compare(catalog: pd.DataFrame, ratings: pd.DataFrame, k: int = 10, **split_kw) -> pd.DataFrame:
    blends = {"content only": 0.0, "hybrid (0.6 CF)": 0.6, "CF only": 1.0}
    return pd.DataFrame(
        {"model": list(blends), f"hit_rate@{k}": [round(hit_rate(catalog, ratings, w, k, **split_kw), 3)
                                                   for w in blends.values()]}
    )
