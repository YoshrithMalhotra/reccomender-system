"""Item-based collaborative filtering and a hybrid content + CF recommender.

Each anime is represented by its column of the user x item ratings matrix,
after subtracting every user's mean rating (so "rated 4 by a harsh critic"
counts for more than "rated 4 by someone who gives everything 5"). Two anime
are similar when the same people liked (or disliked) both: adjusted cosine
similarity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .engine import DEFAULT_DATA, CrunchyrollRecommender, Weights, profile_similarity

LIKE_THRESHOLD = 4  # ratings at or above this count as "liked" when using a user's history
DISLIKE_THRESHOLD = 2  # at or below this count as "disliked"


class ItemCF:
    def __init__(self, ratings: pd.DataFrame, titles: list[str]):
        """`ratings` has columns user_id, title, rating; `titles` fixes the item order."""
        position = {t: i for i, t in enumerate(titles)}
        known = ratings[ratings["title"].isin(position)]
        known = known.drop_duplicates(subset=["user_id", "title"], keep="last")

        users = known["user_id"].astype("category")
        centred = known["rating"] - known.groupby("user_id")["rating"].transform("mean")

        # items x users, so each row is an item vector
        self.item_vectors = csr_matrix(
            (centred.to_numpy(dtype=float), (known["title"].map(position).to_numpy(), users.cat.codes.to_numpy())),
            shape=(len(titles), len(users.cat.categories)),
        )
        self.rating_counts = np.bincount(known["title"].map(position), minlength=len(titles))

    def similarity(self, liked_idx: list[int], disliked_idx: list[int]) -> np.ndarray:
        return profile_similarity(self.item_vectors, liked_idx, disliked_idx)


class HybridRecommender(CrunchyrollRecommender):
    """Blends collaborative filtering with the content-based similarity.

    `cf_weight` is the share given to CF (1.0 = pure CF, 0.0 = pure content).
    Titles with fewer than `min_ratings` ratings lean on content proportionally,
    so new or obscure shows can still be recommended.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        ratings: pd.DataFrame,
        cf_weight: float = 0.6,
        min_ratings: int = 20,
        weights: Weights | None = None,
    ):
        super().__init__(catalog, weights)
        self.ratings = ratings
        self.cf = ItemCF(ratings, self.catalog["title"].tolist())
        self.cf_weight = cf_weight
        self._item_cf_weight = cf_weight * np.minimum(1.0, self.cf.rating_counts / max(min_ratings, 1))

    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_DATA,
        ratings_path: str | Path | None = None,
        cf_weight: float = 0.6,
        weights: Weights | None = None,
    ):
        from .synthetic import DEFAULT_RATINGS

        return cls(pd.read_csv(path), pd.read_csv(ratings_path or DEFAULT_RATINGS), cf_weight, weights=weights)

    def _similarity(self, liked_idx: list[int], disliked_idx: list[int]) -> np.ndarray:
        content = super()._similarity(liked_idx, disliked_idx)
        if self.cf_weight == 0:
            return content
        cf = self.cf.similarity(liked_idx, disliked_idx)
        w = self._item_cf_weight
        return w * cf + (1 - w) * content

    def history(self, user_id) -> pd.DataFrame:
        seen = self.ratings[self.ratings["user_id"] == user_id]
        if seen.empty:
            raise KeyError(f"No ratings for user {user_id}")
        return seen.sort_values("rating", ascending=False)

    def for_user_id(self, user_id, n: int = 10, genres=None, min_rating=None) -> pd.DataFrame:
        """Recommendations for a user in the ratings data, from everything they rated."""
        seen = self.history(user_id)
        liked = seen.loc[seen["rating"] >= LIKE_THRESHOLD, "title"].tolist()
        if not liked:  # nothing rated highly: fall back to their best-rated shows
            liked = seen.loc[seen["rating"] == seen["rating"].max(), "title"].tolist()
        disliked = seen.loc[seen["rating"] <= DISLIKE_THRESHOLD, "title"].tolist()

        liked_idx = [self._idx(t) for t in liked]
        disliked_idx = [self._idx(t) for t in disliked]
        exclude = {self._idx(t) for t in seen["title"]}  # never recommend something already watched
        return self._rank(self._similarity(liked_idx, disliked_idx), exclude, n, genres, min_rating)
