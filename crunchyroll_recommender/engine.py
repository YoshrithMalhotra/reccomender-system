"""Content-based recommender for Crunchyroll anime.

Each title is turned into a feature vector built from three TF-IDF blocks:
genres, tags and the synopsis. Blocks are weighted so that genres and tags
matter more than incidental words in the description. Similarity between
titles is cosine similarity, and the final ranking blends that similarity
with the title's community rating.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "crunchyroll_anime.csv"

REQUIRED_COLUMNS = {"title", "genres"}
OPTIONAL_COLUMNS = {
    "tags": "",
    "description": "",
    "rating": np.nan,
    "year": np.nan,
    "episodes": np.nan,
}


@dataclass
class Weights:
    genres: float = 1.0
    tags: float = 0.8
    description: float = 0.4
    # Share of the final score taken by the (normalised) rating; the rest is similarity.
    rating: float = 0.15


def _split_pipe(value: str) -> list[str]:
    return [part.strip().lower() for part in str(value).split("|") if part.strip()]


def profile_similarity(vectors, liked_idx: list[int], disliked_idx: list[int]) -> np.ndarray:
    """Cosine similarity of every row of `vectors` to the mean of the liked rows,
    pushed away from the mean of the disliked rows."""
    profile = np.asarray(vectors[liked_idx].mean(axis=0)).reshape(1, -1)
    if disliked_idx:
        profile = profile - 0.5 * np.asarray(vectors[disliked_idx].mean(axis=0)).reshape(1, -1)
    if not np.any(profile):
        return np.zeros(vectors.shape[0])
    return cosine_similarity(profile, vectors).ravel()


class CrunchyrollRecommender:
    def __init__(self, catalog: pd.DataFrame, weights: Weights | None = None):
        missing = REQUIRED_COLUMNS - set(catalog.columns)
        if missing:
            raise ValueError(f"catalog is missing required columns: {sorted(missing)}")

        df = catalog.copy()
        for column, default in OPTIONAL_COLUMNS.items():
            if column not in df.columns:
                df[column] = default
        df["tags"] = df["tags"].fillna("")
        df["description"] = df["description"].fillna("")
        df = df.drop_duplicates(subset="title").reset_index(drop=True)

        self.catalog = df
        self.weights = weights or Weights()
        self._title_index = {t.lower(): i for i, t in enumerate(df["title"])}
        # Alternative names (e.g. the romanised Japanese title) also resolve to the row.
        if "alt_title" in df.columns:
            for i, alt in enumerate(df["alt_title"]):
                if isinstance(alt, str) and alt.strip():
                    self._title_index.setdefault(alt.strip().lower(), i)
        # Rows that may appear in results (e.g. only titles licensed by Crunchyroll).
        self.candidate_mask = np.ones(len(df), dtype=bool)
        self._features = self._build_features(df)
        self._rating_score = self._normalise_ratings(df["rating"])

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATA, weights: Weights | None = None):
        return cls(pd.read_csv(path), weights)

    # ------------------------------------------------------------------ features

    def _build_features(self, df: pd.DataFrame):
        w = self.weights
        # Pipe-separated fields are tokenised as whole labels so "slice of life"
        # stays one feature instead of three words.
        genre_vec = TfidfVectorizer(tokenizer=_split_pipe, token_pattern=None, lowercase=False)
        tag_vec = TfidfVectorizer(tokenizer=_split_pipe, token_pattern=None, lowercase=False)
        desc_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

        blocks = []
        for vec, column, weight in (
            (genre_vec, "genres", w.genres),
            (tag_vec, "tags", w.tags),
            (desc_vec, "description", w.description),
        ):
            try:
                blocks.append(vec.fit_transform(df[column].astype(str)) * weight)
            except ValueError:  # column has no usable text (e.g. absent from the CSV)
                continue
        if not blocks:
            raise ValueError("catalog has no genre, tag or description text to learn from")
        self.genre_vocabulary = sorted(getattr(genre_vec, "vocabulary_", {}))
        return normalize(hstack(blocks).tocsr())

    @staticmethod
    def _normalise_ratings(ratings: pd.Series) -> np.ndarray:
        r = pd.to_numeric(ratings, errors="coerce")
        if r.notna().sum() == 0:
            return np.zeros(len(r))
        r = r.fillna(r.median())
        span = r.max() - r.min()
        return ((r - r.min()) / span).to_numpy() if span else np.ones(len(r))

    # ------------------------------------------------------------------ lookup

    def find_title(self, query: str) -> str:
        """Resolve a possibly misspelled or partial title to a catalog title."""
        key = query.strip().lower()
        if key in self._title_index:
            return self.catalog.at[self._title_index[key], "title"]

        substring_hits = [t for t in self._title_index if key in t]
        if len(substring_hits) == 1:
            return self.catalog.at[self._title_index[substring_hits[0]], "title"]

        close = difflib.get_close_matches(key, list(self._title_index), n=1, cutoff=0.5)
        if close:
            return self.catalog.at[self._title_index[close[0]], "title"]
        if substring_hits:
            return self.catalog.at[self._title_index[min(substring_hits, key=len)], "title"]
        raise KeyError(f"No anime matching '{query}' in the catalog")

    def search(self, query: str, limit: int = 10) -> list[str]:
        key = query.strip().lower()
        hits = [t for t in self._title_index if key in t]
        hits += [t for t in difflib.get_close_matches(key, list(self._title_index), n=limit, cutoff=0.4)
                 if t not in hits]
        return [self.catalog.at[self._title_index[t], "title"] for t in hits[:limit]]

    def _idx(self, title: str) -> int:
        return self._title_index[self.find_title(title).lower()]

    # ------------------------------------------------------------------ ranking

    def _similarity(self, liked_idx: list[int], disliked_idx: list[int]) -> np.ndarray:
        """Similarity of every title to a taste profile built from liked/disliked rows."""
        return profile_similarity(self._features, liked_idx, disliked_idx)

    def _rank(
        self,
        similarity: np.ndarray,
        exclude: set[int],
        n: int,
        genres: list[str] | None,
        min_rating: float | None,
    ) -> pd.DataFrame:
        w = self.weights.rating
        score = (1 - w) * similarity + w * self._rating_score

        mask = self.candidate_mask.copy()
        mask[list(exclude)] = False
        if genres:
            wanted = {g.lower() for g in genres}
            mask &= self.catalog["genres"].map(lambda g: wanted <= set(_split_pipe(g))).to_numpy()
        if min_rating is not None:
            mask &= pd.to_numeric(self.catalog["rating"], errors="coerce").fillna(0).to_numpy() >= min_rating

        candidates = np.flatnonzero(mask)
        order = candidates[np.argsort(-score[candidates], kind="stable")][:n]
        result = self.catalog.iloc[order][["title", "genres", "rating", "year"]].copy()
        result["similarity"] = similarity[order].round(3)
        result["score"] = score[order].round(3)
        return result.reset_index(drop=True)

    def similar_to(
        self,
        title: str,
        n: int = 10,
        genres: list[str] | None = None,
        min_rating: float | None = None,
    ) -> pd.DataFrame:
        """Titles most similar to a single anime."""
        idx = self._idx(title)
        return self._rank(self._similarity([idx], []), {idx}, n, genres, min_rating)

    def for_user(
        self,
        liked: list[str],
        disliked: list[str] | None = None,
        n: int = 10,
        genres: list[str] | None = None,
        min_rating: float | None = None,
    ) -> pd.DataFrame:
        """Recommendations from a watch history: liked titles pull, disliked titles push."""
        if not liked:
            raise ValueError("at least one liked title is required")
        liked_idx = [self._idx(t) for t in liked]
        disliked_idx = [self._idx(t) for t in disliked or []]

        sim = self._similarity(liked_idx, disliked_idx)
        return self._rank(sim, set(liked_idx) | set(disliked_idx), n, genres, min_rating)

    def top_rated(self, n: int = 10, genres: list[str] | None = None) -> pd.DataFrame:
        """Cold-start fallback: best-rated titles, optionally within given genres."""
        ranked = self._rank(np.zeros(len(self.catalog)), set(), n, genres, None)
        return ranked.drop(columns=["similarity", "score"])
