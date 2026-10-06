import pandas as pd
import pytest

from crunchyroll_recommender.cli import main
from crunchyroll_recommender.collaborative import HybridRecommender, ItemCF
from crunchyroll_recommender.engine import DEFAULT_DATA
from crunchyroll_recommender.evaluate import compare, holdout_split
from crunchyroll_recommender.synthetic import generate_ratings


@pytest.fixture(scope="module")
def catalog():
    return pd.read_csv(DEFAULT_DATA)


@pytest.fixture(scope="module")
def ratings(catalog):
    return generate_ratings(catalog, n_users=600, seed=1)


def test_synthetic_ratings_shape(catalog, ratings):
    assert set(ratings.columns) == {"user_id", "title", "rating"}
    assert ratings["rating"].between(1, 5).all()
    assert set(ratings["title"]) <= set(catalog["title"])
    assert not ratings.duplicated(["user_id", "title"]).any()
    # Deterministic for a given seed.
    pd.testing.assert_frame_equal(ratings, generate_ratings(catalog, n_users=600, seed=1))


def test_cf_learns_co_watching_across_genres():
    # A and B share no genre, but the same users love both; C is watched by others.
    catalog = pd.DataFrame({
        "title": ["A", "B", "C"],
        "genres": ["Sports", "Mystery", "Sports"],
    })
    rows = [(u, "A", 5) for u in range(10)] + [(u, "B", 5) for u in range(10)]
    rows += [(u, "C", 1) for u in range(10)]
    rows += [(u, "C", 5) for u in range(10, 20)] + [(u, "A", 1) for u in range(10, 20)]
    ratings = pd.DataFrame(rows, columns=["user_id", "title", "rating"])

    content_top = HybridRecommender(catalog, ratings, cf_weight=0.0).similar_to("A", n=1)
    cf_top = HybridRecommender(catalog, ratings, cf_weight=1.0, min_ratings=1).similar_to("A", n=1)
    assert content_top.iloc[0]["title"] == "C"
    assert cf_top.iloc[0]["title"] == "B"


def test_unrated_titles_fall_back_to_content(catalog, ratings):
    unrated = "Tokyo Ghoul"
    sparse = ratings[ratings["title"] != unrated]
    cf = ItemCF(sparse, catalog["title"].tolist())
    assert cf.rating_counts[catalog.index[catalog["title"] == unrated][0]] == 0
    rec = HybridRecommender(catalog, sparse)
    # Pure CF would give it zero similarity; the content side still surfaces it.
    assert unrated in rec.similar_to("Chainsaw Man", n=10)["title"].tolist()


def test_for_user_id_skips_watched(catalog, ratings):
    rec = HybridRecommender(catalog, ratings)
    watched = set(rec.history(1)["title"])
    recs = rec.for_user_id(1, n=10)
    assert len(recs) == 10
    assert not watched & set(recs["title"])
    with pytest.raises(KeyError):
        rec.for_user_id(10**9)


def test_holdout_split_hides_one_liked_title_per_user(ratings):
    train, held = holdout_split(ratings, test_users=50)
    assert len(held) == 50 and held["user_id"].is_unique
    assert (held["rating"] >= 4).all()
    assert len(train) == len(ratings) - 50


def test_cf_beats_content_on_synthetic_data(catalog, ratings):
    scores = compare(catalog, ratings, test_users=150).set_index("model")["hit_rate@10"]
    assert scores["CF only"] > scores["content only"]
    assert scores["hybrid (0.6 CF)"] > scores["content only"]


def test_cli_hybrid_commands(tmp_path, capsys):
    path = tmp_path / "ratings.csv"
    assert main(["--ratings", str(path), "generate-ratings", "--users", "300"]) == 0
    assert main(["--ratings", str(path), "user", "--user-id", "5", "-n", "3"]) == 0
    assert main(["--ratings", str(path), "--mode", "cf", "similar", "Haikyu", "-n", "3"]) == 0
    assert main(["--ratings", str(path), "history", "999999"]) == 1
    assert main(["--ratings", str(tmp_path / "missing.csv"), "user", "--user-id", "5"]) == 1
    capsys.readouterr()
