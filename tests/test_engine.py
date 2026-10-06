import pandas as pd
import pytest

from crunchyroll_recommender import CrunchyrollRecommender
from crunchyroll_recommender.cli import main


@pytest.fixture(scope="module")
def rec():
    return CrunchyrollRecommender.from_csv()


def test_fuzzy_title_lookup(rec):
    assert rec.find_title("jujutsu kaisen") == "Jujutsu Kaisen"
    assert rec.find_title("kaguya") == "Kaguya-sama: Love is War"
    assert rec.find_title("Haikyu") == "Haikyu!!"
    with pytest.raises(KeyError):
        rec.find_title("zzzzqq")


def test_similar_excludes_query_and_finds_same_genre(rec):
    result = rec.similar_to("Haikyu!!", n=3)
    assert "Haikyu!!" not in result["title"].tolist()
    assert result.iloc[0]["title"] == "Kuroko's Basketball"
    assert result["score"].is_monotonic_decreasing


def test_genre_and_rating_filters(rec):
    result = rec.similar_to("Death Note", n=20, genres=["Thriller"], min_rating=4.8)
    assert not result.empty
    assert all("Thriller" in g for g in result["genres"])
    assert (result["rating"] >= 4.8).all()


def test_user_profile_excludes_watched_and_respects_dislikes(rec):
    liked = ["Spy x Family", "Kaguya-sama: Love is War"]
    result = rec.for_user(liked, disliked=["Tokyo Ghoul"], n=10)
    titles = result["title"].tolist()
    assert not set(liked + ["Tokyo Ghoul"]) & set(titles)
    assert "Romance" in result.iloc[0]["genres"] or "Comedy" in result.iloc[0]["genres"]


def test_for_user_requires_liked(rec):
    with pytest.raises(ValueError):
        rec.for_user([])


def test_minimal_catalog_without_optional_columns():
    df = pd.DataFrame({"title": ["A", "B", "C"], "genres": ["Action", "Action|Comedy", "Romance"]})
    rec = CrunchyrollRecommender(df)
    assert rec.similar_to("A", n=1).iloc[0]["title"] == "B"


def test_missing_required_column():
    with pytest.raises(ValueError):
        CrunchyrollRecommender(pd.DataFrame({"title": ["A"]}))


def test_cli(capsys):
    assert main(["similar", "Naruto", "-n", "3"]) == 0
    assert "Naruto Shippuden" in capsys.readouterr().out
    assert main(["similar", "zzzzqq"]) == 1
