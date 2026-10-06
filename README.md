# Crunchyroll Anime Recommender

A content-based recommender for anime on Crunchyroll. Give it a show you liked (or a whole watch history) and it suggests what to watch next.

## How it works

Each anime is turned into a feature vector from three TF-IDF blocks:

| Block | Source | Weight |
|---|---|---|
| Genres | `genres` column (pipe-separated, e.g. `Action\|Fantasy`) | 1.0 |
| Tags | `tags` column (pipe-separated themes such as `isekai`, `time travel`) | 0.8 |
| Synopsis | `description` column (words and bigrams) | 0.4 |

Recommendations are ranked by cosine similarity, blended with the title's rating (15% of the score) so strong shows win ties. With a watch history, the liked titles are averaged into a taste profile, and disliked titles are subtracted from it.

Titles are fuzzy-matched, so `kaguya`, `haikyu` or `jujutsu kaisen` all work.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Shows similar to one title
python -m crunchyroll_recommender similar "Jujutsu Kaisen"

# Recommendations from a watch history
python -m crunchyroll_recommender user --liked "Spy x Family" "Kaguya-sama" --disliked "Tokyo Ghoul"

# Filters: required genres (repeatable), minimum rating, result count
python -m crunchyroll_recommender similar "Death Note" --genre Thriller --min-rating 4.8 -n 5

# Top-rated (for new users with no history)
python -m crunchyroll_recommender top --genre Sports

# Search and list genres
python -m crunchyroll_recommender search slime
python -m crunchyroll_recommender genres
```

From Python:

```python
from crunchyroll_recommender import CrunchyrollRecommender

rec = CrunchyrollRecommender.from_csv()
rec.similar_to("Frieren", n=5)
rec.for_user(["Haikyu!!", "Bocchi the Rock!"], genres=["Comedy"])
```

## Data

`data/crunchyroll_anime.csv` is a hand-curated sample of 70 popular Crunchyroll titles. Ratings are approximate and synopses are written for this project.

To use a bigger catalog (for example a Crunchyroll dataset from Kaggle), point `--data` at a CSV with these columns:

- required: `title`, `genres` (pipe-separated)
- optional: `tags` (pipe-separated), `description`, `rating`, `year`, `episodes`

```bash
python -m crunchyroll_recommender --data my_catalog.csv similar "Naruto"
```

## Tests

```bash
python -m pytest
```
