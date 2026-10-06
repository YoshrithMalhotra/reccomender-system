# Crunchyroll Anime Recommender

A hybrid recommender for anime on Crunchyroll. Give it a show you liked (or a whole watch history) and it suggests what to watch next. It blends two approaches:

- **Content-based**: shows that look alike (genres, tags, synopsis).
- **Collaborative filtering**: shows that the same people liked ("viewers who loved X also loved Y").

## Content-based filtering

Each anime is turned into a feature vector from three TF-IDF blocks:

| Block | Source | Weight |
|---|---|---|
| Genres | `genres` column (pipe-separated, e.g. `Action\|Fantasy`) | 1.0 |
| Tags | `tags` column (pipe-separated themes such as `isekai`, `time travel`) | 0.8 |
| Synopsis | `description` column (words and bigrams) | 0.4 |

Recommendations are ranked by cosine similarity, blended with the title's rating (15% of the score) so strong shows win ties. With a watch history, the liked titles are averaged into a taste profile, and disliked titles are subtracted from it.

Titles are fuzzy-matched, so `kaguya`, `haikyu` or `jujutsu kaisen` all work.

## Collaborative filtering

Item-based CF over the user × anime ratings matrix. Each user's ratings are centred on their own average (a 4 from a harsh critic means more than a 4 from someone who rates everything 5), and two shows are similar when their columns point the same way: adjusted cosine similarity.

The final similarity is a blend:

```
similarity = w · CF + (1 − w) · content       w = 0.6 by default
```

Shows with fewer than 20 ratings get a proportionally smaller `w`, so new or obscure titles are still recommended on content alone instead of vanishing.

### Synthetic ratings

Crunchyroll doesn't publish viewing data, so `data/synthetic_ratings.csv` is **generated** (`crunchyroll_recommender/synthetic.py`): 2,000 fake users, ~31k ratings from 1 to 5. Each user belongs to one or two hidden taste groups (battle shonen, isekai, cozy, romance, mind games, sports, prestige, comedy). Groups are hand-picked title lists that deliberately cross genres. For example, the cozy group mixes Laid-Back Camp, Frieren, Bocchi the Rock!, Haikyu!! and The Apothecary Diaries. Popular shows get watched more, and users rate in-group shows higher.

Regenerate it with a different size or seed:

```bash
python -m crunchyroll_recommender generate-ratings --users 5000 --seed 7
```

To use real data instead, pass `--ratings` a CSV with columns `user_id,title,rating`, where titles match the catalog.

### Does it help?

`evaluate` hides one liked show for each of 300 users, trains on everything else, and checks whether the hidden show appears in that user's top 10:

| Model | Hit rate @10 |
|---|---|
| Content only | 0.62 |
| Hybrid (0.6 CF) | 0.86 |
| CF only | 0.91 |

Caveat: the synthetic users were generated from co-watching groups, which is exactly the pattern CF learns, so these numbers flatter CF. On real data the gap would be smaller. The hybrid is the default because pure CF can't recommend shows nobody has rated yet.

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

# Recommendations for a user in the ratings file, and what they've rated
python -m crunchyroll_recommender user --user-id 42
python -m crunchyroll_recommender history 42

# Pick the model: hybrid (default), cf, or content
python -m crunchyroll_recommender --mode cf similar "Laid-Back Camp"

# Compare the three models
python -m crunchyroll_recommender evaluate

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

import pandas as pd
from crunchyroll_recommender.collaborative import HybridRecommender

hybrid = HybridRecommender.from_csv(cf_weight=0.6)  # bundled catalog + synthetic ratings
hybrid.similar_to("Laid-Back Camp", n=5)
hybrid.for_user_id(42)
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

