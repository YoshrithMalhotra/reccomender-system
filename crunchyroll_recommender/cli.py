"""Command-line interface.

Examples:
    python -m crunchyroll_recommender similar "Jujutsu Kaisen"
    python -m crunchyroll_recommender user --liked "Spy x Family" "Kaguya-sama" --disliked "Tokyo Ghoul"
    python -m crunchyroll_recommender top --genre Sports
    python -m crunchyroll_recommender user --user-id 42
    python -m crunchyroll_recommender evaluate
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .collaborative import HybridRecommender
from .engine import DEFAULT_DATA, CrunchyrollRecommender
from .synthetic import DEFAULT_RATINGS, generate_ratings

MODES = {"content": 0.0, "hybrid": 0.6, "cf": 1.0}


def _print(df: pd.DataFrame) -> None:
    if df.empty:
        print("No recommendations match those filters.")
        return
    with pd.option_context("display.max_colwidth", 45, "display.width", 140):
        print(df.to_string(index=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crunchyroll-recommender", description="Crunchyroll anime recommender")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path to a catalog CSV")
    parser.add_argument("--ratings", default=str(DEFAULT_RATINGS),
                        help="path to a user ratings CSV (user_id,title,rating)")
    parser.add_argument("--mode", choices=MODES, default="hybrid",
                        help="content-based, collaborative filtering, or a blend (default)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p):
        p.add_argument("-n", type=int, default=10, help="number of results")
        p.add_argument("--genre", action="append", help="require a genre (repeatable)")
        p.add_argument("--min-rating", type=float, help="minimum rating")

    p = sub.add_parser("similar", help="anime similar to one title")
    p.add_argument("title")
    add_filters(p)

    p = sub.add_parser("user", help="recommendations from liked/disliked titles or a user's rating history")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--liked", nargs="+")
    who.add_argument("--user-id", type=int, help="a user from the ratings file")
    p.add_argument("--disliked", nargs="+", default=[])
    add_filters(p)

    p = sub.add_parser("history", help="show what a user in the ratings file has rated")
    p.add_argument("user_id", type=int)

    p = sub.add_parser("evaluate", help="compare content, hybrid and CF by hiding one liked title per user")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--test-users", type=int, default=300)

    p = sub.add_parser("generate-ratings", help="(re)generate the synthetic ratings file")
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("top", help="top-rated titles")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--genre", action="append")

    p = sub.add_parser("search", help="search titles in the catalog")
    p.add_argument("query")

    sub.add_parser("genres", help="list available genres")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = pd.read_csv(args.data)

    if args.command == "generate-ratings":
        ratings = generate_ratings(catalog, n_users=args.users, seed=args.seed)
        ratings.to_csv(args.ratings, index=False)
        print(f"Wrote {len(ratings)} ratings from {args.users} users to {args.ratings}")
        return 0

    try:
        ratings = pd.read_csv(args.ratings)
    except FileNotFoundError:
        ratings = None
    if args.command == "evaluate":
        if ratings is None:
            print(f"No ratings file at {args.ratings}", file=sys.stderr)
            return 1
        from .evaluate import compare

        print(compare(catalog, ratings, k=args.k, test_users=args.test_users).to_string(index=False))
        return 0

    if ratings is not None:
        rec = HybridRecommender(catalog, ratings, cf_weight=MODES[args.mode])
    elif args.command == "history" or getattr(args, "user_id", None) is not None:
        print(f"No ratings file at {args.ratings}", file=sys.stderr)
        return 1
    else:
        rec = CrunchyrollRecommender(catalog)

    try:
        if args.command == "similar":
            print(f"Because you watched {rec.find_title(args.title)}:\n")
            _print(rec.similar_to(args.title, args.n, args.genre, args.min_rating))
        elif args.command == "user" and args.user_id is not None:
            _print(rec.for_user_id(args.user_id, args.n, args.genre, args.min_rating))
        elif args.command == "user":
            _print(rec.for_user(args.liked, args.disliked, args.n, args.genre, args.min_rating))
        elif args.command == "history":
            _print(rec.history(args.user_id).reset_index(drop=True))
        elif args.command == "top":
            _print(rec.top_rated(args.n, args.genre))
        elif args.command == "search":
            print("\n".join(rec.search(args.query)) or "No matches.")
        elif args.command == "genres":
            print("\n".join(rec.genre_vocabulary))
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    return 0
