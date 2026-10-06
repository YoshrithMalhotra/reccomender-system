"""Command-line interface.

Examples:
    python -m crunchyroll_recommender similar "Jujutsu Kaisen"
    python -m crunchyroll_recommender user --liked "Spy x Family" "Kaguya-sama" --disliked "Tokyo Ghoul"
    python -m crunchyroll_recommender top --genre Sports
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .engine import DEFAULT_DATA, CrunchyrollRecommender


def _print(df: pd.DataFrame) -> None:
    if df.empty:
        print("No recommendations match those filters.")
        return
    with pd.option_context("display.max_colwidth", 45, "display.width", 140):
        print(df.to_string(index=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crunchyroll-recommender", description="Crunchyroll anime recommender")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path to a catalog CSV")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p):
        p.add_argument("-n", type=int, default=10, help="number of results")
        p.add_argument("--genre", action="append", help="require a genre (repeatable)")
        p.add_argument("--min-rating", type=float, help="minimum rating")

    p = sub.add_parser("similar", help="anime similar to one title")
    p.add_argument("title")
    add_filters(p)

    p = sub.add_parser("user", help="recommendations from a list of liked/disliked titles")
    p.add_argument("--liked", nargs="+", required=True)
    p.add_argument("--disliked", nargs="+", default=[])
    add_filters(p)

    p = sub.add_parser("top", help="top-rated titles")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--genre", action="append")

    p = sub.add_parser("search", help="search titles in the catalog")
    p.add_argument("query")

    sub.add_parser("genres", help="list available genres")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rec = CrunchyrollRecommender.from_csv(args.data)

    try:
        if args.command == "similar":
            print(f"Because you watched {rec.find_title(args.title)}:\n")
            _print(rec.similar_to(args.title, args.n, args.genre, args.min_rating))
        elif args.command == "user":
            _print(rec.for_user(args.liked, args.disliked, args.n, args.genre, args.min_rating))
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
