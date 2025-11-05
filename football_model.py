"""Poisson-based football match outcome predictor.

This module implements a simple model inspired by the approach outlined in
https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/

The model estimates attack and defence strengths for each team alongside a
shared home-advantage parameter. The strengths are learned from historical
results by maximising the Poisson likelihood via gradient ascent.

Example usage (command line):

    python football_model.py data.csv --predict "Arsenal" "Chelsea"

The CSV file must contain at least the following columns:

    date,home_team,away_team,home_goals,away_goals

Additional columns are ignored. Goals should be non-negative integers. The
"date" column may use any format that the Python ``datetime`` module can parse.

The script prints the predicted win/draw/loss probabilities for the supplied
fixture along with the expected goal values for each team.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Match:
    """Container for an individual football match result."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    @property
    def home_points(self) -> int:
        if self.home_goals > self.away_goals:
            return 3
        if self.home_goals == self.away_goals:
            return 1
        return 0

    @property
    def away_points(self) -> int:
        if self.home_goals < self.away_goals:
            return 3
        if self.home_goals == self.away_goals:
            return 1
        return 0


class FootballPoissonModel:
    """Estimate team strengths and predict match outcomes using Poisson goals."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        reg_strength: float = 0.01,
        max_iter: int = 10_000,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.team_index: Dict[str, int] = {}
        self.attack: List[float] = []
        self.defence: List[float] = []
        self.home_advantage: float = 0.0

    def fit(self, matches: Sequence[Match]) -> None:
        """Fit the model to historical match results."""

        teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
        if not teams:
            raise ValueError("No matches supplied")
        self.team_index = {team: idx for idx, team in enumerate(teams)}
        n = len(teams)
        self.attack = [0.0] * n
        self.defence = [0.0] * n
        self.home_advantage = 0.0

        for iteration in range(self.max_iter):
            attack_grad = [0.0] * n
            defence_grad = [0.0] * n
            home_grad = 0.0
            log_likelihood = 0.0

            for match in matches:
                h = self.team_index[match.home_team]
                a = self.team_index[match.away_team]

                lambda_home = math.exp(self.attack[h] + self.defence[a] + self.home_advantage)
                lambda_away = math.exp(self.attack[a] + self.defence[h])

                # Log-likelihood contribution (ignoring constant factorial term)
                log_likelihood += (
                    match.home_goals * math.log(lambda_home)
                    - lambda_home
                    + match.away_goals * math.log(lambda_away)
                    - lambda_away
                )

                diff_home = match.home_goals - lambda_home
                diff_away = match.away_goals - lambda_away

                attack_grad[h] += diff_home
                defence_grad[a] += diff_home

                attack_grad[a] += diff_away
                defence_grad[h] += diff_away

                home_grad += diff_home

            # Apply L2 regularisation to keep ratings bounded.
            for i in range(n):
                attack_grad[i] -= 2.0 * self.reg_strength * self.attack[i]
                defence_grad[i] -= 2.0 * self.reg_strength * self.defence[i]

            home_grad -= 2.0 * self.reg_strength * self.home_advantage

            max_grad = max(
                [abs(g) for g in attack_grad]
                + [abs(g) for g in defence_grad]
                + [abs(home_grad)]
            )

            if max_grad < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration} iterations. LogL={log_likelihood:.2f}")
                break

            step = self.learning_rate / max(len(matches), 1)
            for i in range(n):
                self.attack[i] += step * attack_grad[i]
                self.defence[i] += step * defence_grad[i]

            self.home_advantage += step * home_grad

            # Enforce identifiability by centring attack and defence strengths.
            mean_attack = sum(self.attack) / n
            mean_defence = sum(self.defence) / n
            for i in range(n):
                self.attack[i] -= mean_attack
                self.defence[i] -= mean_defence

            if self.verbose and iteration % 500 == 0:
                print(
                    f"Iter {iteration:5d} | LogL={log_likelihood:10.2f}"
                    f" | max_grad={max_grad:.6f}"
                )
        else:
            if self.verbose:
                print(
                    f"Reached max iterations ({self.max_iter}) without convergence."
                )

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _require_team(self, team: str) -> int:
        if team not in self.team_index:
            known = ", ".join(sorted(self.team_index)) if self.team_index else "none"
            raise KeyError(f"Unknown team '{team}'. Known teams: {known}")
        return self.team_index[team]

    def expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Return the expected goals for the home and away team."""

        h = self._require_team(home_team)
        a = self._require_team(away_team)

        lambda_home = math.exp(self.attack[h] + self.defence[a] + self.home_advantage)
        lambda_away = math.exp(self.attack[a] + self.defence[h])
        return lambda_home, lambda_away

    def score_matrix(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10,
    ) -> List[List[float]]:
        """Return a matrix of scoreline probabilities up to ``max_goals``."""

        lambda_home, lambda_away = self.expected_goals(home_team, away_team)
        probs_home = [poisson_pmf(k, lambda_home) for k in range(max_goals + 1)]
        probs_away = [poisson_pmf(k, lambda_away) for k in range(max_goals + 1)]

        return [[ph * pa for pa in probs_away] for ph in probs_home]

    def outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10,
    ) -> Dict[str, float]:
        """Return win/draw/loss probabilities for the home team."""

        matrix = self.score_matrix(home_team, away_team, max_goals=max_goals)
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for home_goals, row in enumerate(matrix):
            for away_goals, probability in enumerate(row):
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away_win += probability

        return {"home_win": home_win, "draw": draw, "away_win": away_win}


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """Return the probability of k under a Poisson distribution."""

    if k < 0:
        return 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def load_matches(path: Path) -> List[Match]:
    """Load match data from ``path``.

    The CSV is expected to have a header row. Only the required columns are
    accessed; any other columns are ignored.
    """

    matches: List[Match] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV file must contain columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            try:
                match = Match(
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row: {row}") from exc
            matches.append(match)
    if not matches:
        raise ValueError("The CSV file does not contain any matches")
    return matches


def summarise_table(matches: Iterable[Match]) -> Dict[str, Dict[str, float]]:
    """Return aggregate statistics for each team."""

    table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for match in matches:
        table[match.home_team]["played"] += 1
        table[match.home_team]["scored"] += match.home_goals
        table[match.home_team]["conceded"] += match.away_goals
        table[match.home_team]["points"] += match.home_points

        table[match.away_team]["played"] += 1
        table[match.away_team]["scored"] += match.away_goals
        table[match.away_team]["conceded"] += match.home_goals
        table[match.away_team]["points"] += match.away_points
    return table


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to a CSV file containing historical match results.",
    )
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME_TEAM", "AWAY_TEAM"),
        help="Predict the outcome probabilities for the supplied fixture.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
        help="Maximum goals considered when computing the score matrix.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for gradient ascent.",
    )
    parser.add_argument(
        "--reg-strength",
        type=float,
        default=0.02,
        help="L2 regularisation strength to stabilise estimates.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum training iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Gradient tolerance for convergence.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress information.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)

    matches = load_matches(args.csv)
    model = FootballPoissonModel(
        learning_rate=args.learning_rate,
        reg_strength=args.reg_strength,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
    )
    model.fit(matches)

    print("Model trained on", len(matches), "matches.")
    if args.predict:
        home_team, away_team = args.predict
        lambda_home, lambda_away = model.expected_goals(home_team, away_team)
        outcome_probs = model.outcome_probabilities(
            home_team, away_team, max_goals=args.max_goals
        )
        print(f"Expected goals: {home_team} {lambda_home:.2f} - {lambda_away:.2f} {away_team}")
        print(
            "Probabilities:",
            f"home_win={outcome_probs['home_win']:.3f}",
            f"draw={outcome_probs['draw']:.3f}",
            f"away_win={outcome_probs['away_win']:.3f}",
        )
    else:
        table = summarise_table(matches)
        print("Top teams by attack strength:")
        for team, strength in sorted(
            ((team, model.attack[model._require_team(team)]) for team in model.team_index),
            key=lambda item: item[1],
            reverse=True,
        )[:5]:
            record = table[team]
            print(
                f"  {team:20s} attack={strength: .3f}"
                f" | played={int(record['played']):3d}"
                f" | goals_for={int(record['scored']):3d}"
                f" | goals_against={int(record['conceded']):3d}"
            )


if __name__ == "__main__":
    main()
