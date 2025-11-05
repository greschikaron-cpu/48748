# Football Match Outcome Predictor

This repository contains a small Python script that trains a Poisson-based
model to estimate the attack/defence strengths of football teams and predict
the result of future fixtures. The implementation follows the approach
described in [Dashee87's blog post](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/).

## Requirements

The script only depends on the Python standard library and works with Python
3.9 or newer.

## Usage

1. Prepare a CSV file containing historical match results with at least the
   following columns:

   ```csv
   date,home_team,away_team,home_goals,away_goals
   2022-08-05,Crystal Palace,Arsenal,0,2
   2022-08-06,Fulham,Liverpool,2,2
   ```

   Additional columns are ignored.

2. Train the model and predict an upcoming fixture:

   ```bash
   python football_model.py results.csv --predict "Arsenal" "Chelsea"
   ```

   Example output:

   ```
   Model trained on 380 matches.
   Expected goals: Arsenal 1.62 - 1.07 Chelsea
   Probabilities: home_win=0.448 draw=0.270 away_win=0.282
   ```

   Adjust the `--learning-rate`, `--reg-strength`, `--max-iter`, and `--tol`
   options if the model struggles to converge for your data set.

3. Omit `--predict` to print a quick summary of the strongest attacking teams
   observed in the training data.

Run `python football_model.py --help` to see all command-line options.
