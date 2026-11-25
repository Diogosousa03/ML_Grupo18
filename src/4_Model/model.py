import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


#----------------------
# Validation of the Train and Test Files
#----------------------
def rebuild_purchase_dates(date_parts):
    """Purchase date of each row, rebuilt from its parts.

    date_parts: table with the columns purchase_year, purchase_month and purchase_day.
    """
    return pd.to_datetime(
        date_parts.rename(
            columns={
                "purchase_year": "year",
                "purchase_month": "month",
                "purchase_day": "day",
            }
        )
    )


#----------------------
# Baseline Model
#----------------------
CUTOFFS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def f1_by_cutoff(y, probabilities):
    """F1 on these rows at each cut-off of CUTOFFS."""
    return [f1_score(y, probabilities >= c) for c in CUTOFFS]


def best_cutoff(y, probabilities):
    """The cut-off with the highest F1 on these rows."""
    return CUTOFFS[np.argmax(f1_by_cutoff(y, probabilities))]


def fit_with_cutoff(model, X, y):
    """Fit the model on these rows and choose its cut-off on the same rows.

    Returns the fitted model, its probabilities on these rows and the cut-off.
    """
    model = model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    return model, probabilities, best_cutoff(y, probabilities)


def baseline_rule(data):
    """The competition's baseline: a repeat purchase is predicted when purchase_count_365d >= 3."""
    return (data["purchase_count_365d"] >= 3).astype(int)


#----------------------
# Tuning
#----------------------
def score_configurations(prepared, configs, make):
    """Fit `make(config)` for every configuration on every prepared split and score it.

    `prepared` holds one (A_train, A_validation, y_train, y_validation) tuple per
    split, already prepared by the recipe and reduced to the selected columns, so
    every configuration is fitted on the same matrices. Each fit chooses its own
    cut-off on its training rows. Returns one row per (configuration, split), with
    the validation AP and the F1 at that cut-off.
    """
    rows = []
    for index, config in enumerate(configs):
        for split, (A_train, A_validation, y_train, y_validation) in enumerate(prepared):
            model, _, cutoff = fit_with_cutoff(make(config), A_train, y_train)
            p_validation = model.predict_proba(A_validation)[:, 1]
            rows.append({"configuration": index, "split": split, "cut-off": cutoff,
                         "AP": average_precision_score(y_validation, p_validation),
                         "F1": f1_score(y_validation, p_validation >= cutoff)})
    return pd.DataFrame(rows)


def walk_forward(scores, first=2, metric="AP"):
    """The search scored on splits that took no part in the choice: a nested validation in time.

    `scores` is the result of `score_configurations`. For every split from `first`
    on, the configuration with the best mean `metric` on the earlier splits is the
    choice, and its scores on that split are kept. Returns one row per scored split.
    """
    wide = scores.pivot(index="split", columns="configuration", values=metric)
    rows = []
    for split in wide.index[first:]:
        chosen = int(wide.loc[:split - 1].mean().idxmax())
        row = scores[(scores["configuration"] == chosen) & (scores["split"] == split)].iloc[0]
        rows.append({"split": split, "chosen": chosen, "AP": row["AP"], "F1": row["F1"],
                     "cut-off": row["cut-off"]})
    return pd.DataFrame(rows)
