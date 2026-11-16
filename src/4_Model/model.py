import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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
