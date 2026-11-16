"""Feature selectors used by the project."""
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE

# The chosen strategy (L1-penalised logistic regression). Single source of truth:
# the feature-selection notebook saves these values in logs/feature_selection.json
# and the model notebooks read them back.
L1_PARAMS = {"l1_ratio": 1, "solver": "liblinear", "C": 0.1, "max_iter": 1000,
             "threshold": 1e-8, "random_state": RANDOM_STATE}


def make_l1_selector(params=None):
    """A fresh, unfitted L1 selector (SelectFromModel + L1 logistic regression).

    `params` overrides the defaults in `L1_PARAMS`; keys that are not settings
    (such as the "method" description saved in the JSON) are ignored. Calling it
    with no arguments also makes it usable as `select=make_l1_selector`.
    """
    settings = {**L1_PARAMS, **(params or {})}
    return SelectFromModel(
        LogisticRegression(l1_ratio=settings["l1_ratio"], solver=settings["solver"],
                           C=settings["C"], max_iter=settings["max_iter"],
                           random_state=settings["random_state"]),
        threshold=settings["threshold"])