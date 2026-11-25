"""Feature selectors used by the project."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE

# The L1-penalised logistic regression: the embedded method of the feature-selection
# notebook, and the second stage of the strategy it saves in logs/feature_selection.json
# (the multicollinearity filter, then L1).
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


# The strategy saved in logs/feature_selection.json, for the model notebooks, which refit it
# inside their own splits. Same logic as the cells of the feature-selection notebook (3.1.5
# and 3.3.1).
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove one column of each pair whose absolute correlation is above `threshold`.

    Of the two columns of a pair, the one more correlated with the target stays.
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X_df = pd.DataFrame(X_arr).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)

        corr_matrix = X_df.corr().abs()
        target_corr = X_df.apply(lambda col: abs(col.corr(y_series))).fillna(0.0)

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for row in upper.index[upper[col] > self.threshold].tolist():
                # Keep the column of the pair more correlated with the target
                if target_corr[col] >= target_corr[row]:
                    to_drop.add(row)
                else:
                    to_drop.add(col)

        self.support_mask_ = np.array([c not in to_drop for c in X_df.columns], dtype=bool)
        return self

    def transform(self, X):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return X_arr[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_


class CorrL1Selector(BaseEstimator, TransformerMixin):
    """`CorrelationFilter(threshold)`, then the L1 selection on the columns it keeps.

    `l1_params` overrides the L1 settings, as in `make_l1_selector`.
    """

    def __init__(self, threshold=0.90, l1_params=None):
        self.threshold = threshold
        self.l1_params = l1_params

    def fit(self, X, y):
        self.corr_ = CorrelationFilter(threshold=self.threshold).fit(X, y)
        X_filtered = self.corr_.transform(X)
        self.l1_ = make_l1_selector(self.l1_params).fit(X_filtered, y)
        return self

    def get_support(self):
        mask1 = self.corr_.get_support()
        mask2 = self.l1_.get_support()
        combined = np.zeros_like(mask1, dtype=bool)
        combined[np.where(mask1)[0][mask2]] = True
        return combined

    def transform(self, X):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return X_arr[:, self.get_support()]


def make_strategy_selector(settings):
    """A fresh, unfitted selector with the strategy saved in logs/feature_selection.json.

    `settings` is the file's "selector" entry: the correlation threshold and the L1
    settings.
    """
    return CorrL1Selector(threshold=settings["correlation_threshold"], l1_params=settings)