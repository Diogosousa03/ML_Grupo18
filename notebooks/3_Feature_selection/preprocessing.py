import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (needed for IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler



# -------------------------------------------------------------------------
# Loading the Datasets and merging the Datasets train.csv and clientes.csv
# -------------------------------------------------------------------------

trainFile = "../../data/trainDataset/train_2_5_1.csv"
clientFile = "../../data/clientesDataset/clientes_clean.csv"
testFile = "../../data/test.csv"

def load_data():
    train = pd.read_csv(trainFile)
    clientes = pd.read_csv(clientFile)
    test = pd.read_csv(testFile)
    merge = train.merge(clientes, on="customer_id", how="left", validate="many_to_one")
    return merge

RANDOM_STATE = 42


# -------------------------------------------------------------------------
# Columns divisions
# -------------------------------------------------------------------------

target = "repeat_purchase_90d"
identifiers = ["ID", "customer_id"]

# Unique per customer (they name a customer instead of describing one) or pure noise
excluded = ["company_name", "address", "street", "postal_code", "random_noise"]


def columns_type(data):
    features = [c for c in data.columns if c not in [target] + identifiers + excluded]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


# -------------------------------------------------------------------------
# Fill missing function
# -------------------------------------------------------------------------

def fill_missing(train, test, plan):
    """Fill every missing value with imputers fitted on `train` alone.

    `plan` is a list of (columns, imputer) pairs: each imputer is fitted on the
    training rows of its own columns and fills those columns in both halves, so
    the validation rows never help compute a fill.
    """
    train, test = train.copy(), test.copy()
    for columns, imputer in plan:
        blocks = [half[columns].astype(object).fillna(np.nan).infer_objects()
                  for half in (train, test)]
        if isinstance(imputer, (KNNImputer, IterativeImputer)):
            scale = StandardScaler().fit(blocks[0])
            imputer.fit(scale.transform(blocks[0]))
            filled = [scale.inverse_transform(imputer.transform(scale.transform(block)))
                      for block in blocks]
        else:
            imputer.fit(blocks[0])
            filled = [imputer.transform(block) for block in blocks]
        train[columns], test[columns] = filled
    return train, test





def fold_matrices(train, test, recipe):
    """Return (A_train, A_test, names) for ONE split, every step fitted on `train`."""
    numeric, categorical = recipe["numeric"], recipe["categorical"]

    train, test = fill_missing(train, test, recipe["fill"])

    for column in recipe["log1p"]:
        train[column], test[column] = np.log1p(train[column]), np.log1p(test[column])

    for column, minimum in recipe["rare"].items():
        if column in categorical:
            counts = train[column].value_counts()
            frequent = counts[counts >= minimum].index
            train[column] = train[column].where(train[column].isin(frequent), "Other")
            test[column] = test[column].where(test[column].isin(frequent), "Other")

    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    cat_train = encoder.fit_transform(train[categorical].astype("string"))
    cat_test = encoder.transform(test[categorical].astype("string"))

    scaler = StandardScaler().fit(train[numeric])
    num_train, num_test = scaler.transform(train[numeric]), scaler.transform(test[numeric])

    names = np.array(list(encoder.get_feature_names_out(categorical)) + list(numeric))
    return (np.hstack([cat_train, num_train]),
            np.hstack([cat_test, num_test]),
            names)




def purchase_dates(data):
    """Purchase date, rebuilt from its parts (train_2_5_1 no longer carries purchase_date)."""
    parts = data[["purchase_year", "purchase_month", "purchase_day"]]
    return pd.to_datetime(parts.set_axis(["year", "month", "day"], axis=1))


def temporal_splits(data, n_splits=10, gap_days=90, min_train=2000):
    """(training rows, validation rows) pairs in time: train on the past, validate after it.

    The rows are ordered by purchase date and cut into `n_splits` consecutive
    blocks. Training purchases from the `gap_days` before a validation block are
    left out, because their labels depend on purchases inside it, and a split is
    kept only when its training side has at least `min_train` rows. The pairs
    hold row positions, for use with `data.iloc`.
    """
    dates = purchase_dates(data)
    order = np.argsort(dates.to_numpy(), kind="stable")
    gap = pd.Timedelta(days=gap_days)

    splits = []
    for train_block, validation_block in TimeSeriesSplit(n_splits=n_splits).split(order):
        train_rows, validation_rows = order[train_block], order[validation_block]
        start = dates.iloc[validation_rows].min()
        train_rows = train_rows[(dates.iloc[train_rows] < start - gap).to_numpy()]
        if len(train_rows) >= min_train:
            splits.append((train_rows, validation_rows))
    return splits


def held_out_ap(frame, splits, recipe, select=None):
    """Validation AP and F1 on every split, one row per split.

    `select()` builds a fresh, unfitted selector, fitted on each split's
    training matrix alone. Without it, the model sees every column. F1 cuts
    the probabilities at 0.5.
    """
    rows = []
    for train_rows, validation_rows in splits:
        train, validation = frame.iloc[train_rows], frame.iloc[validation_rows]
        A_train, A_validation, names = fold_matrices(train, validation, recipe)
        keep = np.ones(len(names), dtype=bool)
        if select is not None:
            keep = select().fit(A_train, train[target]).get_support()
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(A_train[:, keep], train[target])
        probabilities = model.predict_proba(A_validation[:, keep])[:, 1]
        rows.append({"n": int(keep.sum()),
                    "AP": average_precision_score(validation[target], probabilities),
                    "F1": f1_score(validation[target], probabilities >= 0.5),
                    "kept": list(names[keep])})
    return pd.DataFrame(rows)


def paired_delta(scores, reference):
    """Mean split-by-split difference from `reference`, its SEM, wins and verdict."""
    delta = np.asarray(scores) - np.asarray(reference)
    sem = delta.std(ddof=1) / np.sqrt(len(delta))
    verdict = ("identical on every split" if not delta.any()
               else "inside the noise" if abs(delta.mean()) <= 2 * sem
               else "better" if delta.mean() > 0 else "worse")
    return {"delta": delta.mean(), "SEM": sem,
            "wins": f"{int((delta > 0).sum())}/{len(delta)}", "verdict": verdict}