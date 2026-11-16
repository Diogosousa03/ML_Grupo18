"""Loading the data, splitting the columns by type and building the temporal splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import CLIENT_FILE, TRAIN_FILE, excluded, identifiers, target


def load_data():
    """Training purchases merged with the (clean) customer table."""
    train = pd.read_csv(TRAIN_FILE)
    clientes = pd.read_csv(CLIENT_FILE)
    return train.merge(clientes, on="customer_id", how="left", validate="many_to_one")


def columns_type(data):
    """(numeric columns, categorical columns), without target, identifiers and excluded columns."""
    features = [c for c in data.columns if c not in [target] + identifiers + excluded]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


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