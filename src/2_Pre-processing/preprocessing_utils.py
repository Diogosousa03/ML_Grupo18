"""
Pre-processing auxiliary functions used in the notebook.
"""
import warnings
from collections import namedtuple
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from functools import partial
from IPython.display import display
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit



from pathlib import Path

warnings.filterwarnings("ignore", category=ConvergenceWarning)

Scored = namedtuple("Scored", "ap f1 iterations predictions")

RANDOM_STATE = 42 



def load_raw_datasets(base_dir: str = "../../data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Load the raw datasets from the data folder.
        Returns:
            train (pd.DataFrame): The training dataset.
            clients (pd.DataFrame): The clients dataset.
            test (pd.DataFrame): The test dataset.
    """
    data_dir = Path(base_dir)
    train = pd.read_csv(data_dir / "trainDataset/train_2_5_1.csv")
    clientes = pd.read_csv(data_dir / "clientesDataset/clientes_clean.csv")
    test = pd.read_csv(data_dir / "testDataset/test_2_5_1.csv")
    return train, clientes, test

## 1.1. merging datasets

def merge_train_test_with_customers(train: pd.DataFrame, test: pd.DataFrame, clients: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Merge the train and test datasets with the clients dataset.
        Args:
            train (pd.DataFrame): The training dataset.
            test (pd.DataFrame): The test dataset.
            clients (pd.DataFrame): The clients dataset.
        Returns:
            train_merged (pd.DataFrame): The merged training dataset.
            test_merged (pd.DataFrame): The merged test dataset.
    """
    train_merged = train.merge(clients, on="customer_id", how="left", validate="many_to_one")
    assert len(train_merged) == len(train), "O número de linhas de train mudou após o merge!"

    test_merged = test.merge(clients, on="customer_id", how="left", validate="many_to_one")
    assert len(test_merged) == len(test), "O número de linhas de test mudou após o merge!"

    return train_merged, test_merged


def save_train_test_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: str,
    test_path: str,
) -> None:
    """
            Save the train and test datasets to the data folder.
            Args:
                train (pd.DataFrame): The training dataset.
                test (pd.DataFrame): The test dataset.
                base_dir (str): The base directory where the datasets will be saved.
        """
    for df, file_path in [(train, train_path), (test, test_path)]:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

## 1.2. feature type separation

def separate_feature_types(
    dataframe: pd.DataFrame,
    target_col: str = "repeat_purchase_90d",
    identifiers: Sequence[str] = ("ID", "customer_id"),
    dataset_label: str = "Dataset",
) -> Tuple[pd.DataFrame, Optional[pd.Series], list[str], list[str]]:
    """Numeric and categorical feature columns of `dataframe` (identifiers and target left out).

    Prints the counts and returns the feature table, the target (None when the
    table has no target), and the numeric and categorical column lists.
    """
    excluded_cols = list(identifiers) + (
        [target_col] if target_col in dataframe.columns else []
    )
    features_list = [c for c in dataframe.columns if c not in excluded_cols]

    numeric = [
        c for c in features_list if pd.api.types.is_numeric_dtype(dataframe[c])
    ]
    categorical = [c for c in features_list if c not in numeric]

    X_features = dataframe[numeric + categorical]
    y_target = dataframe[target_col] if target_col in dataframe.columns else None

    print(f"=== Feature Separation ({dataset_label}) ===")
    print(
        f"{len(numeric)} numeric | {len(categorical)} categorical |"
        f" {len(identifiers)} identifiers"
    )
    print(f"categorical: {categorical}\n")

    return X_features, y_target, numeric, categorical

## 1.3. Missing Values Overview

def generate_missing_overview(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    dataset_label: str = "Dataset",
) -> pd.DataFrame:
    """Missing values per feature in X.

    When target `y` is provided, also computes the mean target rate when
    each feature is missing vs. present.
    """
    missing = X.isna().sum()

    overview_dict = {
        "group": [
            "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical"
            for c in X.columns
        ],
        "n_missing": missing,
        "pct_missing": (missing / len(X) * 100).round(1),
    }

    if y is not None:
        overview_dict["target_if_missing"] = [y[X[c].isna()].mean() for c in X.columns]
        overview_dict["target_if_present"] = [y[X[c].notna()].mean() for c in X.columns]

    overview_df = (
        pd.DataFrame(overview_dict)
        .query("n_missing > 0")
        .sort_values("n_missing", ascending=False)
        .round(3)
    )

    print(f"=== MISSING VALUES OVERVIEW ({dataset_label}) ===")
    print(f"rows with at least one missing value: {X.isna().any(axis=1).sum():,} of {len(X):,}\n")
    return overview_df

## 1.4. Evaluation Step
def design_matrix(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Numbers as they are, everything else one-hot. Identifiers never enter."""
    encoder = OneHotEncoder(sparse_output=False, dtype=float)
    encoded = encoder.fit_transform(frame[list(categorical)].astype("string"))
    text = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical),
        index=frame.index,
    )
    return pd.concat([frame[list(numeric)], text], axis=1)

def fill_missing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    plan: Sequence[Tuple[Sequence[str], object]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill every missing value with imputers fitted on `train` alone."""
    train, test = train.copy(), test.copy()
    for columns, imputer in plan:
        cols = list(columns)
        blocks = [
            half[cols].astype(object).fillna(np.nan).infer_objects()
            for half in (train, test)
        ]
        if isinstance(imputer, (KNNImputer, IterativeImputer)):
            scale = StandardScaler().fit(blocks[0])
            imputer.fit(scale.transform(blocks[0]))
            filled = [
                scale.inverse_transform(imputer.transform(scale.transform(block)))
                for block in blocks
            ]
        else:
            imputer.fit(blocks[0])
            filled = [imputer.transform(block) for block in blocks]
        train[cols], test[cols] = filled
    return train, test

def default_imputation_plan(
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> List[Tuple[List[str], SimpleImputer]]:
    """Return baseline median/mode imputation plan."""
    return [
        (list(numeric), SimpleImputer(strategy="median")),
        (list(categorical), SimpleImputer(strategy="most_frequent")),
    ]

def sem(values: np.ndarray) -> float:
    """Standard error of the mean: how far it would move on new splits."""
    return float(values.std(ddof=1) / np.sqrt(len(values)))

def held_out_scores(
    prepare: Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]],
    frame: pd.DataFrame,
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str = "repeat_purchase_90d",
    scaler: Optional[object] = None,
) -> Scored:
    """Score one treatment on each split's untouched validation rows."""
    ap, f1, iterations, predictions = [], [], [], []
    for train_rows, test_rows in splits:
        train, test = prepare(frame.iloc[train_rows], frame.iloc[test_rows])
        matrix = design_matrix(pd.concat([train, test]), numeric, categorical)
        a, b = matrix.iloc[: len(train)], matrix.iloc[len(train) :]
        if scaler is not None:
            fitted = scaler.fit(a)
            a, b = fitted.transform(a), fitted.transform(b)
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(a, train[target])
        probabilities = model.predict_proba(b)[:, 1]
        ap.append(average_precision_score(test[target], probabilities))
        f1.append(f1_score(test[target], probabilities >= 0.5))
        iterations.append(int(model.n_iter_[0]))
        predictions.append((test[target].to_numpy(), probabilities))
    return Scored(np.array(ap), np.array(f1), np.array(iterations), predictions)

## 1.5. Temporal Splits
def build_temporal_splits(
    frame: pd.DataFrame,
    n_splits: int = 10,
    gap_days: int = 90,
    min_train: int = 2000,
    date_cols: Sequence[str] = ("purchase_year", "purchase_month", "purchase_day"),
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.Series]:
    """Build leak-free temporal cross-validation splits with a gap window."""
    purchase_dates = pd.to_datetime(
        frame[list(date_cols)].set_axis(["year", "month", "day"], axis=1)
    )

    order = np.argsort(purchase_dates.to_numpy(), kind="stable")
    gap = pd.Timedelta(days=gap_days)

    time_splits = []
    for train_block, validation_block in TimeSeriesSplit(n_splits=n_splits).split(order):
        train_rows, validation_rows = order[train_block], order[validation_block]
        start = purchase_dates.iloc[validation_rows].min()
        train_rows = train_rows[(purchase_dates.iloc[train_rows] < start - gap).to_numpy()]
        if len(train_rows) >= min_train:
            time_splits.append((train_rows, validation_rows))

    print(f"{len(time_splits)} temporal splits kept of {n_splits}")
    return time_splits, purchase_dates


def summarize_temporal_splits(
    time_splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    purchase_dates: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Summarize date ranges, row counts, and target rates across temporal splits."""
    rows = [
        {
            "training until": purchase_dates.iloc[tr].max().date(),
            "validation from": purchase_dates.iloc[va].min().date(),
            "validation until": purchase_dates.iloc[va].max().date(),
            "training rows": len(tr),
            "validation rows": len(va),
            "training rate": target.iloc[tr].mean(),
            "validation rate": target.iloc[va].mean(),
        }
        for tr, va in time_splits
    ]
    return pd.DataFrame(rows).round(3)

def evaluate_temporal_baseline(
    frame: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    plan: Sequence[Tuple[Sequence[str], object]],
    target_col: str = "repeat_purchase_90d",
) -> Scored:
    """Compute and print the temporal baseline and predict-all-positive floor."""
    baseline = held_out_scores(
        partial(fill_missing, plan=plan),
        frame,
        splits,
        numeric,
        categorical,
        target=target_col,
    )

    target = frame[target_col]
    all_pos_f1 = np.array(
        [f1_score(target.iloc[va], np.ones(len(va))) for _, va in splits]
    )
    all_pos_ap = np.array([target.iloc[va].mean() for _, va in splits])

    print(
        f"temporal baseline    AP {baseline.ap.mean():.4f} +/- {sem(baseline.ap):.4f}"
        f"   F1 {baseline.f1.mean():.4f} +/- {sem(baseline.f1):.4f}"
    )
    print(
        f"predict-all-positive AP {all_pos_ap.mean():.4f}             "
        f"F1 {all_pos_f1.mean():.4f}"
    )
    print(f"AP per split {np.round(baseline.ap, 4)}")

    return baseline


## 1.6. Missing Values Imputation

def apply_imputation_and_save(
    train_in_path: str,
    test_in_path: str,
    train_out_path: str,
    test_out_path: str,
    plan: Sequence[Tuple[Sequence[str], object]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs, apply a missing-value imputation plan, and save the results."""
    df_train = pd.read_csv(train_in_path)
    df_test = pd.read_csv(test_in_path)

    df_filled, df_filled_test = fill_missing(df_train, df_test, plan)
    save_train_test_datasets(df_filled, df_filled_test, train_out_path, test_out_path)

    return df_filled, df_filled_test


def transform_and_save_datasets(
    train_in_path: str,
    test_in_path: str,
    train_out_path: str,
    test_out_path: str,
    transform_fn: Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs, apply a transformation function, and save the results."""
    df_train = pd.read_csv(train_in_path)
    df_test = pd.read_csv(test_in_path)

    df_train_out, df_test_out = transform_fn(df_train, df_test)
    save_train_test_datasets(df_train_out, df_test_out, train_out_path, test_out_path)

    return df_train_out, df_test_out


## 1.7. Outliers

## 1.7. Outliers

DEFAULT_NON_CONTINUOUS_NUMERICS: Tuple[str, ...] = (
    "is_first_purchase",
    "is_weekend",
    "since_2015_or_earlier",
    "purchase_year",
    "purchase_quarter",
    "purchase_month",
    "purchase_day",
    "purchase_dayofweek",
    "customer_since_year",
)


def get_continuous_numeric_cols(
    numeric_cols: Sequence[str],
    exclude: Sequence[str] = DEFAULT_NON_CONTINUOUS_NUMERICS,
) -> List[str]:
    """Filter out binary flags and temporal integer fields from numeric columns."""
    exclude_set = set(exclude)
    return [c for c in numeric_cols if c not in exclude_set]


def _compute_fences(
    values: pd.DataFrame,
    rule: str = "iqr",
    k: float = 1.5,
) -> Tuple[pd.Series, pd.Series]:
    """Compute lower and upper fences per column using 'iqr' or 'zscore'."""
    if rule == "iqr":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr
    mean, sd = values.mean(), values.std(ddof=0)
    return mean - k * sd, mean + k * sd


def outside_fences(
    frame: pd.DataFrame,
    columns: Sequence[str],
    rule: str = "iqr",
    k: float = 1.5,
) -> pd.Series:
    """True for rows with at least one specified column beyond its fences."""
    values = frame[list(columns)]
    low, high = _compute_fences(values, rule=rule, k=k)
    within = values.ge(low) & values.le(high)
    return ~(within | values.isna()).all(axis=1)


def analyze_outliers(
    df_target: pd.DataFrame,
    cols_to_analyze: Sequence[str],
    dataset_label: str = "Train",
    top_n: int = 10,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Summarize row-level and column-level outliers using 1.5x IQR and |Z| > 3."""
    cols = list(cols_to_analyze)
    values = df_target[cols]
    outlier_mask = outside_fences(df_target, columns=cols, rule="iqr", k=1.5)

    n_total = len(df_target)
    n_outliers = int(outlier_mask.sum())
    n_normal = n_total - n_outliers

    print(f"=== OUTLIER DETECTION SUMMARY ({dataset_label.upper()}) ===")
    print(f"• Total analyzed records: {n_total:,}")
    print(f"• Records with at least 1 outlier: {n_outliers:,} ({n_outliers / n_total * 100:.2f}%)")
    print(f"• Normal records (without outliers): {n_normal:,} ({n_normal / n_total * 100:.2f}%)")

    iqr_low, iqr_high = _compute_fences(values, rule="iqr", k=1.5)
    z_low, z_high = _compute_fences(values, rule="zscore", k=3.0)

    outlier_by_col = (
        pd.DataFrame({
            "skewness": values.skew(),
            "% Outliers (1.5x IQR)": ((values < iqr_low) | (values > iqr_high)).mean() * 100,
            "% Outliers (|Z| > 3)": ((values < z_low) | (values > z_high)).mean() * 100,
            "max_value": values.max(),
        })
        .sort_values("% Outliers (1.5x IQR)", ascending=False)
        .round(2)
    )

    print(f"\n=== OUTLIER DETAILED SUMMARY PER VARIABLE ({dataset_label.upper()}) ===")
    display(outlier_by_col.head(top_n))

    print(f"\n--- Sample Rows Identified as Outliers ({dataset_label.upper()}) ---")
    display(df_target.loc[outlier_mask, cols].head())
    print("\n" + "=" * 60 + "\n")

    return outlier_mask, outlier_by_col

def apply_log1p_non_negative(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude_cols: Sequence[str] = (*DEFAULT_NON_CONTINUOUS_NUMERICS, "ID", "customer_id", "repeat_purchase_90d"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select non-negative continuous numeric columns on `train` and apply log1p to both `train` and `test`."""
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    candidate_cols = get_continuous_numeric_cols(numeric_cols, exclude=exclude_cols)
    log1p_cols = [c for c in candidate_cols if (train[c] >= 0).all()]

    train_out, test_out = train.copy(), test.copy()
    train_out[log1p_cols] = np.log1p(train_out[log1p_cols].clip(lower=0))
    test_out[log1p_cols] = np.log1p(test_out[log1p_cols].clip(lower=0))

    gaps_test = int(test_out[log1p_cols].isna().sum().sum())
    print(f"log1p on {len(log1p_cols)} columns | gaps left in the test: {gaps_test}")

    return train_out, test_out

## 1.8. Encoding and Scaling

## 1.8. Encoding & Scaling

DEFAULT_HIGH_CARDINALITY_DROP: Tuple[str, ...] = (
    "company_name",
    "address",
    "street",
    "postal_code",
)

DEFAULT_RARE_CATEGORY_THRESHOLDS: dict[str, int] = {
    "transport_method": 50,
    "payment_type": 20,
    "city": 30,
    "municipality": 30,
    "postal_area": 20,
}


def encode_categorical_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: Sequence[str],
    drop_cols: Sequence[str] = DEFAULT_HIGH_CARDINALITY_DROP,
    thresholds: Optional[dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group rare levels based on `train` counts, one-hot encode, and drop raw categorical/text columns."""
    thresholds = DEFAULT_RARE_CATEGORY_THRESHOLDS if thresholds is None else thresholds
    drop_set = set(drop_cols)
    categorical_filtered = [c for c in categorical_cols if c not in drop_set]

    train_work, test_work = train.copy(), test.copy()
    cols_to_encode = []

    for col in categorical_filtered:
        if col in thresholds:
            counts = train_work[col].value_counts()
            frequent = counts[counts >= thresholds[col]].index
            grouped_col = f"{col}_grouped"
            for df in (train_work, test_work):
                df[grouped_col] = df[col].where(df[col].isin(frequent), "Other")
            cols_to_encode.append(grouped_col)
        else:
            cols_to_encode.append(col)

    encoder = OneHotEncoder(
        sparse_output=False, dtype=float, drop="first", handle_unknown="ignore"
    )
    encoder.fit(train_work[cols_to_encode].fillna("Missing").astype("string"))
    encoded_names = encoder.get_feature_names_out(cols_to_encode)

    encoded_dfs = [
        pd.DataFrame(
            encoder.transform(df[cols_to_encode].fillna("Missing").astype("string")),
            columns=encoded_names,
            index=df.index,
        )
        for df in (train_work, test_work)
    ]

    cols_to_remove = list(set(categorical_filtered) | drop_set | set(cols_to_encode))
    train_out = pd.concat(
        [train_work.drop(columns=cols_to_remove, errors="ignore"), encoded_dfs[0]], axis=1
    )
    test_out = pd.concat(
        [test_work.drop(columns=cols_to_remove, errors="ignore"), encoded_dfs[1]], axis=1
    )

    print(
        f"Encoded design matrix shape: train {encoded_dfs[0].shape} | test {encoded_dfs[1].shape}"
    )
    return train_out, test_out



DEFAULT_NON_FEATURE_COLS: Tuple[str, ...] = ("ID", "customer_id", "repeat_purchase_90d")


def scale_numeric_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    non_feature_cols: Sequence[str] = DEFAULT_NON_FEATURE_COLS,
    target_col: str = "repeat_purchase_90d",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on `train` numeric features and transform both `train` and `test`."""
    excluded = set(non_feature_cols)
    candidate_cols = (
        train.select_dtypes(include=[np.number]).columns
        if numeric_cols is None
        else [c for c in numeric_cols if c in train.columns]
    )

    cols_to_scale = [
        c
        for c in candidate_cols
        if c not in excluded and set(train[c].dropna().unique()) - {0, 1, 0.0, 1.0}
    ]

    scaler = StandardScaler().fit(train[cols_to_scale])

    def _apply_scale(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[cols_to_scale] = out[cols_to_scale].astype(float)
        out[cols_to_scale] = scaler.transform(out[cols_to_scale])
        return out

    train_out = _apply_scale(train)
    test_out = _apply_scale(test)

    train_feats = list(train_out.columns.drop(target_col, errors="ignore"))
    test_feats = list(test_out.columns)

    print(f"Final design matrix: train {train_out.shape} | test {test_out.shape}")
    print(f"Same feature columns in train and test: {train_feats == test_feats}")

    return train_out, test_out