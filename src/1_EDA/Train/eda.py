
"""Shared helpers used by the notebooks (1_DataIngestion&InitialInspection, 2_DataCleaning).
 
Every transformation is a pure function ``DataFrame -> DataFrame`` so that the
same logic can be applied to the train and test files with ``run_train_test``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# ---------------------------------------------------------------- constants
NON_NEGATIVE = [
    "days_since_previous_purchase",
    "prior_mean_purchase_value",
    "prior_mean_gap_days",
    "spend_365d",
]
IDENTIFIERS = ["ID", "customer_id", "purchase_date"]
TARGET = "repeat_purchase_90d"
 
CUSTOMER_BINS = [0, 1, 2, 5, 10, 20, 50, 100, np.inf]
CUSTOMER_LABELS = [
    "1 purchase", "2 purchases", "3-5 purchases", "6-10 purchases",
    "11-20 purchases", "21-50 purchases", "51-100 purchases", ">100 purchases",
]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
 
 
# ------------------------------------------------------------ run helpers
def run_step(transform, input_path, output_path, label="Dataset", **kwargs):
    """Read ``input_path``, apply ``transform(df, **kwargs)``, save to ``output_path``.
 
    Returns the transformed DataFrame.
    """
    df = pd.read_csv(input_path)
    out = transform(df, **kwargs)
    out.to_csv(output_path, index=False)
    print(f"=== {label} ===")
    print(f"  read:  {input_path}")
    print(f"  saved: {output_path}  ({out.shape[0]} rows x {out.shape[1]} columns)\n")
    return out
 
 
def run_train_test(transform, train_in, train_out, test_in, test_out, **kwargs):
    """Apply ``run_step`` to the train and test files. Returns (df_train, df_test)."""
    return (
        run_step(transform, train_in, train_out, "Train", **kwargs),
        run_step(transform, test_in, test_out, "Test", **kwargs),
    )
 
 
# ------------------------------------------------------- basic utilities
def mask_negatives(df, cols=NON_NEGATIVE):
    """Return a copy where negative values of ``cols`` are replaced by NaN."""
    out = df.copy()
    out[cols] = out[cols].mask(out[cols] < 0)
    return out
 
 
def get_numeric_categorical(df, identifiers=IDENTIFIERS, target=TARGET):
    """Split columns into numeric and categorical, excluding identifiers and target."""
    excluded = set(identifiers) | {target}
    numeric = [c for c in df.columns
               if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in excluded and c not in numeric]
    return numeric, categorical
 
 
# ------------------------------------------------------- missing values
def missing_overview(df, label="missing values as the file arrives"):
    """Summary table: rows with nulls, columns with nulls, null cells, worst column."""
    per_col = df.isna().sum()
    rows_null = df.isna().any(axis=1)
    return pd.Series({
        "rows with a null": f"{rows_null.sum():,} of {len(df):,} ({100 * rows_null.mean():.2f}%)",
        "columns with a null": f"{(per_col > 0).sum()} of {df.shape[1]}",
        "null cells": f"{int(per_col.sum()):,}",
        "worst column": f"{per_col.idxmax()} ({per_col.max()} nulls)",
    }).to_frame(label)
 
 
def negative_masking_impact(df, cols=NON_NEGATIVE):
    """Print how masking invalid negatives changes the number of rows/cells with nulls."""
    cleaned = mask_negatives(df, cols)
    rows_before, rows_after = df.isna().any(axis=1).sum(), cleaned.isna().any(axis=1).sum()
    cells_before, cells_after = df.isna().sum().sum(), cleaned.isna().sum().sum()
    print(f"rows with a null: {rows_before:,} -> {rows_after:,} "
          f"({100 * rows_after / len(df):.2f}% of {len(df):,})")
    print(f"null cells:       {cells_before:,} -> {cells_after:,}")
    return cleaned
 
 
def non_negative_validation(df, cols=NON_NEGATIVE):
    """NaNs and remaining negative values in the columns that cannot be negative."""
    return pd.DataFrame({
        "Total NaNs": df[cols].isna().sum(),
        "NaNs Percentage (%)": (df[cols].isna().mean() * 100).round(2),
        "Remaining negative values (< 0)": (df[cols] < 0).sum(),
    })
 
 
def remaining_missing(df, label="Dataset"):
    """Print and return the columns that still contain missing values."""
    counts = df.isnull().sum()
    counts = counts[counts > 0]
    print(f"=== REMAINING MISSING VALUES ({label}) ===")
    if len(counts):
        print(counts)
        print(f"\nTotal columns with nulls: {len(counts)}")
    else:
        print(f"No missing values left in the {label} dataset.")
    print("-" * 50 + "\n")
    return counts
 
 
# ------------------------------------------------------- customer records
def customer_record_report(df, label="Dataset", id_cols=("ID", "random_noise")):
    """Duplicate rows (ignoring ID columns) and distribution of records per customer."""
    dup = df.drop(columns=list(id_cols), errors="ignore").duplicated().sum()
    per_customer = df.groupby("customer_id").size()
    shared = per_customer[per_customer > 1]
    bins = (pd.cut(per_customer, bins=CUSTOMER_BINS, labels=CUSTOMER_LABELS)
            .value_counts().sort_index())
    print(f"--- {label} ---")
    print(f"duplicate rows ignoring {list(id_cols)}: {dup}")
    print(f"surplus rows on 'customer_id': {df.duplicated(subset='customer_id').sum()}")
    print(f"rows sharing a 'customer_id': {shared.sum()} across {len(shared)} customers")
    print("records per customer (binned distribution):", bins.to_dict(), "\n")
    return bins
 
 
# ------------------------------------------------- cleaning transformations
def drop_missing_total(df):
    """Drop rows without 'Total a Pagar' and mask invalid negative values."""
    out = df.dropna(subset=["Total a Pagar"]).copy()
    return mask_negatives(out)
 
 
def impute_structural_missing(df):
    """Add 'is_first_purchase' flag and impute structural (MNAR) missing values."""
    out = df.copy()
    out["is_first_purchase"] = out["days_since_previous_purchase"].isnull().astype(int)
    zero_cols = ["prior_std_purchase_value", "prior_std_gap_days", "prior_mean_gap_days"]
    out[zero_cols] = out[zero_cols].fillna(0)
    minus_one_cols = ["days_since_previous_purchase", "prior_mean_purchase_value",
                      "prior_min_purchase_value", "prior_max_purchase_value"]
    out[minus_one_cols] = out[minus_one_cols].fillna(-1)
    print(f"[flag] is_first_purchase: {out['is_first_purchase'].sum()} rows "
          f"({100 * out['is_first_purchase'].mean():.1f}%)")
    print("[-1 imputation] per column:", (out[minus_one_cols] == -1).sum().to_dict())
    return out
 
 
def correct_prior_mean_purchase_value(df):
    """Recompute prior_mean_purchase_value = prior_total_spend / prior_purchase_count."""
    out = df.copy()
    calculated = np.where(
        (out["prior_purchase_count"] > 0) & (out["prior_total_spend"].notna()),
        out["prior_total_spend"] / out["prior_purchase_count"],
        out["prior_mean_purchase_value"],
    )
    mismatch = (out["prior_purchase_count"] > 0) & (
        ~np.isclose(out["prior_mean_purchase_value"], calculated, atol=1e-4, equal_nan=True)
    )
    out.loc[mismatch, "prior_mean_purchase_value"] = calculated[mismatch]
    print(f"Total cells updated: {mismatch.sum()}")
    return out
 
 
def correct_spend_365d(df):
    """Fix placeholder (>= 1e8) or impossible (> prior_total_spend) values of spend_365d."""
    out = df.copy()
    before = (out["spend_365d"] >= 1e8).sum()
 
    def _fix(row):
        if row["spend_365d"] >= 1e8 or row["spend_365d"] > row["prior_total_spend"]:
            if row["purchase_count_365d"] == 0:
                return 0.0
            if row["purchase_count_365d"] == row["prior_purchase_count"]:
                return row["prior_total_spend"]
            estimated = row["purchase_count_365d"] * row["prior_mean_purchase_value"]
            estimated = min(estimated, row["prior_total_spend"])
            return max(estimated, row["spend_180d"])
        return row["spend_365d"]
 
    out["spend_365d"] = out.apply(_fix, axis=1)
    print(f"Corrupted entries detected: {before} | remaining: {(out['spend_365d'] >= 1e8).sum()}")
    return out
 
 
def correct_max_purchase_value(df):
    """Make sure prior_max_purchase_value never exceeds prior_total_spend."""
    out = df.copy()
    tol = 1e-4
    before = (out["prior_max_purchase_value"] > out["prior_total_spend"] + tol).sum()
 
    def _fix(row):
        if row["prior_max_purchase_value"] > row["prior_total_spend"] + tol:
            if row["prior_purchase_count"] == 1:
                return row["prior_total_spend"]
            max_possible = row["prior_total_spend"] - (
                (row["prior_purchase_count"] - 1) * row["prior_min_purchase_value"]
            )
            return max(row["prior_min_purchase_value"],
                       min(row["prior_total_spend"], max_possible))
        return row["prior_max_purchase_value"]
 
    out["prior_max_purchase_value"] = out.apply(_fix, axis=1)
    after = (out["prior_max_purchase_value"] > out["prior_total_spend"] + tol).sum()
    print(f"Inflated entries detected: {before} | remaining: {after}")
    return out
 
 
def extract_temporal_features(df):
    """Create calendar features from 'purchase_date'."""
    out = df.copy()
    out["purchase_date"] = pd.to_datetime(out["purchase_date"])
    out["purchase_year"] = out["purchase_date"].dt.year
    out["purchase_month"] = out["purchase_date"].dt.month
    out["purchase_day"] = out["purchase_date"].dt.day
    out["purchase_dayofweek"] = out["purchase_date"].dt.dayofweek  # 0 = Monday
    out["purchase_quarter"] = out["purchase_date"].dt.quarter
    out["is_weekend"] = out["purchase_dayofweek"].isin([5, 6]).astype(int)
    return out
 
 
def drop_purchase_date(df):
    """Drop the raw 'purchase_date' column (already decomposed into calendar features)."""
    return df.drop(columns=["purchase_date"], errors="ignore")
 
 
# --------------------------------------------------------------- plotting
def plot_monthly_distribution(df, dataset_label="Train"):
    """Transactions per month (seasonality check). Uses the raw 'purchase_date'."""
    tmp = df.dropna(subset=["purchase_date"]).copy()
    tmp["month_name"] = pd.to_datetime(tmp["purchase_date"]).dt.strftime("%b")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=tmp, x="month_name", order=MONTH_ORDER, color="#2b4c7e", zorder=2)
    plt.title(f"Transaction Distribution by Month ({dataset_label} - Seasonality Check)",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("Number of Transactions", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
 
 
def plot_weekend_distribution_comparison(df_train, df_test):
    """Weekday vs weekend transaction counts, train and test side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, df, title, color in zip(
        axes, (df_train, df_test), ("Train", "Test"), ("#2b4c7e", "#4a7c59")
    ):
        plot_df = df.assign(weekend_label=df["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
        sns.countplot(data=plot_df, x="weekend_label", order=["Weekday", "Weekend"],
                      color=color, zorder=2, ax=ax)
        ax.set_axisbelow(True)
        ax.set_title(f"{title} Dataset: Weekday vs. Weekend", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Period Type", fontsize=10)
        ax.set_ylabel("Number of Transactions", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=1)
    plt.tight_layout()
    plt.show()
 
 
def plot_weekend_retention_proportion(df, target_col=TARGET):
    """90-day repeat purchase rate on weekdays vs weekends (needs the target)."""
    if target_col not in df.columns:
        print(f"Note: '{target_col}' not found. Chart skipped.")
        return
    ct = pd.crosstab(df["is_weekend"], df[target_col], normalize="index") * 100
    ct.index = ["Weekdays (0)", "Weekends (1)"]
    ct.columns = ["No Repeat (0)", "Repeat Purchase (1)"]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#b0b0b0", linestyle="-", linewidth=0.8, zorder=0)
    ax = ct.plot(kind="bar", stacked=True, ax=ax, color=["#b0c4de", "#001f3f"], width=0.5, zorder=3)
    plt.title("Customer Retention Proportion: Weekdays vs. Weekends (Train Dataset)",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Purchase Period", fontsize=11, labelpad=10)
    plt.ylabel("Percentage (%)", fontsize=11, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(title="90-Day Repeat Status", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 100)
    for n, c in enumerate(ct.columns):
        for i, val in enumerate(ct[c]):
            if val > 5:
                ax.text(i, ct.iloc[i, :n].sum() + val / 2, f"{val:.1f}%", ha="center", va="center",
                        color="white" if n == 1 else "black", fontweight="bold", fontsize=10, zorder=4)
    plt.tight_layout()
    plt.show()
 
 
def plot_monthly_engagement_comparison(df_train, df_test, feature="purchase_count_365d"):
    """Mean of ``feature`` by purchase month, train and test side by side."""
    month_map = dict(enumerate(MONTH_ORDER, start=1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, df, title, color in zip(
        axes, (df_train, df_test), ("Train", "Test"), ("#2b4c7e", "#4a7c59")
    ):
        monthly = df.groupby("purchase_month")[feature].mean().reset_index()
        monthly["month_name"] = monthly["purchase_month"].map(month_map)
        sns.barplot(data=monthly, x="month_name", y=feature, order=MONTH_ORDER, color=color, ax=ax)
        ax.set_title(f"{title} Dataset: Mean {feature} by Month", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Purchase Month", fontsize=10)
        ax.set_ylabel(f"Mean {feature}", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
 
 
def plot_mean_transaction_value_comparison(df_train, df_test, value_col="Total a Pagar"):
    """Mean transaction value on business days vs weekends, train and test."""
    labels = {0: "Business Days (0)", 1: "Weekends (1)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, title in zip(axes, (df_train, df_test), ("Train Dataset", "Test Dataset")):
        plot_df = df.assign(period_label=df["is_weekend"].map(labels))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#b0b0b0", linestyle="-", linewidth=0.8, zorder=0)
        sns.barplot(data=plot_df, x="period_label", y=value_col, order=list(labels.values()),
                    hue="period_label", palette=["#001f3f", "#b0c4de"], legend=False,
                    ax=ax, zorder=3, errorbar=None)
        ax.set_title(f"{title}: Mean Spend by Period", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Purchase Period", fontsize=11, labelpad=10)
        ax.set_ylabel(f"Mean {value_col} (EUR)", fontsize=11, labelpad=10)
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"EUR {h:.2f}", (p.get_x() + p.get_width() / 2.0, h), ha="center",
                            va="bottom", xytext=(0, 4), textcoords="offset points",
                            fontsize=10, fontweight="bold", zorder=4)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    plt.tight_layout()
    plt.show()