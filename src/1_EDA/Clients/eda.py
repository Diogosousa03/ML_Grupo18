import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#------------------------------------------------
# Data Ingestion
#------------------------------------------------

def plot_category_bars(values, title, xlabel, ylabel, figsize=(8, 4), margin=1.15):
    """Horizontal bar chart with one bar per category and the value written at the end of each bar.

    values: Series with the categories in the index and the size of each bar in the values
            (for example, the result of value_counts()).
    margin: how much space is left after the largest bar, so the labels are not cut off.
    """
    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x=values.values,
        y=values.index,
        legend=False
    )

    # Add the label with the exact value at the end of each bar
    for container in ax.containers:
        ax.bar_label(container, padding=3)

    # Remove the X-axis
    plt.xticks([])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, values.max() * margin)
    plt.tight_layout()
    plt.show()


#------------------------------------------------
# Data Cleaning
#------------------------------------------------

def customer_location(df):
    """Where each customer is: abroad, in an island region of CT10 (RG17 or RG29),
    in another region of CT10, or in CT10 with no region recorded.

    Returns a Series with one label per customer.
    """
    is_ct10 = df["country_or_region"].str.startswith("Country CT10:")
    no_region = df["commercial_region"].isna()
    island_region = df["commercial_region"].str.startswith(("Region RG17:", "Region RG29:"), na=False)

    return pd.Series(np.select(
        [~is_ct10, no_region, island_region],
        ["foreign", "CT10, no region", "CT10, island region (RG17/RG29)"],
        default="CT10, other region"), index=df.index, name="where the customer is")


def gaps_table(missing_before, missing_after, names):
    """Missing values per column before and after a step, and how many were filled.

    missing_before, missing_after: number of missing values per column (result of isna().sum()).
    names: the three column names of the table (before, filled, after).
    Only the columns that had missing values before are shown.
    """
    before, filled, after = names
    table = pd.DataFrame({
        before: missing_before,
        filled: missing_before - missing_after,
        after: missing_after,
    })
    return table[table[before] > 0]


#------------------------------------------------
# Data Ingestion and Data Cleaning (used in both)
#------------------------------------------------

def cardinality_table(df, names):
    """Number of filled values and of distinct values per column, and the distinct values as a
    percentage of the filled ones (close to 100% means the column works like an identifier).

    names: the three column names of the table (filled, distinct, percentage).
    The table is sorted from the highest to the lowest percentage.
    """
    filled, distinct, pct = names
    table = pd.DataFrame({
        filled: df.notna().sum(),
        distinct: df.nunique(),
    })
    table[pct] = (table[distinct] / table[filled] * 100).round(1)
    return table.sort_values(pct, ascending=False)


def island_status(df):
    """For each customer, "island filled" or "island missing"."""
    return df["island"].isna().map({False: "island filled", True: "island missing"})


def load_purchase_dates(train_file):
    """Read only customer_id and purchase_date from train.csv, with purchase_date as a date."""
    return pd.read_csv(train_file, usecols=["customer_id", "purchase_date"],
                       parse_dates=["purchase_date"])


def get_first_purchase_year(train_dates):
    """Year of the first purchase of each customer (one value per customer_id)."""
    return train_dates.groupby("customer_id")["purchase_date"].min().dt.year.rename("first_purchase_year")


def plot_year_counts(counts, title, xlabel, ylabel, figsize, color=None, rotation=0):
    """Vertical bar chart with the number of customers in each year.

    counts: Series with the years in the index and the number of customers in the values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", color=color, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()
