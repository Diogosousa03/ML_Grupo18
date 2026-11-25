"""Evaluation on temporal splits and paired comparison between techniques."""
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from config import RANDOM_STATE, target
from preprocessing import fold_matrices


def _validation_row(A_train, A_validation, y_train, y_validation, names, keep, selector=None):
    """Train the logistic regression on the kept columns and score it on the validation rows.

    F1 cuts the probabilities at 0.5. `selector` is the fitted selector that
    chose `keep`, kept so its scores (coefficients, importances) can be read later.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(A_train[:, keep], y_train)
    probabilities = model.predict_proba(A_validation[:, keep])[:, 1]
    return {"n": int(keep.sum()),
            "AP": average_precision_score(y_validation, probabilities),
            "F1": f1_score(y_validation, probabilities >= 0.5),
            "kept": list(names[keep]),
            "selector": selector}


def held_out_ap(frame, splits, recipe, select=None, reduce=None):
    """Validation AP and F1 on every split, one row per split.

    `select()` builds a fresh, unfitted selector, fitted on each split's
    training matrix alone. Without it, the model sees every column. F1 cuts
    the probabilities at 0.5. Each row also keeps the columns used (`kept`)
    and the fitted selector (`selector`, None without one).

    `reduce(n)` builds a reducer of n components (such as PCA). It is fitted on
    each split's training matrix, with the fewest components that keep 80% of
    that split's variance, and the model sees the components instead of the
    columns.
    """
    rows = []
    for train_rows, validation_rows in splits:
        train, validation = frame.iloc[train_rows], frame.iloc[validation_rows]
        A_train, A_validation, names = fold_matrices(train, validation, recipe)
        if reduce is not None:
            # A first fit with almost every component tells how many reach 80%
            probe = reduce(A_train.shape[1] - 1).fit(A_train)
            k = int(np.searchsorted(probe.explained_variance_ratio_.cumsum(), 0.80) + 1)
            reducer = reduce(k).fit(A_train)
            A_train, A_validation = reducer.transform(A_train), reducer.transform(A_validation)
            names = np.array([f"component {i}" for i in range(1, k + 1)])
        keep, selector = np.ones(len(names), dtype=bool), None
        if select is not None:
            # The selector gets the matrix WITH its column names, because a filter that
            # judges a categorical as a whole has to know which dummies came from it
            matrix = pd.DataFrame(A_train, columns=names, index=train.index)
            selector = select().fit(matrix, train[target])
            keep = selector.get_support()
        rows.append(_validation_row(A_train, A_validation, train[target], validation[target], names, keep,
                                    selector))
    return pd.DataFrame(rows)


def majority_table(run):
    """How often each stage or member of a combination kept each column, over the splits.

    `run` is a `held_out_ap` result of a combination whose fitted selectors record
    `decisions_` (one column per stage or member, and "kept" for the combination).
    Each entry counts the splits in which that stage still held the column (a
    sequence) or that member kept it (a vote). Only the columns the combination
    kept in most splits are shown.
    """
    selectors = run["selector"]
    counts = pd.concat([fitted.decisions_ for fitted in selectors]).groupby(level=0).sum()
    shown = counts[counts["kept"] > len(selectors) / 2].sort_values(list(counts.columns[::-1]),
                                                                    ascending=False)
    return shown.astype(int).astype(str) + f"/{len(selectors)}"


def per_split(run, read):
    """`read` applied to every split's fitted selector in `run`, one row per split.

    Columns are matched by NAME: a level absent from a training split has no
    column there, so that split leaves it missing rather than zero.
    """
    return pd.DataFrame([pd.Series(read(fitted), index=fitted.feature_names_in_)
                         for fitted in run["selector"]])


def rfe_width_runs(frame, splits, recipe, widths, make_rfe):
    """`held_out_ap` with RFE at every width k in `widths`, as {k: result}.

    `make_rfe()` builds a fresh RFE with step=1 that stops at one column. RFE
    with step=1 removes one column per round, and which column goes does not
    depend on the width where it stops. So one RFE per split, run down to a
    single column, ranks every column, and its k best ranked (`ranking_ <= k`)
    are exactly the columns the same RFE keeps with n_features_to_select=k.
    The result is the same as one `held_out_ap` per width, with one elimination
    per split instead of one per width.
    """
    rows = {k: [] for k in widths}
    for train_rows, validation_rows in splits:
        train, validation = frame.iloc[train_rows], frame.iloc[validation_rows]
        A_train, A_validation, names = fold_matrices(train, validation, recipe)
        rfe = make_rfe()
        if rfe.step != 1 or rfe.n_features_to_select != 1:
            raise ValueError("the ranking gives every width only with step=1, down to one column")
        # The same input as in held_out_ap: the matrix with its column names
        matrix = pd.DataFrame(A_train, columns=names, index=train.index)
        ranking = rfe.fit(matrix, train[target]).ranking_
        for k in widths:
            rows[k].append(_validation_row(A_train, A_validation, train[target], validation[target],
                                           names, ranking <= k))
    return {k: pd.DataFrame(split_rows) for k, split_rows in rows.items()}


def paired_delta(scores, reference):
    """Mean split-by-split difference from `reference`, its SEM, wins and verdict."""
    delta = np.asarray(scores) - np.asarray(reference)
    sem = delta.std(ddof=1) / np.sqrt(len(delta))
    verdict = ("identical on every split" if not delta.any()
               else "inside the noise" if abs(delta.mean()) <= 2 * sem
               else "better" if delta.mean() > 0 else "worse")
    return {"delta": delta.mean(), "SEM": sem,
            "wins": f"{int((delta > 0).sum())}/{len(delta)}", "verdict": verdict}


def stable_columns(run):
    """Columns kept in every split of `run` (the result of `held_out_ap`)."""
    kept = Counter(c for columns in run["kept"] for c in columns)
    return [str(c) for c, n in kept.items() if n == len(run)]


def results_table(runs, reference=None, detail=True, stable=True):
    """One row per technique, built from `held_out_ap` results.

    `runs` maps a name to a `held_out_ap` result. Each row has the mean number of
    columns, the mean AP and F1 and, with `stable=True`, the number of columns kept
    in every split. With a `reference`, it adds for each metric the paired
    difference from it and its verdict; `detail=True` also shows the SEM of that
    difference and the number of splits where the technique is better. The
    reference's own row has no difference.

    A run marked as model-selection validation (a column `evaluation` with that
    value, as in the course practical) had its parameter chosen on these same
    splits. Its scores are optimistic, so no difference from the reference is
    claimed: its verdict reads "no unbiased delta".
    """
    rows = {}
    for name, run in runs.items():
        row = {"columns": run["n"].mean(), "AP": run["AP"].mean(), "F1": run["F1"].mean()}
        if stable:
            row["stable columns"] = len(stable_columns(run))
        chosen_on_these_splits = ("evaluation" in run
                                  and run["evaluation"].eq("model-selection validation").all())
        if reference is not None and run is not reference:
            for metric in ("AP", "F1"):
                if chosen_on_these_splits:
                    row[f"{metric} verdict"] = "no unbiased delta"
                    continue
                r = paired_delta(run[metric], reference[metric])
                row[f"d{metric}"] = r["delta"]
                if detail:
                    row[f"{metric} SEM"] = r["SEM"]
                    row[f"{metric} better on"] = r["wins"]
                row[f"{metric} verdict"] = r["verdict"]
        rows[name] = row
    decimals = {"columns": 1, "AP": 4, "F1": 4, "dAP": 4, "dF1": 4, "AP SEM": 4, "F1 SEM": 4}
    return pd.DataFrame.from_dict(rows, orient="index").round(decimals)


def evaluate_selector(frame, splits, recipe, label, select=None, reference=None, show_splits=False):
    """`held_out_ap` with `select`, shown as tables; returns the `held_out_ap` result.

    The row is named `label` and, with a `reference`, carries the paired difference
    from it (`results_table`). With `show_splits=True` the per-split results are
    shown first.
    """
    run = held_out_ap(frame, splits, recipe, select=select)
    if show_splits:
        display(run[["n", "AP", "F1"]].round(4))
    display(results_table({label: run}, reference=reference))
    return run

def stable_table(run):
    """The columns kept in every split of `run`, as a one-column table."""
    return pd.DataFrame({"kept in every split": stable_columns(run)})