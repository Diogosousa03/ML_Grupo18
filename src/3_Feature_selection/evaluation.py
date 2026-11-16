"""Evaluation on temporal splits and paired comparison between techniques."""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from config import RANDOM_STATE, target
from preprocessing import fold_matrices


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


def report_vs_reference(run, reference, metrics=("AP", "F1")):
    """Print, for each metric, the paired difference of `run` from `reference`.

    Returns the `paired_delta` result of each metric, keyed by metric name.
    """
    results = {}
    for metric in metrics:
        r = paired_delta(run[metric], reference[metric])
        print(f"   d{metric} {r['delta']:+.4f} +/- {r['SEM']:.4f}, better on {r['wins']} -> {r['verdict']}")
        results[metric] = r
    return results


def stable_columns(run):
    """Columns kept in every split of `run` (the result of `held_out_ap`)."""
    kept = Counter(c for columns in run["kept"] for c in columns)
    return [str(c) for c, n in kept.items() if n == len(run)]


def evaluate_selector(frame, splits, recipe, label, select=None, reference=None, show_splits=False):
    """Run `held_out_ap`, print its summary and, if given, the comparison with `reference`.

    `label` names the technique in the printed line. With `show_splits=True`
    the per-split table is printed first. Returns the `held_out_ap` result.
    """
    run = held_out_ap(frame, splits, recipe, select=select)
    if show_splits:
        print(run[["n", "AP", "F1"]])
    print(f"{label}: {run['n'].mean():.0f} columns | AP {run['AP'].mean():.4f} | F1 {run['F1'].mean():.4f}")
    if reference is not None:
        report_vs_reference(run, reference)
    return run