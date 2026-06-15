"""
Per-subject cross-validation helpers.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CLASS_NAMES, N_SPLITS
from .data_io import load_subject_raw_trials
from .pipelines import RIEMANN_CLASSIFIERS


def cv_scores(make_clf, X, y, n_splits=N_SPLITS):
    """Run stratified k-fold CV. Returns (train_scores, test_scores) or (None, None)."""
    min_count     = min((y == i).sum() for i in range(len(CLASS_NAMES)))
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        return None, None
    cv  = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    res = cross_validate(make_clf(), X, y, cv=cv,
                         scoring='accuracy', return_train_score=True)
    return res['train_score'], res['test_score']


def evaluate_subject_riemann(mat_path):
    """Return a result dict for one subject across all RIEMANN_CLASSIFIERS."""
    X, y = load_subject_raw_trials(mat_path)
    if X is None:
        return None
    row = {'subject': mat_path.stem.split('_')[0]}
    for name, make_clf in RIEMANN_CLASSIFIERS.items():
        tr, te = cv_scores(make_clf, X, y)
        if te is None:
            row[f'{name}_train_mean'] = row[f'{name}_train_std'] = np.nan
            row[f'{name}_test_mean']  = row[f'{name}_test_std']  = np.nan
        else:
            row[f'{name}_train_mean'] = tr.mean(); row[f'{name}_train_std'] = tr.std()
            row[f'{name}_test_mean']  = te.mean(); row[f'{name}_test_std']  = te.std()
    return row
