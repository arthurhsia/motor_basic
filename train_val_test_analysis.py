"""
train_val_test_analysis.py

Reports training vs held-out test accuracy for every classifier using
stratified 5-fold CV (sklearn cross_validate with return_train_score=True).

Within each fold:
  - Train acc  : classifier fitted on 4 folds, scored on those same 4 folds
  - Test acc   : same classifier scored on the held-out fold

A large train–test gap indicates overfitting.

Run:
    python3 train_val_test_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM, TSClassifier
    RIEMANN_AVAILABLE = True
except ImportError:
    RIEMANN_AVAILABLE = False

# reuse loaders from the main script
from classify_tongue_hand_rest import (
    load_subject_bandpower,
    load_subject_raw_trials,
    CLASS_NAMES,
)

DATA_DIR = Path(__file__).parent / 'data'
OUT_DIR  = Path(__file__).parent / 'figs'
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5

# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------

def make_lda():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', LinearDiscriminantAnalysis())])

def make_lda_shrink():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))])

def make_svm():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', LinearSVC(max_iter=5000, C=0.1))])

def make_mdm():
    return Pipeline([('cov', Covariances(estimator='lwf')),
                     ('mdm', MDM(metric='riemann'))])

def make_ts_lr():
    return Pipeline([('cov', Covariances(estimator='lwf')),
                     ('ts',  TSClassifier(clf=LogisticRegression(C=0.01, max_iter=1000)))])

# ---------------------------------------------------------------------------
# Classifier registry: name -> (factory, loader, path_glob)
# ---------------------------------------------------------------------------

NPZ_GLOB = '*_mot_th_analyzed.npz'
MAT_GLOB = '*_mot_t_h.mat'

BP_CLFS = {
    'LDA':        make_lda,
    'LDA_shrink': make_lda_shrink,
    'LinearSVM':  make_svm,
}

RIEMANN_CLFS = {
    'MDM':   (make_mdm,   load_subject_raw_trials),
    'TS_LR': (make_ts_lr, load_subject_raw_trials),
} if RIEMANN_AVAILABLE else {}

# ---------------------------------------------------------------------------
# Per-subject CV helper
# ---------------------------------------------------------------------------

def cv_train_test(make_clf, X, y):
    min_c = min((y == i).sum() for i in range(len(CLASS_NAMES)))
    k     = min(N_SPLITS, min_c)
    if k < 2:
        return None, None
    cv  = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    res = cross_validate(make_clf(), X, y, cv=cv,
                         scoring='accuracy', return_train_score=True)
    return res['train_score'].mean(), res['test_score'].mean()


def cv_block_test(make_clf, X, y):
    """Block CV: hold out contiguous temporal chunks; leakage = random_test - block_test."""
    min_c = min((y == i).sum() for i in range(len(CLASS_NAMES)))
    k     = min(N_SPLITS, min_c)
    if k < 2:
        return None
    groups = (np.arange(len(y)) * (k * 2) // len(y)).astype(int)
    cv  = GroupKFold(n_splits=k)
    res = cross_validate(make_clf(), X, y, cv=cv, groups=groups,
                         scoring='accuracy', return_train_score=False)
    return res['test_score'].mean()

# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------

npz_paths = sorted(DATA_DIR.glob(NPZ_GLOB))
mat_paths = sorted(DATA_DIR.glob(MAT_GLOB))
subj_from   = lambda p: p.stem.split('_')[0]
mat_by_subj = {subj_from(p): p for p in mat_paths}

print(f'Subjects: {len(npz_paths)}\n')

rows = []
for npz_path in npz_paths:
    subj = subj_from(npz_path)
    print(f'  {subj}...', end=' ', flush=True)

    X_bp, y_bp = load_subject_bandpower(npz_path)

    row = {'subject': subj}
    for clf_name, make_clf in BP_CLFS.items():
        tr, te = cv_train_test(make_clf, X_bp, y_bp)
        te_b   = cv_block_test(make_clf, X_bp, y_bp)
        row[f'{clf_name}_train']      = tr
        row[f'{clf_name}_test']       = te
        row[f'{clf_name}_block_test'] = te_b

    if RIEMANN_AVAILABLE and subj in mat_by_subj:
        for clf_name, (make_clf, loader) in RIEMANN_CLFS.items():
            X_r, y_r = loader(mat_by_subj[subj])
            if X_r is None:
                row[f'{clf_name}_train']      = np.nan
                row[f'{clf_name}_test']       = np.nan
                row[f'{clf_name}_block_test'] = np.nan
            else:
                tr, te = cv_train_test(make_clf, X_r, y_r)
                te_b   = cv_block_test(make_clf, X_r, y_r)
                row[f'{clf_name}_train']      = tr
                row[f'{clf_name}_test']       = te
                row[f'{clf_name}_block_test'] = te_b

    rows.append(row)
    print('done')

df = pd.DataFrame(rows).set_index('subject')

clf_names = list(BP_CLFS.keys()) + (list(RIEMANN_CLFS.keys()) if RIEMANN_AVAILABLE else [])

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

print('\n' + '=' * 88)
print(f'{"Classifier":<14}  {"Train":>7}  {"Test":>7}  {"Block Test":>10}  '
      f'{"Gap":>6}  {"Leakage":>8}')
print('  (Leakage = Test - Block Test; positive means random-CV was optimistic)')
print('=' * 88)
for clf_name in clf_names:
    tr_col = f'{clf_name}_train'
    te_col = f'{clf_name}_test'
    bl_col = f'{clf_name}_block_test'
    if tr_col not in df.columns:
        continue
    tr_mean = df[tr_col].mean()
    te_mean = df[te_col].mean()
    tr_sem  = df[tr_col].sem()
    te_sem  = df[te_col].sem()
    gap     = tr_mean - te_mean
    if bl_col in df.columns:
        bl_mean    = df[bl_col].mean()
        bl_sem     = df[bl_col].sem()
        leakage    = te_mean - bl_mean
        bl_str     = f'{bl_mean:.3f}±{bl_sem:.3f}'
        leak_str   = f'{leakage:+.3f}'
    else:
        bl_str, leak_str = '       n/a', '     n/a'
    print(f'{clf_name:<14}  {tr_mean:.3f}±{tr_sem:.3f}  '
          f'{te_mean:.3f}±{te_sem:.3f}  {bl_str:>10}  {gap:+.3f}  {leak_str:>8}')
print('Chance: 0.333')

# ---------------------------------------------------------------------------
# Figure: paired train/test/block-test bars per classifier
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                         gridspec_kw={'width_ratios': [1.5, 1]})

# --- left: grouped bar chart (train / random-test / block-test) ---
ax = axes[0]
x        = np.arange(len(clf_names))
w        = 0.25
tr_means = [df[f'{n}_train'].mean() for n in clf_names]
te_means = [df[f'{n}_test'].mean()  for n in clf_names]
bl_means = [df[f'{n}_block_test'].mean() if f'{n}_block_test' in df.columns else np.nan
            for n in clf_names]
tr_sems  = [df[f'{n}_train'].sem() for n in clf_names]
te_sems  = [df[f'{n}_test'].sem()  for n in clf_names]
bl_sems  = [df[f'{n}_block_test'].sem() if f'{n}_block_test' in df.columns else 0
            for n in clf_names]

ax.bar(x - w,   tr_means, w, yerr=tr_sems, capsize=4,
       color='#4e79a7', alpha=0.85, label='Train')
ax.bar(x,       te_means, w, yerr=te_sems, capsize=4,
       color='#f28e2b', alpha=0.85, label='Test (random CV)')
ax.bar(x + w,   bl_means, w, yerr=bl_sems, capsize=4,
       color='#59a14f', alpha=0.85, label='Test (block CV)')
ax.axhline(1/3, color='red', linestyle='--', linewidth=1, label='Chance')
ax.set_xticks(x)
ax.set_xticklabels(clf_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.1)
ax.set_title('Train / random-CV test / block-CV test\n(group mean ± SEM)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- right: overfitting gap + leakage estimate ---
ax2 = axes[1]
gaps     = [df[f'{n}_train'].mean() - df[f'{n}_test'].mean() for n in clf_names]
leakages = [(df[f'{n}_test'].mean() - df[f'{n}_block_test'].mean())
            if f'{n}_block_test' in df.columns else np.nan
            for n in clf_names]
gap_sems  = [(df[f'{n}_train'] - df[f'{n}_test']).sem() for n in clf_names]
bar_cols  = ['#e15759' if g > 0.05 else '#76b7b2' for g in gaps]

ax2.bar(x - w/2, gaps,     w, yerr=gap_sems, capsize=4,
        color=bar_cols, alpha=0.85, label='Overfit gap (train−test)')
ax2.bar(x + w/2, leakages, w, capsize=4,
        color='#ff9da7', alpha=0.85, label='Leakage (test−block_test)')
ax2.axhline(0,    color='black', linewidth=0.8)
ax2.axhline(0.05, color='gray',  linestyle=':', linewidth=1, label='5% threshold')
ax2.set_xticks(x)
ax2.set_xticklabels(clf_names, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Accuracy difference')
ax2.set_title('Overfitting gap & leakage estimate\n(red gap > 5%)')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Train vs test accuracy — Tongue / Hand / Rest (5-fold CV)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
out = OUT_DIR / 'train_test_accuracy.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {out}')

# Save CSV
csv_out = DATA_DIR / 'train_test_results.csv'
df.to_csv(csv_out)
print(f'Saved → {csv_out}')
