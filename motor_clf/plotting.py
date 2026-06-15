"""
Figure generation for classifier comparison results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import OUT_DIR

COLORS = {
    'LDA_shrink': 'darkorange',
    'MDM_mb':     'mediumpurple',
    'TS_pb_LDA':  'seagreen',
    'TS_fb_LDA':  'steelblue',
    'EEGNet':     'crimson',
}

CLF_LABELS = {
    'LDA_shrink': 'LDA\n(shrinkage)',
    'MDM_mb':     'MDM\n(multiband)',
    'TS_pb_LDA':  'TS+pb\n+LDA',
    'TS_fb_LDA':  'TS+fb\n+LDA',
    'EEGNet':     'EEGNet',
}


def save_per_subject_chart(df, clf_names, out_dir=OUT_DIR):
    """Grouped bar chart: one group per subject, one bar per classifier."""
    present  = [n for n in clf_names if f'{n}_test_mean' in df.columns]
    subjects = df.index.tolist()
    n_subj   = len(subjects)
    n_clf    = len(present)

    w     = 0.8 / n_clf
    x     = np.arange(n_subj)
    rng   = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(max(14, n_subj * 0.7), 6))

    for i, name in enumerate(present):
        means = df[f'{name}_test_mean'].fillna(0).values
        stds  = df[f'{name}_test_std'].fillna(0).values
        offset = (i - (n_clf - 1) / 2) * w
        ax.bar(x + offset, means, w * 0.9, yerr=stds, capsize=3,
               color=COLORS.get(name, 'gray'), alpha=0.85,
               label=CLF_LABELS.get(name, name).replace('\n', ' '))

    ax.axhline(1 / 3, color='red', linestyle='--', linewidth=1, label='Chance (33%)')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Per-subject test accuracy — all classifiers\n(Tongue / Hand / Rest, 5-fold CV)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = out_dir / 'per_subject_accuracy.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved per-subject chart → {path}')


def save_bar_chart(df, clf_names, out_dir=OUT_DIR):
    """Per-subject bar chart of test accuracy for each classifier."""
    subjects = df.index.tolist()
    x = np.arange(len(subjects))

    fig, axes = plt.subplots(1, len(clf_names), figsize=(4.5 * len(clf_names), 5), sharey=True)
    if len(clf_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, clf_names):
        means = df[f'{name}_test_mean'].fillna(0).values
        stds  = df[f'{name}_test_std'].fillna(0).values
        ax.bar(x, means, yerr=stds, capsize=4, color=COLORS.get(name, 'gray'), alpha=0.8)
        ax.axhline(1/3, color='red', linestyle='--', linewidth=1, label='Chance (33%)')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{CLF_LABELS.get(name, name)}\nTongue / Hand / Rest')
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / 'classifier_accuracy_barplot.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved bar chart → {path}')


def save_comparison_chart(df, clf_names, out_dir=OUT_DIR):
    """Group-level train vs test accuracy + overfitting gap chart."""
    present = [n for n in clf_names if f'{n}_test_mean' in df.columns]
    x   = np.arange(len(present))
    w   = 0.35
    rng = np.random.default_rng(0)

    tr_means = [df[f'{n}_train_mean'].mean() for n in present]
    te_means = [df[f'{n}_test_mean'].mean()  for n in present]
    tr_sems  = [df[f'{n}_train_mean'].sem()  for n in present]
    te_sems  = [df[f'{n}_test_mean'].sem()   for n in present]
    gaps     = [tr - te for tr, te in zip(tr_means, te_means)]
    gap_sems = [(df[f'{n}_train_mean'] - df[f'{n}_test_mean']).sem() for n in present]
    gap_cols = ['#e15759' if g > 0.05 else '#76b7b2' for g in gaps]

    fig, (ax_acc, ax_gap) = plt.subplots(1, 2, figsize=(13, 5))

    ax_acc.bar(x - w/2, tr_means, w, yerr=tr_sems, capsize=4,
               color='#4e79a7', alpha=0.85, label='Train')
    ax_acc.bar(x + w/2, te_means, w, yerr=te_sems, capsize=4,
               color='#f28e2b', alpha=0.85, label='Test')
    for xi, name in enumerate(present):
        tr_vals = df[f'{name}_train_mean'].dropna().values
        te_vals = df[f'{name}_test_mean'].dropna().values
        jit = rng.uniform(-0.08, 0.08, len(tr_vals))
        ax_acc.scatter(xi - w/2 + jit, tr_vals, color='#4e79a7',
                       edgecolors='black', linewidths=0.4, s=22, zorder=3, alpha=0.7)
        ax_acc.scatter(xi + w/2 + jit, te_vals, color='#f28e2b',
                       edgecolors='black', linewidths=0.4, s=22, zorder=3, alpha=0.7)
    ax_acc.axhline(1/3, color='red', linestyle='--', linewidth=1.2, label='Chance')
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([CLF_LABELS.get(n, n) for n in present], fontsize=10)
    ax_acc.set_ylabel('Accuracy', fontsize=11)
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_title('Train vs test accuracy\n(group mean ± SEM, dots = subjects)', fontsize=11)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(axis='y', alpha=0.3)

    ax_gap.bar(x, gaps, yerr=gap_sems, capsize=4, color=gap_cols, alpha=0.85)
    ax_gap.axhline(0,    color='black', linewidth=0.8)
    ax_gap.axhline(0.05, color='gray',  linestyle=':', linewidth=1, label='5% threshold')
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([CLF_LABELS.get(n, n) for n in present], fontsize=10)
    ax_gap.set_ylabel('Train − Test (overfitting gap)', fontsize=11)
    ax_gap.set_title('Overfitting gap\n(red = gap > 5%)', fontsize=11)
    ax_gap.legend(fontsize=9)
    ax_gap.grid(axis='y', alpha=0.3)

    fig.suptitle('Classifier comparison — Tongue / Hand / Rest (5-fold stratified CV)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = out_dir / 'classifier_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved comparison chart → {path}')
