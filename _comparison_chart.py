import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/Users/arthurhsia/Desktop/motor_basic/figs')
df = pd.read_csv('/Users/arthurhsia/Desktop/motor_basic/data/classifier_results.csv', index_col='subject')

clf_labels = {
    'LDA':        'LDA',
    'LDA_shrink': 'LDA\n(shrinkage)',
    'LinearSVM':  'Linear\nSVM',
    'MDM':        'MDM\n(Riemann)',
    'TS_LR':      'TS+LR\n(Riemann)',
}
colors = {
    'LDA': 'steelblue', 'LDA_shrink': 'darkorange',
    'LinearSVM': 'seagreen', 'MDM': 'mediumpurple', 'TS_LR': 'firebrick',
}

present = [n for n in clf_labels if f'{n}_test_mean' in df.columns]

# ── Figure 1: Train vs Test grouped bar chart (overfitting view) ─────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(present))
w = 0.3
rng = np.random.default_rng(0)

for i, clf in enumerate(present):
    train_vals = df[f'{clf}_train_mean'].dropna().values
    test_vals  = df[f'{clf}_test_mean'].dropna().values
    c = colors[clf]

    # train bar (lighter)
    ax.bar(x[i] - w/2, train_vals.mean(), width=w,
           color=c, alpha=0.35, edgecolor='k', linewidth=0.6, label='_nolegend_')
    # test bar (solid)
    ax.bar(x[i] + w/2, test_vals.mean(), width=w,
           color=c, alpha=0.85, edgecolor='k', linewidth=0.6, label='_nolegend_')

    # subject dots on test bar
    jitter = rng.uniform(-0.08, 0.08, size=len(test_vals))
    ax.scatter(x[i] + w/2 + jitter, test_vals,
               color=c, edgecolors='black', linewidths=0.5, s=25, zorder=4, alpha=0.9)

    # gap annotation
    gap = train_vals.mean() - test_vals.mean()
    ax.annotate(f'Δ{gap:.2f}',
                xy=(x[i], max(train_vals.mean(), test_vals.mean()) + 0.01),
                ha='center', va='bottom', fontsize=8, color='dimgray')

ax.axhline(1/3, color='red', linestyle='--', linewidth=1.2, zorder=1)
ax.text(len(present) - 0.45, 1/3 + 0.01, 'Chance (33%)',
        color='red', fontsize=8, va='bottom')

# legend proxies
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='gray', alpha=0.35, edgecolor='k', label='Train (lighter)'),
    Patch(facecolor='gray', alpha=0.85, edgecolor='k', label='Test (darker, dots = subjects)'),
], fontsize=9, loc='lower right')

ax.set_xticks(x)
ax.set_xticklabels([clf_labels[n] for n in present], fontsize=10)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title(
    'Classifier accuracy — Tongue / Hand / Rest\n'
    'Train (light) vs Test (dark) · Δ = overfitting gap · dots = individual subjects',
    fontsize=11,
)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
path = OUT_DIR / 'classifier_comparison.png'
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {path}')

# ── Figure 2: Test-only bar chart (clean accuracy summary) ───────────────────
fig, ax = plt.subplots(figsize=(7, 5))
test_means = [df[f'{n}_test_mean'].mean() for n in present]
test_sems  = [df[f'{n}_test_mean'].sem()  for n in present]
bar_colors = [colors[n] for n in present]

ax.bar(x, test_means, yerr=test_sems, capsize=5, color=bar_colors, alpha=0.8,
       error_kw={'linewidth': 1.5}, zorder=2)

for xi, clf in enumerate(present):
    vals = df[f'{clf}_test_mean'].dropna().values
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(xi + jitter, vals, color=bar_colors[xi],
               edgecolors='black', linewidths=0.5, s=30, zorder=3, alpha=0.85)

ax.axhline(1/3, color='red', linestyle='--', linewidth=1.2, label='Chance (33%)', zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([clf_labels[n] for n in present], fontsize=10)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title(
    'Classifier test accuracy — Tongue / Hand / Rest\n'
    'Group mean ± SEM, dots = individual subjects',
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
path = OUT_DIR / 'classifier_accuracy_barplot.png'
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {path}')
