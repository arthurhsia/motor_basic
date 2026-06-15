"""
feature_analysis.py

Analyses what features drive classification across methods and why
broadband-covariance Riemannian classifiers underperform band-power ones.

Four panels
-----------
1. PSD profiles per condition (averaged across subjects, log-scale)
   – shows LFB power >> HFB power, so raw covariance is dominated by LFB
2. Per-channel selectivity: |r| distributions for HFB vs LFB, hand vs tongue
   – shows HFB carries more discriminative signal per channel
3. Linear classifier feature weights: fraction of |coef| on HFB vs LFB
   – shows all linear classifiers rely primarily on HFB
4. Ablation: HFB-only vs LFB-only vs combined accuracy (LDA-shrink)
   – directly quantifies each band's contribution

Run:
    python3 feature_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / 'data'
OUT_DIR  = Path(__file__).parent / 'figs'
OUT_DIR.mkdir(exist_ok=True)

CODE_TONGUE      = 11
CODE_HAND        = 12
CODE_REST_TONGUE = 110
CODE_REST_HAND   = 120
VALID_CODES  = {CODE_TONGUE, CODE_HAND, CODE_REST_TONGUE, CODE_REST_HAND}
CLASS_NAMES  = ['tongue', 'hand', 'rest']
LABEL_MAP    = {CODE_TONGUE: 'tongue', CODE_HAND: 'hand',
                CODE_REST_TONGUE: 'rest', CODE_REST_HAND: 'rest'}

FREQS     = np.arange(200)   # 0–199 Hz (1 Hz bins)
LFB_SLICE = slice(7, 32)     # 8–32 Hz
HFB_SLICE = slice(75, 100)   # 76–100 Hz
N_SPLITS  = 5

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_npz(path):
    d  = np.load(path)
    keep   = np.isin(d['tr_sc'], list(VALID_CODES))
    HFB    = d['HFB_trials'][:, keep]  # (n_ch, n_trials)
    LFB    = d['LFB_trials'][:, keep]
    tr_sc  = d['tr_sc'][keep]
    y      = np.array([CLASS_NAMES.index(LABEL_MAP[c]) for c in tr_sc])
    X_hfb  = HFB.T                      # (n_trials, n_ch)
    X_lfb  = LFB.T
    X_both = np.hstack([X_hfb, X_lfb])  # (n_trials, 2*n_ch)
    return X_hfb, X_lfb, X_both, y, d


def cv_accuracy(X, y, make_clf, n_splits=N_SPLITS):
    min_c = min((y == i).sum() for i in range(len(CLASS_NAMES)))
    k     = min(n_splits, min_c)
    if k < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return cross_val_score(make_clf(), X, y, cv=cv, scoring='accuracy').mean()


def make_lda_shrink():
    return Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ])


def make_clf_for_weights(name):
    if name == 'LDA':
        return Pipeline([('sc', StandardScaler()),
                         ('clf', LinearDiscriminantAnalysis())])
    if name == 'LDA_shrink':
        return Pipeline([('sc', StandardScaler()),
                         ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))])
    if name == 'LinearSVM':
        return Pipeline([('sc', StandardScaler()),
                         ('clf', LinearSVC(max_iter=5000, C=0.1))])

# ---------------------------------------------------------------------------
# Collect per-subject data
# ---------------------------------------------------------------------------

npz_paths = sorted(DATA_DIR.glob('*_mot_th_analyzed.npz'))
print(f'Analysing {len(npz_paths)} subjects...')

# --- Panel 1: PSD profiles ---
psd_conditions = {
    'Hand move':    'mean_PSD_handmove',
    'Hand rest':    'mean_PSD_handrest',
    'Tongue move':  'mean_PSD_tonguemove',
    'Tongue rest':  'mean_PSD_tonguerest',
}
psd_accum = {k: [] for k in psd_conditions}  # will hold per-subject channel-mean PSDs

# --- Panel 2: r-value distributions ---
r_records = []   # rows: (subj, channel, band, movement, |r|)

# --- Panel 3: classifier weights ---
clf_names    = ['LDA', 'LDA_shrink', 'LinearSVM']
weight_fracs = {n: {'HFB': [], 'LFB': []} for n in clf_names}

# --- Panel 4: ablation ---
ablation = {'HFB-only': [], 'LFB-only': [], 'Both': []}

for npz_path in npz_paths:
    subj = npz_path.stem.split('_')[0]
    X_hfb, X_lfb, X_both, y, d = load_npz(npz_path)

    # --- PSD (channel-mean, normalised by overall mean PSD) ---
    overall = d['mean_PSD'].mean(axis=1)   # (200,)
    for label, key in psd_conditions.items():
        cond_psd = d[key].mean(axis=1)     # (200,)
        psd_accum[label].append(cond_psd / (overall + 1e-12))

    # --- r-values ---
    n_ch = d['r_hand_HFB'].shape[0]
    for m in range(n_ch):
        r_records.append({'band': 'HFB', 'movement': 'hand',   '|r|': abs(d['r_hand_HFB'][m])})
        r_records.append({'band': 'LFB', 'movement': 'hand',   '|r|': abs(d['r_hand_LFB'][m])})
        r_records.append({'band': 'HFB', 'movement': 'tongue', '|r|': abs(d['r_tongue_HFB'][m])})
        r_records.append({'band': 'LFB', 'movement': 'tongue', '|r|': abs(d['r_tongue_LFB'][m])})

    # --- classifier weights: HFB fraction vs LFB fraction ---
    for clf_name in clf_names:
        clf = make_clf_for_weights(clf_name)
        try:
            clf.fit(X_both, y)
            coef = clf.named_steps['clf'].coef_  # (n_classes or n_pairs, 2*n_ch)
            abs_w    = np.abs(coef).mean(axis=0)  # (2*n_ch,) mean over discriminants
            n_ch     = X_hfb.shape[1]
            hfb_w    = abs_w[:n_ch].sum()
            lfb_w    = abs_w[n_ch:].sum()
            total_w  = hfb_w + lfb_w + 1e-12
            weight_fracs[clf_name]['HFB'].append(hfb_w / total_w)
            weight_fracs[clf_name]['LFB'].append(lfb_w / total_w)
        except Exception:
            pass

    # --- ablation ---
    ablation['HFB-only'].append(cv_accuracy(X_hfb,  y, make_lda_shrink))
    ablation['LFB-only'].append(cv_accuracy(X_lfb,  y, make_lda_shrink))
    ablation['Both'].append(    cv_accuracy(X_both, y, make_lda_shrink))

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Feature analysis: what drives classification across methods',
             fontsize=13, fontweight='bold')
ax1, ax2, ax3, ax4 = axes.flat

# ---------------------------------------------------------------------------
# Panel 1 — PSD profiles
# ---------------------------------------------------------------------------
colors_psd = {
    'Hand move':   '#e41a1c',
    'Hand rest':   '#fb9a99',
    'Tongue move': '#377eb8',
    'Tongue rest': '#a6cee3',
}
for label, arr_list in psd_accum.items():
    mat = np.array(arr_list)        # (n_subj, 200)
    mu  = mat.mean(axis=0)
    se  = mat.std(axis=0) / np.sqrt(len(arr_list))
    ax1.semilogy(FREQS, mu, color=colors_psd[label], label=label, linewidth=1.5)
    ax1.fill_between(FREQS, mu - se, mu + se,
                     color=colors_psd[label], alpha=0.15)

ax1.axvspan(8,  32,  alpha=0.12, color='gold',     label='LFB (8–32 Hz)')
ax1.axvspan(76, 100, alpha=0.15, color='limegreen', label='HFB (76–100 Hz)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Normalised PSD (log scale)')
ax1.set_title('Panel 1  PSD per condition\n'
              '→ LFB power >> HFB power; broadband covariance is dominated by LFB')
ax1.legend(fontsize=7, ncol=2)
ax1.set_xlim(0, 150)

# ---------------------------------------------------------------------------
# Panel 2 — |r| distributions
# ---------------------------------------------------------------------------
df_r = pd.DataFrame(r_records)
groups = [('HFB', 'hand'), ('LFB', 'hand'), ('HFB', 'tongue'), ('LFB', 'tongue')]
labels_r  = ['HFB\nhand', 'LFB\nhand', 'HFB\ntongue', 'LFB\ntongue']
colors_r  = ['#e41a1c', '#fb9a99', '#377eb8', '#a6cee3']
data_r    = [df_r[(df_r.band == b) & (df_r.movement == m)]['|r|'].values
             for b, m in groups]

bplot = ax2.boxplot(data_r, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
for patch, col in zip(bplot['boxes'], colors_r):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(labels_r, fontsize=9)
ax2.set_ylabel('|r|  (point-biserial correlation, move vs rest)')
ax2.set_title('Panel 2  Per-channel selectivity\n'
              '→ HFB channels are more discriminative than LFB channels')
ax2.grid(axis='y', alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 3 — classifier weight fractions
# ---------------------------------------------------------------------------
x3 = np.arange(len(clf_names))
hfb_means = [np.mean(weight_fracs[n]['HFB']) for n in clf_names]
lfb_means = [np.mean(weight_fracs[n]['LFB']) for n in clf_names]
hfb_sems  = [np.std(weight_fracs[n]['HFB']) / np.sqrt(len(weight_fracs[n]['HFB']))
             for n in clf_names]
lfb_sems  = [np.std(weight_fracs[n]['LFB']) / np.sqrt(len(weight_fracs[n]['LFB']))
             for n in clf_names]

w = 0.35
bars_hfb = ax3.bar(x3 - w/2, hfb_means, w, yerr=hfb_sems, capsize=4,
                   color='limegreen', alpha=0.8, label='HFB (76–100 Hz)')
bars_lfb = ax3.bar(x3 + w/2, lfb_means, w, yerr=lfb_sems, capsize=4,
                   color='gold',      alpha=0.8, label='LFB (8–32 Hz)')
ax3.set_xticks(x3)
ax3.set_xticklabels(clf_names, fontsize=10)
ax3.set_ylabel('Fraction of total |weight|')
ax3.set_ylim(0, 1.0)
ax3.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
ax3.set_title('Panel 3  Classifier weight: HFB vs LFB fraction\n'
              '→ All linear classifiers rely primarily on HFB features')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 4 — ablation
# ---------------------------------------------------------------------------
abl_names  = ['HFB-only', 'LFB-only', 'Both']
abl_means  = [np.nanmean(ablation[k]) for k in abl_names]
abl_sems   = [np.nanstd(ablation[k]) / np.sqrt(np.sum(~np.isnan(ablation[k])))
              for k in abl_names]
abl_colors = ['limegreen', 'gold', 'steelblue']

x4 = np.arange(len(abl_names))
ax4.bar(x4, abl_means, yerr=abl_sems, capsize=5,
        color=abl_colors, alpha=0.8)

rng = np.random.default_rng(1)
for xi, key in enumerate(abl_names):
    vals = np.array(ablation[key])
    vals = vals[~np.isnan(vals)]
    ax4.scatter(xi + rng.uniform(-0.15, 0.15, len(vals)), vals,
                color=abl_colors[xi], edgecolors='black',
                linewidths=0.5, s=28, zorder=3, alpha=0.85)

ax4.axhline(1/3, color='red', linestyle='--', linewidth=1.2, label='Chance (33%)')
ax4.set_xticks(x4)
ax4.set_xticklabels(['HFB only\n(76–100 Hz)', 'LFB only\n(8–32 Hz)', 'HFB + LFB\n(both)'],
                    fontsize=9)
ax4.set_ylabel('Accuracy (LDA-shrinkage, 5-fold CV)')
ax4.set_ylim(0, 1.05)
ax4.set_title('Panel 4  Ablation: HFB-only vs LFB-only vs both\n'
              '→ HFB alone captures most of the discriminative signal')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# print group summaries
print('\nAblation group means (LDA-shrinkage):')
for key in abl_names:
    vals = np.array(ablation[key])
    print(f'  {key:12s}: {np.nanmean(vals):.3f} ± {np.nanstd(vals)/np.sqrt(np.sum(~np.isnan(vals))):.3f} SEM')

print('\nClassifier weight fractions (group mean):')
for clf_name in clf_names:
    h = np.mean(weight_fracs[clf_name]['HFB'])
    l = np.mean(weight_fracs[clf_name]['LFB'])
    print(f'  {clf_name:12s}: HFB={h:.2f}  LFB={l:.2f}')

fig.tight_layout()
out = OUT_DIR / 'feature_analysis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {out}')
