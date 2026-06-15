"""
noise_analysis_extended.py — three diagnostic analyses across all 19 subjects.

Generates figures in figs/noise/:
  1. electrode_maps.png     — per-subject HFB R² per channel, top-k highlighted
  2. subject_quality.png    — per-subject signal quality summary + accuracy correlation
  3. stationarity.png       — trial-by-trial HFB drift + first vs last half test
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import mannwhitneyu, pearsonr, kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from motor_clf.config import (
    DATA_DIR, SRATE, WIN_START, WIN_LEN, K_CHAN,
    VALID_CODES, LABEL_MAP, CLASS_NAMES,
)

OUT   = Path('figs/noise')
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_subject(mat_path):
    d    = sio.loadmat(str(mat_path))
    raw  = d['data'].astype(float)
    stim = d['stim'].flatten().astype(int)
    car  = raw - raw.mean(axis=1, keepdims=True)

    diff       = np.concatenate([[0], np.diff(stim)])
    changes    = np.where(np.concatenate([[True], stim[1:] != stim[:-1]]))[0]
    codes      = np.zeros(len(changes), dtype=int)
    codes[1:]  = stim[changes[1:]]

    # label rest periods
    for i, (onset, code) in enumerate(zip(changes, codes)):
        pass  # already handled in data_io; here we just collect valid onsets

    trials, labels = [], []
    for onset, code in zip(changes, codes):
        if code not in VALID_CODES: continue
        s = onset + WIN_START
        e = s + WIN_LEN
        if e > len(car): continue
        trials.append(car[s:e, :].T)           # (n_ch, T)
        labels.append(CLASS_NAMES.index(LABEL_MAP[code]))

    if not trials:
        return None, None, None, None

    X   = np.stack(trials)    # (n_trials, n_ch, T)
    y   = np.array(labels)
    return X, y, car, stim


def hfb_r2(X, y, lo=70, hi=200):
    """OVR R² per channel using HFB variance — same as HFBChannelSelector."""
    nyq = SRATE / 2
    sos = sig.butter(4, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
    pwr = sig.sosfiltfilt(sos, X, axis=-1).var(axis=-1)  # (n_trials, n_ch)
    Xc  = pwr - pwr.mean(0); Xstd = Xc.std(0) + 1e-12
    r2  = np.zeros(X.shape[1])
    for cls in np.unique(y):
        yb = (y == cls).astype(float); yb -= yb.mean()
        r  = (Xc * yb[:, None]).mean(0) / (Xstd * (yb.std() + 1e-12))
        r2 = np.maximum(r2, r ** 2)
    return r2


def hfb_power_per_trial(X):
    """Mean HFB (70-200 Hz) power per trial, averaged across channels."""
    nyq = SRATE / 2
    sos = sig.butter(4, [70 / nyq, 200 / nyq], btype='bandpass', output='sos')
    return sig.sosfiltfilt(sos, X, axis=-1).var(axis=-1).mean(axis=1)  # (n_trials,)


# ── load all subjects ─────────────────────────────────────────────────────────

mat_paths   = sorted(DATA_DIR.glob('*_mot_t_h.mat'))
subj_names  = [p.stem.split('_')[0] for p in mat_paths]
n_subj      = len(mat_paths)
print(f'Loading {n_subj} subjects...')

all_r2     = []   # list of (n_ch,) arrays — variable length per subject
all_nch    = []   # channel count per subject
quality    = []   # list of dicts for analysis 2
drift_data = []   # list of arrays for analysis 3

for si, (p, subj) in enumerate(zip(mat_paths, subj_names)):
    print(f'  {subj}', end=' ', flush=True)
    X, y, car, stim = load_subject(p)
    if X is None:
        print('skip')
        all_r2.append(None); all_nch.append(0)
        quality.append(None); drift_data.append(None)
        continue

    n_ch = X.shape[1]
    all_nch.append(n_ch)

    # analysis 1: R² per channel
    r2 = hfb_r2(X, y)
    all_r2.append(r2)

    # analysis 2: signal quality metrics
    # bad channels: RMS amplitude outliers (too high = saturated, too low = dead)
    # uses MAD so one extreme channel doesn't inflate the threshold
    ch_rms     = car.std(axis=0)                         # (n_ch,)
    med_rms    = np.median(ch_rms)
    mad_rms    = np.median(np.abs(ch_rms - med_rms))
    n_bad_ch   = int(((ch_rms > med_rms + 4 * mad_rms) |
                       (ch_rms < med_rms - 4 * mad_rms)).sum())

    # outlier trials: trial RMS more than 4 MADs above median for that subject
    trial_rms  = X.std(axis=(1, 2))
    med_tr     = np.median(trial_rms)
    mad_tr     = np.median(np.abs(trial_rms - med_tr))
    n_outlier  = int((trial_rms > med_tr + 4 * mad_tr).sum())

    hfb_pwr   = hfb_power_per_trial(X)
    hfb_ch    = X.var(axis=-1)                          # (n_trials, n_ch)
    corr_mat  = np.corrcoef(hfb_ch)                     # (n_trials, n_trials)
    mean_r    = corr_mat[np.triu_indices(len(corr_mat), k=1)].mean()

    quality.append({
        'subject':       subj,
        'n_trials':      len(y),
        'n_ch':          n_ch,
        'mean_rms':      trial_rms.mean(),
        'n_bad_ch':      n_bad_ch,
        'n_outlier_tr':  n_outlier,
        'inter_trial_r': mean_r,
    })

    drift_data.append(hfb_pwr)
    print(f'ok ({n_ch} ch)')

print()

# ── load accuracy from CSV ────────────────────────────────────────────────────

csv_path = DATA_DIR / 'classifier_results.csv'
acc = {}
if csv_path.exists():
    df_res = pd.read_csv(csv_path, index_col='subject')
    for name in ['LDA_shrink', 'TS_pb_LDA', 'TS_fb_LDA']:
        col = f'{name}_test_mean'
        if col in df_res.columns:
            acc[name] = df_res[col].to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Responsive electrode map — all subjects on one figure
# ─────────────────────────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(n_subj / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8))
axes = axes.flatten()

for si, (subj, r2, nc) in enumerate(zip(subj_names, all_r2, all_nch)):
    ax = axes[si]
    if r2 is None:
        ax.set_visible(False); continue
    k       = min(K_CHAN, nc)
    top_k   = np.argsort(r2)[::-1][:k]
    top_set = set(top_k)
    colors  = ['#e15759' if i in top_set else 'steelblue' for i in range(nc)]
    ax.bar(np.arange(nc), r2, color=colors, width=1.0, linewidth=0)
    ax.axhline(r2[top_k[-1]], color='red', ls='--', lw=0.8)
    ax.set_title(f'{subj}  ({nc} ch, top-{k} red)', fontsize=8, fontweight='bold')
    ax.set_xlabel('Channel', fontsize=7); ax.set_ylabel('OVR R²', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(-1, nc)

for ax in axes[n_subj:]:
    ax.set_visible(False)

fig.suptitle('HFB OVR R² per channel — responsive electrode map\n'
             '(red = top-15 selected; dashed = selection threshold)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'electrode_maps.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved electrode_maps.png')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-subject signal quality summary + accuracy correlation
# ─────────────────────────────────────────────────────────────────────────────
q_valid  = [q for q in quality if q is not None]
subjs_q  = [q['subject'] for q in q_valid]
x_idx    = np.arange(len(subjs_q))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 2a: inter-trial HFB correlation per subject (higher = more consistent)
irs = [q['inter_trial_r'] for q in q_valid]
bar_colors = ['#e15759' if r < np.percentile(irs, 25) else
              '#76b7b2' if r > np.percentile(irs, 75) else '#4e79a7'
              for r in irs]
axes[0, 0].bar(x_idx, irs, color=bar_colors)
axes[0, 0].axhline(np.mean(irs), color='black', ls='--', lw=1.2,
                   label=f'Mean = {np.mean(irs):.3f}')
axes[0, 0].set_xticks(x_idx); axes[0, 0].set_xticklabels(subjs_q, rotation=45, ha='right', fontsize=8)
axes[0, 0].set_title('Inter-trial HFB consistency\n(higher = more reproducible neural response)')
axes[0, 0].set_ylabel('Mean pairwise trial correlation'); axes[0, 0].legend(fontsize=8)

# 2b: bad channel count per subject
bad_chs = [q['n_bad_ch'] for q in q_valid]
axes[0, 1].bar(x_idx, bad_chs, color=['#e15759' if b > 0 else 'steelblue' for b in bad_chs])
axes[0, 1].set_xticks(x_idx); axes[0, 1].set_xticklabels(subjs_q, rotation=45, ha='right', fontsize=8)
axes[0, 1].set_title('Amplitude-outlier channels per subject\n(channel RMS > median ± 4×MAD — dead or saturated)')
axes[0, 1].set_ylabel('N outlier channels')

# 2c: outlier trial count per subject
out_tr = [q['n_outlier_tr'] for q in q_valid]
axes[1, 0].bar(x_idx, out_tr, color=['#e15759' if o > 0 else 'steelblue' for o in out_tr])
axes[1, 0].set_xticks(x_idx); axes[1, 0].set_xticklabels(subjs_q, rotation=45, ha='right', fontsize=8)
axes[1, 0].set_title('High-artifact trials per subject\n(trial RMS > median + 4×MAD — movement/noise artifact)')
axes[1, 0].set_ylabel('N outlier trials')

# 2d: inter-trial consistency vs accuracy (scatter)
ax = axes[1, 1]
clf_colors = {'LDA_shrink': 'darkorange', 'TS_pb_LDA': 'seagreen', 'TS_fb_LDA': 'steelblue'}
for clf_name, acc_dict in acc.items():
    irs_matched, accs_matched, subjs_matched = [], [], []
    for q in q_valid:
        s = q['subject']
        if s in acc_dict:
            irs_matched.append(q['inter_trial_r'])
            accs_matched.append(acc_dict[s])
            subjs_matched.append(s)
    if not irs_matched: continue
    r, p = pearsonr(irs_matched, accs_matched)
    ax.scatter(irs_matched, accs_matched, label=f'{clf_name} (r={r:.2f}, p={p:.3f})',
               color=clf_colors.get(clf_name, 'gray'), alpha=0.8, s=50)
    # regression line
    z = np.polyfit(irs_matched, accs_matched, 1)
    xl = np.linspace(min(irs_matched), max(irs_matched), 50)
    ax.plot(xl, np.polyval(z, xl), color=clf_colors.get(clf_name, 'gray'), lw=1.5, ls='--')

ax.set_xlabel('Inter-trial HFB consistency'); ax.set_ylabel('Test accuracy')
ax.set_title('Signal quality vs decoding accuracy\n(correlation between neural consistency and CV accuracy)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle('Per-subject signal quality summary', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'subject_quality.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved subject_quality.png')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Trial stationarity check
# ─────────────────────────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(n_subj / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8))
axes = axes.flatten()

stationarity_results = []

for si, (subj, hfb_pwr) in enumerate(zip(subj_names, drift_data)):
    ax = axes[si]
    if hfb_pwr is None:
        ax.set_visible(False); continue

    n_tr = len(hfb_pwr)
    x    = np.arange(n_tr)

    # normalise to z-score so subjects are comparable
    z = (hfb_pwr - hfb_pwr.mean()) / (hfb_pwr.std() + 1e-12)

    # rolling mean (window = 20% of trials)
    win = max(3, n_tr // 5)
    roll = np.convolve(z, np.ones(win) / win, mode='valid')
    x_roll = np.arange(win // 2, win // 2 + len(roll))

    ax.scatter(x, z, s=12, alpha=0.5, color='steelblue')
    ax.plot(x_roll, roll, color='navy', lw=1.5, label='Running mean')
    ax.axhline(0, color='black', lw=0.6, ls=':')

    # first vs last half Mann-Whitney U
    mid = n_tr // 2
    stat, p = mannwhitneyu(z[:mid], z[mid:], alternative='two-sided')
    drift_detected = p < 0.05
    col = '#e15759' if drift_detected else '#76b7b2'
    ax.set_facecolor('#fff5f5' if drift_detected else 'white')
    ax.set_title(f'{subj}  p={p:.3f}{"  ⚠ drift" if drift_detected else "  ✓ stable"}',
                 fontsize=8, fontweight='bold', color=col if drift_detected else 'black')
    ax.set_xlabel('Trial index', fontsize=7); ax.set_ylabel('HFB power (z)', fontsize=7)
    ax.tick_params(labelsize=6)
    stationarity_results.append({'subject': subj, 'drift_p': p, 'drift': drift_detected})

for ax in axes[n_subj:]:
    ax.set_visible(False)

n_drift = sum(r['drift'] for r in stationarity_results)
fig.suptitle(f'Trial stationarity — HFB power across session\n'
             f'(Mann-Whitney U first vs last half; {n_drift}/{len(stationarity_results)} subjects show drift p<0.05)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'stationarity.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved stationarity.png')

# ── print text summary ────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('SIGNAL QUALITY SUMMARY')
print(f'{"="*60}')
print(f'{"Subject":<8} {"Trials":>7} {"BadCh":>6} {"OutTr":>6} {"ITCorr":>8} {"Drift":>6}')
print('-' * 60)
drift_dict = {r['subject']: r['drift'] for r in stationarity_results}
for q in q_valid:
    s     = q['subject']
    drift = '⚠' if drift_dict.get(s, False) else '✓'
    print(f'{s:<8} {q["n_trials"]:>7} {q["n_bad_ch"]:>6} '
          f'{q["n_outlier_tr"]:>6} {q["inter_trial_r"]:>8.3f} {drift:>6}')
print(f'{"="*60}')
print(f'\nAll figures saved to {OUT}/')
