"""
noise_analysis.py — ECoG data quality and noise characterisation.

Generates figures in figs/noise/:
  1. raw_traces.png       — raw multi-channel traces + one zoomed trial
  2. psd.png              — power spectral density (all channels + median)
  3. channel_stats.png    — per-channel RMS, kurtosis, line-noise ratio
  4. event_locked.png     — HFB and beta power time-locked to trial onset
  5. trial_quality.png    — per-trial RMS outlier detection
  6. car_effect.png       — before vs after CAR on PSD
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kurtosis

# ── config ────────────────────────────────────────────────────────────────────
from motor_clf.config import DATA_DIR, SRATE, WIN_START, WIN_LEN, VALID_CODES, LABEL_MAP, CLASS_NAMES

OUT = Path('figs/noise')
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {'tongue': '#e15759', 'hand': '#4e79a7', 'rest': '#76b7b2'}

# ── load one representative subject ──────────────────────────────────────────
mat_paths = sorted(DATA_DIR.glob('*_mot_t_h.mat'))
p = mat_paths[0]
subj = p.stem.split('_')[0]
print(f'Analysing subject: {subj}')

d    = sio.loadmat(str(p))
raw  = d['data'].astype(float)          # (T, C) — pre-CAR
stim = d['stim'].flatten().astype(int)
T, C = raw.shape

car  = raw - raw.mean(axis=1, keepdims=True)   # common average reference

# trial onsets
diff       = np.concatenate([[0], np.diff(stim)])
changes    = np.where(np.concatenate([[True], stim[1:] != stim[:-1]]))[0]
codes      = np.zeros(len(changes), dtype=int)
codes[1:]  = stim[changes[1:]]
trials     = [(onset, LABEL_MAP[c]) for onset, c in zip(changes, codes)
              if c in VALID_CODES]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw traces
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# 1a: 30-s overview, 6 channels spread by RMS
rms      = raw.std(axis=0)
ch_pick  = np.round(np.linspace(0, C - 1, 6)).astype(int)
t_ov     = np.arange(30 * SRATE) / SRATE
scale    = 3000
for i, ch in enumerate(ch_pick):
    axes[0].plot(t_ov, raw[:30 * SRATE, ch] + i * scale, lw=0.4, color='steelblue')
axes[0].set_title(f'{subj} — raw ECoG overview (6 channels, 30 s)', fontsize=10)
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude (offset per ch)')
axes[0].set_yticks([i * scale for i in range(6)])
axes[0].set_yticklabels([f'ch {ch}' for ch in ch_pick], fontsize=7)

# 1b: single trial, all channels
onset, label = trials[0]
s, e = onset + WIN_START, onset + WIN_START + WIN_LEN
t_tr = np.arange(WIN_LEN) / SRATE * 1000  # ms
scale2 = 2000
for i in range(C):
    axes[1].plot(t_tr, car[s:e, i] + i * scale2, lw=0.3, alpha=0.7, color='steelblue')
axes[1].set_title(f'Single trial ({label}, post-CAR) — all {C} channels', fontsize=10)
axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Channels (offset)')
axes[1].axvline(0, color='red', lw=1, ls='--', label='win start')

# 1c: same trial, 4 channels, raw vs CAR overlay
for i, ch in enumerate(ch_pick[:4]):
    offset = i * 8000
    axes[2].plot(t_tr, raw[s:e, ch] + offset, lw=0.6, color='gray',  alpha=0.6, label='raw' if i == 0 else '')
    axes[2].plot(t_tr, car[s:e, ch] + offset, lw=0.6, color='steelblue', label='CAR' if i == 0 else '')
axes[2].set_title('Raw vs CAR — 4 channels from same trial', fontsize=10)
axes[2].set_xlabel('Time (ms)'); axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'raw_traces.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved raw_traces.png')

# ─────────────────────────────────────────────────────────────────────────────
# 2. PSD — all channels + median, pre and post CAR
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, signal, label_str in [(axes[0], raw, 'Raw (pre-CAR)'),
                               (axes[1], car, 'Post-CAR')]:
    psds = []
    for ch in range(C):
        f, p_ch = sig.welch(signal[:, ch], fs=SRATE, nperseg=SRATE * 2)
        psds.append(p_ch)
    psds = np.array(psds)
    for ch in range(C):
        ax.semilogy(f, psds[ch], lw=0.4, alpha=0.3, color='steelblue')
    ax.semilogy(f, np.median(psds, axis=0), lw=1.8, color='navy', label='Median')
    ax.axvspan(8,  32,  alpha=0.08, color='orange', label='Beta (8–32 Hz)')
    ax.axvspan(70, 200, alpha=0.08, color='green',  label='HFB (70–200 Hz)')
    ax.axvline(60, color='red', lw=1, ls='--', label='60 Hz')
    ax.axvline(120, color='red', lw=0.7, ls=':')
    ax.set_xlim(1, 250); ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)'); ax.set_title(label_str)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle(f'{subj} — Power Spectral Density', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'psd.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved psd.png')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Channel statistics — RMS, kurtosis, 60 Hz line noise ratio
# ─────────────────────────────────────────────────────────────────────────────
rms_ch  = car.std(axis=0)
kurt_ch = np.array([kurtosis(car[:, ch]) for ch in range(C)])

# line noise ratio: power at 60 Hz / broadband baseline
line_ratios = []
for ch in range(C):
    f, p_ch = sig.welch(car[:, ch], fs=SRATE, nperseg=SRATE * 2)
    p60   = p_ch[(f >= 58)  & (f <= 62)].mean()
    pbase = p_ch[(f >= 30)  & (f <= 55)].mean()
    line_ratios.append(p60 / (pbase + 1e-12))
line_ratios = np.array(line_ratios)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ch_idx = np.arange(C)

# RMS
axes[0].bar(ch_idx, rms_ch, color=['#e15759' if r > np.percentile(rms_ch, 90) else 'steelblue'
                                    for r in rms_ch])
axes[0].axhline(np.percentile(rms_ch, 90), color='red', ls='--', lw=1, label='90th pct')
axes[0].set_title('Channel RMS amplitude (post-CAR)'); axes[0].set_xlabel('Channel')
axes[0].set_ylabel('RMS (a.u.)'); axes[0].legend(fontsize=8)

# Kurtosis
axes[1].bar(ch_idx, kurt_ch, color=['#e15759' if k > 5 else 'steelblue' for k in kurt_ch])
axes[1].axhline(5, color='red', ls='--', lw=1, label='Kurtosis > 5 (spiky)')
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_title('Kurtosis (epileptic spikes → high kurtosis)'); axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Excess kurtosis'); axes[1].legend(fontsize=8)

# Line noise
axes[2].bar(ch_idx, line_ratios, color=['#e15759' if r > 3 else 'steelblue' for r in line_ratios])
axes[2].axhline(3, color='red', ls='--', lw=1, label='3× baseline = noisy')
axes[2].set_title('60 Hz line noise ratio (power at 60 Hz / baseline)'); axes[2].set_xlabel('Channel')
axes[2].set_ylabel('Ratio'); axes[2].legend(fontsize=8)

noisy_rms  = np.where(rms_ch  > np.percentile(rms_ch, 90))[0]
noisy_kurt = np.where(kurt_ch > 5)[0]
noisy_line = np.where(line_ratios > 3)[0]
print(f'High RMS channels:       {noisy_rms}')
print(f'High kurtosis channels:  {noisy_kurt}')
print(f'High line noise channels:{noisy_line}')

fig.suptitle(f'{subj} — Per-channel noise profile', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'channel_stats.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved channel_stats.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Event-locked HFB and beta power
# ─────────────────────────────────────────────────────────────────────────────
def bandpower_trace(signal, lo, hi, fs=SRATE):
    """Bandpass → square → smooth → return (T,) power trace."""
    sos = sig.butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    bp  = sig.sosfiltfilt(sos, signal, axis=0)
    pwr = bp ** 2
    # smooth with 200ms Gaussian
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(pwr, sigma=0.1 * fs, axis=0)

WIN_PRE = 500   # ms pre-onset baseline
EPOCH   = WIN_PRE + WIN_LEN
t_axis  = np.arange(-WIN_PRE, WIN_LEN) / SRATE * 1000   # ms

hfb_pwr = bandpower_trace(car, 70, 200)
bet_pwr = bandpower_trace(car, 8,  32)

epochs_hfb = {c: [] for c in CLASS_NAMES}
epochs_bet = {c: [] for c in CLASS_NAMES}
for onset, label in trials:
    s = onset - WIN_PRE
    e = onset + WIN_LEN
    if s < 0 or e > T: continue
    epochs_hfb[label].append(hfb_pwr[s:e, :].mean(axis=1))
    epochs_bet[label].append(bet_pwr[s:e, :].mean(axis=1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for band_name, epochs_dict, ax in [('HFB (70–200 Hz)', epochs_hfb, axes[0]),
                                    ('Beta (8–32 Hz)',  epochs_bet, axes[1])]:
    for label in CLASS_NAMES:
        ep = np.array(epochs_dict[label])
        if len(ep) == 0: continue
        mn  = ep.mean(axis=0)
        sem = ep.std(axis=0) / np.sqrt(len(ep))
        # normalise to pre-stimulus baseline
        base = mn[:WIN_PRE].mean()
        mn   = mn / (base + 1e-12)
        sem  = sem / (base + 1e-12)
        ax.plot(t_axis, mn, lw=1.5, color=COLORS[label], label=f'{label} (n={len(ep)})')
        ax.fill_between(t_axis, mn - sem, mn + sem, color=COLORS[label], alpha=0.2)
    ax.axvline(0,   color='black', lw=1.2, ls='--', label='Trial onset')
    ax.axvspan(0, WIN_START, color='gray', alpha=0.08, label=f'Pre-window (0–{WIN_START}ms)')
    ax.axhline(1,   color='black', lw=0.6, ls=':')
    ax.set_title(f'Event-locked {band_name}\n(mean across channels, normalised to baseline)')
    ax.set_xlabel('Time relative to onset (ms)'); ax.set_ylabel('Relative power')
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle(f'{subj} — Event-locked band power', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'event_locked.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved event_locked.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Trial-level quality — detect outlier trials by RMS
# ─────────────────────────────────────────────────────────────────────────────
trial_rms = []
trial_labels = []
for onset, label in trials:
    s = onset + WIN_START
    e = s + WIN_LEN
    if e > T: continue
    trial_rms.append(car[s:e, :].std())
    trial_labels.append(label)

trial_rms = np.array(trial_rms)
thresh    = np.percentile(trial_rms, 95)
n_outlier = (trial_rms > thresh).sum()

fig, ax = plt.subplots(figsize=(12, 4))
colors_t = [COLORS[l] for l in trial_labels]
ax.bar(np.arange(len(trial_rms)), trial_rms, color=colors_t, alpha=0.8)
ax.axhline(thresh, color='red', ls='--', lw=1.5, label=f'95th pct threshold ({thresh:.0f})')
ax.set_title(f'{subj} — Per-trial RMS (red=tongue, blue=hand, teal=rest)\n'
             f'{n_outlier} potential outlier trials (>{thresh:.0f})')
ax.set_xlabel('Trial index'); ax.set_ylabel('RMS amplitude (post-CAR)')
ax.legend(fontsize=9)

# custom legend for classes
from matplotlib.patches import Patch
handles = [Patch(color=COLORS[c], label=c) for c in CLASS_NAMES]
handles.append(plt.Line2D([0], [0], color='red', ls='--', label='95th pct'))
ax.legend(handles=handles, fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'trial_quality.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved trial_quality.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Inter-channel correlation (spatial noise structure)
# ─────────────────────────────────────────────────────────────────────────────
corr = np.corrcoef(car.T)   # (C, C)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im0 = axes[0].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
axes[0].set_title('Inter-channel correlation (post-CAR)\n(high off-diagonal = shared noise)')
axes[0].set_xlabel('Channel'); axes[0].set_ylabel('Channel')
plt.colorbar(im0, ax=axes[0])

# distribution of off-diagonal correlations
off_diag = corr[np.triu_indices(C, k=1)]
axes[1].hist(off_diag, bins=40, color='steelblue', edgecolor='white', lw=0.4)
axes[1].axvline(off_diag.mean(), color='red', lw=1.5,
                label=f'Mean r = {off_diag.mean():.3f}')
axes[1].axvline(0, color='black', lw=0.8, ls='--')
axes[1].set_title('Distribution of pairwise channel correlations')
axes[1].set_xlabel('Pearson r'); axes[1].set_ylabel('Count')
axes[1].legend(fontsize=9)

fig.suptitle(f'{subj} — Spatial noise structure', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'channel_correlation.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved channel_correlation.png')

print(f'\nAll noise analysis figures saved to {OUT}/')
print(f'\nSummary for {subj}:')
print(f'  Channels with high RMS (>90th pct): {noisy_rms}')
print(f'  Channels with high kurtosis (>5):   {noisy_kurt}')
print(f'  Channels with 60Hz line noise (>3x):{noisy_line}')
print(f'  Outlier trials (>95th pct RMS):     {n_outlier}/{len(trial_rms)}')
print(f'  Mean inter-channel correlation:      {off_diag.mean():.3f}')
