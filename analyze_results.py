"""
analyze_results.py

Load per-subject .npz results and produce:
  1. PSD plots     — move vs rest spectra for a chosen subject/electrode
  2. Hand vs tongue selectivity scatter — per subject or pooled
  3. Group-level summary — r² values across all subjects plotted in MNI space

Run:
    python analyze_results.py
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / 'data'
LOCS_DIR    = SCRIPT_DIR / 'locs'
OUT_DIR     = SCRIPT_DIR / 'analysis'
OUT_DIR.mkdir(exist_ok=True)

SUBJECTS = [
    'bp', 'ca', 'cc', 'de', 'fp', 'gc', 'gf',
    'hh', 'hl', 'jc', 'jf', 'jm', 'jp', 'jt',
    'rh', 'rr', 'ug', 'wc', 'zt',
]
FREQ = np.arange(200)   # 0–199 Hz axis


# ---------------------------------------------------------------------------
# 1. PSD plots: move vs rest for a given subject
# ---------------------------------------------------------------------------

def plot_psd(subj, elec_idx=None, p_thresh=0.05):
    """
    Plot mean PSD (hand-move / hand-rest / tongue-move / tongue-rest) for
    `subj`. If elec_idx is None, uses the electrode with the highest
    significant hand-HFB r².  Saves to analysis/{subj}_PSD.png.
    """
    d = np.load(DATA_DIR / f'{subj}_mot_th_analyzed.npz', allow_pickle=True)

    if elec_idx is None:
        sig = (d['p_hand_HFB'] < p_thresh) * d['r_hand_HFB']
        elec_idx = int(np.argmax(sig))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    bands = {
        'Hand':   ('mean_PSD_handmove',   'mean_PSD_handrest',   'steelblue', 'lightblue'),
        'Tongue': ('mean_PSD_tonguemove', 'mean_PSD_tonguerest', 'firebrick', 'lightsalmon'),
    }

    for ax, (label, (move_key, rest_key, c_move, c_rest)) in zip(axes, bands.items()):
        move = d[move_key]
        rest = d[rest_key]
        if move.size == 0 or rest.size == 0:
            ax.set_title(f'{label}: no data')
            continue
        ax.semilogy(FREQ, move[:, elec_idx], color=c_move, label=f'{label} move')
        ax.semilogy(FREQ, rest[:, elec_idx], color=c_rest, label=f'{label} rest', linestyle='--')
        ax.axvspan(8,  32,  alpha=0.08, color='green',  label='LFB (8–32 Hz)')
        ax.axvspan(75, 100, alpha=0.08, color='orange', label='HFB (75–99 Hz)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'{subj} — {label} — electrode {elec_idx}')
        ax.legend(fontsize=7)

    fig.tight_layout()
    out = OUT_DIR / f'{subj}_PSD_elec{elec_idx}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ---------------------------------------------------------------------------
# 2. Hand vs tongue selectivity scatter (one panel per subject)
# ---------------------------------------------------------------------------

def plot_selectivity(subj, p_thresh=0.05):
    """
    Scatter r_hand_HFB vs r_tongue_HFB for every electrode.
    Points significant for hand = blue, tongue = red, both = purple.
    Saves to analysis/{subj}_selectivity.png.
    """
    d = np.load(DATA_DIR / f'{subj}_mot_th_analyzed.npz', allow_pickle=True)
    rh = d['r_hand_HFB']
    rt = d['r_tongue_HFB']
    ph = d['p_hand_HFB']
    pt = d['p_tongue_HFB']

    sig_h = ph < p_thresh
    sig_t = pt < p_thresh

    colors = np.where(
        sig_h & sig_t,  'purple',
        np.where(sig_h, 'steelblue',
        np.where(sig_t, 'firebrick', 'lightgray'))
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rh, rt, c=colors, s=30, edgecolors='none', alpha=0.8)
    lim = max(np.abs(np.concatenate([rh, rt])).max(), 0.1)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Hand HFB signed r²')
    ax.set_ylabel('Tongue HFB signed r²')
    ax.set_title(f'{subj} — hand vs tongue selectivity (HFB)')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue', label='Hand only'),
        Patch(color='firebrick', label='Tongue only'),
        Patch(color='purple',    label='Both'),
        Patch(color='lightgray', label='n.s.'),
    ], fontsize=7, loc='upper left')

    fig.tight_layout()
    out = OUT_DIR / f'{subj}_selectivity.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ---------------------------------------------------------------------------
# 3. Group-level r² summary in MNI space
# ---------------------------------------------------------------------------

def plot_group_mni(band='HFB', movement='hand', p_thresh=0.05, min_subjects=1):
    """
    Pool all electrodes across subjects. Plot in MNI space (axial projection)
    colored by signed r², only showing electrodes significant in >= min_subjects.
    Saves to analysis/group_{movement}_{band}_MNI.png.
    """
    all_xyz, all_r = [], []

    for subj in SUBJECTS:
        npz_path  = DATA_DIR / f'{subj}_mot_th_analyzed.npz'
        locs_path = LOCS_DIR / f'{subj}_electrodes.mat'
        if not npz_path.exists() or not locs_path.exists():
            continue

        d    = np.load(npz_path, allow_pickle=True)
        elec = sio.loadmat(str(locs_path))['electrodes']  # (n, 3) MNI

        r_key = f'r_{movement}_{band}'
        p_key = f'p_{movement}_{band}'
        r = d[r_key]
        p = d[p_key]

        sig = p < p_thresh
        all_xyz.append(elec[sig])
        all_r.append(r[sig])

    if not all_xyz:
        print('No significant electrodes found.')
        return

    xyz = np.vstack(all_xyz)
    r   = np.concatenate(all_r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    views = [('Axial (x–y)',   0, 1, 'x (mm)', 'y (mm)'),
             ('Coronal (x–z)', 0, 2, 'x (mm)', 'z (mm)')]

    vmax = np.abs(r).max() if r.size else 1.0

    for ax, (title, xi, yi, xl, yl) in zip(axes, views):
        sc = ax.scatter(xyz[:, xi], xyz[:, yi], c=r, cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, s=20, edgecolors='k', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label='signed r²')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'Group — {movement} {band} — {title}  (n={len(r)})')
        ax.set_aspect('equal')

    fig.tight_layout()
    out = OUT_DIR / f'group_{movement}_{band}_MNI.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ---------------------------------------------------------------------------
# 4. HFB vs LFB spatial specificity: hand vs tongue overlay in MNI space
# ---------------------------------------------------------------------------

def plot_spatial_specificity(p_thresh=0.05):
    """
    2-column figure: left = HFB, right = LFB.
    Each column shows hand (blue) and tongue (red) significant electrodes
    overlaid in MNI space (axial + coronal), sized by |r²|.
    A spatial overlap score (Jaccard on a voxel grid) is printed per band.
    Saves to analysis/spatial_specificity_HFB_vs_LFB.png.
    """
    data = {}   # band -> {'hand': (xyz, r), 'tongue': (xyz, r)}

    for band in ('HFB', 'LFB'):
        hand_xyz, hand_r, tong_xyz, tong_r = [], [], [], []
        for subj in SUBJECTS:
            npz_path  = DATA_DIR / f'{subj}_mot_th_analyzed.npz'
            locs_path = LOCS_DIR / f'{subj}_electrodes.mat'
            if not npz_path.exists() or not locs_path.exists():
                continue
            d    = np.load(npz_path, allow_pickle=True)
            elec = sio.loadmat(str(locs_path))['electrodes']

            sig_h = d[f'p_hand_{band}']   < p_thresh
            sig_t = d[f'p_tongue_{band}'] < p_thresh
            hand_xyz.append(elec[sig_h]);  hand_r.append(d[f'r_hand_{band}'][sig_h])
            tong_xyz.append(elec[sig_t]);  tong_r.append(d[f'r_tongue_{band}'][sig_t])

        data[band] = {
            'hand':   (np.vstack(hand_xyz) if hand_xyz else np.zeros((0, 3)),
                       np.concatenate(hand_r) if hand_r else np.array([])),
            'tongue': (np.vstack(tong_xyz) if tong_xyz else np.zeros((0, 3)),
                       np.concatenate(tong_r) if tong_r else np.array([])),
        }

    # spatial overlap: discretise MNI into 10 mm voxels, compute Jaccard
    def jaccard(xyz_h, xyz_t, vox=10):
        if xyz_h.shape[0] == 0 or xyz_t.shape[0] == 0:
            return float('nan')
        def voxels(xyz):
            return set(map(tuple, (xyz / vox).astype(int)))
        vh, vt = voxels(xyz_h), voxels(xyz_t)
        return len(vh & vt) / len(vh | vt)

    views = [('Axial (x–y)',   0, 1, 'x (mm)', 'y (mm)'),
             ('Coronal (x–z)', 0, 2, 'x (mm)', 'z (mm)')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Spatial specificity: HFB vs LFB for hand vs tongue\n'
        '(blue = hand, red = tongue; size ∝ |r²|; only p < 0.05)',
        fontsize=12, fontweight='bold'
    )

    for col, band in enumerate(('HFB', 'LFB')):
        h_xyz, h_r = data[band]['hand']
        t_xyz, t_r = data[band]['tongue']
        j = jaccard(h_xyz, t_xyz)

        for row, (view_title, xi, yi, xl, yl) in enumerate(views):
            ax = axes[row, col]
            if h_xyz.shape[0]:
                ax.scatter(h_xyz[:, xi], h_xyz[:, yi],
                           s=np.abs(h_r) * 120 + 8,
                           c='steelblue', alpha=0.55, edgecolors='none', label='Hand')
            if t_xyz.shape[0]:
                ax.scatter(t_xyz[:, xi], t_xyz[:, yi],
                           s=np.abs(t_r) * 120 + 8,
                           c='firebrick', alpha=0.55, edgecolors='none', label='Tongue')
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            title = f'{band} — {view_title}'
            if row == 0:
                title += f'\nSpatial overlap (Jaccard) = {j:.2f}  (lower = more specific)'
            ax.set_title(title, fontsize=9)
            ax.set_aspect('equal')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, markerscale=0.6)

        print(f'  {band}: hand n={h_xyz.shape[0]}, tongue n={t_xyz.shape[0]}, '
              f'Jaccard overlap = {j:.3f}')

    fig.tight_layout()
    out = OUT_DIR / 'spatial_specificity_HFB_vs_LFB.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ---------------------------------------------------------------------------
# 5. Overall summary plots (across all subjects)
# ---------------------------------------------------------------------------

def plot_summary(p_thresh=0.05):
    """
    Six-panel overall summary figure (GridSpec 4 rows × 2 cols):
      A. Significant electrode counts per subject       [row 0, full width]
      B-HFB. HFB r² distributions hand vs tongue       [row 1, left]
      B-LFB. LFB r² distributions hand vs tongue       [row 1, right]
      C-HFB. HFB selectivity breakdown                 [row 2, left]
      C-LFB. LFB selectivity breakdown                 [row 2, right]
      D. Group-average dB power change (hand)          [row 3, full width]
    Saves to analysis/summary.png.
    """
    from matplotlib.gridspec import GridSpec

    sig_counts   = {k: [] for k in ('hand_HFB', 'tongue_HFB', 'hand_LFB', 'tongue_LFB')}
    all_r        = {k: [] for k in ('hand_HFB', 'tongue_HFB', 'hand_LFB', 'tongue_LFB')}
    sel_hfb      = {'hand_only': 0, 'tongue_only': 0, 'both': 0, 'neither': 0}
    sel_lfb      = {'hand_only': 0, 'tongue_only': 0, 'both': 0, 'neither': 0}
    best_db      = []
    valid_subjs  = []

    for subj in SUBJECTS:
        npz_path = DATA_DIR / f'{subj}_mot_th_analyzed.npz'
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        valid_subjs.append(subj)

        for cond in ('hand_HFB', 'tongue_HFB', 'hand_LFB', 'tongue_LFB'):
            r = d[f'r_{cond}']
            p = d[f'p_{cond}']
            sig_counts[cond].append((p < p_thresh).sum())
            all_r[cond].extend(r.tolist())

        # HFB selectivity
        sig_h = d['p_hand_HFB']   < p_thresh
        sig_t = d['p_tongue_HFB'] < p_thresh
        sel_hfb['hand_only']   += int(( sig_h & ~sig_t).sum())
        sel_hfb['tongue_only'] += int((~sig_h &  sig_t).sum())
        sel_hfb['both']        += int(( sig_h &  sig_t).sum())
        sel_hfb['neither']     += int((~sig_h & ~sig_t).sum())

        # LFB selectivity
        sig_h_l = d['p_hand_LFB']   < p_thresh
        sig_t_l = d['p_tongue_LFB'] < p_thresh
        sel_lfb['hand_only']   += int(( sig_h_l & ~sig_t_l).sum())
        sel_lfb['tongue_only'] += int((~sig_h_l &  sig_t_l).sum())
        sel_lfb['both']        += int(( sig_h_l &  sig_t_l).sum())
        sel_lfb['neither']     += int((~sig_h_l & ~sig_t_l).sum())

        # best hand-HFB electrode: dB change = 10*log10(move/rest)
        best = int(np.argmax(d['r_hand_HFB'] * (d['p_hand_HFB'] < p_thresh)))
        move_psd = d['mean_PSD_handmove']
        rest_psd = d['mean_PSD_handrest']
        if move_psd.size and rest_psd.size:
            safe_rest = np.where(rest_psd[:, best] == 0, 1e-12, rest_psd[:, best])
            best_db.append(10 * np.log10(move_psd[:, best] / safe_rest))

    # ---- layout ----
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Overall summary across all subjects', fontsize=13, fontweight='bold')
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.2, 1, 1, 1], hspace=0.45, wspace=0.35)

    # A: significant electrode counts (full width)
    ax = fig.add_subplot(gs[0, :])
    x  = np.arange(len(valid_subjs))
    w  = 0.2
    colors_bar = {'hand_HFB': 'steelblue', 'tongue_HFB': 'firebrick',
                  'hand_LFB': 'cornflowerblue', 'tongue_LFB': 'salmon'}
    for i, (cond, color) in enumerate(colors_bar.items()):
        ax.bar(x + i * w, sig_counts[cond], width=w, color=color, label=cond.replace('_', ' '))
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(valid_subjs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('# significant electrodes (p < 0.05)')
    ax.set_title('A. Significant electrodes per subject')
    ax.legend(fontsize=7, ncol=4)

    # B-HFB: r² distributions
    ax = fig.add_subplot(gs[1, 0])
    for cond, color in [('hand_HFB', 'steelblue'), ('tongue_HFB', 'firebrick')]:
        ax.hist(np.array(all_r[cond]), bins=40, alpha=0.6, color=color,
                label=cond.replace('_', ' '), density=True)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Signed r²')
    ax.set_ylabel('Density')
    ax.set_title('B. HFB r² distribution\n(expect positive — broadband increase)')
    ax.legend(fontsize=7)

    # B-LFB: r² distributions
    ax = fig.add_subplot(gs[1, 1])
    for cond, color in [('hand_LFB', 'cornflowerblue'), ('tongue_LFB', 'salmon')]:
        ax.hist(np.array(all_r[cond]), bins=40, alpha=0.6, color=color,
                label=cond.replace('_', ' '), density=True)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Signed r²')
    ax.set_ylabel('Density')
    ax.set_title('B. LFB r² distribution\n(expect negative — ERD/suppression)')
    ax.legend(fontsize=7)

    # C-HFB: selectivity breakdown
    def _sel_bars(ax, sel, title):
        labels = ['Hand only', 'Tongue only', 'Both', 'Neither']
        vals   = [sel['hand_only'], sel['tongue_only'], sel['both'], sel['neither']]
        clrs   = ['steelblue', 'firebrick', 'purple', 'lightgray']
        bars   = ax.bar(labels, vals, color=clrs, edgecolor='k', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('# electrodes (all subjects)')
        ax.set_title(title)

    _sel_bars(fig.add_subplot(gs[2, 0]), sel_hfb, 'C. HFB selectivity breakdown (pooled)')
    _sel_bars(fig.add_subplot(gs[2, 1]), sel_lfb,  'C. LFB selectivity breakdown (pooled)')

    # D: dB change (full width)
    ax = fig.add_subplot(gs[3, :])
    if best_db:
        db_mean = np.mean(best_db, axis=0)
        db_sem  = np.std(best_db,  axis=0) / np.sqrt(len(best_db))
        ax.plot(FREQ, db_mean, color='steelblue', lw=1.5, label='Hand move vs rest')
        ax.fill_between(FREQ, db_mean - db_sem, db_mean + db_sem, alpha=0.25, color='steelblue')
        ax.axhline(0, color='k', lw=0.8, linestyle='--')
        ax.axvspan(8,  32,  alpha=0.10, color='green',  label='LFB 8–32 Hz (ERD expected ↓)')
        ax.axvspan(75, 100, alpha=0.10, color='orange', label='HFB 75–99 Hz (increase expected ↑)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power change (dB)')
    ax.set_title(f'D. Group-avg dB change at best hand-HFB electrode (n={len(best_db)}, +dB = more power during move)')
    ax.legend(fontsize=7)

    out = OUT_DIR / 'summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out.name}')


# ---------------------------------------------------------------------------
# Run all analyses
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== PSD plots (best electrode per subject) ===')
    for subj in SUBJECTS:
        try:
            plot_psd(subj)
        except Exception as e:
            print(f'  {subj}: {e}')

    print('\n=== Hand vs tongue selectivity ===')
    for subj in SUBJECTS:
        try:
            plot_selectivity(subj)
        except Exception as e:
            print(f'  {subj}: {e}')

    print('\n=== Group MNI maps ===')
    for movement in ('hand', 'tongue'):
        for band in ('HFB', 'LFB'):
            try:
                plot_group_mni(band=band, movement=movement)
            except Exception as e:
                print(f'  {movement} {band}: {e}')

    print('\n=== Spatial specificity (HFB vs LFB) ===')
    try:
        plot_spatial_specificity()
    except Exception as e:
        print(f'  spatial_specificity: {e}')

    print('\n=== Overall summary ===')
    try:
        plot_summary()
    except Exception as e:
        print(f'  summary: {e}')

    print('\nDone — results in analysis/')
