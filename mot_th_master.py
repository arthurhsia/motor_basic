"""
mot_th_master.py
Python port of mot_th_master.m (Miller, KJ 2015).

Calculates broadband (HFB, 76-100 Hz) and low-frequency (LFB, 8-32 Hz)
move-vs-rest power differences for hand and tongue movements across subjects,
saving per-subject results and brain-surface figures.

Run:
    python3 mot_th_master.py
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')               # headless rendering for saved figures
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_ind
from pathlib import Path

from tail_gauss_plot_redux import tail_gauss_plot_redux

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def car(data):
    """Common average re-reference: subtract cross-channel mean at each sample."""
    return data - data.mean(axis=1, keepdims=True)


def signed_r2(x, y):
    """Signed r^2 (point-biserial): correlates combined values against a binary group label."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    values = np.concatenate([x, y])
    labels = np.concatenate([np.ones(len(x)), np.zeros(len(y))])
    r = np.corrcoef(values, labels)[0, 1]
    return float(np.sign(r) * r ** 2)


def compute_block_psd(signal, srate=1000):
    """
    Welch PSD matching MATLAB psd(x, nfft, fs, window, noverlap) call:
        psd(x, 1000, 1000, 250, 100)
    Returns the first 200 frequency bins (0–199 Hz).
    """
    if len(signal) < 250:       # too short for the default window — skip
        return np.zeros(200)
    _, pxx = welch(
        signal.astype(float),
        fs=srate,
        window='hann',
        nperseg=250,
        noverlap=100,
        nfft=srate,
    )
    return pxx[:200]   # keep 0–199 Hz


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def mot_th_master():
    srate = 1000
    BF_correct = False      # set True to Bonferroni-correct p-values

    subjects = [
        'bp', 'ca', 'cc', 'de', 'fp', 'gc', 'gf',
        'hh', 'hl', 'jc', 'jf', 'jm', 'jp', 'jt',
        'rh', 'rr', 'ug', 'wc', 'zt',
    ]

    script_dir = Path(__file__).parent
    os.makedirs(script_dir / 'figs', exist_ok=True)

    for subj in subjects:
        print(f'Subject {subj}')

        # --- load data and electrode locations --------------------------------
        data_mat = sio.loadmat(str(script_dir / 'data' / f'{subj}_mot_t_h.mat'))
        elec_mat = sio.loadmat(str(script_dir / 'locs' / f'{subj}_electrodes.mat'))

        data = data_mat['data'].astype(float)          # (time, channels)
        stim = data_mat['stim'].flatten().astype(int)  # (time,)
        electrodes = elec_mat['electrodes']            # (n_elec, 3)

        # --- common average re-reference --------------------------------------
        data = car(data)

        # --- determine block size from stimulus transitions -------------------
        diff_stim = np.concatenate([[0], np.diff(stim)])
        ends_hand   = np.where(diff_stim == -12)[0]   # hand block ends
        ends_tongue = np.where(diff_stim == -11)[0]   # tongue block ends
        starts_hand   = np.where(diff_stim == 12)[0]  # hand block starts
        starts_tongue = np.where(diff_stim == 11)[0]  # tongue block starts

        if len(ends_hand) == len(starts_hand):
            blocksize = int(round(np.abs(ends_hand - starts_hand).mean()))
        elif len(ends_tongue) == len(starts_tongue):
            blocksize = int(round(np.abs(ends_tongue - starts_tongue).mean()))
        else:
            print(f'  WARNING: cannot determine blocksize for {subj}, skipping.')
            continue

        if blocksize % 1000 != 0:
            print(f'  WARNING: blocksize {blocksize} is not a multiple of 1000, skipping.')
            continue

        # --- label rest blocks by the movement that preceded them -------------
        N = len(stim)
        prev_stim = np.concatenate([np.zeros(blocksize, dtype=int), stim[:N - blocksize]])
        stim[(prev_stim == 12) & (stim == 0)] = 120   # hand rest
        stim[(prev_stim == 11) & (stim == 0)] = 110   # tongue rest

        # --- catalogue consecutive trials (same stim code) -------------------
        changes = np.concatenate([[True], stim[1:] != stim[:-1]])
        trial_ids = np.cumsum(changes)      # 1-indexed trial number per sample
        trial_starts = np.where(changes)[0]

        # tr_sc[k] = stim code for trial k+1; initialised with 0 like MATLAB
        tr_sc = np.zeros(len(trial_starts), dtype=int)
        tr_sc[0] = 0
        tr_sc[1:] = stim[trial_starts[1:]]

        max_trial = int(trial_ids.max())
        num_chans = data.shape[1]

        # --- compute Welch PSD for every trial --------------------------------
        mean_PSD = np.zeros((200, num_chans))
        all_PSD_list = []

        for cur_trial in range(1, max_trial + 1):
            idx = np.where(trial_ids == cur_trial)[0]
            curr_data = data[idx, :]

            if len(curr_data) >= blocksize:
                curr_data = curr_data[srate // 2: blocksize, :]

            block_PSD = np.column_stack([
                compute_block_psd(curr_data[:, p], srate) for p in range(num_chans)
            ])                                  # (200, num_chans)
            mean_PSD += block_PSD
            all_PSD_list.append(block_PSD)

        mean_PSD /= max_trial
        all_PSD = np.stack(all_PSD_list, axis=2)   # (200, num_chans, num_trials)

        # --- band-limited power relative to overall mean ----------------------
        # Avoid division by zero
        safe_mean = np.where(mean_PSD == 0, 1e-12, mean_PSD)
        all_PSD_norm = all_PSD / safe_mean[:, :, np.newaxis]

        # LFB: 8–32 Hz (MATLAB rows 8:32, 1-indexed → Python [7:32])
        # HFB: 76–100 Hz (MATLAB rows 76:100, 1-indexed → Python [75:100])
        LFB_trials = all_PSD_norm[7:32,  :, :].sum(axis=0)   # (num_chans, num_trials)
        HFB_trials = all_PSD_norm[75:100, :, :].sum(axis=0)  # (num_chans, num_trials)

        # --- signed r^2 and t-test for each channel ---------------------------
        hand_idx   = np.where(tr_sc == 12)[0]
        hrest_idx  = np.where(tr_sc == 120)[0]
        tong_idx   = np.where(tr_sc == 11)[0]
        trest_idx  = np.where(tr_sc == 110)[0]

        r_hand_HFB   = np.zeros(num_chans)
        r_hand_LFB   = np.zeros(num_chans)
        p_hand_HFB   = np.ones(num_chans)
        p_hand_LFB   = np.ones(num_chans)
        r_tongue_HFB = np.zeros(num_chans)
        r_tongue_LFB = np.zeros(num_chans)
        p_tongue_HFB = np.ones(num_chans)
        p_tongue_LFB = np.ones(num_chans)

        for m in range(num_chans):
            if len(hand_idx) > 1 and len(hrest_idx) > 1:
                r_hand_HFB[m] = signed_r2(HFB_trials[m, hand_idx],  HFB_trials[m, hrest_idx])
                r_hand_LFB[m] = signed_r2(LFB_trials[m, hand_idx],  LFB_trials[m, hrest_idx])
                _, p_hand_HFB[m] = ttest_ind(HFB_trials[m, hand_idx],  HFB_trials[m, hrest_idx])
                _, p_hand_LFB[m] = ttest_ind(LFB_trials[m, hand_idx],  LFB_trials[m, hrest_idx])

            if len(tong_idx) > 1 and len(trest_idx) > 1:
                r_tongue_HFB[m] = signed_r2(HFB_trials[m, tong_idx], HFB_trials[m, trest_idx])
                r_tongue_LFB[m] = signed_r2(LFB_trials[m, tong_idx], LFB_trials[m, trest_idx])
                _, p_tongue_HFB[m] = ttest_ind(HFB_trials[m, tong_idx], HFB_trials[m, trest_idx])
                _, p_tongue_LFB[m] = ttest_ind(LFB_trials[m, tong_idx], LFB_trials[m, trest_idx])

        if BF_correct:
            p_hand_HFB   = np.minimum(p_hand_HFB   * num_chans, 1.0)
            p_hand_LFB   = np.minimum(p_hand_LFB   * num_chans, 1.0)
            p_tongue_HFB = np.minimum(p_tongue_HFB * num_chans, 1.0)
            p_tongue_LFB = np.minimum(p_tongue_LFB * num_chans, 1.0)

        # --- condition-average PSDs -------------------------------------------
        mean_PSD_handmove   = all_PSD[:, :, hand_idx].mean(axis=2)  if len(hand_idx)  > 0 else None
        mean_PSD_tonguemove = all_PSD[:, :, tong_idx].mean(axis=2)  if len(tong_idx)  > 0 else None
        mean_PSD_handrest   = all_PSD[:, :, hrest_idx].mean(axis=2) if len(hrest_idx) > 0 else None
        mean_PSD_tonguerest = all_PSD[:, :, trest_idx].mean(axis=2) if len(trest_idx) > 0 else None

        # --- brain-surface figures --------------------------------------------
        for movement in ('hand', 'tongue'):
            for band in ('HFB', 'LFB'):
                r_vals = {
                    ('hand',   'HFB'): r_hand_HFB,
                    ('hand',   'LFB'): r_hand_LFB,
                    ('tongue', 'HFB'): r_tongue_HFB,
                    ('tongue', 'LFB'): r_tongue_LFB,
                }[(movement, band)]

                p_vals = {
                    ('hand',   'HFB'): p_hand_HFB,
                    ('hand',   'LFB'): p_hand_LFB,
                    ('tongue', 'HFB'): p_tongue_HFB,
                    ('tongue', 'LFB'): p_tongue_LFB,
                }[(movement, band)]

                wts = r_vals * (p_vals < 0.05)

                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111, projection='3d')
                tail_gauss_plot_redux(electrodes, wts, ax=ax)
                ax.set_title(
                    f"{subj}, {movement}-{band}, max r²={np.max(np.abs(wts)):.3f}",
                    fontsize=9,
                )
                out_path = script_dir / 'figs' / f'{subj}_GMap_{movement}_{band}.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'  Saved {out_path.name}')

        # --- save results -----------------------------------------------------
        save_path = script_dir / 'data' / f'{subj}_mot_th_analyzed.npz'
        np.savez(
            save_path,
            mean_PSD=mean_PSD,
            mean_PSD_handmove=mean_PSD_handmove   if mean_PSD_handmove   is not None else np.array([]),
            mean_PSD_tonguemove=mean_PSD_tonguemove if mean_PSD_tonguemove is not None else np.array([]),
            mean_PSD_handrest=mean_PSD_handrest   if mean_PSD_handrest   is not None else np.array([]),
            mean_PSD_tonguerest=mean_PSD_tonguerest if mean_PSD_tonguerest is not None else np.array([]),
            r_hand_HFB=r_hand_HFB,
            r_hand_LFB=r_hand_LFB,
            r_tongue_HFB=r_tongue_HFB,
            r_tongue_LFB=r_tongue_LFB,
            p_hand_HFB=p_hand_HFB,
            p_hand_LFB=p_hand_LFB,
            p_tongue_HFB=p_tongue_HFB,
            p_tongue_LFB=p_tongue_LFB,
            HFB_trials=HFB_trials,
            LFB_trials=LFB_trials,
            tr_sc=tr_sc,
            blocksize=np.array([blocksize]),
        )
        print(f'  Saved {save_path.name}')
        print('-------------------------------------------')


if __name__ == '__main__':
    mot_th_master()
