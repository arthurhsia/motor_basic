"""
Data loaders for the ECoG motor task.

load_subject_bandpower() — reads pre-computed HFB/LFB scalars from .npz
load_subject_raw_trials() — reads raw multi-channel trial segments from .mat
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

from . import config as _config
from .config import VALID_CODES, LABEL_MAP, CLASS_NAMES, SRATE


def load_subject_bandpower(npz_path: Path):
    """Return (X, y) from pre-computed HFB/LFB band-power features.

    X : (n_trials, 2 * n_channels) — first half HFB, second half LFB
    y : (n_trials,) int
    """
    d     = np.load(npz_path)
    HFB   = d['HFB_trials']   # (n_channels, n_trials)
    LFB   = d['LFB_trials']
    tr_sc = d['tr_sc']
    keep  = np.isin(tr_sc, list(VALID_CODES))
    HFB, LFB, tr_sc = HFB[:, keep], LFB[:, keep], tr_sc[keep]
    X = np.hstack([HFB.T, LFB.T])
    y = np.array([CLASS_NAMES.index(LABEL_MAP[c]) for c in tr_sc])
    return X, y


def _get_blocksize(stim):
    diff = np.concatenate([[0], np.diff(stim)])
    ends_h,   starts_h = np.where(diff == -12)[0], np.where(diff ==  12)[0]
    ends_t,   starts_t = np.where(diff == -11)[0], np.where(diff ==  11)[0]
    if len(ends_h) == len(starts_h) and len(ends_h) > 0:
        return int(round(np.abs(ends_h - starts_h).mean()))
    if len(ends_t) == len(starts_t) and len(ends_t) > 0:
        return int(round(np.abs(ends_t - starts_t).mean()))
    return None


def load_subject_raw_trials(mat_path: Path, win_start=None, win_len=None):
    """Return (X, y) using raw multi-channel trial segments for Riemannian.

    X : (n_trials, n_channels, win_len)
    y : (n_trials,) int
    win_start/win_len default to motor_clf.config values when None.
    """
    if win_start is None: win_start = _config.WIN_START
    if win_len   is None: win_len   = _config.WIN_LEN

    d    = sio.loadmat(str(mat_path))
    data = d['data'].astype(float)
    stim = d['stim'].flatten().astype(int)

    data = data - data.mean(axis=1, keepdims=True)   # CAR

    blocksize = _get_blocksize(stim)
    if blocksize is None or blocksize % 1000 != 0:
        return None, None

    N = len(stim)
    prev = np.concatenate([np.zeros(blocksize, dtype=int), stim[:N - blocksize]])
    stim[(prev == 12) & (stim == 0)] = 120
    stim[(prev == 11) & (stim == 0)] = 110

    changes      = np.concatenate([[True], stim[1:] != stim[:-1]])
    starts       = np.where(changes)[0]
    codes        = np.zeros(len(starts), dtype=int)
    codes[1:]    = stim[starts[1:]]

    segments, labels = [], []
    for onset, code in zip(starts, codes):
        if code not in VALID_CODES:
            continue
        s = onset + win_start
        e = s + win_len
        if e > len(data):
            continue
        segments.append(data[s:e, :].T)   # (n_channels, win_len)
        labels.append(CLASS_NAMES.index(LABEL_MAP[code]))

    if not segments:
        return None, None

    X = np.stack(segments, axis=0)
    y = np.array(labels)

    # reject artifact trials: RMS > median + 4*MAD across trials
    trial_rms = X.std(axis=(1, 2))
    med = np.median(trial_rms)
    mad = np.median(np.abs(trial_rms - med))
    keep = trial_rms <= med + 4 * mad
    n_rejected = int((~keep).sum())
    if n_rejected:
        X, y = X[keep], y[keep]

    return X, y
