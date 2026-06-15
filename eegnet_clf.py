"""
eegnet_clf.py — EEGNet within-subject classification

Same protocol as classify_tongue_hand_rest.py:
  - Within-subject 5-fold stratified CV
  - WIN_START=500 ms, WIN_LEN=2500 ms post-onset
  - All 47 channels (EEGNet learns its own spatial filter)

Run:
    python3 eegnet_clf.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from braindecode.models import EEGNetv4
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

from motor_clf.config import (
    DATA_DIR, OUT_DIR, CLASS_NAMES, N_SPLITS, SRATE, WIN_START, WIN_LEN,
)
from motor_clf.data_io import load_subject_raw_trials
from motor_clf.plotting import save_bar_chart, save_comparison_chart

N_CLASSES  = len(CLASS_NAMES)
N_CHANNELS = 47   # all channels — EEGNet learns spatial filter
N_TIMES    = WIN_LEN
BATCH_SIZE = 16
MAX_EPOCHS = 150
LR         = 1e-3
WEIGHT_DECAY = 1e-2


def make_eegnet():
    model = EEGNetv4(
        n_chans=N_CHANNELS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length='auto',
        F1=8,
        D=2,
        F2=16,
        kernel_length=int(SRATE * 0.25),   # 250 ms temporal filter
        drop_prob=0.5,                      # higher dropout to combat overfitting
    )
    clf = NeuralNetClassifier(
        module=model,
        max_epochs=MAX_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=WEIGHT_DECAY,
        criterion=nn.CrossEntropyLoss,
        train_split=None,
        verbose=0,
        device='cpu',
        callbacks=[
            LRScheduler(policy=CosineAnnealingLR, T_max=MAX_EPOCHS),
        ],
    )
    return clf


def cv_eegnet(X, y, n_splits=N_SPLITS):
    """Within-subject stratified k-fold CV for EEGNet.

    X : (n_trials, n_channels, n_times)  float32
    y : (n_trials,)  int64
    Returns (train_scores, test_scores) or (None, None).
    """
    min_count     = min((y == i).sum() for i in range(N_CLASSES))
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        return None, None

    X32 = X.astype(np.float32)
    y64 = y.astype(np.int64)

    cv       = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    tr_scores, te_scores = [], []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X32, y64)):
        clf = make_eegnet()
        clf.fit(X32[train_idx], y64[train_idx])
        tr_scores.append(clf.score(X32[train_idx], y64[train_idx]))
        te_scores.append(clf.score(X32[test_idx],  y64[test_idx]))

    return np.array(tr_scores), np.array(te_scores)


def main():
    npz_paths = sorted(DATA_DIR.glob('*_mot_th_analyzed.npz'))
    mat_paths = sorted(DATA_DIR.glob('*_mot_t_h.mat'))
    if not mat_paths:
        raise FileNotFoundError(f'No .mat files found in {DATA_DIR}')

    subj_from   = lambda p: p.stem.split('_')[0]
    mat_by_subj = {subj_from(p): p for p in mat_paths}

    print('EEGNet — within-subject 5-fold CV')
    print(f'  Channels={N_CHANNELS}, n_times={N_TIMES}, kernel={int(SRATE*0.25)} ms')
    print(f'  Epochs={MAX_EPOCHS} (cosine LR decay), lr={LR}, wd={WEIGHT_DECAY}, batch={BATCH_SIZE}')
    print(f'  Window: {WIN_START}–{WIN_START+WIN_LEN} ms post-onset')
    print()

    rows = []
    subjects = [subj_from(p) for p in npz_paths if subj_from(p) in mat_by_subj]

    for subj in subjects:
        print(f'  {subj}', end='', flush=True)
        X, y = load_subject_raw_trials(mat_by_subj[subj])
        if X is None:
            print(' — skipped (load failed)')
            continue

        tr, te = cv_eegnet(X, y)
        if te is None:
            print(' — skipped (too few trials)')
            continue

        print(f'  train={tr.mean():.3f}  test={te.mean():.3f}  gap={tr.mean()-te.mean():.3f}')
        rows.append({
            'subject':                subj,
            'EEGNet_train_mean':      tr.mean(),
            'EEGNet_train_std':       tr.std(),
            'EEGNet_test_mean':       te.mean(),
            'EEGNet_test_std':        te.std(),
        })

    if not rows:
        print('No results.')
        return

    df = pd.DataFrame(rows).set_index('subject')

    print()
    print('=' * 45)
    print(f'{"Classifier":<12} {"Train":>8} {"Test":>8} {"Gap":>8}')
    print('=' * 45)
    tr_m = df['EEGNet_train_mean'].mean()
    te_m = df['EEGNet_test_mean'].mean()
    print(f'{"EEGNet":<12} {tr_m:>8.3f} {te_m:>8.3f} {tr_m-te_m:>8.3f}')
    print('=' * 45)

    # merge with existing results CSV if present
    csv_path = DATA_DIR / 'classifier_results.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path, index_col='subject')
        df_existing = df_existing.join(df, how='left')
        df_existing.to_csv(csv_path)
        print(f'\nMerged into existing results CSV → {csv_path}')

        all_clf = ['LDA_shrink', 'MDM_mb', 'TS_pb_LDA', 'EEGNet']
        present = [n for n in all_clf if f'{n}_test_mean' in df_existing.columns]
        save_bar_chart(df_existing, present, OUT_DIR)
        save_comparison_chart(df_existing, present, OUT_DIR)
    else:
        df.to_csv(csv_path)
        print(f'\nSaved results CSV → {csv_path}')
        save_bar_chart(df, ['EEGNet'], OUT_DIR)
        save_comparison_chart(df, ['EEGNet'], OUT_DIR)


if __name__ == '__main__':
    main()
