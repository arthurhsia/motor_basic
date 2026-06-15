"""
classify_tongue_hand_rest.py — entry point

Orchestrates the full classification pipeline:
  1. Window sweep (Riemannian only) → best WIN_START / WIN_LEN
  2. k sweep → best channel count for LDA and Riemannian paths
  3. m sweep → best tangent feature count for TS_pb_LDA
  4. Final per-subject CV with best hyperparameters
  5. Save CSV + figures

All classes and helpers live in the motor_clf/ package:
  motor_clf/config.py       — constants and hyperparameters
  motor_clf/transformers.py — sklearn transformer classes
  motor_clf/pipelines.py    — pipeline factory functions
  motor_clf/data_io.py      — data loaders (.npz and .mat)
  motor_clf/evaluation.py   — cross-validation helpers
  motor_clf/plotting.py     — figure generation

Run:
    python3 classify_tongue_hand_rest.py
"""

import numpy as np
import pandas as pd

import motor_clf.config as _cfg
from motor_clf.config import (
    DATA_DIR, OUT_DIR, CLASS_NAMES, K_CHAN, M_FEAT, BANDS, EXCLUDED_SUBJECTS,
)
from motor_clf.data_io import load_subject_bandpower, load_subject_raw_trials
from motor_clf.evaluation import cv_scores, evaluate_subject_riemann
from motor_clf.pipelines import (
    make_lda_shrinkage, make_mdm_mb, make_ts_pb_lda,
    RIEMANN_CLASSIFIERS, RIEMANN_AVAILABLE,
)
from motor_clf.plotting import save_bar_chart, save_comparison_chart, save_per_subject_chart


def main():
    npz_paths = sorted(DATA_DIR.glob('*_mot_th_analyzed.npz'))
    mat_paths = sorted(DATA_DIR.glob('*_mot_t_h.mat'))
    if not npz_paths:
        raise FileNotFoundError(f'No .npz files found in {DATA_DIR}')

    subj_from   = lambda p: p.stem.split('_')[0]
    mat_by_subj = {subj_from(p): p for p in mat_paths}

    if EXCLUDED_SUBJECTS:
        print('Excluded subjects:')
        for s, reason in EXCLUDED_SUBJECTS.items():
            print(f'  {s}: {reason}')
        print()
        npz_paths = [p for p in npz_paths if p.stem.split('_')[0] not in EXCLUDED_SUBJECTS]
        mat_paths = [p for p in mat_paths if p.stem.split('_')[0] not in EXCLUDED_SUBJECTS]

    print(f'Found {len(npz_paths)} subjects\n')

    # --- pipeline summary -------------------------------------------------------
    bands_str = '  +  '.join(f'{lo}–{hi} Hz' for lo, hi in BANDS)
    print('=' * 65)
    print('PIPELINES')
    print('=' * 65)
    print('LDA_shrink  (BP path — pre-computed .npz features):')
    print(f'  HFB + LFB band-power per channel (.npz)')
    print(f'  → OVR R² channel selection (top-{K_CHAN} HFB + top-{K_CHAN} LFB)')
    print(f'  → StandardScaler → Shrinkage LDA  (lsqr, auto)')
    print()
    print(f'MDM_mb  (Riemannian — block-diagonal):')
    print(f'  Raw ECoG → CAR → HFB R² channel selection (top-{K_CHAN})')
    print(f'  → Block-diagonal cov  [{bands_str}] → MDM')
    print()
    print(f'TS_pb_LDA  (Riemannian — per-band tangent, 2 bands):')
    print(f'  Raw ECoG → CAR → HFB R² channel selection (top-{K_CHAN})')
    print(f'  → per band [{bands_str}]: each → own tangent space → concat (240-d)')
    print(f'  → OVR R² feature selection (top-{M_FEAT}) → Shrinkage LDA')
    print('=' * 65)
    print()

    # window locked: WIN_START=500 ms, WIN_LEN=2500 ms (tuned by prior sweep)
    best_win_start, best_win_len = _cfg.WIN_START, _cfg.WIN_LEN
    print(f'Window: {best_win_start}–{best_win_start + best_win_len} ms post-onset (locked)\n')

    # --- k sweep: LDA_shrink ---------------------------------------------------
    all_bp_Xy = [load_subject_bandpower(p) for p in npz_paths]
    K_GRID    = [8, 12, 13, 14, 15]
    hdr = f'{"k":>4} | {"LDA_tr":>7} {"LDA_te":>7} {"gap":>7}'
    print('LDA_shrink k sweep:')
    print('=' * len(hdr)); print(hdr); print('-' * len(hdr))
    best_lda_te, best_k_lda = 0, K_GRID[0]
    for k in K_GRID:
        trs, tes = [], []
        for X, y in all_bp_Xy:
            tr, te = cv_scores(lambda kk=k: make_lda_shrinkage(kk), X, y)
            if te is not None: trs.append(tr.mean()); tes.append(te.mean())
        mt, mte = np.mean(trs), np.mean(tes)
        print(f'{k:>4} | {mt:>7.3f} {mte:>7.3f} {mt-mte:>7.3f}', flush=True)
        if mte > best_lda_te: best_lda_te, best_k_lda = mte, k
    print('=' * len(hdr))
    print(f'Best k={best_k_lda}  (test={best_lda_te:.3f})\n')

    # --- LDA_shrink final run --------------------------------------------------
    bp_rows = []
    for npz_path in npz_paths:
        subj   = subj_from(npz_path)
        print(f'Processing {subj}...', flush=True)
        X, y   = load_subject_bandpower(npz_path)
        counts = {cn: int((y == i).sum()) for i, cn in enumerate(CLASS_NAMES)}
        row    = {'subject': subj, **{f'n_{c}': v for c, v in counts.items()}}
        tr, te = cv_scores(lambda kk=best_k_lda: make_lda_shrinkage(kk), X, y)
        if te is None:
            row['LDA_shrink_train_mean'] = row['LDA_shrink_train_std'] = np.nan
            row['LDA_shrink_test_mean']  = row['LDA_shrink_test_std']  = np.nan
        else:
            row['LDA_shrink_train_mean'] = tr.mean(); row['LDA_shrink_train_std'] = tr.std()
            row['LDA_shrink_test_mean']  = te.mean(); row['LDA_shrink_test_std']  = te.std()
        bp_rows.append(row)

    df = pd.DataFrame(bp_rows).set_index('subject')

    # --- load Riemannian data with best window ---------------------------------
    best_k_r, best_pb_m = K_CHAN, M_FEAT   # locked: k=15, m=15
    all_mat_Xy = []
    if RIEMANN_AVAILABLE and mat_paths:
        for npz_path in npz_paths:
            subj = subj_from(npz_path)
            if subj not in mat_by_subj: continue
            Xm, ym = load_subject_raw_trials(mat_by_subj[subj],
                                             win_start=best_win_start, win_len=best_win_len)
            if Xm is not None: all_mat_Xy.append((Xm, ym))

    # # --- k sweep: Riemannian (commented out — k locked at 15) ----------------
    # hdr2 = f'{"k":>4} | {"MDM_tr":>7} {"MDM_te":>7} {"gap":>7} | {"TSp_tr":>7} {"TSp_te":>7} {"gap":>7}'
    # print('Riemannian k sweep:')
    # print('=' * len(hdr2)); print(hdr2); print('-' * len(hdr2))
    # best_k_r, best_combined = K_GRID[0], -np.inf
    # for k in K_GRID:
    #     mdm_trs, mdm_tes, pb_trs, pb_tes = [], [], [], []
    #     for Xm, ym in all_mat_Xy:
    #         tr, te = cv_scores(lambda kk=k: make_mdm_mb(kk), Xm, ym)
    #         if te is not None: mdm_trs.append(tr.mean()); mdm_tes.append(te.mean())
    #         tr, te = cv_scores(lambda kk=k: make_ts_pb_lda(kk, M_FEAT), Xm, ym)
    #         if te is not None: pb_trs.append(tr.mean()); pb_tes.append(te.mean())
    #     mmt, mmte = np.mean(mdm_trs), np.mean(mdm_tes)
    #     pbt, pbte = np.mean(pb_trs),  np.mean(pb_tes)
    #     print(f'{k:>4} | {mmt:>7.3f} {mmte:>7.3f} {mmt-mmte:>7.3f} | {pbt:>7.3f} {pbte:>7.3f} {pbt-pbte:>7.3f}', flush=True)
    #     if mmte + pbte > best_combined:
    #         best_combined, best_k_r = mmte + pbte, k
    # print('=' * len(hdr2))
    # print(f'Best k={best_k_r}\n')

    # # --- m sweep: TS_pb_LDA (commented out — m locked at 15) -----------------
    # M_GRID = list(range(14, 20))
    # hdr3   = f'{"m":>4} | {"TSp_tr":>7} {"TSp_te":>7} {"gap":>7}'
    # print(f'TS_pb_LDA m sweep (k={best_k_r}):')
    # print('=' * len(hdr3)); print(hdr3); print('-' * len(hdr3))
    # best_pb_m_te, best_pb_m = 0, M_GRID[0]
    # for m in M_GRID:
    #     pb_trs, pb_tes = [], []
    #     for Xm, ym in all_mat_Xy:
    #         tr, te = cv_scores(lambda kk=best_k_r, mm=m: make_ts_pb_lda(kk, mm), Xm, ym)
    #         if te is not None: pb_trs.append(tr.mean()); pb_tes.append(te.mean())
    #     pbt, pbte = np.mean(pb_trs), np.mean(pb_tes)
    #     print(f'{m:>4} | {pbt:>7.3f} {pbte:>7.3f} {pbt-pbte:>7.3f}', flush=True)
    #     if pbte > best_pb_m_te: best_pb_m_te, best_pb_m = pbte, m
    # print('=' * len(hdr3))
    # print(f'Best m={best_pb_m}  (test={best_pb_m_te:.3f})\n')

    # --- Riemannian final run --------------------------------------------------
    RIEMANN_CLASSIFIERS['MDM_mb']    = lambda: make_mdm_mb(k=best_k_r)
    RIEMANN_CLASSIFIERS['TS_pb_LDA'] = lambda: make_ts_pb_lda(k=best_k_r, m=best_pb_m)

    riemann_rows = []
    if RIEMANN_AVAILABLE and mat_paths:
        print('\nRunning Riemannian classifiers...', flush=True)
        for npz_path in npz_paths:
            subj = subj_from(npz_path)
            if subj not in mat_by_subj: continue
            print(f'  {subj}', flush=True)
            row = evaluate_subject_riemann(mat_by_subj[subj])
            if row: riemann_rows.append(row)

    if riemann_rows:
        df_r = pd.DataFrame(riemann_rows).set_index('subject')
        df   = df.join(df_r[[c for c in df_r.columns
                              if c.endswith('_mean') or c.endswith('_std')]], how='left')

    riemann_names = list(RIEMANN_CLASSIFIERS.keys()) if riemann_rows else []
    all_clf_names = ['LDA_shrink'] + riemann_names

    # --- print summary ---------------------------------------------------------
    print('=' * 65)
    print(f'{"Classifier":<15} {"Train":>8} {"Test":>8} {"Gap":>8}')
    print('=' * 65)
    for name in all_clf_names:
        tr = df[f'{name}_train_mean'].mean()
        te = df[f'{name}_test_mean'].mean()
        print(f'{name:<15} {tr:>8.3f} {te:>8.3f} {tr-te:>8.3f}')
    print('=' * 65)

    # --- save CSV + figures ----------------------------------------------------
    csv_path = DATA_DIR / 'classifier_results.csv'
    df.to_csv(csv_path)
    print(f'\nSaved results CSV → {csv_path}')

    save_bar_chart(df, all_clf_names, OUT_DIR)
    save_comparison_chart(df, all_clf_names, OUT_DIR)
    save_per_subject_chart(df, all_clf_names, OUT_DIR)


if __name__ == '__main__':
    main()
