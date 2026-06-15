"""
Custom sklearn-compatible transformers for ECoG motor classification.
All transformers follow the fit(X, y) / transform(X) API and are safe
to use inside cross_validate() — no data leakage across CV folds.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis as _kurtosis
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
except ImportError:
    pass

from .config import BANDS


class BPChannelSelector(BaseEstimator, TransformerMixin):
    """Select top-k channels by OVR R² on HFB power (band-power path).

    Input:  (n_trials, 2*n_ch) — first half HFB, second half LFB scalars
    Output: (n_trials, 2*k)    — top-k HFB + same-k LFB channels
    """

    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        n_ch = X.shape[1] // 2
        Xh   = X[:, :n_ch]
        Xc   = Xh - Xh.mean(axis=0)
        Xstd = Xc.std(axis=0) + 1e-12
        r2   = np.zeros(n_ch)
        for cls in np.unique(y):
            yb = (y == cls).astype(float); yb -= yb.mean()
            r  = (Xc * yb[:, None]).mean(0) / (Xstd * (yb.std() + 1e-12))
            r2 = np.maximum(r2, r ** 2)
        self.selected_ = np.argsort(r2)[::-1][:min(self.k, n_ch)]
        return self

    def transform(self, X):
        n_ch = X.shape[1] // 2
        return X[:, np.concatenate([self.selected_, self.selected_ + n_ch])]


class HFBChannelSelector(BaseEstimator, TransformerMixin):
    """Select top-k channels by OVR R² on HFB power; return raw signal.

    Input:  (n_trials, n_ch, T) — raw ECoG
    Output: (n_trials, k, T)   — raw ECoG for selected channels only

    Channels are ranked on HFB (70–200 Hz) variance so selection is driven
    by the most discriminative band. Downstream steps apply all bands.
    """

    def __init__(self, k=15, srate=1000, order=4):
        self.k = k; self.srate = srate; self.order = order

    def fit(self, X, y):
        # step 1: reject spiky/epileptic channels (kurtosis outliers within subject)
        kurt    = np.array([_kurtosis(X[:, ch, :].ravel()) for ch in range(X.shape[1])])
        med_k   = np.median(kurt)
        mad_k   = np.median(np.abs(kurt - med_k)) + 1e-12
        good    = np.where(kurt <= med_k + 4 * mad_k)[0]
        X_good  = X[:, good, :]

        # step 2: HFB R² on surviving channels
        nyq = self.srate / 2
        sos = butter(self.order, [70 / nyq, 200 / nyq], btype='bandpass', output='sos')
        pwr = sosfiltfilt(sos, X_good, axis=-1).var(axis=-1)
        Xc  = pwr - pwr.mean(0); Xstd = Xc.std(0) + 1e-12
        r2  = np.zeros(len(good))
        for cls in np.unique(y):
            yb = (y == cls).astype(float); yb -= yb.mean()
            r  = (Xc * yb[:, None]).mean(0) / (Xstd * (yb.std() + 1e-12))
            r2 = np.maximum(r2, r ** 2)
        # flag suspiciously high R² — likely artifact, not neural signal
        artifact = r2 > 0.9
        if artifact.any():
            good    = good[~artifact]
            r2      = r2[~artifact]
            X_good  = X_good[:, ~artifact, :]

        top = np.argsort(r2)[::-1][:min(self.k, len(good))]
        self.selected_ = good[top]
        return self

    def transform(self, X):
        return X[:, self.selected_, :]


class BlockDiagBandCov(BaseEstimator, TransformerMixin):
    """Filter into multiple bands, compute LWF covariance per band, assemble block-diagonal.

    Input:  (n_trials, n_ch, T) — raw ECoG (channel-selected)
    Output: (n_trials, B*n_ch, B*n_ch) — block-diagonal SPD matrix

    Used by MDM_mb. Each band's k×k covariance occupies a diagonal block;
    off-diagonal blocks are zero. Result is SPD and avoids cross-band noise.
    """

    def __init__(self, bands=None, srate=1000, order=4, estimator='lwf'):
        self.bands = bands or BANDS
        self.srate = srate; self.order = order; self.estimator = estimator

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        n, nc, _ = X.shape
        nb  = len(self.bands); nyq = self.srate / 2
        out = np.zeros((n, nb * nc, nb * nc))
        cov = Covariances(estimator=self.estimator)
        for b, (lo, hi) in enumerate(self.bands):
            sos = butter(self.order, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
            C   = cov.fit_transform(sosfiltfilt(sos, X, axis=-1))
            s, e = b * nc, (b + 1) * nc
            out[:, s:e, s:e] = C
        return out


class PerBandTangentConcat(BaseEstimator, TransformerMixin):
    """Project each band's covariance to its own tangent space, then concatenate.

    Input:  (n_trials, n_ch, T) — raw ECoG (channel-selected)
    Output: (n_trials, B * n_ch*(n_ch+1)//2)

    Compared to projecting a block-diagonal SPD to a single tangent space,
    this avoids the B²·n_ch² − B·n_ch² near-zero cross-band features.
    Each band gets its own Riemannian mean fitted on training data only.
    """

    def __init__(self, bands=None, srate=1000, order=4, estimator='lwf'):
        self.bands = bands or BANDS
        self.srate = srate; self.order = order; self.estimator = estimator

    def fit(self, X, y=None):
        nyq = self.srate / 2
        self.sos_list_, self.ts_list_ = [], []
        cov = Covariances(estimator=self.estimator)
        for lo, hi in self.bands:
            sos = butter(self.order, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
            self.sos_list_.append(sos)
            C  = cov.fit_transform(sosfiltfilt(sos, X, axis=-1))
            self.ts_list_.append(TangentSpace(metric='riemann').fit(C))
        return self

    def transform(self, X):
        cov = Covariances(estimator=self.estimator)
        vecs = []
        for sos, ts in zip(self.sos_list_, self.ts_list_):
            C = cov.fit_transform(sosfiltfilt(sos, X, axis=-1))
            vecs.append(ts.transform(C))
        return np.concatenate(vecs, axis=1)


class TangentFeatureSelector(BaseEstimator, TransformerMixin):
    """Select top-m tangent-space features by OVR R².

    Input:  (n_trials, n_features) — vectorised tangent vectors
    Output: (n_trials, m)
    """

    def __init__(self, m=15):
        self.m = m

    def fit(self, X, y):
        Xc = X - X.mean(0); Xstd = Xc.std(0) + 1e-12
        r2 = np.zeros(X.shape[1])
        for cls in np.unique(y):
            yb = (y == cls).astype(float); yb -= yb.mean()
            r  = (Xc * yb[:, None]).mean(0) / (Xstd * (yb.std() + 1e-12))
            r2 = np.maximum(r2, r ** 2)
        self.selected_ = np.argsort(r2)[::-1][:min(self.m, X.shape[1])]
        return self

    def transform(self, X):
        return X[:, self.selected_]
