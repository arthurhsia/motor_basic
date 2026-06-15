"""
Pipeline factory functions.
Each make_*() returns a fresh sklearn Pipeline ready for cross_validate().
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pyriemann.classification import MDM
    RIEMANN_AVAILABLE = True
except ImportError:
    RIEMANN_AVAILABLE = False

from .config import BANDS, FILTER_BANK_BANDS, K_CHAN, M_FEAT
from .transformers import (
    BPChannelSelector,
    HFBChannelSelector,
    BlockDiagBandCov,
    PerBandTangentConcat,
    TangentFeatureSelector,
)



def make_lda_shrinkage(k=K_CHAN):
    """Band-power path: HFB+LFB scalars → R² channel sel → shrinkage LDA."""
    return Pipeline([
        ('sel',    BPChannelSelector(k=k)),
        ('scaler', StandardScaler()),
        ('clf',    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ])


def make_mdm_mb(k=K_CHAN):
    """Riemannian: raw ECoG → channel sel → block-diagonal cov → MDM."""
    return Pipeline([
        ('sel', HFBChannelSelector(k=k)),
        ('cov', BlockDiagBandCov(bands=BANDS)),
        ('mdm', MDM(metric='riemann')),
    ])


def make_ts_pb_lda(k=K_CHAN, m=M_FEAT):
    """Riemannian: raw ECoG → channel sel → per-band tangent concat → R² sel → LDA."""
    return Pipeline([
        ('sel', HFBChannelSelector(k=k)),
        ('pbt', PerBandTangentConcat(bands=BANDS)),
        ('tfs', TangentFeatureSelector(m=m)),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ])


def make_ts_fb_lda(k=K_CHAN, m=M_FEAT):
    """Filter-bank Riemannian: 6 narrow bands → per-band tangent → R² sel → LDA."""
    return Pipeline([
        ('sel', HFBChannelSelector(k=k)),
        ('pbt', PerBandTangentConcat(bands=FILTER_BANK_BANDS)),
        ('tfs', TangentFeatureSelector(m=m)),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ])


BP_CLASSIFIERS = {
    'LDA_shrink': make_lda_shrinkage,
}

RIEMANN_CLASSIFIERS = {
    'MDM_mb':    lambda: make_mdm_mb(k=K_CHAN),
    'TS_pb_LDA': lambda: make_ts_pb_lda(k=K_CHAN, m=M_FEAT),
}
