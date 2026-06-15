from pathlib import Path

# Directories
DATA_DIR = Path(__file__).parent.parent / 'data'
OUT_DIR  = Path(__file__).parent.parent / 'figs'
OUT_DIR.mkdir(exist_ok=True)

# Trial event codes in the .mat stim channel
CODE_TONGUE      = 11
CODE_HAND        = 12
CODE_REST_TONGUE = 110
CODE_REST_HAND   = 120

LABEL_MAP = {
    CODE_TONGUE:      'tongue',
    CODE_HAND:        'hand',
    CODE_REST_TONGUE: 'rest',
    CODE_REST_HAND:   'rest',
}
CLASS_NAMES = ['tongue', 'hand', 'rest']
VALID_CODES = set(LABEL_MAP.keys())

# Signal parameters
SRATE     = 1000   # Hz
WIN_START = 500    # ms after trial onset
WIN_LEN   = 2500   # ms window length (500–3000 ms, full trial — tuned by window sweep)

# Cross-validation
N_SPLITS = 5       # stratified k-fold

# Frequency bands for multi-band Riemannian pipelines: (lo_hz, hi_hz)
BANDS = [(8, 32), (70, 200)]   # beta/alpha  +  HFB (2-band baseline)

# Narrow filter-bank bands for TS_fb_LDA (32–70 Hz skipped — mixed signal)
FILTER_BANK_BANDS = [
    ( 8,  13),   # alpha
    (13,  20),   # low beta
    (20,  32),   # high beta
    (70,  110),  # low HFB
    (110, 150),  # mid HFB
    (150, 200),  # high HFB
]

# Hyperparameters (tuned by sweep in main)
K_CHAN = 15   # OVR R² channel selection — top-k channels
M_FEAT = 15   # OVR R² tangent feature selection — top-m features

# Subjects excluded from analysis (with reason)
EXCLUDED_SUBJECTS = {
    'jf': 'R²≈1.0 artifact channel + step-change drift (p=0.000) — likely EMG contamination',
}
