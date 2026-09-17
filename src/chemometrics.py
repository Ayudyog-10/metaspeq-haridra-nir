"""
Chemometrics core for the METASPEQ Haridra NIR models.

Preprocessing, PLS model construction, and — importantly — validation that
splits at the *sample* level rather than the spectrum level.

Sixteen replicate scans were acquired per physical sample. Splitting those
scans at random puts replicates of the same packed cuvette on both sides of
the split, so the model is scored on spectra it has effectively already
seen. That inflates apparent performance and tells you nothing about how the
model will behave on a new sample. Every validation routine here groups by
sample_id.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

# Wavelength window used in the original Mendine study.
DEFAULT_BAND = (1074.0, 1574.0)
DEFAULT_N_COMPONENTS = 8


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: removes multiplicative scatter and baseline
    offset row by row. First choice for powders with variable packing."""
    spectra = np.asarray(spectra, dtype=float)
    mu = spectra.mean(axis=1, keepdims=True)
    sd = spectra.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (spectra - mu) / sd


def savgol(spectra: np.ndarray, window: int = 11, poly: int = 2,
           deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay smoothing or derivative."""
    return savgol_filter(np.asarray(spectra, dtype=float),
                         window_length=window, polyorder=poly,
                         deriv=deriv, axis=1)


def preprocess(spectra: np.ndarray, method: str = "snv") -> np.ndarray:
    """Apply a named preprocessing pipeline.

    Whatever is used to build a model must be used identically at prediction
    time. A mismatch produces confident nonsense with no error message.
    """
    method = method.lower()
    if method == "none":
        return np.asarray(spectra, dtype=float)
    if method == "snv":
        return snv(spectra)
    if method == "snv+sg":
        return savgol(snv(spectra), 11, 2, 0)
    if method == "snv+sg1":
        return savgol(snv(spectra), 15, 2, 1)
    if method == "snv+sg2":
        return savgol(snv(spectra), 21, 2, 2)
    raise ValueError(f"unknown preprocessing method: {method!r}")


PREPROCESSING_CHOICES = ["snv", "snv+sg", "snv+sg1", "snv+sg2", "none"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_starch_dataset(path: str | Path | None = None):
    """Return (wavelengths, X absorbance, y reference %, sample_id groups)."""
    path = Path(path) if path else repo_root() / "data" / "haridra_starch_absorbance.csv"
    df = pd.read_csv(path)
    wl = np.array([float(c) for c in df.columns[2:]])
    X = df.iloc[:, 2:].to_numpy(dtype=float)
    y = df["starch_pct"].to_numpy(dtype=float)
    # "S12_4" -> sample S12. The replicate index after the underscore must not
    # be allowed to separate scans of the same physical sample.
    groups = df["sample_id"].astype(str).str.split("_").str[0].to_numpy()
    return wl, X, y, groups


def band_mask(wavelengths: np.ndarray, band=DEFAULT_BAND) -> np.ndarray:
    lo, hi = band
    return (wavelengths >= lo) & (wavelengths <= hi)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    r2: float
    rmse: float
    bias: float
    sep: float
    rpd: float
    slope: float
    intercept: float

    def as_dict(self):
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_pred - y_true
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    sep = float(resid.std(ddof=1))
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    return Metrics(
        n=len(y_true),
        r2=1.0 - ss_res / ss_tot,
        rmse=float(np.sqrt((resid ** 2).mean())),
        bias=float(resid.mean()),
        sep=sep,
        rpd=float(y_true.std(ddof=1) / sep) if sep else float("inf"),
        slope=float(slope),
        intercept=float(intercept),
    )


def interpret_rpd(rpd: float) -> str:
    if rpd < 2.0:
        return "Not usable for quantification"
    if rpd < 3.0:
        return "Rough screening only — distinguishes high from low"
    if rpd < 5.0:
        return "Adequate for quality-control screening and in-process control"
    if rpd < 8.0:
        return "Adequate for QC and, with supporting validation, release quantification"
    return "Suitable for any application"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def leave_one_sample_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         n_components: int = DEFAULT_N_COMPONENTS):
    """Cross-validate with all replicate scans of a sample withheld together.

    Returns the out-of-fold predictions. This is the honest analogue of what
    happens in routine use: the model meets a sample it has never seen.
    """
    unique = np.unique(groups)
    splitter = GroupKFold(n_splits=len(unique))
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx]).ravel()
    return preds


def component_scan(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   max_components: int = 14) -> pd.DataFrame:
    """RMSECV against model complexity, honestly validated.

    Use this to choose the number of latent variables. Pick the point where
    the curve flattens; do not keep adding components for a marginal gain.
    """
    rows = []
    for k in range(1, max_components + 1):
        m = compute_metrics(y, leave_one_sample_out(X, y, groups, k))
        rows.append({"n_components": k, "r2": m.r2, "rmsecv": m.rmse,
                     "rpd": m.rpd, "bias": m.bias})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class StarchModel:
    pls: PLSRegression
    wavelengths: np.ndarray
    band: tuple
    preprocessing: str
    n_components: int
    cv_metrics: dict
    calibration_metrics: dict

    def predict(self, spectra: np.ndarray) -> np.ndarray:
        """Predict starch % from raw absorbance over the full 256-point grid."""
        spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
        mask = band_mask(self.wavelengths, self.band)
        Xp = preprocess(spectra[:, mask], self.preprocessing)
        return self.pls.predict(Xp).ravel()

    def in_calibration_range(self, value: float) -> bool:
        lo, hi = self.cv_metrics["ref_min"], self.cv_metrics["ref_max"]
        return lo <= value <= hi


def train_starch_model(n_components: int = DEFAULT_N_COMPONENTS,
                       preprocessing: str = "snv",
                       band: tuple = DEFAULT_BAND) -> StarchModel:
    wl, X, y, groups = load_starch_dataset()
    mask = band_mask(wl, band)
    Xp = preprocess(X[:, mask], preprocessing)

    cv_pred = leave_one_sample_out(Xp, y, groups, n_components)
    cv = compute_metrics(y, cv_pred).as_dict()
    cv["ref_min"], cv["ref_max"] = float(y.min()), float(y.max())
    cv["n_samples"] = int(len(np.unique(groups)))

    pls = PLSRegression(n_components=n_components, scale=False).fit(Xp, y)
    cal = compute_metrics(y, pls.predict(Xp).ravel()).as_dict()

    return StarchModel(pls=pls, wavelengths=wl, band=band,
                       preprocessing=preprocessing, n_components=n_components,
                       cv_metrics=cv, calibration_metrics=cal)
