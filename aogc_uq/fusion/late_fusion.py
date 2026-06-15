"""Late-fusion of step-level signals into one uncertainty.

Signals come in as a dict {name: array} or a 2-D array [n_steps, n_signals].
NaNs (a signal undefined for some steps, e.g. no claims to check) are imputed by
column mean. Everything here is CPU/instant — no large-model training (LogReg
fits only n_signals coefficients).
"""

from __future__ import annotations

import numpy as np


def _matrix(signals, names=None):
    if isinstance(signals, dict):
        names = names or list(signals)
        X = np.column_stack([np.asarray(signals[n], dtype=float) for n in names])
        return X, list(names)
    X = np.asarray(signals, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X, (list(names) if names else [f"s{i}" for i in range(X.shape[1])])


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def _ranknorm(x):
    """Normalized rank in [0,1]; NaNs take the median rank (neutral)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    filled = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0, x)
    ranks = np.argsort(np.argsort(filled, kind="mergesort"), kind="mergesort")
    return ranks / (n - 1)


def rank_fuse(signals):
    """Training-free: average normalized rank across signals. Returns [n] in [0,1]."""
    X, _ = _matrix(signals)
    R = np.column_stack([_ranknorm(X[:, j]) for j in range(X.shape[1])])
    return R.mean(axis=1)


def mean_fuse(signals):
    """Training-free: average min-max-normalized signals (NaN columns ignored per row)."""
    X, _ = _matrix(signals)
    M = np.column_stack([minmax(X[:, j]) for j in range(X.shape[1])])
    with np.errstate(invalid="ignore"):
        out = np.nanmean(M, axis=1)
    return np.where(np.isfinite(out), out, 0.0)


class LogRegFusion:
    """Supervised late-fusion: logistic regression on standardized signals.

    Fits on a small dev set with labels (is_error). ``fuse`` returns P(error).
    Exposes per-signal ``coef_`` for the orthogonality / contribution analysis.
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.names = None
        self._mean = None
        self._std = None
        self._clf = None
        self.coef_ = None

    def fit(self, signals, labels):
        from sklearn.linear_model import LogisticRegression

        X, names = _matrix(signals)
        y = np.asarray(labels).astype(int)
        self.names = names
        self._mean = np.nanmean(X, axis=0)
        self._mean = np.where(np.isfinite(self._mean), self._mean, 0.0)
        Xi = np.where(np.isnan(X), self._mean, X)
        self._std = Xi.std(axis=0)
        self._std[self._std == 0] = 1.0
        Xs = (Xi - self._mean) / self._std
        self._clf = LogisticRegression(C=self.C, max_iter=1000).fit(Xs, y)
        self.coef_ = dict(zip(names, self._clf.coef_[0].tolist()))
        return self

    def fuse(self, signals):
        if self._clf is None:
            raise RuntimeError("call fit() before fuse()")
        X, _ = _matrix(signals, names=self.names)
        Xi = np.where(np.isnan(X), self._mean, X)
        Xs = (Xi - self._mean) / self._std
        return self._clf.predict_proba(Xs)[:, 1]
