"""Stage 2a: per-source Isolation Forest on rule-residual rows (label-free fit)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

IF_PARAMS = dict(n_estimators=200, max_samples=256, contamination="auto", n_jobs=-1)


def fit_if(M, fit_mask, seed):
    """Fit on the masked rows only (train residual). No labels used."""
    return IsolationForest(random_state=seed, **IF_PARAMS).fit(M.loc[fit_mask])


def score(model, M):
    """Higher = more anomalous."""
    return pd.Series(-model.score_samples(M), index=M.index, name="anomaly_score")


def oof_scores(M, fit_mask, seed, n_chunks=5):
    """Out-of-fold scores for the fit rows (time-ordered chunks), full-model scores elsewhere."""
    full = fit_if(M, fit_mask, seed)
    s = score(full, M)
    chunks = np.array_split(M.index[fit_mask], n_chunks)
    for k, ch in enumerate(chunks):
        rest = np.concatenate([c for j, c in enumerate(chunks) if j != k])
        m_k = IsolationForest(random_state=seed, **IF_PARAMS).fit(M.loc[rest])
        s.loc[ch] = -m_k.score_samples(M.loc[ch])
    return s, full


def save(model, path):
    joblib.dump(model, path)