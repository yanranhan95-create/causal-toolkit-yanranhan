import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression, LinearRegression

def _fit_pscore(X, t):
    try:
        lr = LogisticRegression(penalty='none', max_iter=2000, random_state=0)
        lr.fit(X, t)
    except Exception:
        lr = LogisticRegression(penalty=None, max_iter=2000, random_state=0)
        lr.fit(X, t)
    e = lr.predict_proba(X)[:, 1]
    eps = 1e-6
    e = np.clip(e, eps, 1 - eps)
    return e

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    data = df[[T, Y]].join(pd.DataFrame(dmatrix(ps_formula, df, return_type='dataframe'))).dropna()
    t = data[T].values
    y = data[Y].values
    X = data.drop(columns=[T, Y]).values
    e = _fit_pscore(X, t)
    ate = np.mean(t * y / e - (1 - t) * y / (1 - e))
    return float(ate)

def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    data = df[[T, Y]].join(pd.DataFrame(dmatrix(formula, df, return_type='dataframe'))).dropna()
    t = data[T].values
    y = data[Y].values
    X = data.drop(columns=[T, Y]).values
    e = _fit_pscore(X, t)
    mask1 = (t == 1)
    mask0 = ~mask1
    m1 = LinearRegression().fit(X[mask1], y[mask1])
    m0 = LinearRegression().fit(X[mask0], y[mask0])
    mu1 = m1.predict(X)
    mu0 = m0.predict(X)
    term1 = t * (y - mu1) / e + mu1
    term0 = (1 - t) * (y - mu0) / (1 - e) + mu0
    ate = np.mean(term1 - term0)
    return float(ate)
