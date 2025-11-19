#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# S-Learner
def s_learner_discrete(train, test, X, T, y):
    model = LGBMRegressor()
    model.fit(train[X + [T]], train[y])

    test_t1 = test.copy()
    test_t1[T] = 1
    test_t0 = test.copy()
    test_t0[T] = 0

    cate = model.predict(test_t1[X + [T]]) - model.predict(test_t0[X + [T]])
    test_result = test.copy()
    test_result["cate"] = cate
    return test_result


# T-Learner
def t_learner_discrete(train, test, X, T, y):
    m0 = LGBMRegressor()
    m1 = LGBMRegressor()

    m0.fit(train[train[T] == 0][X], train[train[T] == 0][y])
    m1.fit(train[train[T] == 1][X], train[train[T] == 1][y])

    cate = m1.predict(test[X]) - m0.predict(test[X])
    test_result = test.copy()
    test_result["cate"] = cate
    return test_result


# X-Learner
def x_learner_discrete(train, test, X, T, y):
    m0 = LGBMRegressor()
    m1 = LGBMRegressor()
    m0.fit(train[train[T] == 0][X], train[train[T] == 0][y])
    m1.fit(train[train[T] == 1][X], train[train[T] == 1][y])

    train0 = train[train[T] == 0].copy()
    train1 = train[train[T] == 1].copy()
    train0["tau0"] = m1.predict(train0[X]) - train0[y]
    train1["tau1"] = train1[y] - m0.predict(train1[X])

    tau0_model = LGBMRegressor()
    tau1_model = LGBMRegressor()
    tau0_model.fit(train0[X], train0["tau0"])
    tau1_model.fit(train1[X], train1["tau1"])

    ps_model = LogisticRegression(penalty=None)
    ps_model.fit(train[X], train[T])
    e_x = ps_model.predict_proba(test[X])[:, 1]

    cate = e_x * tau0_model.predict(test[X]) + (1 - e_x) * tau1_model.predict(test[X])
    test_result = test.copy()
    test_result["cate"] = cate
    return test_result


# Double-ML
def double_ml_cate(train, test, X, T, y):
    t_model = LGBMRegressor()
    y_model = LGBMRegressor()

    t_res = train[T] - cross_val_predict(t_model, train[X], train[T], cv=5)
    y_res = train[y] - cross_val_predict(y_model, train[X], train[y], cv=5)

    y_star = y_res / t_res
    w = t_res ** 2

    cate_model = LGBMRegressor()
    cate_model.fit(train[X], y_star, sample_weight=w)

    test_result = test.copy()
    test_result["cate"] = cate_model.predict(test[X])
    return test_result


# In[20]:


def simple_data():
    np.random.seed(42)
    n = 1000
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
    return df.iloc[:800].copy(), df.iloc[800:].copy()


# In[22]:


train, test = simple_data()
X, T, y = ["x1", "x2"], "t", "y"

result = t_learner_discrete(train, test, X, T, y)
print(result.head())
print("CATE mean:", result["cate"].mean())


