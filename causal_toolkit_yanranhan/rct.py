import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm

def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    treated = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    ate = treated.mean() - control.mean()
    n1, n0 = len(treated), len(control)
    var1, var0 = treated.var(ddof=1), control.var(ddof=1)
    se = np.sqrt(var1 / n1 + var0 / n0)

    z = norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    return ate, ci_lower, ci_upper


def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    treated = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    ate = treated.mean() - control.mean()
    n1, n0 = len(treated), len(control)
    var1, var0 = treated.var(ddof=1), control.var(ddof=1)
    se = np.sqrt(var1 / n1 + var0 / n0)

    t_stat = ate / se
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return ate, t_stat, p_value
