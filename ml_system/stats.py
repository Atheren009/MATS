"""
Statistical tests for model validation.
Implements Deflated Sharpe Ratio (DSR) and Superior Predictive Ability (SPA) block bootstrap.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

def deflated_sharpe_ratio(
    returns: pd.Series,
    expected_sr: float = 0.0,
    n_trials: int = 1,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 26  # 15-min equity bars: 6.5h/day × 4 bars/h × 252 days = 6552
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR) adjusting for non-normality and multiple testing.
    Uses the Bailey & Lopez de Prado (2014) approach.
    """
    if len(returns) < 3 or returns.std() == 0:
        return 0.0

    # Annualised SR
    mean_ret = returns.mean() - (risk_free_rate / periods_per_year)
    std_ret = returns.std()
    sr = (mean_ret / std_ret) * np.sqrt(periods_per_year)

    # Moments
    sk = skew(returns)
    ku = kurtosis(returns, fisher=False)  # Pearson's kurtosis

    # Estimate expected maximum SR from multiple trials (Euler-Mascheroni approximation)
    if n_trials > 1:
        emc = 0.5772156649
        max_sr_expected = np.sqrt((1 - emc) * norm.ppf(1 - 1.0/n_trials) + emc * norm.ppf(1 - 1.0/(n_trials * np.e)))
    else:
        max_sr_expected = expected_sr

    # Adjusted variance of SR
    t = len(returns)
    var_sr = (1 - (sk * sr) + ((ku - 1) / 4) * (sr**2)) / (t - 1)
    
    if var_sr <= 0:
        return 0.0
        
    # Probabilistic Sharpe Ratio (PSR)
    dsr = norm.cdf((sr - max_sr_expected) / np.sqrt(var_sr))
    return dsr

def block_bootstrap_spa(
    challenger_returns: pd.Series,
    baseline_returns: pd.Series,
    block_size: int = 96,  # 1 day of 15m bars
    n_bootstraps: int = 1000
) -> float:
    """
    Superior Predictive Ability (SPA) using block bootstrap.
    Calculates the p-value that the challenger outperforms the baseline.
    Null Hypothesis: Challenger Mean Return <= Baseline Mean Return.
    Returns the p-value (smaller means challenger is significantly better).
    """
    # Align series
    df = pd.concat([challenger_returns, baseline_returns], axis=1).fillna(0.0)
    chal_ret = df.iloc[:, 0].values
    base_ret = df.iloc[:, 1].values
    
    t = len(df)
    if t < block_size:
        return 1.0
        
    # Test statistic (mean return difference)
    d = chal_ret - base_ret
    actual_diff = np.mean(d)
    
    # Recentered differences for the bootstrap under the null hypothesis
    # (assuming expected difference is 0 or less)
    d_centered = d - actual_diff
    
    boot_diffs = np.zeros(n_bootstraps)
    n_blocks = t // block_size + 1
    
    for i in range(n_bootstraps):
        # Sample random start indices for blocks
        starts = np.random.randint(0, t - block_size, size=n_blocks)
        boot_sample = np.concatenate([d_centered[s : s + block_size] for s in starts])[:t]
        boot_diffs[i] = np.mean(boot_sample)
        
    # P-value: proportion of bootstrap samples where the mean diff exceeds our actual observed diff
    p_val = np.sum(boot_diffs >= actual_diff) / n_bootstraps
    return float(p_val)
