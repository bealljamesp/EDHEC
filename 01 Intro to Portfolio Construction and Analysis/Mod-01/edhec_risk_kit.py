import numpy as np
import pandas as pd
from scipy.stats import norm


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns and outputs a DataFrame with columns for the wealth index, previous peaks, and percentage drawdown."""
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame(
        {"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdown}
    )


def get_ffme_returns():
    """
    Load the Fama-French data on returns of the top and bottom deciles by market cap."""
    me_m = pd.read_csv(
        "../labs/data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header=0,
        index_col=0,
        parse_dates=True,
        na_values=-99.99,
    )
    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = ["SmallCap", "LargeCap"]
    return rets / 100


def get_hfi_returns():
    """
    Load the EDHEC Hedge Fund Index returns.
    """
    hfi = pd.read_csv(
        "../labs/data/edhec-hedgefundindices.csv",
        header=0,
        index_col=0,
        parse_dates=True,
        na_values=-99.99,
    )
    hfi = hfi / 100
    return hfi


def semideviation_0(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def semideviation_mean(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess = r - r.mean()  # We demean the returns
    excess_negative = excess[excess < 0]  # We take only the returns below the mean
    excess_negative_square = (
        excess_negative**2
    )  # We square the demeaned returns below the mean
    n_negative = (excess < 0).sum()  # number of returns under the mean
    return (excess_negative_square.sum() / n_negative) ** 0.5  # semideviation


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level, i.e. returns the number such that "level" percent of the returns fall below that number, and the rest above.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def skewness(r):
    """
    Alternative to scipy skewness.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy kurtosis."""
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal or not. The null hypothesis is that the data is normally distributed. Rejection of the null at the given level means the data is not normal.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        # .apply() handles the column-by-column execution flawlessly here
        return r.apply(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
