import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm


def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns and outputs a DataFrame with columns for the wealth index, previous peaks, and percentage drawdown.
    """
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


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns.
    """
    ind = pd.read_csv(
        "../data/ind30_m_vw_rets.csv",
        header=0,
        index_col=0,
        parse_dates=True,
        na_values=-99.99,
    )
    ind = ind / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


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


def annualized_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std() * (periods_per_year**0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computes the annualized Sharpe ratio of a set of returns
    """
    # Convert the annual riskfree rate to per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = excess_ret.mean() * periods_per_year
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def annualized_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from constituent weights and covariance matrix
    """
    return (weights.T @ covmat @ weights) ** 0.5


def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)


def plot_ef(n_points, er, cov, style=".-"):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return with minimum volatility.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er),
    }
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    results = scipy_minimize(
        fun=portfolio_vol,
        x0=init_guess,
        args=(cov,),
        method="SLSQP",
        options={"disp": False},
        constraints=(return_is_target, weights_sum_to_1),
        bounds=bounds,
    )
    return results.x


def optimal_weights(n_points, er, cov):
    """
    Returns the weights of the minimum volatility portfolio for a range of target returns.
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


# def optimal_weights(n_points, er, cov):
#     target_rets = np.linspace(er.min(), er.max(), n_points)
#     weights = []
#     for tr in target_rets:
#         # Catch edge cases where min/max bounds clip the SLSQP solver
#         w = minimize_vol(tr, er, cov)
#         weights.append(w)
#     return weights
