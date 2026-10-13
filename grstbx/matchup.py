'''
Matchups between satellite retrievals and in situ measurements: pairing of the data in time,
agreement statistics and scatter plots.
'''

import numpy as np
import pandas as pd
import xarray as xr

from .utils import Plotting


def matchup_metrics(x, y):
    """
    Agreement statistics between in situ (``x``) and satellite (``y``) values.

    - slope, intercept and Pearson's r of the linear regression of y vs x,
    - rmse and bias (mean of y - x), in the units of the data,
    - mape: mean absolute percentage error (%),
    - eps: median symmetric accuracy (%) and beta: signed symmetric percentage bias (%)
      (Morley et al., 2018; Eqs. 8-9 in Pahlevan et al., 2020), computed on the
      strictly positive pairs.

    The pairs with a non-finite value are discarded.

    :param x: in situ values (array-like)
    :param y: satellite values (array-like), same length as ``x``
    :return: dict with the keys 'N', 'slope', 'intercept', 'r', 'rmse', 'bias', 'mape', 'eps', 'beta'
    """
    from scipy import stats

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    regr = stats.linregress(x, y)
    pos = (x > 0) & (y > 0)
    log_ratio = np.log10(y[pos] / x[pos])
    med = np.median(log_ratio)
    return {'N': len(x),
            'slope': regr.slope,
            'intercept': regr.intercept,
            'r': regr.rvalue,
            'rmse': np.sqrt(np.mean((y - x) ** 2)),
            'bias': np.mean(y - x),
            'mape': 100 * np.mean(np.abs(y - x) / np.abs(x)),
            'eps': 100 * (10 ** np.median(np.abs(log_ratio)) - 1),
            'beta': 100 * np.sign(med) * (10 ** np.abs(med) - 1)}


def plot_matchup(ax, x, y, vmin, vmax, fmt='.4f', xerr=None, yerr=None, **kwargs):
    """
    Scatter plot of y vs x with error bars, 1:1 and regression lines, and the agreement statistics
    (:func:`matchup_metrics`).

    :param ax: matplotlib axes
    :param x: in situ values
    :param y: satellite values
    :param vmin: lower limit of both axes
    :param vmax: upper limit of both axes
    :param fmt: format of the intercept, rmse and bias
    :param xerr: error bars of x (see matplotlib errorbar), none if None
    :param yerr: error bars of y (see matplotlib errorbar), none if None
    :param kwargs: passed to ``ax.scatter``
    :return: (scatter PathCollection, dict of the agreement statistics)
    """
    import matplotlib.ticker as mticker

    m = matchup_metrics(x, y)
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', color='grey', lw=0.6, capsize=2, zorder=1)
    sc = ax.scatter(x, y, zorder=2, **kwargs)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    Plotting.set_layout(ax)  # square axes and 1:1 line
    ax.axline((0, m['intercept']), slope=m['slope'], ls='--', lw=1.2, c='grey')
    text = '\n'.join([f"$y = {m['slope']:.2f}x {m['intercept']:+{fmt}}$",
                      f"$r = {m['r']:.3f}$",
                      f"$rmse = {m['rmse']:{fmt}}$",
                      f"$bias = {m['bias']:{fmt}}$",
                      f"$\\epsilon = {m['eps']:.1f}\\%$",
                      f"$\\beta = {m['beta']:+.1f}\\%$",
                      f"$N = {m['N']}$"])
    ax.text(0.97, 0.03, text, fontsize=10, ha='right', va='bottom', transform=ax.transAxes)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.minorticks_on()
    return sc, m


def _to_frame(obj, name):
    """1-D xarray / pandas object -> DataFrame indexed by time (the Series and DataArray as column ``name``)."""
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        obj = obj.reset_coords(drop=True)
        obj = obj.to_series() if isinstance(obj, xr.DataArray) else obj.to_dataframe()
    if isinstance(obj, pd.Series):
        obj = obj.rename(name).to_frame()
    if obj.index.nlevels > 1:
        raise ValueError(f'one time dimension expected, got {list(obj.index.names)}')
    return obj


def get_matchups(sat, insitu, max_lag='1D', floor='1D'):
    """
    Pair each in situ measurement with the closest satellite image acquired within ``max_lag``.

    :param sat: satellite values indexed by acquisition time (1-D): xarray Dataset / DataArray
                (e.g. per-image statistics with 'median', 'q25', 'q75') or pandas DataFrame / Series
    :param insitu: in situ values indexed by sampling time (1-D): xarray DataArray or pandas Series;
                   the missing values are dropped
    :param max_lag: maximum time lag between the satellite and in situ data (str or pandas.Timedelta)
    :param floor: the times of both data sets are rounded down to this frequency before pairing (default
                  '1D': in situ sampling dates often have no time of day, the date of the image is used);
                  None to pair the exact times
    :return: pandas.DataFrame with one row per matchup: 'date' (in situ time), 'insitu' (in situ value),
             'time' (satellite time) and the satellite variables
    """
    sat = _to_frame(sat, 'sat')
    sat_vars = list(sat.columns)
    sat = sat.rename_axis('time').reset_index().sort_values('time')
    obs = _to_frame(insitu, 'insitu').rename_axis('date').reset_index().dropna().sort_values('date')

    key = '_matchup_key'
    sat[key] = sat['time'].astype('datetime64[ns]')
    obs[key] = obs['date'].astype('datetime64[ns]')
    if floor is not None:
        sat[key] = sat[key].dt.floor(floor)
        obs[key] = obs[key].dt.floor(floor)
    matchups = pd.merge_asof(obs, sat, on=key, direction='nearest', tolerance=pd.Timedelta(max_lag))
    return matchups.dropna(subset=sat_vars, how='all').drop(columns=key).reset_index(drop=True)
