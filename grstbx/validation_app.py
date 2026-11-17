"""
Dashboard of the matchups of GRS with AERONET-OC (table of :func:`grstbx.validation.matchup_table`), with panel,
holoviews and bokeh: the quality criteria and the selections (stations, bands, in situ level and normalisation,
period, platforms, aerosol models and GRS versions) are applied interactively, and the views follow:

- **Overview**: map of the stations (number of matchups), overall statistics, statistics by station;
- **Scatter**: satellite vs in situ Rrs per band (median and standard deviation of the window, 1:1 and
  regression lines, statistics), or all the bands together; hover: station, date, lag, valid pixels...;
- **Statistics**: tables by band, by station and by station and band, heat map of a statistic;
- **Errors**: distribution of the residuals (satellite - in situ) by band and by station;
- **Spectra**: spectra of a matchup (satellite median +- std, AERONET-OC) with the quicklooks of the L2A product;
- **Time series**: residuals in time;
- **Drivers**: residuals against AOT, sunglint, geometry, wind, lag, homogeneity of the window...

In a notebook::

    from grstbx import validation
    validation.dashboard(table)            # or validation.dashboard(validation.read_table('matchups.csv'))

As a web application::

    grstbx-validation dashboard matchups.csv [other_version.csv ...] --show
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from bokeh.models import NumeralTickFormatter
from holoviews.plotting.util import process_cmap

from .matchup import matchup_metrics
from .validation import QC, quality

METRICS = ['N', 'slope', 'intercept', 'r', 'rmse', 'bias', 'mae', 'eps', 'beta', 'medape', 'mape', 'rmse_log',
           'bias_log', 'mae_log']
METRIC_LABELS = {'eps': 'MdSA (%)', 'beta': 'SSPB (%)', 'medape': 'MedAPE (%)', 'mape': 'MAPE (%)'}
INSITU = {'Lwn / F0 (directional, as measured)': 'Rrs_insitu', 'Lwn f/Q (Morel et al. 2002)': 'Rrs_insitu_fQ',
          'Lwn IOP (Lee et al. 2011)': 'Rrs_insitu_IOP'}
DRIVERS = {'AOT(550) GRS': 'aot550_sat', 'AOT(550) AERONET-OC': 'aot550_insitu', 'sunglint BRDFg': 'BRDFg',
           'sun zenith (deg)': 'sza', 'view zenith (deg)': 'vza', 'relative azimuth (deg)': 'raa',
           'wind GRS (m/s)': 'wind_sat', 'wind AERONET-OC (m/s)': 'wind_insitu', 'lag (min)': 'lag_min',
           'valid fraction of the window': 'frac_valid', 'CV Rrs(560) of the window': 'cv560',
           'cloud cover of the tile (%)': 'cloud_cover', 'in situ Rrs': 'x', 'date': 'time'}
COLOR_BY = {'band': 'wl', 'station': 'station', 'lag (min)': 'lag_min', 'AOT(550) GRS': 'aot550_sat',
            'sunglint BRDFg': 'BRDFg', 'in situ level': 'level', 'platform': 'platform', 'GRS version': 'version',
            'aerosol model': 'aerosol_model'}
# web style (after the Bokeh demos, e.g. https://demo.bokeh.org/chaotic-motion): warm paper page, plots on a
# cream background with faint grid lines, serif titles; earth palette gold, cream, brick red, sage, olive
PALETTE = {'gold': '#D5A03F', 'cream': '#E3DECB', 'red': '#A3301E', 'sage': '#B9B482', 'olive': '#696F45',
           'ink': '#2A2620', 'paper': '#FFFDF9', 'warm': '#F6F3EA',
           'accent': '#C96B50'}       # accent of the widgets and tabs: lighter terracotta than the brick red
SERIF = 'Georgia, "Times New Roman", serif'
EARTH = [PALETTE[c] for c in ('olive', 'red', 'gold', 'sage')] + ['#3F4329', '#6E2214', '#9C7424', '#8C8858']
CUSTOM_CMAPS = {'earth': EARTH, 'earth gradient': [PALETTE[c] for c in ('olive', 'sage', 'gold', 'red')]}
MD_CSS = (f'h1, h2, h3, h4 {{font-family: {SERIF}; font-weight: 400; letter-spacing: -0.02em; '
          f'color: {PALETTE["olive"]}; margin: 0.6em 0 0.3em}} '
          f'h3 {{border-left: 3px solid {PALETTE["accent"]}; padding-left: 8px}}')
BUTTON_CSS = (f'.bk-btn, .bk-btn.bk-btn-default {{background: {PALETTE["gold"]} !important; '
              f'border-color: {PALETTE["gold"]} !important; color: {PALETTE["ink"]} !important}} '
              f'.bk-btn:hover {{background: {PALETTE["sage"]} !important; border-color: {PALETTE["sage"]} !important}}')
TABS_CSS = (f'.bk-tab {{font-family: {SERIF}; font-size: 15px; color: {PALETTE["ink"]}}} '
            f'.bk-tab.bk-active {{color: {PALETTE["accent"]}; border-color: {PALETTE["accent"]}; '
            f'background-color: {PALETTE["paper"]}}}')
# colormaps of the points: 'auto' is earth (up to 8 categories) or Category20 for the categories (station...),
# viridis for the numbers (AOT...)
CATEGORICAL_CMAPS = ['earth', 'Category10', 'Category20', 'Set1', 'Set2', 'Dark2', 'Paired', 'glasbey_dark']
CONTINUOUS_CMAPS = ['earth gradient', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'Spectral_r',
                    'RdYlBu_r',
                    'coolwarm', 'jet']
FRAME = 260            # side (px) of the square frame of the scatter plots per band; 'all bands': 2 x FRAME


def _style(plot, element):
    """Bokeh hook: plot style of the dashboard (cream frame, faint grid, serif title, no outline)."""
    fig = plot.state
    fig.background_fill_color = PALETTE['cream']
    fig.border_fill_color = PALETTE['paper']
    fig.outline_line_color = None
    for grid in fig.xgrid + fig.ygrid:
        grid.grid_line_color, grid.grid_line_alpha, grid.grid_line_width = PALETTE['ink'], 0.12, 0.5
    for axis in fig.xaxis + fig.yaxis:
        axis.axis_line_color = axis.major_tick_line_color = axis.minor_tick_line_color = PALETTE['ink']
        axis.axis_line_alpha = axis.major_tick_line_alpha = axis.minor_tick_line_alpha = 0.5
        axis.axis_label_text_font_style = 'normal'
        if type(axis).__name__ == 'LinearAxis':          # 0.005, 0.0005... instead of the scientific notation
            axis.formatter = NumeralTickFormatter(format='0.[0000]')
    if fig.title is not None:
        fig.title.text_font, fig.title.text_font_style = SERIF, 'normal'
        fig.title.text_color = PALETTE['ink']
    for legend in fig.legend:
        legend.background_fill_color, legend.background_fill_alpha = PALETTE['paper'], 0.85
        legend.border_line_color = None


STYLE = dict(hooks=[_style])


def _hv(obj, fixed=False):
    """
    HoloViews pane of a view: axes not linked to the other plots (the in situ Rrs ``x`` of the scatter plots is not the
    easting of the map, and each view sets its ranges), fixed size for the square plots.
    """
    return pn.pane.HoloViews(obj, linked_axes=False, sizing_mode='fixed' if fixed else 'stretch_width')


def _md(text, **kwargs):
    """Markdown pane with the headings of the dashboard."""
    return pn.pane.Markdown(text, stylesheets=[MD_CSS], **kwargs)


def _borders(left=5, right=5, top=5, bottom=5):
    """
    Bokeh hook: style of the dashboard and minimum borders around the frame (axes, title, colour bar inside them),
    for tight grids of panels of fixed size.
    """
    def hook(plot, element):
        fig = plot.state
        _style(plot, element)
        fig.min_border_left, fig.min_border_right, fig.min_border_top, fig.min_border_bottom = left, right, top, bottom
    return hook


def _square(size, xlabel=True, ylabel=True, colorbar=False):
    """
    Width, height and borders of a plot with a square frame of ``size`` px: the borders hold the axes, title and
    colour bar, so that the frame keeps its size whatever the labels.
    """
    left, right, top, bottom = 50 + 18 * ylabel, 15 + 95 * colorbar, 28, 32 + 18 * xlabel
    return dict(width=size + left + right, height=size + top + bottom,
                hooks=[_borders(left, right, top, bottom), _equal_zoom])


def _equal_zoom(plot, element):
    """Bokeh hook: the box zoom keeps the square aspect, so that x and y keep the same range."""
    for tool in plot.state.tools:
        if type(tool).__name__ == 'BoxZoomTool':
            tool.match_aspect = True


BORDER = dict(hooks=[_borders()])
HOVER = ['station', 'date', 'wl', 'level', 'lag_min', 'n_valid', 'frac_valid', 'cv560', 'aot550_sat', 'BRDFg',
         'aerosol_model']


def _prepare(table):
    df = table.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df.time.dt.strftime('%Y-%m-%d %H:%M')
    df['wl'] = df.wl.astype(int)
    for col in ['version', 'platform', 'level', 'station', 'aerosol_model']:
        df[col] = df[col].fillna('').astype(str) if col in df else ''
    df['aerosol_model'] = df.aerosol_model.replace('', 'unknown')
    for col in ['Rrs_insitu_fQ', 'Rrs_insitu_IOP', 'aot550_insitu', 'wind_insitu', 'cloud_cover', 'BRDFg',
                'aot550_sat', 'wind_sat', 'sza', 'vza', 'raa', 'Rrs_sat_std']:
        if col not in df:
            df[col] = np.nan
    return df


def _stats_text(m):
    if not m or not m.get('N'):
        return 'N = 0'
    return (f"N = {m['N']}  r = {m['r']:.3f}\ny = {m['slope']:.2f}x {m['intercept']:+.5f}\n"
            f"RMSE = {m['rmse']:.5f}  bias = {m['bias']:+.5f}\nMdSA = {m['eps']:.1f}%  SSPB = {m['beta']:+.1f}%")


def _metrics(df, by=None, min_n=3):
    rows = {}
    if len(df) >= min_n:
        rows['all'] = matchup_metrics(df.x, df.Rrs_sat)
    if by:
        for key, group in df.groupby(by):
            if len(group) >= min_n:
                rows[key] = matchup_metrics(group.x, group.Rrs_sat)
    out = pd.DataFrame.from_dict(rows, orient='index')
    if out.empty:
        return pd.DataFrame(columns=METRICS)
    out = out[[c for c in METRICS if c in out]].rename(columns=METRIC_LABELS)
    out.index = [' / '.join(map(str, k)) if isinstance(k, tuple) else str(k) for k in out.index]
    out.index.name = by if isinstance(by, str) else ' / '.join(by) if by else ''
    return out.round(5)


def _palette(values, cmap):
    """Colour of each category of ``values`` (sorted), the same in all the panels."""
    cats = sorted(pd.unique(values.dropna()), key=str)
    if cmap == 'auto':
        cmap = 'earth' if len(cats) <= len(EARTH) else 'Category20'
    colors = process_cmap(CUSTOM_CMAPS.get(cmap, cmap), ncolors=max(len(cats), 1),
                          categorical=cmap in CATEGORICAL_CMAPS)
    return dict(zip(cats, colors))


def _points(df, x, vdims, color, cmap='auto', alpha=0.8, clim=None, palette=None, legend=True, size=6):
    """
    Points of ``df`` coloured by the column ``color``: one layer per category (legend, colours fixed by
    ``palette`` across the panels) or a colour scale (limits ``clim`` shared by the panels).
    """
    style = dict(size=size, alpha=alpha, tools=['hover'], line_color=PALETTE['ink'], line_width=0.3,
                 muted_alpha=0.05)
    vdims = list(dict.fromkeys(v for v in vdims if v in df))
    if df[color].dtype == object:
        palette = palette or _palette(df[color], cmap)
        layers = {str(c): hv.Scatter(g, x, vdims, label=str(c)).opts(color=palette.get(c, 'grey'), **style)
                  for c, g in df.groupby(color, sort=True)}
        return hv.Overlay(list(layers.values())).opts(show_legend=legend, legend_position='top_left',
                                                      legend_opts={'click_policy': 'mute'})
    return hv.Scatter(df, x, vdims + ([color] if color not in vdims else [])).opts(
        color=color, cmap='viridis' if cmap == 'auto' else CUSTOM_CMAPS.get(cmap, cmap), clim=clim or (None, None),
        colorbar=legend,
        clabel=color, **style)


def _table(df, height=300):
    return pn.widgets.Tabulator(df.reset_index(), show_index=False, height=height, sizing_mode='stretch_width',
                                disabled=True, layout='fit_data_table', theme='simple')


class MatchupDashboard:
    """
    Interactive analysis of a matchup table (see the module documentation).

    :param table: table of :func:`grstbx.validation.matchup_table` (several GRS versions can be concatenated)
    :param title: title of the page
    :param basemap: basemap of the map of the stations (:data:`grstbx.visual.BASEMAPS`)
    :param insitu: in situ Rrs to choose from, {label: column} (default: :data:`INSITU`, AERONET-OC); the
                   columns missing from the table are left out
    :param network: name of the in situ network in the labels of the widgets
    """

    def __init__(self, table, title='GRS L2A vs AERONET-OC', basemap='OpenStreetMap', insitu=None,
                 network='AERONET-OC'):
        hv.extension('bokeh', logo=False)
        pn.extension('tabulator', sizing_mode='stretch_width', notifications=True)
        self.title = title
        self.basemap = basemap
        self.data = _prepare(table)
        df = self.data
        stations = sorted(df.station.unique())
        bands = sorted(df.wl.unique())
        versions = sorted(df.version.unique())
        platforms = sorted(df.platform.unique())
        aerosols = sorted(df.aerosol_model.unique())
        tmin, tmax = df.time.min().to_pydatetime(), df.time.max().to_pydatetime()
        w = pn.widgets
        self.w_versions = w.MultiChoice(name='GRS versions', options=versions, value=versions[-1:])
        self.w_stations = w.MultiChoice(name='Stations', options=stations, value=stations)
        self.w_bands = w.CheckButtonGroup(name='Bands (nm)', options=bands, value=bands, button_type='primary',
                                          button_style='outline')
        insitu = {k: v for k, v in (insitu or INSITU).items() if v in table and table[v].notna().any()}
        self.w_insitu = w.Select(name='In situ Rrs', options=insitu or {'in situ': 'Rrs_insitu'},
                                 value='Rrs_insitu')
        levels = sorted(df.level.unique())
        self.network = network
        self.w_levels = w.CheckButtonGroup(name=f'{network} level', options=levels,
                                           value=[lv for lv in QC['levels'][:1] if lv in levels] or levels,
                                           button_type='primary', button_style='outline')
        self.w_platforms = w.CheckButtonGroup(name='Platforms', options=platforms, value=platforms,
                                              button_type='primary', button_style='outline')
        self.w_aerosols = w.MultiChoice(name='Aerosol models of GRS', options=aerosols, value=aerosols)
        self.w_dates = w.DateRangeSlider(name='Period', start=tmin, end=tmax, value=(tmin, tmax))
        self.w_lag = w.IntSlider(name='Maximum lag (min)', start=5, end=int(max(60, df.lag_min.abs().max())),
                                 step=5, value=QC['max_lag_min'])
        self.w_valid = w.FloatSlider(name='Minimum valid fraction of the window', start=0, end=1, step=0.05,
                                     value=QC['min_valid'])
        self.w_cv = w.FloatSlider(name='Maximum CV of Rrs(560) in the window', start=0.02, end=1, step=0.01,
                                  value=QC['max_cv'])
        self.w_aot = w.FloatSlider(name='Maximum AOT(550) of GRS (1: no limit)', start=0.05, end=1, step=0.05,
                                   value=1)
        self.w_glint = w.FloatSlider(name='Maximum sunglint BRDFg (0.1: no limit)', start=0.001, end=0.1,
                                     step=0.001, value=0.1, format='0.000')
        self.w_color = w.Select(name='Colour by', options=COLOR_BY, value='wl')
        self.w_log = w.Checkbox(name='Logarithmic axes', value=False)
        self.w_cmap = w.Select(name='Colormap', value='auto', groups={
            'auto': ['auto'], 'categories': CATEGORICAL_CMAPS, 'continuous': CONTINUOUS_CMAPS})
        self.w_alpha = w.FloatSlider(name='Opacity of the points', start=0.1, end=1, step=0.05, value=0.8)
        self.w_driver = w.Select(name='Driver', options=DRIVERS, value='aot550_sat')
        self.w_metric = w.Select(name='Statistic of the heat map', value='MdSA (%)',
                                 options=[METRIC_LABELS.get(m, m) for m in METRICS if m != 'N'])
        self.w_scatter_mode = w.RadioButtonGroup(options=['per band', 'all bands'], value='per band',
                                                 button_type='primary', button_style='outline')
        self.filters = [self.w_versions, self.w_stations, self.w_bands, self.w_insitu, self.w_levels,
                        self.w_platforms, self.w_aerosols, self.w_dates, self.w_lag, self.w_valid, self.w_cv,
                        self.w_aot, self.w_glint]
        self.download = w.FileDownload(callback=self._csv, filename='matchups_selection.csv',
                                       label='Download the selected matchups (CSV)', button_type='default',
                                       stylesheets=[BUTTON_CSS])

    # --------------------------------------------------------------------------------------------------------
    def selection(self, *_):
        """Matchups meeting the criteria and selections of the widgets; ``x``: in situ Rrs, ``res``: residual."""
        df = self.data
        df = df[df.version.isin(self.w_versions.value) & df.station.isin(self.w_stations.value)
                & df.wl.isin(self.w_bands.value) & df.platform.isin(self.w_platforms.value)
                & df.aerosol_model.isin(self.w_aerosols.value)]
        start, end = self.w_dates.value
        df = df[(df.time >= pd.Timestamp(start)) & (df.time <= pd.Timestamp(end) + pd.Timedelta('1D'))]
        df = df.assign(Rrs_insitu=df[self.w_insitu.value])
        ok = quality(df, max_lag_min=self.w_lag.value, min_valid=self.w_valid.value, max_cv=self.w_cv.value,
                     levels=self.w_levels.value, max_aot=None if self.w_aot.value >= 1 else self.w_aot.value,
                     max_glint=None if self.w_glint.value >= 0.1 else self.w_glint.value)
        df = df[ok].copy()
        df['x'] = df.Rrs_insitu
        df['res'] = df.Rrs_sat - df.x
        return df

    def _csv(self):
        buffer = io.StringIO()
        self.selection().drop(columns=['x', 'res']).to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    # --------------------------------------------------------------------------------------------------------
    def view_counts(self, *_):
        df = self.selection()
        return _md(
            f"**{df.image.nunique()}** images, **{len(df)}** matchups (image x band), "
            f"**{df.station.nunique()}** station{'s' if df.station.nunique() > 1 else ''}, "
            f"{df.time.min():%Y-%m-%d} to {df.time.max():%Y-%m-%d}"
            if len(df) else '**No matchup** with these criteria', margin=(0, 10))

    def view_map(self, *_):
        from .visual import basemap_tiles
        from holoviews.util.transform import lon_lat_to_easting_northing

        df = self.selection()
        all_st = self.data.groupby('station')[['lon', 'lat']].first()
        n = df.groupby('station').image.nunique().reindex(all_st.index).fillna(0)
        x, y = lon_lat_to_easting_northing(all_st.lon.values, all_st.lat.values)
        points = pd.DataFrame({'x': x, 'y': y, 'station': all_st.index, 'images': n.values.astype(int),
                               'size': 6 + 3 * np.sqrt(n.values)})
        # extent of the stations, at least 600 km wide (a single station: the map around it)
        pad = max(3e5, 0.1 * max(np.ptp(x), np.ptp(y)))
        xlim, ylim = (x.min() - pad, x.max() + pad), (y.min() - pad, y.max() + pad)
        stations = hv.Points(points, ['x', 'y'], ['station', 'images', 'size']).opts(
            size='size', color='images', cmap=CUSTOM_CMAPS['earth gradient'], colorbar=len(points) > 1,
            tools=['hover'], line_color=PALETTE['ink'], clabel='images with matchups')
        return _hv((basemap_tiles(self.basemap) * stations).opts(
            height=420, responsive=True, xaxis=None, yaxis=None, xlim=xlim, ylim=ylim,
            title='Stations (size, colour: images with matchups)', **STYLE))

    def view_overall(self, *_):
        df = self.selection()
        return pn.Column(_md('#### Statistics by band'), _table(_metrics(df, 'wl'), 330),
                         _md('#### Statistics by station'), _table(_metrics(df, 'station'), 400))

    # --------------------------------------------------------------------------------------------------------
    def _color_scale(self, df, color):
        """Palette (categories) or limits (numbers) of the colours, from all the matchups of the view."""
        if df[color].dtype == object:
            return dict(palette=_palette(df[color], self.w_cmap.value))
        return dict(clim=(float(df[color].min()), float(df[color].max())) if df[color].notna().any() else None)

    def _scatter(self, df, title, color, size=FRAME, legend=True, xlabel=True, ylabel=True, scale=None):
        """Scatter plot GRS vs the in situ Rrs in a square frame of ``size`` px, same limits on both axes."""
        scale = scale or self._color_scale(df, color) if not df.empty else {}
        frame = _square(size, xlabel, ylabel, colorbar=legend and 'clim' in scale)
        if df.empty:
            return hv.Curve([]).opts(title=f'{title}: no matchup', **frame)
        log = self.w_log.value
        lo = max(1e-5, float(np.nanmin([df.x.min(), df.Rrs_sat.min()]))) if log else \
            float(min(0, df.x.min(), df.Rrs_sat.min()))
        hi = float(np.nanmax([df.x.max(), df.Rrs_sat.max()])) * 1.1
        m = matchup_metrics(df.x, df.Rrs_sat) if len(df) > 2 else {}
        scatter = _points(df, 'x', ['Rrs_sat', color] + HOVER, color, self.w_cmap.value, self.w_alpha.value,
                          legend=legend, **scale)
        err = hv.ErrorBars(df.assign(e=df.Rrs_sat_std.fillna(0)), 'x', ['Rrs_sat', 'e']).opts(
            color=PALETTE['ink'], alpha=0.35 * self.w_alpha.value, line_width=0.8, lower_head=None, upper_head=None)
        one = hv.Curve([(lo, lo), (hi, hi)]).opts(color=PALETTE['ink'], line_dash='dashed', line_width=1)
        layers = err * scatter * one
        if m:
            xs = np.linspace(lo, hi, 50)
            layers *= hv.Curve((xs, m['slope'] * xs + m['intercept'])).opts(color=PALETTE['red'], line_width=1.5)
            # bottom right: the points are along the 1:1 line
            layers *= hv.Text(hi * 0.97 if not log else hi * 0.8, lo + 0.03 * (hi - lo) if not log else lo * 1.3,
                              _stats_text(m), halign='right', valign='bottom', fontsize=8)
        return layers.opts(title=title, xlim=(lo, hi), ylim=(lo, hi), logx=log, logy=log,
                           xlabel=f'Rrs {self.network} (sr-1)' if xlabel else '', ylabel='Rrs GRS (sr-1)' if ylabel else '',
                           show_legend=legend, legend_position='top_left', toolbar='above',
                           fontsize={'title': 10, 'labels': 9, 'ticks': 8, 'legend': 8}, **frame)

    def view_scatter(self, *_):
        df = self.selection()
        color = self.w_color.value
        if self.w_scatter_mode.value == 'all bands':
            return _hv(self._scatter(df, 'All bands', color, size=2 * FRAME), fixed=True)
        if color == 'wl':          # one band per panel: the stations are coloured instead
            color = 'station'
        groups = list(df.groupby('wl'))
        if not groups:
            return _md('No matchup')
        # tight grid: same colours and frame in all the panels, axis labels on the outer panels, one legend
        # (first panel) or one colour bar (last panel of each row)
        ncols = min(3, len(groups))
        nrows = -(-len(groups) // ncols)
        scale = self._color_scale(df, color)
        categorical = df[color].dtype == object
        plots = []
        for i, (wl, group) in enumerate(groups):
            row, col = divmod(i, ncols)
            last_in_row = col == ncols - 1 or i == len(groups) - 1
            plots.append(self._scatter(group, f'{wl} nm', color, scale=scale,
                                       legend=(i == 0) if categorical else last_in_row,
                                       xlabel=row == nrows - 1 or i + ncols >= len(groups), ylabel=col == 0))
        return _hv(hv.Layout(plots).cols(ncols).opts(shared_axes=False, merge_tools=True), fixed=True)

    # --------------------------------------------------------------------------------------------------------
    def view_statistics(self, *_):
        df = self.selection()
        by_both = _metrics(df, ['station', 'wl'])
        metric = self.w_metric.value
        heat = _md('')
        if len(by_both) > 1 and metric in by_both:
            cells = by_both.drop(index='all', errors='ignore').reset_index()
            first = cells.columns[0]
            cells[['station', 'wl']] = cells[first].str.split(' / ', expand=True)
            cells['wl'] = cells.wl.astype(int)
            heat = _hv(hv.HeatMap(cells, ['wl', 'station'], [metric, 'N']).opts(
                cmap='RdBu_r' if metric in ('bias', 'SSPB (%)', 'intercept') else 'viridis', colorbar=True,
                tools=['hover'], height=40 + 22 * cells.station.nunique(), responsive=True,
                title=f'{metric} by station and band', xlabel='band (nm)',
                symmetric=metric in ('bias', 'SSPB (%)', 'intercept'), **STYLE))
        return pn.Column(heat, _md('#### By band'), _table(_metrics(df, 'wl'), 330),
                         _md('#### By station'), _table(_metrics(df, 'station'), 400),
                         _md('#### By station and band'), _table(by_both, 500),
                         _md('#### By aerosol model of GRS and band'),
                         _table(_metrics(df, ['aerosol_model', 'wl']), 400))

    def view_errors(self, *_):
        df = self.selection()
        if df.empty:
            return _md('No matchup')
        opts = dict(tools=['hover'], height=350, responsive=True, ylabel=f'Rrs GRS - {self.network} (sr-1)',
                    whisker_color=PALETTE['ink'], box_line_color=PALETTE['ink'], outlier_color=PALETTE['ink'],
                    **STYLE)
        by_band = hv.BoxWhisker(df, 'wl', 'res').opts(title='Residuals by band', xlabel='band (nm)',
                                                     box_fill_color=PALETTE['sage'], **opts)
        by_station = hv.BoxWhisker(df, 'station', 'res').opts(title='Residuals by station', xrotation=45,
                                                             box_fill_color=PALETTE['gold'], **opts)
        rel = df[df.x > 0].assign(rel=lambda d: 100 * d.res / d.x)
        by_band_rel = hv.BoxWhisker(rel, 'wl', 'rel').opts(
            title='Relative residuals by band', xlabel='band (nm)', ylabel=f'(GRS - {self.network}) / {self.network} (%)',
            box_fill_color=PALETTE['olive'], whisker_color=PALETTE['ink'], box_line_color=PALETTE['ink'],
            outlier_color=PALETTE['ink'], tools=['hover'], height=350, responsive=True, **STYLE)
        hists = []
        colors = _palette(df.wl, self.w_cmap.value)
        for wl, group in df.groupby('wl'):
            counts, edges = np.histogram(group.res, bins=30)
            hists.append(hv.Histogram((edges, counts), label=f'{wl} nm').opts(
                fill_color=colors[wl], line_color=colors[wl], alpha=0.45))
        hist = hv.Overlay(hists).opts(title='Distribution of the residuals', height=350, responsive=True,
                                      xlabel=f'Rrs GRS - {self.network} (sr-1)', legend_position='right', **STYLE)
        by_aerosol = hv.BoxWhisker(df, ['aerosol_model', 'wl'], 'res').opts(
            title='Residuals by aerosol model of GRS and band', xrotation=45, box_fill_color=PALETTE['accent'],
            box_alpha=0.6,
            xlabel='aerosol model, band (nm)', **opts)
        return pn.Column(*(_hv(plot) for plot in (by_band, by_band_rel, by_station, by_aerosol, hist)))

    # --------------------------------------------------------------------------------------------------------
    def view_spectra(self, *_):
        df = self.selection()
        if df.empty:
            return _md('No matchup')
        images = (df.groupby('image').agg(station=('station', 'first'), date=('date', 'first'),
                                           level=('level', 'first'), lag_min=('lag_min', 'first'),
                                           n_valid=('n_valid', 'first'), cv560=('cv560', 'first'),
                                           aot550=('aot550_sat', 'first'), l2a=('l2a', 'first'))
                  .sort_values(['station', 'date']).reset_index())
        chooser = pn.widgets.Tabulator(images.drop(columns='l2a').round(4), show_index=False, height=330,
                                       selectable=1, selection=[0], disabled=True, sizing_mode='stretch_width',
                                       layout='fit_data_table', theme='simple', pagination='local', page_size=12)

        def spectrum(selection):
            if not selection:
                return _md('Select a matchup in the table')
            row = images.iloc[selection[0]]
            sel = df[df.image == row.image].sort_values('wl')
            sat = hv.Curve(sel, 'wl', 'Rrs_sat', label='GRS (median)').opts(color=PALETTE['red'], line_width=2)
            spread = hv.Spread(sel.assign(e=sel.Rrs_sat_std.fillna(0)), 'wl', ['Rrs_sat', 'e'],
                               label='GRS +- std').opts(color=PALETTE['gold'], alpha=0.35)
            ins = hv.Scatter(sel, 'wl_insitu', 'x', label=self.network).opts(color=PALETTE['olive'], size=9,
                                                                         marker='diamond')
            wls = pd.concat([sel.wl, sel.wl_insitu])
            plot = (spread * sat * ins).opts(
                xlim=(wls.min() - 20, wls.max() + 20),
                title=f'{row.station} {row.date} (lag {row.lag_min:.0f} min, {row.level})', height=380,
                responsive=True, xlabel='wavelength (nm)', ylabel='Rrs (sr-1)', legend_position='top_right',
                show_grid=True, **STYLE)
            folder = Path(row.l2a).parent
            name = Path(row.l2a).stem
            pngs = [pn.pane.PNG(str(folder / f'{name}_ql_{kind}.png'), height=260, sizing_mode='fixed')
                    for kind in ('l1c_rgb', 'l2a_rgb', 'sunglint') if (folder / f'{name}_ql_{kind}.png').exists()]
            return pn.Column(_hv(plot), pn.Row(*pngs) if pngs else _md('(no quicklook)'))

        return pn.Column(chooser, pn.bind(spectrum, chooser.param.selection))

    def view_timeseries(self, *_):
        df = self.selection()
        if df.empty:
            return _md('No matchup')
        plots = []
        palette = _palette(df.station, self.w_cmap.value)
        for i, (wl, group) in enumerate(df.groupby('wl')):
            plots.append((_points(group, 'time', ['res', 'station', 'date', 'lag_min'], 'station',
                                  alpha=self.w_alpha.value,
                                  palette=palette, legend=i == 0, size=5)
                          * hv.HLine(0).opts(color=PALETTE['ink'], line_width=0.8)).opts(
                height=220, responsive=True, title=f'{wl} nm', ylabel=f'GRS - {self.network}', xlabel='',
                show_legend=i == 0, legend_position='top_left', **BORDER))
        return _hv(hv.Layout(plots).cols(1).opts(shared_axes=False, merge_tools=True))

    def view_drivers(self, *_):
        df = self.selection()
        driver = self.w_driver.value
        if df.empty or driver not in df or df[driver].isna().all():
            return _md('No matchup, or no value of this driver')
        plots = []
        palette = _palette(df.station, self.w_cmap.value)
        label = {v: k for k, v in DRIVERS.items()}.get(driver, driver)
        groups = list(df.groupby('wl'))
        ncols = min(3, len(groups))
        for i, (wl, group) in enumerate(groups):
            plots.append((_points(group, driver, ['res', 'station', 'date'], 'station', alpha=self.w_alpha.value,
                                  palette=palette, legend=i == 0, size=5)
                          * hv.HLine(0).opts(color=PALETTE['ink'], line_width=0.8)).opts(
                width=FRAME + 120, height=FRAME, title=f'{wl} nm',
                ylabel=f'GRS - {self.network}' if i % ncols == 0 else '',
                xlabel=label if i + ncols >= len(groups) else '', show_legend=i == 0, legend_position='top_left',
                **BORDER))
        return _hv(hv.Layout(plots).cols(ncols).opts(shared_axes=False, merge_tools=True), fixed=True)

    # --------------------------------------------------------------------------------------------------------
    def sidebar(self):
        return [
            _md('### Selection'),
            *([self.w_versions] if len(self.w_versions.options) > 1 else []),
            self.w_stations,
            _md('Bands (nm)', margin=(0, 10)), self.w_bands,
            self.w_insitu,
            _md(f'{self.network} level', margin=(0, 10)), self.w_levels,
            *([_md('Platforms', margin=(0, 10)), self.w_platforms]
              if len(self.w_platforms.options) > 1 else []),
            *([self.w_aerosols] if len(self.w_aerosols.options) > 1 else []),
            self.w_dates,
            _md('### Quality criteria'),
            self.w_lag, self.w_valid, self.w_cv, self.w_aot, self.w_glint,
            _md('### Display'),
            self.w_color, self.w_cmap, self.w_alpha, self.w_log,
            self.download,
        ]

    def main(self):
        deps = self.filters
        bind = lambda f, *extra: pn.bind(f, *deps, *extra)        # noqa: E731
        tabs = pn.Tabs(
            ('Overview', pn.Column(bind(self.view_map), bind(self.view_overall))),
            ('Scatter', pn.Column(self.w_scatter_mode, bind(self.view_scatter, self.w_color, self.w_log,
                                                             self.w_scatter_mode, self.w_cmap, self.w_alpha))),
            ('Statistics', pn.Column(self.w_metric, bind(self.view_statistics, self.w_metric))),
            ('Errors', bind(self.view_errors)),
            ('Spectra', bind(self.view_spectra)),
            ('Time series', bind(self.view_timeseries, self.w_cmap, self.w_alpha)),
            ('Drivers', pn.Column(self.w_driver, bind(self.view_drivers, self.w_driver, self.w_cmap, self.w_alpha))),
            dynamic=True, sizing_mode='stretch_width', stylesheets=[TABS_CSS])
        return pn.Column(bind(self.view_counts), tabs, sizing_mode='stretch_width')

    def template(self):
        """Web page (panel FastListTemplate), e.g. served by ``grstbx-validation dashboard``."""
        return pn.template.FastListTemplate(
            title=self.title, sidebar=self.sidebar(), main=[self.main()], sidebar_width=360,
            header_background=PALETTE['olive'], header_color=PALETTE['paper'], accent_base_color=PALETTE['accent'],
            header_accent_base_color=PALETTE['gold'], background_color=PALETTE['warm'],
            neutral_color=PALETTE['cream'], corner_radius=2, shadow=False,
            font='Arial, Helvetica, sans-serif', theme_toggle=False,
            raw_css=[f'#header .title, .title {{font-family: {SERIF}; font-weight: 400; letter-spacing: -0.02em}}'])

    def inline(self):
        """Sidebar and views in a row, for a notebook cell (the templates are not rendered in notebooks)."""
        return pn.Row(pn.Column(*self.sidebar(), width=340), self.main(), sizing_mode='stretch_width')


def create_dashboard(table, template=None, **kwargs):
    """
    Dashboard of a matchup table.

    :param table: table of :func:`grstbx.validation.matchup_table` (or :func:`~grstbx.validation.read_table`)
    :param template: True: web page (to serve), False: row for a notebook cell; default: a row in a notebook,
                     a page otherwise
    :param kwargs: options of :class:`MatchupDashboard`
    """
    board = MatchupDashboard(table, **kwargs)
    if template is None:
        template = pn.state.curdoc is not None and pn.state.curdoc.session_context is not None
    return board.template() if template else board.inline()
