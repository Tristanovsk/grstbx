"""
Dashboard of the matchups of GRS with AERONET-OC (table of :func:`grstbx.validation.matchup_table`), with panel,
holoviews and bokeh: the quality criteria and the selections (stations, bands, in situ level and normalisation,
period, platforms, GRS versions) are applied interactively, and the views follow:

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
            'sunglint BRDFg': 'BRDFg', 'in situ level': 'level', 'platform': 'platform', 'GRS version': 'version'}
HOVER = ['station', 'date', 'wl', 'level', 'lag_min', 'n_valid', 'frac_valid', 'cv560', 'aot550_sat', 'BRDFg']


def _prepare(table):
    df = table.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df.time.dt.strftime('%Y-%m-%d %H:%M')
    df['wl'] = df.wl.astype(int)
    for col in ['version', 'platform', 'level', 'station']:
        df[col] = df[col].astype(str) if col in df else ''
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


def _table(df, height=300):
    return pn.widgets.Tabulator(df.reset_index(), show_index=False, height=height, sizing_mode='stretch_width',
                                disabled=True, layout='fit_data_table', theme='simple')


class MatchupDashboard:
    """
    Interactive analysis of a matchup table (see the module documentation).

    :param table: table of :func:`grstbx.validation.matchup_table` (several GRS versions can be concatenated)
    :param title: title of the page
    :param basemap: basemap of the map of the stations (:data:`grstbx.visual.BASEMAPS`)
    """

    def __init__(self, table, title='GRS L2A vs AERONET-OC', basemap='OpenStreetMap'):
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
        tmin, tmax = df.time.min().to_pydatetime(), df.time.max().to_pydatetime()
        w = pn.widgets
        self.w_versions = w.MultiChoice(name='GRS versions', options=versions, value=versions[-1:])
        self.w_stations = w.MultiChoice(name='Stations', options=stations, value=stations)
        self.w_bands = w.CheckButtonGroup(name='Bands (nm)', options=bands, value=bands, button_type='primary',
                                          button_style='outline')
        self.w_insitu = w.Select(name='In situ Rrs', options=INSITU, value='Rrs_insitu')
        self.w_levels = w.CheckButtonGroup(name='AERONET-OC level', options=['lev20', 'lev15'],
                                           value=list(QC['levels'][:1]), button_type='primary', button_style='outline')
        self.w_platforms = w.CheckButtonGroup(name='Platforms', options=platforms, value=platforms,
                                              button_type='primary', button_style='outline')
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
        self.w_driver = w.Select(name='Driver', options=DRIVERS, value='aot550_sat')
        self.w_metric = w.Select(name='Statistic of the heat map', value='MdSA (%)',
                                 options=[METRIC_LABELS.get(m, m) for m in METRICS if m != 'N'])
        self.w_scatter_mode = w.RadioButtonGroup(options=['per band', 'all bands'], value='per band',
                                                 button_type='primary', button_style='outline')
        self.filters = [self.w_versions, self.w_stations, self.w_bands, self.w_insitu, self.w_levels,
                        self.w_platforms, self.w_dates, self.w_lag, self.w_valid, self.w_cv, self.w_aot, self.w_glint]
        self.download = w.FileDownload(callback=self._csv, filename='matchups_selection.csv',
                                       label='Download the selected matchups (CSV)', button_type='success')

    # --------------------------------------------------------------------------------------------------------
    def selection(self, *_):
        """Matchups meeting the criteria and selections of the widgets; ``x``: in situ Rrs, ``res``: residual."""
        df = self.data
        df = df[df.version.isin(self.w_versions.value) & df.station.isin(self.w_stations.value)
                & df.wl.isin(self.w_bands.value) & df.platform.isin(self.w_platforms.value)]
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
        return pn.pane.Markdown(
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
        return (basemap_tiles(self.basemap) * hv.Points(points, ['x', 'y'], ['station', 'images', 'size']).opts(
            size='size', color='images', cmap='viridis', colorbar=True, tools=['hover'], line_color='black',
            clabel='images with matchups')).opts(height=420, responsive=True, xaxis=None, yaxis=None, xlim=xlim,
                                                  ylim=ylim, title='Stations (size, colour: images with matchups)')

    def view_overall(self, *_):
        df = self.selection()
        return pn.Column(pn.pane.Markdown('#### Statistics by band'), _table(_metrics(df, 'wl'), 330),
                         pn.pane.Markdown('#### Statistics by station'), _table(_metrics(df, 'station'), 400))

    # --------------------------------------------------------------------------------------------------------
    def _scatter(self, df, title, color, cmap=None, height=330, width=330):
        if df.empty:
            return hv.Curve([]).opts(title=f'{title}: no matchup', height=height, width=width)
        log = self.w_log.value
        lo = max(1e-5, float(np.nanmin([df.x.min(), df.Rrs_sat.min()]))) if log else \
            float(min(0, df.x.min(), df.Rrs_sat.min()))
        hi = float(np.nanmax([df.x.max(), df.Rrs_sat.max()])) * 1.1
        m = matchup_metrics(df.x, df.Rrs_sat) if len(df) > 2 else {}
        vdims = ['Rrs_sat', color] + [h for h in HOVER if h not in (color,)]
        scatter = hv.Scatter(df, 'x', vdims).opts(
            color=color, cmap=cmap or ('Category20' if df[color].dtype == object else 'viridis'), size=6,
            alpha=0.8, tools=['hover'], line_color='black', line_width=0.3,
            colorbar=df[color].dtype != object, show_legend=df[color].dtype == object)
        err = hv.ErrorBars(df.assign(e=df.Rrs_sat_std.fillna(0)), 'x', ['Rrs_sat', 'e']).opts(
            color='grey', alpha=0.5, line_width=0.8)
        one = hv.Curve([(lo, lo), (hi, hi)]).opts(color='black', line_dash='dashed', line_width=1)
        layers = err * scatter * one
        if m:
            xs = np.linspace(lo, hi, 50)
            layers *= hv.Curve((xs, m['slope'] * xs + m['intercept'])).opts(color='red', line_width=1.2)
            # bottom right: the points are along the 1:1 line
            layers *= hv.Text(hi * 0.97 if not log else hi * 0.8, lo + 0.03 * (hi - lo) if not log else lo * 1.3,
                              _stats_text(m), halign='right', valign='bottom', fontsize=8)
        return layers.opts(title=title, xlim=(lo, hi), ylim=(lo, hi), logx=log, logy=log, height=height,
                           width=width, xlabel='Rrs AERONET-OC (sr-1)', ylabel='Rrs GRS (sr-1)',
                           show_legend=df[color].dtype == object, legend_position='top_left')

    def view_scatter(self, *_):
        df = self.selection()
        color = self.w_color.value
        if self.w_scatter_mode.value == 'all bands':
            return self._scatter(df, 'All bands', color, height=620, width=680)
        if color == 'wl':          # one band per panel: the stations are coloured instead
            color = 'station'
        plots = [self._scatter(group, f'{wl} nm', color) for wl, group in df.groupby('wl')]
        if not plots:
            return pn.pane.Markdown('No matchup')
        return hv.Layout(plots).cols(3).opts(shared_axes=False)

    # --------------------------------------------------------------------------------------------------------
    def view_statistics(self, *_):
        df = self.selection()
        by_both = _metrics(df, ['station', 'wl'])
        metric = self.w_metric.value
        heat = pn.pane.Markdown('')
        if len(by_both) > 1 and metric in by_both:
            cells = by_both.drop(index='all', errors='ignore').reset_index()
            first = cells.columns[0]
            cells[['station', 'wl']] = cells[first].str.split(' / ', expand=True)
            cells['wl'] = cells.wl.astype(int)
            heat = hv.HeatMap(cells, ['wl', 'station'], [metric, 'N']).opts(
                cmap='RdBu_r' if metric in ('bias', 'SSPB (%)', 'intercept') else 'viridis', colorbar=True,
                tools=['hover'], height=40 + 22 * cells.station.nunique(), responsive=True,
                title=f'{metric} by station and band', xlabel='band (nm)',
                symmetric=metric in ('bias', 'SSPB (%)', 'intercept'))
        return pn.Column(heat, pn.pane.Markdown('#### By band'), _table(_metrics(df, 'wl'), 330),
                         pn.pane.Markdown('#### By station'), _table(_metrics(df, 'station'), 400),
                         pn.pane.Markdown('#### By station and band'), _table(by_both, 500))

    def view_errors(self, *_):
        df = self.selection()
        if df.empty:
            return pn.pane.Markdown('No matchup')
        opts = dict(tools=['hover'], height=350, responsive=True, ylabel='Rrs GRS - AERONET-OC (sr-1)')
        by_band = hv.BoxWhisker(df, 'wl', 'res').opts(title='Residuals by band', xlabel='band (nm)',
                                                     box_fill_color='#9ecae1', **opts)
        by_station = hv.BoxWhisker(df, 'station', 'res').opts(title='Residuals by station', xrotation=45,
                                                             box_fill_color='#fdae6b', **opts)
        rel = df[df.x > 0].assign(rel=lambda d: 100 * d.res / d.x)
        by_band_rel = hv.BoxWhisker(rel, 'wl', 'rel').opts(
            title='Relative residuals by band', xlabel='band (nm)', ylabel='(GRS - AERONET-OC) / AERONET-OC (%)',
            box_fill_color='#a1d99b', tools=['hover'], height=350, responsive=True)
        hists = []
        for wl, group in df.groupby('wl'):
            counts, edges = np.histogram(group.res, bins=30)
            hists.append(hv.Histogram((edges, counts), label=f'{wl} nm').opts(alpha=0.4))
        hist = hv.Overlay(hists).opts(title='Distribution of the residuals', height=350, responsive=True,
                                      xlabel='Rrs GRS - AERONET-OC (sr-1)', legend_position='right')
        return pn.Column(by_band, by_band_rel, by_station, hist)

    # --------------------------------------------------------------------------------------------------------
    def view_spectra(self, *_):
        df = self.selection()
        if df.empty:
            return pn.pane.Markdown('No matchup')
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
                return pn.pane.Markdown('Select a matchup in the table')
            row = images.iloc[selection[0]]
            sel = df[df.image == row.image].sort_values('wl')
            sat = hv.Curve(sel, 'wl', 'Rrs_sat', label='GRS (median)').opts(color='#1f77b4', line_width=2)
            spread = hv.Spread(sel.assign(e=sel.Rrs_sat_std.fillna(0)), 'wl', ['Rrs_sat', 'e'],
                               label='GRS +- std').opts(color='#1f77b4', alpha=0.2)
            ins = hv.Scatter(sel, 'wl_insitu', 'x', label='AERONET-OC').opts(color='black', size=8, marker='diamond')
            wls = pd.concat([sel.wl, sel.wl_insitu])
            plot = (spread * sat * ins).opts(
                xlim=(wls.min() - 20, wls.max() + 20),
                title=f'{row.station} {row.date} (lag {row.lag_min:.0f} min, {row.level})', height=380,
                responsive=True, xlabel='wavelength (nm)', ylabel='Rrs (sr-1)', legend_position='top_right',
                show_grid=True)
            folder = Path(row.l2a).parent
            name = Path(row.l2a).stem
            pngs = [pn.pane.PNG(str(folder / f'{name}_ql_{kind}.png'), height=260, sizing_mode='fixed')
                    for kind in ('l1c_rgb', 'l2a_rgb', 'sunglint') if (folder / f'{name}_ql_{kind}.png').exists()]
            return pn.Column(plot, pn.Row(*pngs) if pngs else pn.pane.Markdown('(no quicklook)'))

        return pn.Column(chooser, pn.bind(spectrum, chooser.param.selection))

    def view_timeseries(self, *_):
        df = self.selection()
        if df.empty:
            return pn.pane.Markdown('No matchup')
        plots = []
        for wl, group in df.groupby('wl'):
            plots.append(hv.Scatter(group, 'time', ['res', 'station', 'date', 'lag_min']).opts(
                color='station', cmap='Category20', size=5, tools=['hover'], height=220, responsive=True,
                title=f'{wl} nm', ylabel='GRS - AERONET-OC', show_legend=False)
                * hv.HLine(0).opts(color='black', line_width=0.8))
        return hv.Layout(plots).cols(1).opts(shared_axes=False)

    def view_drivers(self, *_):
        df = self.selection()
        driver = self.w_driver.value
        if df.empty or driver not in df or df[driver].isna().all():
            return pn.pane.Markdown('No matchup, or no value of this driver')
        plots = []
        for wl, group in df.groupby('wl'):
            plots.append(hv.Scatter(group, driver, ['res', 'station', 'date']).opts(
                color='station', cmap='Category20', size=5, tools=['hover'], height=260, width=380,
                title=f'{wl} nm', ylabel='GRS - AERONET-OC', show_legend=False)
                * hv.HLine(0).opts(color='black', line_width=0.8))
        return hv.Layout(plots).cols(3).opts(shared_axes=False)

    # --------------------------------------------------------------------------------------------------------
    def sidebar(self):
        return [
            pn.pane.Markdown('### Selection'),
            *([self.w_versions] if len(self.w_versions.options) > 1 else []),
            self.w_stations,
            pn.pane.Markdown('Bands (nm)', margin=(0, 10)), self.w_bands,
            self.w_insitu,
            pn.pane.Markdown('AERONET-OC level', margin=(0, 10)), self.w_levels,
            *([pn.pane.Markdown('Platforms', margin=(0, 10)), self.w_platforms]
              if len(self.w_platforms.options) > 1 else []),
            self.w_dates,
            pn.pane.Markdown('### Quality criteria'),
            self.w_lag, self.w_valid, self.w_cv, self.w_aot, self.w_glint,
            pn.pane.Markdown('### Display'),
            self.w_color, self.w_log,
            self.download,
        ]

    def main(self):
        deps = self.filters
        bind = lambda f, *extra: pn.bind(f, *deps, *extra)        # noqa: E731
        tabs = pn.Tabs(
            ('Overview', pn.Column(bind(self.view_map), bind(self.view_overall))),
            ('Scatter', pn.Column(self.w_scatter_mode, bind(self.view_scatter, self.w_color, self.w_log,
                                                             self.w_scatter_mode))),
            ('Statistics', pn.Column(self.w_metric, bind(self.view_statistics, self.w_metric))),
            ('Errors', bind(self.view_errors)),
            ('Spectra', bind(self.view_spectra)),
            ('Time series', bind(self.view_timeseries)),
            ('Drivers', pn.Column(self.w_driver, bind(self.view_drivers, self.w_driver))),
            dynamic=True, sizing_mode='stretch_width')
        return pn.Column(bind(self.view_counts), tabs, sizing_mode='stretch_width')

    def template(self):
        """Web page (panel FastListTemplate), e.g. served by ``grstbx-validation dashboard``."""
        return pn.template.FastListTemplate(title=self.title, sidebar=self.sidebar(), main=[self.main()],
                                            sidebar_width=360, header_background='#1f4e79')

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
