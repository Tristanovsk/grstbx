import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from grstbx import validation as val
from grstbx.matchup import matchup_metrics


def l2a_product(n=21, rrs=0.01, version='3.1.1'):
    wl = [443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
    rng = np.random.default_rng(0)
    values = rrs * (1 + 0.01 * rng.standard_normal((len(wl), n, n)))
    values[wl.index(865)] *= 0.1
    ds = xr.Dataset({'Rrs': (('wl', 'y', 'x'), values),
                     'mask': (('y', 'x'), np.zeros((n, n), dtype='uint8')),
                     'aot550': (('y', 'x'), np.full((n, n), 0.1)),
                     'BRDFg': (('y', 'x'), np.full((n, n), 0.002))},
                    coords={'wl': wl, 'y': np.arange(n)[::-1] * 20., 'x': np.arange(n) * 20.})
    ds.attrs['version'] = version
    return ds


def test_window_mask():
    ring = val.window_mask(21, 21, 'ring', (4, 7))
    assert not ring[10, 10] and ring[10, 14] and ring[10, 17] and not ring[10, 18]
    assert int(ring.sum()) == 15 ** 2 - 7 ** 2
    assert int(val.window_mask(21, 21, 'box', 3).sum()) == 9
    with pytest.raises(ValueError):
        val.window_mask(5, 5, 'disk')


def test_extract_window_platform_and_mask():
    ds = l2a_product()
    ds['Rrs'].loc[dict(wl=865, y=ds.y[10], x=ds.x[14])] = 0.1     # bright platform pixel in the ring
    ds['mask'][10, 15] = 1                                          # invalid pixel
    out = val.extract_window(ds)
    assert out['n_window'] == 176 and out['n_valid'] == 174
    assert out['Rrs_sat_560'] == pytest.approx(0.01, rel=0.01)
    assert out['cv560'] < 0.05 and out['aot550_sat'] == pytest.approx(0.1)
    ds['mask'][:] = 1
    empty = val.extract_window(ds)
    assert empty['n_valid'] == 0 and np.isnan(empty['Rrs_sat_560'])


def test_aerosol_model():
    assert val.aerosol_model({'opac_model': 'WASO_rh0'}) == ('WASO_rh0', '{"WASO_rh0": 1.0}')
    model, mix = val.aerosol_model({'opac_model': 'WASO_rh0',
                                    'aerosol_combination': '{"MACL_rh70": 0.1, "WASO_rh0": 0.9}'})
    assert model == '0.90 WASO_rh0 + 0.10 MACL_rh70' and json.loads(mix) == {'MACL_rh70': 0.1, 'WASO_rh0': 0.9}
    assert val.aerosol_model({}) == ('', '')


def test_band_pairs():
    pairs = val.band_pairs([412, 443, 490, 555, 667, 779, 870, 1020])
    assert pairs == {443: 443, 490: 490, 560: 555, 665: 667, 783: 779, 865: 870}


def aeronet_file(path, site='SITE', t0='2024-07-01T10:00'):
    times = pd.to_datetime(t0) + pd.to_timedelta([-90, -20, 10, 40], 'min')
    wl = np.array([443., 490., 560., 667., 865.])
    rrs = np.tile([0.010, 0.011, 0.012, 0.004, 0.0005], (4, 1)).astype('float32')
    ds = xr.Dataset({'subsource': ('obs', [site] * 4), 'time': ('obs', times),
                     'qc_level': ('obs', np.array([2, 1.5, 2, 2], dtype='float32')),
                     'sza': ('obs', np.full(4, 30.)), 'vza': ('obs', np.full(4, 40.)), 'raa': ('obs', np.full(4, 90.)),
                     'wind_speed': ('obs', np.full(4, 3.)),
                     'src_aot_490': ('obs', np.full(4, 0.2)), 'src_aot_667': ('obs', np.full(4, 0.1)),
                     'Rrs': (('obs', 'wl'), rrs), 'Rrs_fQ': (('obs', 'wl'), 0.9 * rrs)},
                    coords={'wl': wl})
    ds.to_netcdf(path)
    return path


def test_read_aeronet_oc_and_nearest(tmp_path):
    path = aeronet_file(tmp_path / 'aeronet_oc.nc')
    ins = val.read_aeronet_oc('SITE', path)
    assert list(ins.level) == ['lev20', 'lev15', 'lev20', 'lev20']
    assert {'Rrs_443', 'Rrs_fQ_865', 'aot550_insitu', 'wind_insitu'} <= set(ins.columns)
    assert ins.aot550_insitu.iloc[0] == pytest.approx(0.2 * (550 / 490) ** (np.log(0.5) / np.log(667 / 490)))
    # overpass at 10:00: the Level 1.5 record at -20 min is nearer, the Level 2.0 one at +10 min is taken
    near = val._nearest(ins, [pd.Timestamp('2024-07-01T10:00'), pd.Timestamp('2024-07-02')], '1h')
    assert near.time_insitu.iloc[0] == pd.Timestamp('2024-07-01T10:10') and near.level.iloc[0] == 'lev20'
    assert pd.isna(near.time_insitu.iloc[1])
    assert val.read_aeronet_oc('SITE', path, levels=(2.0,)).level.eq('lev20').all()
    # two sites of one station measuring at the same time: no error, the Level 2.0 record taken
    twin = pd.concat([ins.iloc[[1]].assign(level='lev20', site='TWIN'), ins])
    near = val._nearest(twin, [pd.Timestamp('2024-07-01T09:40')], '1h')
    assert near.site.iloc[0] == 'TWIN' and near.level.iloc[0] == 'lev20'


def test_matchup_table_quality_metrics(tmp_path):
    insitu = aeronet_file(tmp_path / 'aeronet_oc.nc')
    subset = tmp_path / 'sentinel-2' / 'l1c' / 'SITE' / '2024' / '07' / '01' / \
        'S2A_MSIL1C_20240701T100000_N0510_R022_T32TQR_20240701T120000_SITE.nc'
    items = pd.DataFrame([{'id': subset.stem, 'time': pd.Timestamp('2024-07-01T10:00'), 'station': 'SITE',
                           'sites': ['SITE'], 'lon': 12.5, 'lat': 45.3, 'tile': '32TQR', 'platform': 'sentinel-2a',
                           'cloud_cover': 5., 'subset': str(subset)}])
    l2a = val.l2a_path(subset, '3.1.1', tmp_path)
    assert l2a.parts[-6:-2] == ('SITE', '2024', '07', '01') and l2a.name.endswith('_SITE_V3.1.1.nc')
    assert 'MSIL2AGRS' in l2a.name
    # extraction from the file: version and aerosol model read from the attributes
    product = l2a_product()
    product.attrs['opac_model'] = 'COAV_rh70'
    l2a.parent.mkdir(parents=True)
    product.to_netcdf(l2a)
    extractions = val.extract_l2a([l2a], workers=1)
    assert extractions.aerosol_model.iloc[0] == 'COAV_rh70' and extractions.version.iloc[0] == '3.1.1'
    table = val.matchup_table(items, extractions, version='3.1.1', root=tmp_path, insitu_path=insitu)
    assert list(table.wl) == [443, 490, 560, 665, 865]
    assert list(table.wl_insitu) == [443, 490, 560, 667, 865]
    assert (table.lag_min == 10).all() and (table.level == 'lev20').all()
    assert (table.aerosol_model == 'COAV_rh70').all()
    assert val.add_aerosol_model(table.drop(columns=['aerosol_model'])).aerosol_model.eq('COAV_rh70').all()
    assert table.Rrs_insitu_fQ.iloc[0] == pytest.approx(0.009)
    assert table.Rrs_insitu.iloc[0] == pytest.approx(0.010)
    assert table.residual.iloc[0] == pytest.approx(table.Rrs_sat.iloc[0] - table.Rrs_insitu.iloc[0])
    assert val.quality(table).all()
    # per record, the nearest in situ band with a value: 555 nm when 560 nm is empty (other site of the station)
    ins = xr.open_dataset(insitu).load()
    wl = np.array([443., 490., 555., 560., 667., 865.])
    for var in ('Rrs', 'Rrs_fQ'):
        values = np.insert(ins[var].values, 2, ins[var].values[:, 2], axis=1)
        values[2, 3] = np.nan          # 560 nm missing in the matched record (+10 min) only
        ins = ins.drop_vars(var).assign({var: (('obs', 'wl2'), values)})
    ins = ins.drop_vars('wl').rename({'wl2': 'wl'}).assign_coords(wl=wl)
    ins.to_netcdf(tmp_path / 'aeronet_oc_555.nc')
    table2 = val.matchup_table(items, extractions, version='3.1.1', root=tmp_path,
                               insitu_path=tmp_path / 'aeronet_oc_555.nc')
    assert list(table2.wl_insitu) == [443, 490, 555, 667, 865] and table2.wl_insitu.dtype.kind == 'i'
    assert table2.Rrs_insitu.notna().all()
    assert not val.quality(table, max_lag_min=5).any()
    assert not val.quality(table, levels=('lev15',)).any()
    metrics = val.metrics_table(table, by='wl', min_n=1)
    assert metrics.loc['all', 'N'] == 5 and 443 in metrics.index
    # round trip
    path = val.save_table(table, tmp_path / 'matchups.csv')
    back = val.read_table(path)
    assert back.time.dtype.kind == 'M' and len(back) == len(table) and back.sites.iloc[0] == 'SITE'


def test_matchup_table_without_l2a(tmp_path):
    items = pd.DataFrame([{'id': 'x', 'time': pd.Timestamp('2024-07-01'), 'station': 'SITE', 'sites': ['SITE'],
                           'subset': str(tmp_path / 'sentinel-2/l1c/SITE/2024/07/01/x_SITE.nc')}])
    assert val.matchup_table(items, version='3.1.1', root=tmp_path, verbose=False).empty


def test_matchup_metrics_log_and_ape():
    x = np.array([0.01, 0.02, 0.04])
    m = matchup_metrics(x, 2 * x)
    assert m['bias_log'] == pytest.approx(2) and m['mae_log'] == pytest.approx(2)
    assert m['rmse_log'] == pytest.approx(np.log10(2)) and m['medape'] == pytest.approx(100)
    assert m['mae'] == pytest.approx(np.mean(x))


def test_read_items(tmp_path):
    folder = tmp_path / 'stac' / 's2-l1c-aeronet-oc'
    folder.mkdir(parents=True)
    item = {'id': 'x_SITE', 'properties': {'datetime': '2024-07-01T10:00:00Z', 'aeronet_oc:station': 'SITE',
                                           'aeronet_oc:sites': ['SITE'], 'grid:code': 'MGRS-32TQR'},
            'assets': {'data': {'href': 's3://aeronet-oc/sentinel-2/l1c/SITE/2024/07/01/x_SITE.nc'}}}
    (folder / 'items.json').write_text(json.dumps({'type': 'FeatureCollection', 'features': [item]}))
    (tmp_path / 'stac' / 'catalog.json').write_text('{}')
    items = val.read_items(tmp_path / 'stac', root=tmp_path)
    assert items.subset.iloc[0] == str(tmp_path / 'sentinel-2/l1c/SITE/2024/07/01/x_SITE.nc')
    assert items.tile.iloc[0] == '32TQR'
    assert val.read_items(tmp_path / 'stac', stations=['OTHER'], root=tmp_path).empty


def synthetic_table(n=40):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        for wl in (443, 490, 560, 665, 865):
            x = 0.01 * rng.uniform(0.2, 2)
            rows.append({'station': 'A' if i % 2 else 'B', 'image': f'img{i}', 'l2a': f'/nowhere/img{i}.nc',
                         'time': pd.Timestamp('2020-01-01') + pd.Timedelta(days=10 * i), 'wl': wl, 'wl_insitu': wl,
                         'Rrs_sat': x * 1.05 + 1e-4 * rng.standard_normal(), 'Rrs_sat_std': 1e-4,
                         'Rrs_insitu': x, 'Rrs_insitu_fQ': 0.9 * x, 'level': 'lev20' if i % 3 else 'lev15',
                         'lag_min': 10. * (i % 7), 'n_valid': 150, 'frac_valid': 0.9, 'cv560': 0.05,
                         'aot550_sat': 0.1, 'BRDFg': 0.001, 'platform': 'sentinel-2a', 'version': '3.1.1',
                         'lon': 12.5 + (i % 2), 'lat': 45.3, 'sza': 40., 'site': 'A',
                         'aerosol_model': 'WASO_rh0' if i % 4 else 'COAV_rh70'})
    return pd.DataFrame(rows)


def test_dashboard_views():
    pytest.importorskip('panel')
    from grstbx.validation_app import MatchupDashboard

    board = MatchupDashboard(synthetic_table())
    selection = board.selection()
    assert set(selection.level) == {'lev20'} and (selection.lag_min <= 60).all()
    board.w_levels.value = ['lev20', 'lev15']
    assert len(board.selection()) > len(selection)
    for view in (board.view_counts, board.view_overall, board.view_scatter, board.view_statistics,
                 board.view_errors, board.view_spectra, board.view_timeseries, board.view_drivers):
        assert view() is not None
    board.w_scatter_mode.value = 'all bands'
    board.w_color.value = 'station'
    board.w_log.value = True
    assert board.view_scatter() is not None
    board.w_color.value = 'aerosol_model'
    assert board.view_scatter() is not None and board.view_statistics() is not None
    # colormaps and opacity; square frames of the same size in the grid of the bands, same colours in all panels
    import holoviews as hv
    from grstbx.validation_app import FRAME, _palette
    board.w_scatter_mode.value, board.w_log.value = 'per band', False
    for color, cmap in [('station', 'earth'), ('aot550_sat', 'earth gradient'), ('station', 'Set1'),
                        ('aot550_sat', 'turbo')]:
        board.w_color.value, board.w_cmap.value, board.w_alpha.value = color, cmap, 0.5
        grid = hv.render(board.view_scatter().object)
        figs = [child[0] for child in grid.children]
        for fig in figs:
            assert fig.height - fig.min_border_top - fig.min_border_bottom == FRAME
            assert fig.width - fig.min_border_left - fig.min_border_right == FRAME
        assert board.view_timeseries() is not None and board.view_drivers() is not None
    palette = _palette(board.selection().station, 'earth')
    assert len(set(palette.values())) == len(palette)
    board.w_aerosols.value = ['COAV_rh70']
    assert set(board.selection().aerosol_model) == {'COAV_rh70'}
    board.w_stations.value = []
    assert board.selection().empty and board.view_scatter() is not None and board.view_errors() is not None
    assert board.inline() is not None and board.template() is not None
    assert board._csv().getvalue().startswith('station,')
