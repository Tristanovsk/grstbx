import json
import os
import sys
from pathlib import Path

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


def gaussian_srf(bands=(443, 490, 560, 665, 865), fwhm=20.):
    wl_hr = np.arange(400, 1000)
    sigma = fwhm / 2.355
    values = np.exp(-0.5 * ((wl_hr[None] - np.array(bands)[:, None]) / sigma) ** 2)
    return xr.DataArray(values, dims=('wl', 'wl_hr'), coords={'wl': list(bands), 'wl_hr': wl_hr})


def hypernets_file(path, site='BEFR', t0='2024-07-01T10:00'):
    times = pd.to_datetime(t0) + pd.to_timedelta([-90, -20, 10, 40], 'min')
    wl = np.arange(350., 1101.)
    rrs = np.tile(0.01 + 1e-5 * (wl - 350), (4, 1)).astype('float32')     # linear: band value = Rrs(centre)
    rrs[:, wl > 900] = np.nan                                               # spectra stop at 900 nm
    ds = xr.Dataset({'subsource': ('obs', [f'{site}_H_HYPSTAR_v1.2_zenodo'] * 3 + ['OTHR_H_HYPSTAR_v1.2_zenodo']),
                     'time': ('obs', times), 'qc_flag': ('obs', np.array([0, 1, 0, 0], dtype='int8')),
                     'sza': ('obs', np.full(4, 30.)), 'vza': ('obs', np.full(4, 40.)), 'raa': ('obs', np.full(4, 135.)),
                     'wind_speed': ('obs', np.full(4, 3.)), 'Rrs': (('obs', 'wl'), rrs)},
                    coords={'wl': wl})
    ds.to_netcdf(path)
    return path


def test_convolve_srf_and_read_hypernets(tmp_path):
    srf = val.band_srf(gaussian_srf(bands=(443, 560, 865, 945)))
    assert np.allclose(srf.sum('wl_hr'), 1)
    path = hypernets_file(tmp_path / 'hypernets.nc')
    dir_norm = xr.DataArray(np.full((3, 2), 1.1), dims=('sza', 'wl'), coords={'sza': [10, 30, 60], 'wl': [350, 1100]})
    ins = val.read_hypernets('BEFR', path, srf=srf, dir_norm=dir_norm)
    assert len(ins) == 2 and set(ins.site) == {'BEFR'} and set(ins.level) == {'qc0'}    # qc_flag 1, other site out
    assert ins.Rrs_560.iloc[0] == pytest.approx(0.01 + 1e-5 * 210, rel=1e-4)
    assert ins.Rrs_nadir_560.iloc[0] == pytest.approx(1.1 * ins.Rrs_560.iloc[0])
    assert 'Rrs_945' not in ins or ins.Rrs_945.isna().all()                   # band beyond the in situ spectra
    assert len(val.read_hypernets('BEFR', path, srf=srf, qc_flags=(0, 1))) == 3
    with pytest.raises(ValueError):
        val.read_hypernets('BEFR', path)


def test_matchup_table_hypernets_reader(tmp_path):
    import functools

    insitu = hypernets_file(tmp_path / 'hypernets.nc')
    subset = tmp_path / 'sentinel-2' / 'l1c' / 'BEFR' / '2024' / '07' / '01' / \
        'S2A_MSIL1C_20240701T100000_N0510_R108_T31TFJ_20240701T120000_BEFR.nc'
    items = pd.DataFrame([{'id': subset.stem, 'time': pd.Timestamp('2024-07-01T10:00'), 'station': 'BEFR',
                           'sites': ['BEFR'], 'lon': 5.1, 'lat': 43.4, 'tile': '31TFJ', 'platform': 'sentinel-2a',
                           'cloud_cover': 5., 'subset': str(subset)}])
    l2a = val.l2a_path(subset, '3.1.1', tmp_path)
    l2a.parent.mkdir(parents=True)
    l2a_product().to_netcdf(l2a)
    extractions = val.extract_l2a([l2a], workers=1)
    dir_norm = xr.DataArray(np.full((2, 2), 0.95), dims=('sza', 'wl'), coords={'sza': [10, 60], 'wl': [350, 1100]})
    reader = functools.partial(val.read_hypernets, path=insitu, srf=gaussian_srf(), dir_norm=dir_norm)
    table = val.matchup_table(items, extractions, version='3.1.1', root=tmp_path, reader=reader)
    assert list(table.wl) == [443, 490, 560, 665, 865] and (table.wl == table.wl_insitu).all()
    assert (table.lag_min == 10).all() and (table.level == 'qc0').all()
    assert np.allclose(table.Rrs_insitu_nadir, 0.95 * table.Rrs_insitu)
    assert 'Rrs_insitu_fQ' not in table
    assert not val.quality(table).any()                 # default levels: AERONET-OC
    assert val.quality(table, levels=None).all() and val.quality(table, levels=('qc0',)).all()


def test_dashboard_hypernets_labels():
    pytest.importorskip('panel')
    from grstbx.validation_app import MatchupDashboard

    table = synthetic_table().drop(columns='Rrs_insitu_fQ').assign(level='qc0')
    table['Rrs_insitu_nadir'] = 0.95 * table.Rrs_insitu
    insitu = {'HYPERNETS (as measured)': 'Rrs_insitu', 'HYPERNETS at nadir': 'Rrs_insitu_nadir',
              'f/Q': 'Rrs_insitu_fQ'}
    board = MatchupDashboard(table, insitu=insitu, network='HYPERNETS')
    assert list(board.w_insitu.options) == ['HYPERNETS (as measured)', 'HYPERNETS at nadir']
    assert board.w_levels.options == ['qc0'] and board.w_levels.value == ['qc0']
    assert len(board.selection()) > 0
    board.w_insitu.value = 'Rrs_insitu_nadir'
    assert np.allclose(board.selection().x, board.selection().Rrs_insitu_nadir)
    for view in (board.view_scatter, board.view_errors, board.view_spectra):
        assert view() is not None


def test_run_grs_fmin_error_is_no_water(tmp_path, monkeypatch):
    import sys
    import types

    class Process:
        no_water = False

        def execute(self, *args, **kwargs):
            raise ValueError(self.message)

    fake = types.ModuleType('grs')
    fake.Process = Process
    monkeypatch.setitem(sys.modules, 'grs', fake)
    out = tmp_path / 'l2a' / 'x' / 'x.nc'
    Process.message = 'zero-size array to reduction operation fmin which has no identity'
    (out.parent).mkdir(parents=True)
    (tmp_path / 'l2a' / 'x' / 'x.err').write_text('old failure')
    assert val.run_grs('subset.nc', 'cams.nc', out) == 'no_water'
    assert val.l2a_state(out) == 'no_water' and not (out.parent / 'x.err').exists()
    assert 'fmin' in json.loads((out.parent / 'x_no_water.txt').read_text())['error']
    # any other error: .err file and raised again
    out2 = tmp_path / 'l2a' / 'y' / 'y.nc'
    Process.message = 'something else'
    with pytest.raises(ValueError):
        val.run_grs('subset.nc', 'cams.nc', out2)
    assert val.l2a_state(out2) == 'failed'


FAKE_GRS = '''
import os, signal
from pathlib import Path
import numpy as np
import xarray as xr
__version__ = '3.1.1'


class Process:
    no_water = False

    def execute(self, subset, cams_file=None, resolution=20, **options):
        Path(os.environ['FAKE_PIDS']).open('a').write(f'{Path(subset).stem} {os.getpid()}\\n')
        if 'KILL' in subset:
            os.kill(os.getpid(), signal.SIGKILL)       # as by the system when it runs out of memory
        if 'FAIL' in subset:
            raise RuntimeError('GRS failed')

    def write_output(self):
        n = 21
        ds = xr.Dataset({'Rrs': (('wl', 'y', 'x'), np.full((2, n, n), 0.01)),
                         'mask': (('y', 'x'), np.zeros((n, n), 'uint8'))},
                        coords={'wl': [560, 865], 'y': 4800000. - 20 * np.arange(n), 'x': 600000. + 20 * np.arange(n)})
        ds.rio.write_crs(32632).assign_attrs(version=__version__, acquisition_date='2024-07-01T10:00:00Z').to_netcdf(self.odir)
'''


def test_process_subsets_batched(tmp_path, monkeypatch):
    """Subsets processed several per process; the images of a killed process run again one per process."""
    pkg = tmp_path / 'fake_pkgs' / 'grs'
    pkg.mkdir(parents=True)
    (pkg / '__init__.py').write_text(FAKE_GRS)
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(tmp_path / 'fake_pkgs')] + sys.path))
    monkeypatch.setenv('FAKE_PIDS', str(tmp_path / 'pids.txt'))
    cams = tmp_path / 'cams'
    cams.mkdir()
    (cams / 'cams_forecast_2024-07.nc').write_text('')
    rows = []
    for k, tag in enumerate(['A', 'B', 'KILL', 'C', 'FAIL']):
        subset = tmp_path / 'sentinel-2' / 'l1c' / 'SITE' / '2024' / '07' / f'0{k + 1}' / \
            f'S2A_MSIL1C_2024070{k + 1}T100000_N0510_R022_T32TQR_2024070{k + 1}T120000_{tag}_SITE.nc'
        rows.append({'subset': str(subset), 'time': pd.Timestamp(f'2024-07-0{k + 1}T10:00')})
    runs = val.process_subsets(pd.DataFrame(rows), version='3.1.1', root=tmp_path, cams_dir=cams, workers=1,
                               images_per_process=2, verbose=False)
    assert runs.state.tolist() == ['done', 'done', 'failed', 'done', 'failed']
    assert 'status -9' in Path(runs.l2a[2][:-3] + '.err').read_text()          # killed, alone, by SIGKILL
    assert 'GRS failed' in Path(runs.l2a[4][:-3] + '.err').read_text()
    pids = {}
    for line in (tmp_path / 'pids.txt').read_text().split('\n'):
        if line:
            name, pid = line.split()
            pids.setdefault(name.split('_')[-2], []).append(pid)
    assert pids['A'] == pids['B']                       # same process for the images of a batch
    assert pids['C'][0] not in pids['KILL']             # left by the killed process: run alone
    assert len(pids['KILL']) == 2                       # in its batch, then alone
    # nothing left to run
    again = val.process_subsets(pd.DataFrame(rows), version='3.1.1', root=tmp_path, cams_dir=cams, verbose=False)
    assert again.state.tolist() == ['done', 'done', 'failed', 'done', 'failed']
