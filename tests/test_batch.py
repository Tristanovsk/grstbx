"""Batch processing helpers (grstbx.batch) on synthetic product trees: no processing, GRS and GRSl2bgen
are replaced by stand-ins."""
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from grstbx import batch

matplotlib.use('Agg')

L1C = 'S2A_MSIL1C_20250801T104041_N0511_R008_T31TEJ_20250801T131318.SAFE'
L2A = 'S2A_MSIL2AGRS_20250801T104041_N0511_R008_T31TEJ_20250801T131318_V3.1.0'
L2B = 'S2A_MSIL2B_20250801T104041_N0511_R008_T31TEJ_20250801T131318_V3.1.0'


def make_level1(root, tile, date, name, cloud):
    path = root / tile / date.replace('-', '/') / name
    path.mkdir(parents=True)
    (path / 'MTD_MSIL1C.xml').write_text(f'<a><Cloud_Coverage_Assessment>{cloud}</Cloud_Coverage_Assessment></a>')
    return path


def make_zarr(store, levels=2, complete=True):
    """Zarr L2A store with the root metadata and the Rrs and mask arrays of each level."""
    layout = [{'group': str(k), 'scale': 2**k} for k in range(levels)]
    store.mkdir(parents=True)
    (store / 'zarr.json').write_text(json.dumps({'attributes': {'multiscales': [{'layout': layout}]},
                                                 'zarr_format': 3, 'node_type': 'group'}))
    for k in range(levels):
        for var in ['.', 'Rrs', 'mask']:
            if not complete and k == levels - 1 and var == 'mask':
                continue
            (store / str(k) / var).mkdir(parents=True, exist_ok=True)
            (store / str(k) / var / 'zarr.json').write_text('{}')
    return store


def make_netcdf(folder, name, variables=('Rrs', 'mask')):
    folder.mkdir(parents=True)
    ds = xr.Dataset({v: (('y', 'x'), np.zeros((2, 2))) for v in variables})
    ds.to_netcdf(folder / f'{name}.nc')
    xr.Dataset({'aod550': (('y', 'x'), np.zeros((2, 2)))}).to_netcdf(folder / f'{name}_anc.nc')
    return folder


@pytest.fixture
def level1(tmp_path):
    """Three Level-1 images of tile 31TEJ, the last one cloudy."""
    root = tmp_path / 'L1C'
    make_level1(root, '31TEJ', '2025-08-01', L1C, 35.2)
    make_level1(root, '31TEJ', '2025-08-01', L1C.replace('S2A', 'S2B').replace('R008', 'R108'), 16.2)
    make_level1(root, '31TEJ', '2025-08-02', L1C.replace('20250801T104041', '20250802T105051'), 82.5)
    cams = tmp_path / 'cams'
    cams.mkdir()
    (cams / 'cams_2025-08.nc').write_text('')
    return root


def test_names():
    assert batch.l2a_name(L1C, suffix='_V3.1.0') == L2A
    assert batch.l2a_name('LC08_L1TP_199027_20250801_20250805_02_T1', suffix='_V2.2.0') == \
        'LC08_L2AGRS_199027_20250801_20250805_02_T1_V2.2.0'
    assert batch.l2b_name(f'/data/{L2A}.zarr') == L2B
    assert batch.l2b_name(f'/data/{L2A.replace("L2AGRS", "L2A")}/') == L2B
    assert batch.image_name(f'{L2A}.zarr') == L2A and batch.image_name(L1C) == L1C[:-5]
    assert batch.item_path(f'/out/{L2B}/{L2B}.nc') == f'/out/{L2B}/{L2B}.json'


def test_list_images(level1):
    images = batch.list_images(level1, ['31TEJ'])
    assert len(images) == 3 and list(images.columns) == ['tile', 'name', 'path', 'cloud_cover']
    assert images.index[0] == pd.Timestamp('2025-08-01')
    assert images['cloud_cover'].round(3).tolist() == [0.352, 0.162, 0.825]
    assert len(images.loc['2025-08-02':]) == 1


def test_complete_products(tmp_path):
    store = make_zarr(tmp_path / L2A / f'{L2A}.zarr')
    assert batch.l2a_format(tmp_path / L2A) == 'zarr'
    assert batch.zarr_complete(store) and batch.l2a_complete(tmp_path / L2A)
    partial = make_zarr(tmp_path / 'partial' / 'partial.zarr', complete=False)
    assert not batch.zarr_complete(partial)
    assert not batch.l2a_complete(tmp_path / 'missing.zarr')

    name = L2A.replace('L2AGRS', 'L2A')
    folder = make_netcdf(tmp_path / 'nc' / name, name)
    assert batch.l2a_format(folder) == 'nc' and batch.l2a_complete(folder)
    assert batch.l2a_complete(folder / f'{name}.nc')
    (folder / f'{name}_anc.nc').write_bytes(b'truncated')
    assert not batch.netcdf_complete(folder)
    other = make_netcdf(tmp_path / 'nc2' / name, name, variables=('Rrs',))
    assert not batch.l2a_complete(other)

    l2b = tmp_path / L2B / f'{L2B}.nc'
    l2b.parent.mkdir()
    l2b.write_text('')
    assert batch.l2b_complete(l2b, stac=False) and not batch.l2b_complete(l2b)
    (tmp_path / L2B / f'{L2B}.json').write_text('{}')
    assert batch.l2b_complete(l2b)


def test_list_l2a_products(tmp_path):
    day = tmp_path / '31TEJ' / '2025' / '08' / '01'
    make_zarr(day / L2A / f'{L2A}.zarr')
    old = L2A.replace('V3.1.0', 'V3.0.0')
    make_zarr(day / old / f'{old}.zarr', complete=False)
    products = batch.list_l2a_products(tmp_path)
    assert len(products) == 2 and products['complete'].tolist().count(True) == 1
    assert set(products['version']) == {'V3.1.0', 'V3.0.0'} and set(products['tile']) == {'31TEJ'}
    latest = batch.select_versions(products)
    assert latest['version'].tolist() == ['V3.1.0']
    assert batch.select_versions(products, 'V3.0.0')['version'].tolist() == ['V3.0.0']


def new_batch(tmp_path, **kwargs):
    options = dict(cams_file_template=str(tmp_path / 'cams' / 'cams_{date:%Y-%m}.nc'),
                   max_cloud_cover=0.6, version='3.1.0')
    return batch.Batch(str(tmp_path / 'L2A'), str(tmp_path / 'L2B'), **options | kwargs)


def test_plan_level1(tmp_path, level1):
    images = batch.list_images(level1, ['31TEJ'])
    run = new_batch(tmp_path)
    plan = run.plan_level1(images)
    assert plan['l2a'].iloc[0] == str(tmp_path / 'L2A' / '31TEJ' / '2025/08/01' / L2A / f'{L2A}.zarr')
    assert plan['l2b'].iloc[0] == str(tmp_path / 'L2B' / '31TEJ' / '2025/08/01' / L2B / f'{L2B}.nc')
    assert plan['profile'].iloc[0].endswith(f'profiles/31TEJ/{L1C[:-5]}_profile.json')
    assert plan['cams_ok'].all() and plan['cloudy'].tolist() == [False, False, True]
    assert plan['run_l2a'].all() and plan['run_l2b'].all()
    assert 'L2A: 0 done, 2 to compute, 1 too cloudy' in run.describe(plan)

    # L2A of the first image done: reused; L2B of the second done: nothing to run
    make_zarr(tmp_path / 'L2A' / '31TEJ' / '2025/08/01' / L2A / f'{L2A}.zarr')
    l2b = plan['l2b'].iloc[1]
    os.makedirs(os.path.dirname(l2b))
    open(batch.item_path(l2b), 'w').close()
    plan = run.plan_level1(images)
    assert plan[['l2a_done', 'l2b_done', 'run_l2a', 'run_l2b']].values.tolist() == \
        [[True, False, False, True], [False, True, False, False], [False, False, True, True]]
    plan = new_batch(tmp_path, overwrite_l2b=True).plan_level1(images)
    assert plan['run_l2b'].all() and plan['run_l2a'].tolist() == [False, True, True]
    plan = new_batch(tmp_path, overwrite_l2a=True).plan_level1(images)
    assert plan['run_l2a'].all() and plan['run_l2b'].all()

    # L2A only, GRS 2.x: NetCDF products, no L2B step
    run = batch.Batch(str(tmp_path / 'L2A_nc'), version='2.2.0')
    plan = run.plan_level1(images)
    assert plan['l2a'].iloc[0].endswith(f'{L2A[:-6]}V2.2.0.nc') and 'l2b' not in plan
    assert plan['run_l2a'].all() and not plan['run_l2b'].any() and plan['cams_ok'].all()


def test_plan_l2a(tmp_path):
    day = tmp_path / 'L2A' / '31TEJ' / '2025' / '08' / '01'
    make_zarr(day / L2A / f'{L2A}.zarr')
    old = L2A.replace('20250801T104041', '20250801T102559')
    make_zarr(day / old / f'{old}.zarr', complete=False)
    run = batch.Batch(l2b_dir=str(tmp_path / 'L2B'))
    plan = run.plan_l2a(batch.list_l2a_products(tmp_path / 'L2A'))
    assert plan['l2b'].tolist()[1] == str(tmp_path / 'L2B' / '31TEJ' / '2025/08/01' / L2B / f'{L2B}.nc')
    assert plan['run_l2b'].tolist() == [False, True] and not plan['run_l2a'].any()
    with pytest.raises(ValueError):
        batch.Batch(str(tmp_path / 'L2A')).plan_l2a(batch.list_l2a_products(tmp_path / 'L2A'))


@pytest.fixture
def stub_processors(monkeypatch):
    """GRS and GRSl2bgen replaced by stand-ins writing complete products; failures on request."""
    calls, fail = [], set()

    def run_grs(level1, l2a, **kwargs):
        calls.append('L2A')
        if 'L2A' in fail:
            raise RuntimeError('GRS failed')
        make_zarr(Path(l2a))

    def run_grsl2bgen(l2a, l2b, **kwargs):
        calls.append('L2B')
        if 'L2B' in fail:
            raise RuntimeError('GRSl2bgen failed')
        os.makedirs(os.path.dirname(l2b), exist_ok=True)
        open(l2b, 'w').close()
        open(batch.item_path(l2b), 'w').close()

    monkeypatch.setattr(batch, 'run_grs', run_grs)
    monkeypatch.setattr(batch, 'run_grsl2bgen', run_grsl2bgen)
    return calls, fail


def test_run(tmp_path, level1, stub_processors, capsys):
    calls, fail = stub_processors
    images = batch.list_images(level1, ['31TEJ'])
    run = new_batch(tmp_path, memory_interval=0.05)
    plan = run.plan_level1(images)
    summary = run.run(plan)
    assert summary['status'].tolist() == ['done', 'done', 'cloudy']
    assert summary['l2a'].tolist() == ['done', 'done', ''] and calls == ['L2A', 'L2B'] * 2
    assert '[3/3] cloudy' in capsys.readouterr().out
    profiles = batch.read_profiles(plan['profile'])
    assert profiles['step'].tolist() == ['L2A', 'L2B'] * 2 and (profiles['status'] == 'done').all()
    assert (profiles['rss_peak_mb'] >= profiles['rss_start_mb']).all()
    profile = batch.read_profile(plan['profile'].iloc[0])
    assert profile['status'] == 'done' and profile['l1'] == plan['path'].iloc[0] and profile['memory']
    assert batch.plot_memory(plan['profile'].iloc[0]).get_title() == L1C[:-5]

    # second run: all done or cloudy; then the L2B products again, from the existing L2A products
    calls.clear()
    assert run.run(run.plan_level1(images), verbose=False)['status'].tolist() == ['skipped', 'skipped', 'cloudy']
    rerun = new_batch(tmp_path, overwrite_l2b=True)
    summary = rerun.run(rerun.plan_level1(images), verbose=False)
    assert summary['l2a'].tolist() == ['reused', 'reused', ''] and calls == ['L2B', 'L2B']

    # failures: L2B not run after a failed L2A, a failed L2B keeps its L2A
    calls.clear()
    fail.add('L2A')
    failing = new_batch(tmp_path, overwrite_l2a=True, max_cloud_cover=1)
    record = failing.process(failing.plan_level1(images).iloc[2])
    assert record['status'] == 'L2A failed' and 'GRS failed' in record['error'] and calls == ['L2A']
    steps = batch.read_profile(failing.plan_level1(images)['profile'].iloc[2])['steps']
    assert [(s['step'], s['status']) for s in steps] == [('L2A', 'failed'), ('L2B', 'not run')]
    fail.clear()
    fail.add('L2B')
    record = rerun.process(rerun.plan_level1(images).iloc[0])
    assert record['status'] == 'L2B failed' and record['l2a'] == 'reused'

    path = run.save_summary(summary, prefix='test')
    assert os.path.basename(path).startswith('test_') and len(pd.read_csv(path)) == 3


def test_no_cams(tmp_path, level1, stub_processors):
    images = batch.list_images(level1, ['31TEJ'])
    run = new_batch(tmp_path, cams_file_template=str(tmp_path / 'cams' / 'missing_{date:%Y}.nc'))
    record = run.process(run.plan_level1(images).iloc[0])
    assert record['status'] == 'no_cams' and 'missing_2025.nc' in record['error']


def test_profiler_step_failure(tmp_path):
    profiler = batch.Profiler('image.SAFE', interval=0.01)
    with pytest.raises(ZeroDivisionError), profiler.step('L2A'):
        1 / 0
    profiler.skip('L2B', 'not run')
    path = profiler.write(str(tmp_path / 'p' / 'image_profile.json'), status='L2A failed')
    profile = batch.read_profile(path)
    assert [s['status'] for s in profile['steps']] == ['failed', 'not run']
    assert profile['steps'][0]['wall_s'] >= 0 and profile['cpus'] == os.cpu_count()


def test_commands(tmp_path, level1):
    images = batch.list_images(level1, ['31TEJ'])
    run = new_batch(tmp_path, aot_resolution=480)
    commands = run.commands(run.plan_level1(images))
    assert len(commands) == 2                              # the cloudy image is left out
    line = commands[0]
    assert line.startswith(f'[ -e {tmp_path}/L2B/31TEJ/2025/08/01/{L2B}/{L2B}.json ] || {{')
    assert '--aot_resolution 480' in line and '--no_clobber' in line and '--cams_file' in line
    assert f'GRSl2bgen {tmp_path}/L2A/31TEJ/2025/08/01/{L2A} -o {tmp_path}/L2B/31TEJ/2025/08/01/{L2B}.nc' in line
    assert f'{L2A}.zarr/0/mask/zarr.json ] && /usr/bin/time -v' in line
    assert '_profile_L2A.txt grs' in line and '_profile_L2B.txt GRSl2bgen' in line

    # L2A done: GRSl2bgen only; GRS only: no L2B step, no guard
    make_zarr(tmp_path / 'L2A' / '31TEJ' / '2025/08/01' / L2A / f'{L2A}.zarr')
    line = run.commands(run.plan_level1(images))[0]
    assert ' grs ' not in line and 'GRSl2bgen' in line
    only = batch.Batch(str(tmp_path / 'L2A_nc'), version='2.2.0', profile_dir=False)
    line = only.commands(only.plan_level1(images))[0]
    assert line.startswith('{ grs ') and 'GRSl2bgen' not in line and '/usr/bin/time' not in line
