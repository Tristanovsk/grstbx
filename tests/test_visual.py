import numpy as np
import pandas as pd
import xarray as xr

from grstbx.visual import _LRUCache


def test_lru_cache_bounded():
    cache = _LRUCache(max_bytes=3 * 800)
    for key in range(5):
        cache.put(key, np.zeros(100))             # 800 bytes each
    assert list(cache) == [2, 3, 4]
    assert cache.nbytes == 2400
    cache.get(2)                                   # 2 becomes the most recent
    cache.put(5, (np.zeros(100), np.zeros(0)))
    assert list(cache) == [4, 2, 5]


def polygon_viewer():
    from shapely.geometry import box

    from conftest import l2b_product
    from grstbx.visual import Multiscale, ViewParam

    times = pd.to_datetime(['2025-01-03', '2025-01-08', '2025-01-13'])
    cube = xr.concat([l2b_product(t, chla=float(i + 1), size=40)[['Chla_OC2nasa']] for i, t in enumerate(times)], 'time')
    cube['Chla_OC2nasa'] = cube.Chla_OC2nasa.where(cube.time != times[2])     # no valid pixel at the last date
    data = Multiscale.from_xarray(cube.rio.write_crs('EPSG:32631'))
    viewer = ViewParam(data, params=['Chla_OC2nasa'], reproject=False)
    x0, y0 = float(cube.x[20]), float(cube.y[30])
    return viewer, box(x0, y0, float(cube.x[35]), float(cube.y[5]))


def set_polygon(viewer, polygon):
    x, y = polygon.exterior.xy
    viewer.aoi_stream.event(data={'xs': [list(x)], 'ys': [list(y)]})


def test_polygon_statistics():
    from shapely.geometry import Polygon, box

    viewer, polygon = polygon_viewer()
    stats = viewer.data.polygon_statistics(1, 'Chla_OC2nasa', polygon)
    assert stats['median'] == stats['q25'] == stats['q75'] == 2 and stats['valid'] == 1
    x0, y0, x1, y1 = polygon.bounds
    triangle = Polygon([(x0, y0), (x1, y0), (x0, y1)])
    assert viewer.data.polygon_statistics(1, 'Chla_OC2nasa', triangle)['pixels'] < stats['pixels']
    assert viewer.data.polygon_statistics(1, 'Chla_OC2nasa', box(0, 0, 10, 10)) is None


def test_polygon_timeseries():
    viewer, polygon = polygon_viewer()
    set_polygon(viewer, polygon)
    table = viewer.polygon_timeseries('Chla_OC2nasa')
    assert table['median'].iloc[:2].tolist() == [1, 2] and np.isnan(table['median'].iloc[2])


def test_polygon_series_of_the_viewer():
    import asyncio

    import panel as pn

    viewer, polygon = polygon_viewer()
    panel = viewer._polygon_panel(pn.widgets.Select(value='Chla_OC2nasa', options=['Chla_OC2nasa']))
    status = panel.select(pn.pane.Markdown)[0]
    widgets = (status, panel.select(pn.indicators.Progress)[0], panel.select(pn.widgets.Button)[1],
               panel.select(pn.pane.HoloViews)[0])
    asyncio.run(viewer._polygon_series('Chla_OC2nasa', widgets))
    assert 'Draw a polygon' in status.object
    set_polygon(viewer, polygon)
    asyncio.run(viewer._polygon_series('Chla_OC2nasa', widgets))
    assert '2 dates plotted, 1 with less than' in status.object
    assert len(viewer.polygon_table) == 3 and widgets[3].object is not None
    import holoviews as hv

    assert hv.render(widgets[3].object).yaxis[0].axis_label == 'Chla_OC2nasa'


def test_polygons_read_together_give_the_same_statistics(monkeypatch):
    from shapely.affinity import translate
    from shapely.geometry import box

    viewer, polygon = polygon_viewer()
    data = viewer.data
    x0, y0, x1, y1 = polygon.bounds
    near = translate(polygon.buffer(-(x1 - x0) / 4), xoff=(x1 - x0) / 8)       # inside the box of the first
    ex0, ey0, _, _ = data.bounds(1)
    res = data.resolution(1)
    far = box(ex0 + res, ey0 + res, ex0 + 5 * res, ey0 + 5 * res)    # small, in the opposite corner
    polygons = {0: polygon, 1: near, 2: far}
    windows = [[members for _, members in data._window_groups([(i, p, None) for i, p in polygons.items()])]]
    assert [[m[0] for m in members] for members in windows[0]] == [[0, 1], [2]]   # the distant one apart

    alone = {index: data.polygon_statistics(1, 'Chla_OC2nasa', p) for index, p in polygons.items()}
    data._cache.clear()
    data._cache.nbytes = 0
    loads = []
    original = xr.DataArray.load
    monkeypatch.setattr(xr.DataArray, 'load', lambda self, **kw: loads.append(self.shape) or original(self, **kw))
    together = data.polygons_statistics(1, 'Chla_OC2nasa', polygons)
    assert together == alone and len(loads) == 2
    # masks kept for the other dates on the same grid, statistics kept for this date
    assert sum(key[0] == 'mask' for key in data._cache) == 3
    data.polygons_statistics(1, 'Chla_OC2nasa', polygons)
    assert len(loads) == 2


def test_polygon_series_reads_the_dates_in_parallel():
    import asyncio
    import threading
    import time

    import panel as pn

    viewer, polygon = polygon_viewer()
    set_polygon(viewer, polygon)
    threads, original = set(), viewer.data.polygons_statistics

    def slow(*args, **kwargs):
        threads.add(threading.get_ident())
        time.sleep(0.2)
        return original(*args, **kwargs)

    viewer.data.polygons_statistics = slow
    panel = viewer._polygon_panel(pn.widgets.Select(value='Chla_OC2nasa', options=['Chla_OC2nasa']))
    widgets = (panel.select(pn.pane.Markdown)[0], panel.select(pn.indicators.Progress)[0],
               panel.select(pn.widgets.Button)[1], panel.select(pn.pane.HoloViews)[0])
    t0 = time.time()
    asyncio.run(viewer._polygon_series('Chla_OC2nasa', widgets))
    assert time.time() - t0 < 0.5 and len(threads) == 3     # 3 dates of 0.2 s at the same time
    assert viewer.polygon_table['median'].iloc[:2].tolist() == [1, 2]


def test_pixel_timeseries_reads_the_dates_in_parallel():
    import threading
    import time

    viewer, polygon = polygon_viewer()
    data = viewer.data
    threads, original = set(), data.pixel

    def slow(*args, **kwargs):
        threads.add(threading.get_ident())
        time.sleep(0.2)
        return original(*args, **kwargs)

    data.pixel = slow
    x, y = polygon.centroid.x, polygon.centroid.y
    t0 = time.time()
    series = data.timeseries('Chla_OC2nasa', x, y)
    assert time.time() - t0 < 0.5 and len(threads) == 3     # 3 dates of 0.2 s at the same time
    assert series.index.is_monotonic_increasing and series.iloc[:2].tolist() == [1, 2]
    assert np.isnan(series.iloc[2])
    assert data.timeseries('Chla_OC2nasa', x, y, indexes=[2, 0]).index.tolist() == [data.times[2], data.times[0]]
    assert data.timeseries('Chla_OC2nasa', 0, 0).empty      # outside the image


def test_minimum_share_of_valid_pixels_of_the_polygon_series():
    import asyncio

    import panel as pn
    from shapely.geometry import box

    viewer, polygon = polygon_viewer()
    cube_x0, _, _, _ = viewer.data.bounds(0)
    x0, y0, x1, y1 = polygon.bounds
    # half of the polygon on the left quarter without data: 50 % of valid pixels
    half = box(cube_x0 + 20 * 5, y0, cube_x0 + 20 * 15, y1)
    set_polygon(viewer, half)
    panel = viewer._polygon_panel(pn.widgets.Select(value='Chla_OC2nasa', options=['Chla_OC2nasa']))
    status = panel.select(pn.pane.Markdown)[0]
    widgets = (status, panel.select(pn.indicators.Progress)[0], panel.select(pn.widgets.Button)[1],
               panel.select(pn.pane.HoloViews)[0])
    min_valid = panel.select(pn.widgets.FloatInput)[0]
    assert min_valid.value == 10
    asyncio.run(viewer._polygon_series('Chla_OC2nasa', widgets))
    assert viewer.polygon_table.valid.round(2).tolist()[:2] == [0.5, 0.5]
    assert '2 dates plotted, 1 with less than 10 %' in status.object

    min_valid.value = 60          # the plot is updated without reading the dates again
    assert '0 dates plotted, 3 with less than 60 %' in status.object
    assert viewer.polygon_min_valid == 0.6 and viewer.polygon_table['median'].isna().all()
    min_valid.value = 2.5
    assert '2 dates plotted, 1 with less than 2.5 %' in status.object
    assert viewer.polygon_table['median'].iloc[:2].tolist() == [1, 2]


def test_first_view_on_the_valid_pixels():
    from grstbx.visual import Multiscale, ViewParam

    size = 1000
    x, y = 700010. + 20 * np.arange(size), 4900000. - 10 - 20 * np.arange(size)
    values = np.full((2, size, size), np.nan)
    values[0, 100:110, 600:620] = 1.             # a small lake; no valid pixel at the second date
    values[0, [5, 990, 995], [990, 3, 500]] = 1.  # and a few scattered pixels far from it
    cube = xr.Dataset({'chl': (('time', 'y', 'x'), values)},
                      coords={'time': pd.to_datetime(['2025-01-03', '2025-01-08']), 'x': x, 'y': y})
    viewer = ViewParam(Multiscale.from_xarray(cube.rio.write_crs('EPSG:32631')), params=['chl'])
    (xmin, xmax), (ymin, ymax) = viewer._extent(0)
    # centred on the lake, at least valid_min_pixels wide, with the margin
    assert xmin < x[600] < x[619] < xmax and ymin < y[109] < y[100] < ymax
    assert np.isclose(xmax - xmin, 20 * viewer.valid_min_pixels * (1 + 2 * viewer.valid_margin))
    assert xmax - xmin < 0.2 * 20 * size
    # no valid pixel: the whole image
    (xmin, xmax), (ymin, ymax) = viewer._extent(1)
    assert (xmin, xmax, ymin, ymax) == (x[0] - 10, x[-1] + 10, y[-1] - 10, y[0] + 10)


def test_state_of_the_map():
    viewer, _ = polygon_viewer()
    viewer.visu()
    date, cmap = viewer.widgets['date'], viewer.widgets['colormap']
    state = viewer.state()
    assert state['date'] == viewer.datetimes[0].strftime('%Y-%m-%dT%H:%M:%S') and 'view' not in state
    changes = []
    viewer.on_state_change(changes.append)
    other = next(name for name in cmap.options if name != cmap.value_name)
    viewer.set_state({'date': '2025-01-08', 'layer': 'Chla_OC2nasa', 'cmap': other, 'log': '1', 'range': '0.5,3',
                      'opacity': '0.4', 'reverse': 'true', 'basemap': 'no such basemap', 'layer ': 'x'})
    assert date.value == 1 and cmap.value_name == other and viewer.widgets['log'].value
    assert viewer.widgets['range'].value == (0.5, 3) and viewer.widgets['opacity'].value == 0.4
    assert changes and changes[-1]['date'].startswith('2025-01-08') and changes[-1]['range'] == '0.5,3'
    viewer.set_state({'date': 'not a date', 'layer': 'unknown', 'range': 'a,b'})     # left out
    assert date.value == 1 and viewer.widgets['range'].value == (0.5, 3)
    # the view follows the map
    x0, x1 = viewer.data.bounds(0)[0], viewer.data.bounds(0)[2]
    y0, y1 = viewer.data.bounds(0)[1], viewer.data.bounds(0)[3]
    viewer._range_stream.event(x_range=(x0, x1), y_range=(y0, y1))
    assert len(changes[-1]['view'].split(',')) == 4


def test_dates_from_the_player_and_the_time_series():
    import panel as pn

    viewer, _ = polygon_viewer()
    layout = viewer.visu()
    player = next(w for w in layout.select(pn.widgets.Player))
    player.value = 2
    assert viewer.widgets['date'].value == 2
    viewer.widgets['date'].value = 1
    assert player.value == 1
    # a click near a date of a time series plot (ms since 1970 on the Bokeh time axes)
    assert viewer._nearest_date(pd.Timestamp('2025-01-12').value / 1e6) == 2
    assert viewer._nearest_date(np.datetime64('2025-01-04')) == 0


def test_first_view_from_the_previous_map():
    import pyproj

    from conftest import l2b_product
    from grstbx.visual import Multiscale, ViewParam

    cube = l2b_product(pd.Timestamp('2025-01-03'), size=40)[['Chla_OC2nasa']].expand_dims(time=[pd.Timestamp('2025-01-03')])
    data = Multiscale.from_xarray(cube.rio.write_crs('EPSG:32631'))
    to_lonlat = pyproj.Transformer.from_crs(32631, 4326, always_xy=True)
    xmin, ymin, xmax, ymax = data.bounds(0)
    inside = to_lonlat.transform_bounds(xmin, ymin, (xmin + xmax) / 2, (ymin + ymax) / 2)
    (vx0, vx1), (vy0, vy1) = ViewParam(data, view=inside)._extent(0)
    tol = 0.05 * (xmax - xmin)       # the lon/lat box of the view is a little larger
    assert abs(vx0 - xmin) < tol and abs(vx1 - (xmin + xmax) / 2) < tol
    assert abs(vy0 - ymin) < tol and abs(vy1 - (ymin + ymax) / 2) < tol
    # a view far from the image: its valid pixels
    assert ViewParam(data, view=(-60, -10, -59, -9))._extent(0) == ViewParam(data)._extent(0)


def test_click_on_the_time_series_selects_the_date():
    import holoviews as hv
    import panel as pn

    hv.extension('bokeh')
    viewer, polygon = polygon_viewer()
    layout = viewer.visu()
    plot = hv.renderer('bokeh').get_plot(list(layout.select(pn.pane.HoloViews))[1].object)
    taps = [c for sub in plot.subplots.values() for c in sub.callbacks if type(c).__name__ == 'TapCallback']
    assert taps, 'the time series plot has no Tap callback'
    for stream in taps[0].streams:
        stream.event(x=np.datetime64('2025-01-12'), y=1.)
    assert viewer.widgets['date'].value == 2
    # the plot of the polygons too
    set_polygon(viewer, polygon)
    table = viewer.polygon_timeseries('Chla_OC2nasa')
    plot = hv.renderer('bokeh').get_plot(viewer._polygon_plot(table, 'Chla'))
    assert any(type(c).__name__ == 'TapCallback' for sub in plot.subplots.values() for c in sub.callbacks)
