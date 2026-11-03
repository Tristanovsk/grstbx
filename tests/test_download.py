import os
import urllib.parse

import fsspec
import pandas as pd
import pytest

from grstbx import download

NAMES = ['S2A_MSIL1C_20240702T102601_N0510_R108_T31TFJ_20240702T141221.SAFE',
         # two processings of the same acquisition: the last one is kept
         'S2B_MSIL1C_20240707T102559_N0500_R108_T31TFJ_20240707T100000.SAFE',
         'S2B_MSIL1C_20240707T102559_N0510_R108_T31TFJ_20240707T123051.SAFE']


def product(name, cloud):
    sensing = pd.Timestamp(name.split('_')[2])
    return {'Id': name[:8], 'Name': name, 'ContentLength': 100, 'Online': True,
            'S3Path': f'/eodata/Sentinel-2/MSI/L1C/{sensing:%Y/%m/%d}/{name}',
            'ContentDate': {'Start': f'{sensing:%Y-%m-%dT%H:%M:%S}.024000Z'},
            'GeoFootprint': None,
            'Attributes': [{'Name': 'tileId', 'Value': '31TFJ'}, {'Name': 'cloudCover', 'Value': cloud}]}


@pytest.fixture
def catalogue(monkeypatch):
    """Catalogue of two pages; the requested URLs are kept."""
    urls = []
    pages = {None: {'value': [product(NAMES[0], 0.5), product(NAMES[1], 6.)], '@odata.nextLink': 'next'},
             'next': {'value': [product(NAMES[2], 6.7)]}}

    def read(url, timeout=120):
        urls.append(url)
        return pages['next' if url == 'next' else None]

    monkeypatch.setattr(download, '_read_json', read)
    return urls


def test_search(catalogue):
    images = download.search(['31TFJ'], '2024-07-01', '2024-07-31', max_cloud_cover=20)
    assert list(images.name) == [NAMES[0], NAMES[2]] and list(images.cloud_cover) == [0.5, 6.7]
    assert images.time[0] == pd.Timestamp('2024-07-02 10:26:01.024') and images.baseline[1] == '0510'
    query = urllib.parse.unquote(catalogue[0])
    assert "contains(Name,'_T31TFJ_')" in query and 'Value le 20.00' in query and "'S2MSI1C'" in query
    assert 'ContentDate/Start ge 2024-07-01T00:00:00.000Z' in query
    assert 'ContentDate/Start lt 2024-08-01T00:00:00.000Z' in query     # end date included
    assert len(download.search('31TFJ', '2024-07-01', '2024-07-31', last_processing=False)) == 3


def test_search_by_area(catalogue):
    download.search(area=(5.0, 43.4, 5.2, 43.5), start='2024-07-01', end='2024-07-31', level='L2A')
    query = urllib.parse.unquote(catalogue[0])
    assert "Intersects(area=geography'SRID=4326;POLYGON ((5.2 43.4" in query and "'S2MSI2A'" in query
    assert 'contains(Name' not in query
    with pytest.raises(ValueError, match='tiles or an area'):
        download.search(start='2024-07-01', end='2024-07-31')


def test_level1_dir_and_commands(catalogue):
    assert download.level1_dir('/L1C', NAMES[0]) == '/L1C/31TFJ/2024/07/02'
    images = download.search('31TFJ', '2024-07-01', '2024-07-31')
    assert download.s3cmd_commands(images, '/L1C')[0] == (
        f's3cmd -c ~/.s3cfg sync s3://eodata/Sentinel-2/MSI/L1C/2024/07/02/{NAMES[0]} /L1C/31TFJ/2024/07/02/')


def test_storage_options_from_s3cfg(tmp_path):
    cfg = tmp_path / 's3cfg'
    cfg.write_text('[default]\naccess_key = KEY\nsecret_key = SECRET\nhost_base = eodata.dataspace.copernicus.eu\n')
    options = download.cdse_storage_options(s3cfg=str(cfg))
    assert options['key'] == 'KEY' and options['secret'] == 'SECRET'
    assert options['client_kwargs']['endpoint_url'] == download.S3_ENDPOINT
    assert download.cdse_storage_options(profile='cdse') == {'client_kwargs': {'endpoint_url': download.S3_ENDPOINT},
                                                             'profile': 'cdse'}


def test_download(catalogue, tmp_path):
    images = download.search('31TFJ', '2024-07-01', '2024-07-31')
    fs = fsspec.filesystem('memory')
    fs.store.clear()
    first, second = (row.s3_path.lstrip('/') for row in images.itertuples())
    for path in (first, second):
        fs.pipe(f'/{path}/manifest.safe', b'<xml/>')
        fs.pipe(f'/{path}/GRANULE/B02.jp2', b'band')
    fs.rm(f'/{second}/manifest.safe')               # incomplete in the bucket
    images['s3_path'] = ['/' + first, '/' + second]
    summary = download.download(images, str(tmp_path), fs=fs, workers=2)
    assert list(summary.status) == ['downloaded', 'failed'] and 'manifest' in summary.error[1]
    path = summary.path[0]
    assert path == str(tmp_path / '31TFJ/2024/07/02' / NAMES[0])
    assert open(os.path.join(path, 'GRANULE', 'B02.jp2')).read() == 'band'
    assert not os.path.exists(summary.path[1]) and not os.path.exists(summary.path[1] + '.part')
    assert list(download.download(images.head(1), str(tmp_path), fs=fs).status) == ['exists']

    from grstbx.batch import list_images

    assert list(list_images(str(tmp_path), ['31TFJ']).name) == [NAMES[0]]
