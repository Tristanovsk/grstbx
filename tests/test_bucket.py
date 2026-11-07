"""grstbx-s3-bucket: the shell script shipped with grstbx, run with a fake rclone and a temporary HOME."""
import os
import subprocess

import pytest

from grstbx import bucket


def test_script_shipped():
    path = bucket.script_path()
    assert os.path.isfile(path)
    assert 'rclone serve s3' in open(path).read()


def test_environment():
    env = bucket.environment(write=True, bucket='b', root='/r', address='h:1', profile='p', base={})
    assert env == {'WRITE': '1', 'BUCKET': 'b', 'S3_ROOT': '/r', 'ADDRESS': 'h:1', 'PROFILE': 'p'}
    assert bucket.environment(base={'BUCKET': 'kept'}) == {'WRITE': '0', 'BUCKET': 'kept'}


def test_print_script(capsys):
    assert bucket.main(['--script']) == 0
    assert capsys.readouterr().out.strip() == bucket.script_path()


@pytest.mark.parametrize('write', [False, True])
def test_script_runs(tmp_path, write):
    """Links, key, AWS profile and rclone options, without any server."""
    fake = tmp_path / 'bin'
    fake.mkdir()
    (fake / 'rclone').write_text('#!/usr/bin/env bash\necho "rclone $*"\n')
    (fake / 'rclone').chmod(0o755)
    products = tmp_path / 'L2A'
    products.mkdir()
    home, root = tmp_path / 'home', tmp_path / 's3'
    home.mkdir()
    env = bucket.environment(write=write, bucket='test-bucket', root=root, address='127.0.0.1:9999', profile='test',
                             base={'PATH': f'{fake}:{os.environ["PATH"]}', 'HOME': str(home)})
    out = subprocess.run(['bash', bucket.script_path(), f'sentinel-2/l2a={products}'], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert os.readlink(root / 'test-bucket' / 'sentinel-2' / 'l2a') == str(products)
    assert (root / 'test-bucket' / 'stac').is_dir()
    assert '[profile test]' in (home / '.aws' / 'config').read_text()
    assert 'endpoint_url = http://127.0.0.1:9999' in (home / '.aws' / 'config').read_text()
    assert '[test]' in (home / '.aws' / 'credentials').read_text()
    line = out.strip().splitlines()[-1]
    assert line.startswith(f'rclone serve s3 {root} --addr 127.0.0.1:9999')
    assert ('--read-only' in line) is not write
