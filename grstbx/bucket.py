"""
Local S3 bucket of the GRS products: command ``grstbx-s3-bucket``.

The folders of products (and of the Level-1C images) are served in place as an S3 bucket by
``rclone serve s3``, through the shell script ``s3_bucket.sh`` shipped with grstbx (``grstbx/data``):
nothing is copied, the folders are linked into the bucket. The script also writes the key of the server and
the AWS profile (``grstbx-local``) read by s3fs, the batches and the viewers. See the tutorial *Products in
an S3 bucket*.

::

    grstbx-s3-bucket                 # read-only, default layout
    grstbx-s3-bucket --write         # writable: the batches write their products to the bucket
    grstbx-s3-bucket sentinel-2/l1c=/data/L1C sentinel-2/l2a=/data/L2A
"""
import argparse
import os
import shutil
import sys

import importlib_resources

SCRIPT = 's3_bucket.sh'


def script_path():
    """Path of the shell script ``s3_bucket.sh`` shipped with grstbx."""
    return str(importlib_resources.files('grstbx.data').joinpath(SCRIPT))


def environment(write=False, bucket=None, root=None, address=None, profile=None, base=None):
    """Environment of the script: its options as the variables WRITE, BUCKET, S3_ROOT, ADDRESS, PROFILE."""
    env = dict(os.environ if base is None else base)
    env['WRITE'] = '1' if write else '0'
    for name, value in (('BUCKET', bucket), ('S3_ROOT', root), ('ADDRESS', address), ('PROFILE', profile)):
        if value:
            env[name] = str(value)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='grstbx-s3-bucket',
        description='Serve the folders of products as a local S3 bucket (rclone serve s3), in place: the folders '
                    'are linked into the bucket, nothing is copied. Runs in the foreground until Ctrl-C.')
    parser.add_argument('links', nargs='*', metavar='PREFIX=FOLDER',
                        help='folders to link into the bucket, as <prefix>=<folder> (a folder alone is linked by '
                             'its name); default: sentinel-2/l1c, sentinel-2/l2a and sentinel-2/l2b of '
                             '/data/satellite/Sentinel-2')
    parser.add_argument('-w', '--write', action='store_true',
                        help='writable bucket: the batches write their products to it, i.e. into the linked '
                             'folders (default: read-only)')
    parser.add_argument('--bucket', help='bucket name (default: grstbx-s3)')
    parser.add_argument('--root', help='root folder of the buckets served by rclone (default: /data/satellite/s3)')
    parser.add_argument('--address', help='address of the server (default: 127.0.0.1:9000)')
    parser.add_argument('--profile', help='AWS profile written with the key and the endpoint (default: grstbx-local)')
    parser.add_argument('--script', action='store_true',
                        help='print the path of the shell script s3_bucket.sh and exit')
    args = parser.parse_args(argv)

    path = script_path()
    if args.script:
        print(path)
        return 0
    if shutil.which('rclone') is None:
        parser.error('rclone is not installed (https://rclone.org/install, or conda install -c conda-forge rclone)')
    env = environment(args.write, args.bucket, args.root, args.address, args.profile)
    sys.stdout.flush()
    os.execvpe('bash', ['bash', path, *args.links], env)   # the server replaces this process (Ctrl-C stops it)


if __name__ == '__main__':
    sys.exit(main())
