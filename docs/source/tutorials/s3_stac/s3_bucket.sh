#!/usr/bin/env bash
# Local S3 bucket of the GRS products, served in place by `rclone serve s3`: nothing is copied, the folders
# of the products are linked into the bucket. Read-only, on the local machine only, with a generated key.
# The server runs in the foreground until it is stopped (Ctrl-C): keep this terminal open, or use tmux.
#
#   bash s3_bucket.sh                                       # default layout below
#   bash s3_bucket.sh sentinel-2/l2a=/data/L2A sentinel-2/l2b=/data/L2B   # other folders: <prefix>=<folder>
#   BUCKET=my-bucket bash s3_bucket.sh /data/L2A /data/L2B  # a folder alone is linked by its name
#   WRITE=1 bash s3_bucket.sh                               # writable: batches write their products
#                                                           # to the bucket, i.e. into the linked folders
#
# Default layout, one prefix per sensor and level (whatever the format of the products):
#   s3://grstbx-s3/sentinel-2/l2a/    -> /data/satellite/Sentinel-2/L2A_zarr   GRS L2A, <tile>/YYYY/MM/DD/<name>/
#   s3://grstbx-s3/sentinel-2/l2b/    -> /data/satellite/Sentinel-2/L2B_zarr   GRSl2bgen L2B, same tree
#   s3://grstbx-s3/stac/              STAC catalogue of the products (grstbx.stac.build_catalog)
#
# It creates (and never overwrites an existing key):
#   $S3_ROOT/$BUCKET/          the bucket: links to the folders of products, and stac/ for the catalogue
#   ~/.config/grstbx/s3.env    key of the server (private)
#   ~/.aws/credentials, config profile $PROFILE, with the endpoint: read by s3fs and the AWS tools
set -euo pipefail

BUCKET=${BUCKET:-grstbx-s3}
S3_ROOT=${S3_ROOT:-/data/satellite/s3}
ADDRESS=${ADDRESS:-127.0.0.1:9000}
PROFILE=${PROFILE:-grstbx-local}
WRITE=${WRITE:-0}
LINKS=("$@")
[ ${#LINKS[@]} -gt 0 ] || LINKS=(sentinel-2/l2a=/data/satellite/Sentinel-2/L2A_zarr
                                 sentinel-2/l2b=/data/satellite/Sentinel-2/L2B_zarr)

mkdir -p "$S3_ROOT/$BUCKET/stac"
for link in "${LINKS[@]}"; do
    if [[ $link == *=* ]]; then prefix=${link%%=*} folder=${link#*=}; else prefix=$(basename "$link") folder=$link; fi
    mkdir -p "$(dirname "$S3_ROOT/$BUCKET/$prefix")"
    ln -sfn "$folder" "$S3_ROOT/$BUCKET/$prefix"
done

# key of the server, given to rclone by the environment (a command line can be read by the other users)
env_file=$HOME/.config/grstbx/s3.env
if [ ! -f "$env_file" ]; then
    mkdir -p "$(dirname "$env_file")"
    (umask 077; printf 'RCLONE_AUTH_KEY=%s,%s\n' "$(openssl rand -hex 10)" "$(openssl rand -hex 20)" > "$env_file")
fi
key=$(sed -n 's/^RCLONE_AUTH_KEY=\(.*\),.*/\1/p' "$env_file")
secret=$(sed -n 's/^RCLONE_AUTH_KEY=.*,\(.*\)/\1/p' "$env_file")

mkdir -p "$HOME/.aws"
(umask 077; touch "$HOME/.aws/credentials" "$HOME/.aws/config")
if ! grep -q "^\[$PROFILE\]" "$HOME/.aws/credentials"; then
    printf '[%s]\naws_access_key_id = %s\naws_secret_access_key = %s\n' "$PROFILE" "$key" "$secret" \
        >> "$HOME/.aws/credentials"
fi
if ! grep -q "^\[profile $PROFILE\]" "$HOME/.aws/config"; then
    printf '[profile %s]\nregion = us-east-1\nendpoint_url = http://%s\n' "$PROFILE" "$ADDRESS" >> "$HOME/.aws/config"
fi

if [ "$WRITE" = 1 ]; then mode=() access=writable; else mode=(--read-only) access=read-only; fi
echo "s3://$BUCKET served on http://$ADDRESS ($access), profile $PROFILE"
# rclone reads this variable as CSV: the quotes keep the comma of the pair
export RCLONE_AUTH_KEY="\"$key,$secret\""
# --etag-hash '': no MD5 of the files at each request; --dir-cache-time: a new catalogue is seen within a minute
exec rclone serve s3 "$S3_ROOT" --addr "$ADDRESS" --copy-links "${mode[@]}" --etag-hash '' --dir-cache-time 1m
