#!/usr/bin/env bash
# Local S3 bucket of the GRS products, served in place by `rclone serve s3`: nothing is copied, the folders
# of the products are linked into the bucket. Read-only, on the local machine only, with a generated key.
# The server runs in the foreground until it is stopped (Ctrl-C): keep this terminal open, or use tmux.
#
#   bash s3_bucket.sh                                       # default folders below
#   BUCKET=my-bucket bash s3_bucket.sh /data/L2A /data/L2B  # other folders of products
#
# It creates (and never overwrites an existing key):
#   $S3_ROOT/$BUCKET/          the bucket: links to the folders of products, and stac/ for the catalogue
#   ~/.config/grstbx/s3.env    key of the server (private)
#   ~/.aws/credentials, config profile $PROFILE, with the endpoint: read by s3fs and the AWS tools
set -euo pipefail

BUCKET=${BUCKET:-grstbx-sentinel2}
S3_ROOT=${S3_ROOT:-/data/satellite/s3}
ADDRESS=${ADDRESS:-127.0.0.1:9000}
PROFILE=${PROFILE:-grstbx-local}
FOLDERS=("$@")
[ ${#FOLDERS[@]} -gt 0 ] || FOLDERS=(/data/satellite/Sentinel-2/L2A_zarr /data/satellite/Sentinel-2/L2B_zarr)

mkdir -p "$S3_ROOT/$BUCKET/stac"
for folder in "${FOLDERS[@]}"; do
    ln -sfn "$folder" "$S3_ROOT/$BUCKET/$(basename "$folder")"
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

echo "s3://$BUCKET served on http://$ADDRESS, profile $PROFILE"
# rclone reads this variable as CSV: the quotes keep the comma of the pair
export RCLONE_AUTH_KEY="\"$key,$secret\""
# --etag-hash '': no MD5 of the files at each request; --dir-cache-time: a new catalogue is seen within a minute
exec rclone serve s3 "$S3_ROOT" --addr "$ADDRESS" --copy-links --read-only --etag-hash '' --dir-cache-time 1m
