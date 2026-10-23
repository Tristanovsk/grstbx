#!/usr/bin/env bash
# Batch processing of Sentinel-2 L1C images into GRS 3.x L2A products (Zarr stores), one grs
# process per image: the memory is fully released after each image, and several images can be
# processed in parallel. Re-run the script to resume a stopped batch: complete products are skipped.
#
# usage: bash grs_batch_v3.sh
set -uo pipefail

# ---- configuration ------------------------------------------------------------------------------
idir=/data/satellite/Sentinel-2/L1C                     # <idir>/<tile>/YYYY/MM/DD/<image>.SAFE
odir=/data/satellite/Sentinel-2/L2A_zarr                # <odir>/<tile>/YYYY/MM/DD/<name>/<name>.zarr
tiles="31TEJ"                                           # e.g. "31TEJ 31TFJ"
start=2025-08-01 end=2025-08-04                         # date range (inclusive)
cams_template=/data/cams/world/cams_forecast_%Y-%m.nc   # CAMS file of the month (date format of `date`)
dem_template=/data/dem/COP-DEM_GLO-30-DGED_TILE.tif     # DEM of the tile (TILE replaced), optional
resolution=20                                           # m (the default of grs is 60 m)
aot_resolution=480                                      # m, multiple of resolution
max_cloud_cover=0.6                                     # 0-1: skip the images at least that cloudy
scale_aot=1
levname=L2AGRS
suffix=""                                               # "": _V<grs version>
jobs=1                                                  # images processed in parallel (about 13 GB each)
# --------------------------------------------------------------------------------------------------

version=$(grs --version | sed 's/^grs_//')
[ -n "$suffix" ] || suffix="_V$version"
logs=$odir/logs                                         # error.log and profile of each image
summary=$odir/grs_batch_$(date +%Y%m%dT%H%M%S).csv
mkdir -p "$logs"
echo "date,tile,image,status,wall_time,max_rss_kb,product" > "$summary"
export idir odir resolution aot_resolution max_cloud_cover scale_aot levname suffix logs summary \
       cams_template dem_template

# L2A product complete: root metadata and, for each level of the pyramid, the Rrs and mask arrays
zarr_complete() {
    [ -f "$1/zarr.json" ] && [ -d "$1/0" ] || return 1
    for level in "$1"/[0-9]*; do
        [ -f "$level/Rrs/zarr.json" ] && [ -f "$level/mask/zarr.json" ] || return 1
    done
}

process_one() {
    local image=$1 tile day base name outdir product cams dem cc log status
    base=$(basename "$image")
    tile=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$image")")")")")
    day=$(echo "$image" | grep -oE '[0-9]{4}/[0-9]{2}/[0-9]{2}/[^/]+$' | cut -d/ -f1-3)
    name=${base%.SAFE}; name=${name/L1C/$levname}$suffix     # as grs.run.set_ofile
    outdir=$odir/$tile/$day/$name
    product=$outdir/$name.zarr
    cams=$(date -d "${day//\//-}" +"$cams_template")
    dem=${dem_template//TILE/$tile}
    log=$logs/$tile/${base%.SAFE}
    mkdir -p "$log"

    cc=$(grep -oP '<Cloud_Coverage_Assessment>\s*\K[0-9.]+' "$image/MTD_MSIL1C.xml" 2>/dev/null)
    if zarr_complete "$product"; then status=skipped
    elif [ ! -f "$cams" ]; then status=no_cams
    elif [ -n "$cc" ] && awk -v c="$cc" -v m="$max_cloud_cover" 'BEGIN {exit !(c / 100 >= m)}'; then
        status=cloudy
    else
        rm -rf "$product"                                       # partial product of a failed run
        local options=(--cams_file "$cams" --odir "$odir/$tile/$day" --resolution "$resolution"
                       --aot_resolution "$aot_resolution" --scale_aot "$scale_aot" --levname "$levname"
                       --suffix "$suffix")
        [ -f "$dem" ] && options+=(--dem_file "$dem")
        # error.log is written to the current folder: one folder per image
        (cd "$log" && /usr/bin/time -v -o profile.txt grs "$image" "${options[@]}" > grs.out 2>&1)
        # grs exits with 0 even when the processing fails: the product is checked instead
        if zarr_complete "$product"; then status=done; else status=failed; rm -rf "$product"; fi
    fi

    local wall="" rss=""
    if [ "$status" = done ] || [ "$status" = failed ]; then
        wall=$(grep -oP 'Elapsed \(wall clock\) time .*: \K.*' "$log/profile.txt")
        rss=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$log/profile.txt")
    fi
    echo "$day,$tile,$base,$status,$wall,$rss,$product" >> "$summary"
    echo "$(date +%T) $status $base ${wall:+($wall, $((${rss:-0} / 1024)) MB)}"
}
export -f process_one zarr_complete

# images of the tiles and date range, one per line, processed $jobs at a time
for tile in $tiles; do
    find "$idir/$tile" -mindepth 4 -maxdepth 4 -name 'S2*_MSIL1C_*.SAFE' 2>/dev/null
done | sort | awk -v s="$start" -v e="$end" -F/ '{d = $(NF-3) "-" $(NF-2) "-" $(NF-1)} d >= s && d <= e' |
    xargs -r -P "$jobs" -I{} bash -c 'process_one "$1"' _ {}

echo "summary: $summary"
cut -d, -f4 "$summary" | tail -n +2 | sort | uniq -c
