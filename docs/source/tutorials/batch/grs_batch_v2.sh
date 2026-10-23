#!/usr/bin/env bash
# Batch processing of Sentinel-2 L1C images into GRS 2.x L2A products (NetCDF files), one grs
# process per image: the memory is fully released after each image, and several images can be
# processed in parallel. Re-run the script to resume a stopped batch: complete products are skipped.
#
# usage: bash grs_batch_v2.sh
set -uo pipefail

# ---- configuration ------------------------------------------------------------------------------
idir=/data/satellite/Sentinel-2/L1C                     # <idir>/<tile>/YYYY/MM/DD/<image>.SAFE
odir=/data/satellite/Sentinel-2/L2A_nc                  # <odir>/<tile>/YYYY/MM/DD/<name>/<name>.nc
tiles="31TEJ"                                           # e.g. "31TEJ 31TFJ"
start=2025-08-01 end=2025-08-04                         # date range (inclusive)
cams_template=/data/cams/world/cams_forecast_%Y-%m.nc   # CAMS file of the month (date format of `date`)
dem_template=/data/dem/COP-DEM_GLO-30-DGED_TILE.tif     # DEM of the tile (TILE replaced), optional
resolution=20                                           # m (the default of grs is 60 m)
max_cloud_cover=0.6                                     # 0-1: skip the images at least that cloudy
scale_aot=1
levname=L2AGRS
suffix=""                                               # "": _V<grs version>
jobs=1                                                  # images processed in parallel (about 32 GB each)
# --------------------------------------------------------------------------------------------------

# memory freed by the processing threads reused from one image to the next (set by grs 3.x itself)
export MALLOC_ARENA_MAX=2
version=$(grs --version | sed 's/^grs_//')
[ -n "$suffix" ] || suffix="_V$version"
logs=$odir/logs                                         # error.log and profile of each image
summary=$odir/grs_batch_$(date +%Y%m%dT%H%M%S).csv
mkdir -p "$logs"
echo "date,tile,image,status,wall_time,max_rss_kb,product" > "$summary"
export idir odir resolution max_cloud_cover scale_aot levname suffix logs summary \
       cams_template dem_template

# NetCDF file complete: it exists and, if ncdump is installed, its header can be read
nc_complete() {
    [ -s "$1" ] || return 1
    command -v ncdump > /dev/null || return 0
    ncdump -h "$1" > /dev/null 2>&1
}

# L2A product complete: main file with the Rrs and mask variables, and ancillary file
l2a_complete() {
    nc_complete "$1.nc" && nc_complete "$1_anc.nc" || return 1
    command -v ncdump > /dev/null || return 0
    ncdump -h "$1.nc" | grep -q ' Rrs(' && ncdump -h "$1.nc" | grep -q ' mask('
}

process_one() {
    local image=$1 tile day base name outdir product cams dem cc log status
    base=$(basename "$image")
    tile=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$image")")")")")
    day=$(echo "$image" | grep -oE '[0-9]{4}/[0-9]{2}/[0-9]{2}/[^/]+$' | cut -d/ -f1-3)
    name=${base%.SAFE}; name=${name/L1C/$levname}$suffix     # as procutils.misc.set_ofile
    outdir=$odir/$tile/$day/$name
    product=$outdir/$name                                       # <product>.nc and <product>_anc.nc
    cams=$(date -d "${day//\//-}" +"$cams_template")
    dem=${dem_template//TILE/$tile}
    log=$logs/$tile/${base%.SAFE}
    mkdir -p "$log"

    cc=$(grep -oP '<Cloud_Coverage_Assessment>\s*\K[0-9.]+' "$image/MTD_MSIL1C.xml" 2>/dev/null)
    if l2a_complete "$product"; then status=skipped
    elif [ ! -f "$cams" ]; then status=no_cams
    elif [ -n "$cc" ] && awk -v c="$cc" -v m="$max_cloud_cover" 'BEGIN {exit !(c / 100 >= m)}'; then
        status=cloudy
    else
        rm -f "$product.nc" "${product}_anc.nc"                 # partial product of a failed run
        local options=(--cams_file "$cams" --odir "$odir/$tile/$day" --resolution "$resolution"
                       --scale_aot "$scale_aot" --levname "$levname" --suffix "$suffix")
        [ -f "$dem" ] && options+=(--dem_file "$dem")
        # error.log is written to the current folder: one folder per image
        (cd "$log" && /usr/bin/time -v -o profile.txt grs "$image" "${options[@]}" > grs.out 2>&1)
        # grs exits with 0 even when the processing fails: the product and error.log are checked instead
        if l2a_complete "$product" && ! grep -q 'Fatal error' "$log/error.log" 2>/dev/null; then
            status=done
        else
            status=failed
            rm -f "$product.nc" "${product}_anc.nc"
        fi
    fi

    local wall="" rss=""
    if [ "$status" = done ] || [ "$status" = failed ]; then
        wall=$(grep -oP 'Elapsed \(wall clock\) time .*: \K.*' "$log/profile.txt")
        rss=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$log/profile.txt")
    fi
    echo "$day,$tile,$base,$status,$wall,$rss,$product.nc" >> "$summary"
    echo "$(date +%T) $status $base ${wall:+($wall, $((${rss:-0} / 1024)) MB)}"
}
export -f process_one nc_complete l2a_complete

# images of the tiles and date range, one per line, processed $jobs at a time
for tile in $tiles; do
    find "$idir/$tile" -mindepth 4 -maxdepth 4 -name 'S2*_MSIL1C_*.SAFE' 2>/dev/null
done | sort | awk -v s="$start" -v e="$end" -F/ '{d = $(NF-3) "-" $(NF-2) "-" $(NF-1)} d >= s && d <= e' |
    xargs -r -P "$jobs" -I{} bash -c 'process_one "$1"' _ {}

echo "summary: $summary"
cut -d, -f4 "$summary" | tail -n +2 | sort | uniq -c
