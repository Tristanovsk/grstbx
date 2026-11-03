# Download of Sentinel-2 images

The batches of {doc}`batch` read the Level-1C images in the tree `<odir>/<tile>/YYYY/MM/DD/<image>.SAFE`.
{mod}`grstbx.download` fills this tree from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) (CDSE):

- {func}`~grstbx.download.search` finds the images of a period in the public OData catalogue of CDSE, by
  tile or by area (images whose footprint intersects it), with a maximum cloud cover; Level-1C by default,
  or Level-2A (`level='L2A'`). When the archive holds several processings of an acquisition, the last one
  is kept. No account is needed;
- {func}`~grstbx.download.download` copies the images from the S3 interface of CDSE (bucket `eodata`),
  several at a time, into the tree of the batches. Each image is written to `<image>.SAFE.part` and renamed
  at the end: an interrupted download leaves no incomplete image, and the images already on disk
  ({func}`~grstbx.download.on_disk`) are skipped, so that running it again resumes it;
- {func}`~grstbx.download.s3cmd_commands` gives the same download as `s3cmd sync` commands, to run from a
  shell or a scheduler.

```python
from grstbx import download

images = download.search(['31TFJ'], '2024-07-01', '2024-08-31', max_cloud_cover=20)
images = download.search(area=(4.75, 43.25, 5.25, 43.56), start='2024-07-01', end='2024-07-15')
summary = download.download(images, '/data/satellite/Sentinel-2/L1C', workers=4)
```

The download needs s3fs (`pip install "grstbx[remote]"`) and the S3 keys of a CDSE account, created at
<https://eodata-s3keysmanager.dataspace.copernicus.eu>. {func}`~grstbx.download.cdse_storage_options` reads
them from the `~/.s3cfg` file of s3cmd (`access_key`, `secret_key`, as for the `s3cmd` commands), from an
AWS profile (`cdse_storage_options(profile='cdse')`) or from the environment (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`): the notebooks hold no key. A full Level-1C tile is 0.6 to 0.9 GB.

The cloud cover of the search is the one of the catalogue, in % of the tile; {func}`grstbx.batch.list_images`
reads it again from the metadata of each image on disk, as a fraction (0-1).

The first notebook searches by tile and by area (an area at the edge of two tiles), downloads the images
and lists them for the batches. The second one downloads the images of a list of events, one row per site,
tile and period (`Noctiluca_liste_grs.txt`, blooms of Noctiluca on the European coasts). They are rendered
with the outputs of a run on the catalogue in November 2026, without the download itself (`run_download =
False`).

```{toctree}
:maxdepth: 1

download/sentinel2_download
download/sentinel2_download_from_list
```
