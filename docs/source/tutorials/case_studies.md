# Case studies

Applications of grstbx to time series of GRS images. The notebooks are rendered with the outputs
of their last run: the input data are not distributed with grstbx, set the paths of their
configuration cell to run them on your own data. The notebooks are in
[`notebook/case_study/`](https://github.com/Tristanovsk/grstbx/tree/main/notebook/case_study).

- **Clear Lake (California)**: multi-temporal RGB composites over the shaded relief computed for
  the sun position of each acquisition, pixel classification flags and spectra of a cyanobacteria
  bloom.
- **Bagré reservoir (Burkina Faso)**: matchups between five years of Sentinel-2 images and the
  suspended particulate matter (SPM) measured at the Kaporé station, comparison of SPM algorithms
  from the literature with a local band-ratio model, SPM time series and maps.

```{toctree}
:maxdepth: 1

case_studies/grstbx_rgb_dem_multitemp
case_studies/grstbx_l2a_datacube_matchup
```
