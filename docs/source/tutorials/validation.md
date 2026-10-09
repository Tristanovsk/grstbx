# Validation with in situ data

These notebooks compare GRS products with in situ measurements in the Berre lagoon (France),
where the HYPERNETS station `BEFR` is installed. They are rendered with the outputs of their last
run: the input data are not distributed with grstbx, set the paths of their configuration cell to
run them on your own data. The notebooks are in
[`notebook/validation/`](https://github.com/Tristanovsk/grstbx/tree/master/notebook/validation).

- **L2A**: remote-sensing reflectance against the HYPERNETS hyperspectral radiometer, with the
  normalization of the in situ data to nadir viewing.
- **L2B**: chlorophyll-a and suspended particulate matter against the samples of the GIPREB
  monitoring network.

```{toctree}
:maxdepth: 1

validation/grstbx_l2a_hypernets_matchup
validation/grstbx_l2b_hypernets_matchup
```
