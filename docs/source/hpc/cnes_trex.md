# CNES TREX

Notebooks to run the GRS chain and to export its products on **TREX**, the HPC cluster of CNES, following the
TREX guide of the L1C-L2B chain (Overleaf *L1C-L2B processing chain on TREX*):

- the Sentinel-2 L1C images are read from the **CNES datalake** (S3 object storage), with the temporary
  credentials of a read-only role, renewed during the batch, and the restoration of the images archived on
  tier 3;
- the products are written to `/work/datalake/watcal/Sentinel-2/{L1C,L2A,L2B}/<tile>/YYYY/MM/DD/`, with the
  daily CAMS files and the COP-DEM GLO-30 tiles of `/work/datalake`;
- the notebooks run interactively (JupyterHub of TREX) or as a **SLURM job array** written by the notebook
  itself and run by python on the compute nodes (one task per tile and month for the batch, per year for the
  export). The scripts set `GRS_BATCH_JOB=1` / `EXPORT_BATCH_JOB=1`, so that the notebook knows it runs as a
  batch task: JupyterHub of TREX also runs the Jupyter server in a SLURM job, so `SLURM_JOB_ID` cannot tell.

| notebook | environment | what it does |
|---|---|---|
| {doc}`cnes_trex/grs_datalake_trex_batch` | conda `GRSv2` | L1C → L2A (GRS 2.2) → L2B (GRSl2bgen) from the datalake |
| {doc}`cnes_trex/grs_datalake_trex_batch_v3` | venv `grs_v3_venv` | the same batch with GRS 3.x |
| {doc}`cnes_trex/grstbx_trex_export_gdrive` | venv `grs_v3_venv` | export of an area and a period (L1C, L2A, L2B) to Google Drive: exports on the compute nodes, upload from a login node through the CNES proxy |

The datalake steps (credentials, tier-3 restorations) run in the conda env `s3-env` of the computing centre,
called by the notebooks. Set the ARN of the datalake role (`DATALAKE_ROLE_ARN`, see
`/softs/tools/datalake/roles_list.sh`) and the SLURM account of your project (`-A`) before running them. The
notebooks are rendered without outputs.

The batch outside TREX is described in {doc}`../tutorials/batch`, the export of the local bucket in
{doc}`../tutorials/export`.

```{toctree}
:maxdepth: 1

cnes_trex/grs_datalake_trex_batch
cnes_trex/grs_datalake_trex_batch_v3
cnes_trex/grstbx_trex_export_gdrive
```
