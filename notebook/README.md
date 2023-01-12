# easy access from CNES HPC
## PBS job
qsub -I -X -l walltime=05:00:00 -l select=1:ncpus=8:mem=20000MB:os=rh7

## conda env
module load conda
conda activate grstbx

## call jupyter-lab
jupyter-lab notebook

### Then replace 'lab' with 'tree' in your browser
http://localhost:8888/lab --> http://localhost:8888/tree

# Access through the CNES jupyterhub:
https://jupyterhub.cnes.fr 