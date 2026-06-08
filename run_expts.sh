#!/bin/bash

#SBATCH --account=torch_pr_154_courant
#SBATCH --gres=gpu:h200:4                # requests 1 V100 GPU
#SBATCH --cpus-per-task=4                # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=diffusion
#SBATCH --output=slurm_outputs/cell_diff.out

module purge
cd /scratch/rg5218/SiT/

SINGULARITY_FILE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY_FILE=overlay-25GB-500K.ext3:ro

singularity exec --nv --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-0.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-100.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-10.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-110.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-120.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-130.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-140.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-150.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-160.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-170.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-180.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-190.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-200.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-20.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-210.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-220.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-230.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-30.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-40.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-50.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-60.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-70.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-80.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22-90.sqf:ro --overlay /projects/work/public/ml-datasets/funk22/sqfs/funk22_lmdb_shuffled.sqf:ro --overlay $OVERLAY_FILE $SINGULARITY_FILE /bin/bash -c "source /ext3/env.sh; ./train.sh"