#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1


cd 
module load CCEnv arch/avx512 nixpkgs/16.09
module load python/3.7.0
source sch_bridge/bin/activate
cd $SCRATCH/GP_Sinkhorn/script/
python EB_Dataset.py
